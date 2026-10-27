"""
scenario.py

Time-varying scenario model for oracle-free resiliency label generation.

A ScenarioState captures the health of a service at a given moment.
States evolve via a Markov chain — the model is never told the scenario;
it must infer it from observable signals (rolling fail-rate, consecutive
failures, retry success rate, latency trend, CB state, etc.).

The Markov chain encodes the realistic lifecycle:
  HEALTHY → TRANSIENT → DEGRADED → SERVICE_DOWN → RECOVERING → HEALTHY

Optimal policies (used only by the counterfactual labeler, never by the model):
  HEALTHY:   retry=1, medium timeout, CB closed, no fallback
  TRANSIENT: retry=2, long timeout,   CB closed, cached fallback
  DEGRADED:  retry=1, long timeout,   CB half-open, cached fallback
  DOWN:      retry=0, short timeout,  CB open, alternate fallback
  RECOVERING:retry=1, medium timeout, CB half-open, cached fallback
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List

import numpy as np


# ── Scenario states ────────────────────────────────────────────────────────────

class ScenarioState(IntEnum):
    HEALTHY           = 0
    TRANSIENT_FAILURE = 1
    DEGRADED          = 2
    SERVICE_DOWN      = 3
    RECOVERING        = 4


SCENARIO_NAMES = ["HEALTHY", "TRANSIENT", "DEGRADED", "DOWN", "RECOVERING"]

# ── Markov transition matrix ───────────────────────────────────────────────────

_TRANSITION: dict[ScenarioState, dict[ScenarioState, float]] = {
    ScenarioState.HEALTHY: {
        ScenarioState.HEALTHY:           0.88,
        ScenarioState.TRANSIENT_FAILURE: 0.12,
    },
    ScenarioState.TRANSIENT_FAILURE: {
        ScenarioState.HEALTHY:           0.30,
        ScenarioState.TRANSIENT_FAILURE: 0.48,
        ScenarioState.DEGRADED:          0.22,
    },
    ScenarioState.DEGRADED: {
        ScenarioState.TRANSIENT_FAILURE: 0.25,
        ScenarioState.DEGRADED:          0.45,
        ScenarioState.SERVICE_DOWN:      0.30,
    },
    ScenarioState.SERVICE_DOWN: {
        ScenarioState.SERVICE_DOWN:      0.72,
        ScenarioState.RECOVERING:        0.28,
    },
    ScenarioState.RECOVERING: {
        ScenarioState.RECOVERING:        0.38,
        ScenarioState.HEALTHY:           0.52,
        ScenarioState.SERVICE_DOWN:      0.10,
    },
}

# ── Observable signal parameters per scenario ──────────────────────────────────
_OBS_PARAMS: dict[ScenarioState, dict] = {
    ScenarioState.HEALTHY: dict(
        base_fail_rate=0.05, fail_noise=0.03, max_consec=1,
        retry_succ_rate=0.93, latency_dir=-0.5,
    ),
    ScenarioState.TRANSIENT_FAILURE: dict(
        base_fail_rate=0.30, fail_noise=0.10, max_consec=4,
        retry_succ_rate=0.62, latency_dir=0.1,
    ),
    ScenarioState.DEGRADED: dict(
        base_fail_rate=0.62, fail_noise=0.10, max_consec=9,
        retry_succ_rate=0.28, latency_dir=0.6,
    ),
    ScenarioState.SERVICE_DOWN: dict(
        base_fail_rate=0.97, fail_noise=0.02, max_consec=25,
        retry_succ_rate=0.02, latency_dir=1.0,
    ),
    ScenarioState.RECOVERING: dict(
        base_fail_rate=0.40, fail_noise=0.15, max_consec=6,
        retry_succ_rate=0.48, latency_dir=-0.4,
    ),
}

# ── Optimal policy per scenario (ground truth — kept out of the model) ─────────
OPTIMAL_POLICY: dict[ScenarioState, tuple] = {
    ScenarioState.HEALTHY:           (1, 1, 0, 0),
    ScenarioState.TRANSIENT_FAILURE: (2, 2, 0, 2),
    ScenarioState.DEGRADED:          (1, 2, 1, 2),
    ScenarioState.SERVICE_DOWN:      (0, 0, 2, 1),
    ScenarioState.RECOVERING:        (1, 1, 1, 2),
}

# ── Circuit-breaker state machine ──────────────────────────────────────────────

CB_CLOSED    = 0
CB_HALF_OPEN = 1
CB_OPEN      = 2

class CBStateMachine:
    """
    Tracks circuit-breaker state as a feedback signal.

    The CB trips open when the failure rate over the last WINDOW steps
    exceeds OPEN_THRESHOLD.  After COOLDOWN steps open it probes with
    one request (HALF_OPEN); success closes it, failure re-opens it.

    This gives the model a closed-loop signal: "the system already knows
    things are bad and has opened the CB" rather than having to infer it
    from symptoms alone.
    """
    WINDOW          = 15   # failure-rate window (steps)
    OPEN_THRESHOLD  = 0.60 # failure rate that trips the CB open
    COOLDOWN        = 5    # steps in OPEN before moving to HALF_OPEN

    def __init__(self) -> None:
        self.state:    int       = CB_CLOSED
        self.duration: int       = 0
        self._history: list[int] = []   # 1=fail, 0=success per step

    @property
    def failures_last_window(self) -> int:
        return sum(self._history)

    def step(self, failed: bool) -> None:
        """Advance CB state given the outcome of the current request."""
        self._history.append(int(failed))
        if len(self._history) > self.WINDOW:
            self._history.pop(0)

        fail_rate = self.failures_last_window / max(len(self._history), 1)

        if self.state == CB_CLOSED:
            if fail_rate >= self.OPEN_THRESHOLD:
                self.state    = CB_OPEN
                self.duration = 0
            else:
                self.duration += 1

        elif self.state == CB_OPEN:
            if self.duration >= self.COOLDOWN:
                self.state    = CB_HALF_OPEN
                self.duration = 0
            else:
                self.duration += 1

        elif self.state == CB_HALF_OPEN:
            # One probe: success → close, failure → reopen
            if not failed:
                self.state    = CB_CLOSED
                self.duration = 0
            else:
                self.state    = CB_OPEN
                self.duration = 0

    def snapshot(self) -> tuple[int, int, int]:
        """Return (cb_state, cb_duration, failures_last_window) before this step."""
        return self.state, self.duration, self.failures_last_window


# ── Scenario observation (what the model receives) ─────────────────────────────

SCENARIO_OBS_DIM = 9   # 6 original + 3 CB-state feedback features

@dataclass
class ScenarioObservation:
    """
    Observable signals derived from scenario state — model input only.

    The first 6 features are symptom signals (fail rate, latency, etc.).
    The last 3 are CB state feedback: what the breaker is currently doing,
    how long it has been there, and how many failures it has seen recently.
    Together they give the model both open-loop symptoms AND closed-loop
    system state, enabling it to learn CB transition logic from data.
    """
    scenario:           ScenarioState   # ground truth (NOT given to the model)
    t:                  int
    rolling_fail_rate:  float           # [0, 1]
    consec_failures:    int             # raw count
    retry_succ_rate:    float           # [0, 1]
    latency_trend:      float           # [-1, +1]
    time_since_success: int             # steps since last observed success
    scenario_duration:  int             # steps in current scenario

    # CB feedback (3 new features)
    cb_state:           int   = 0       # 0=closed, 1=half_open, 2=open
    cb_duration:        int   = 0       # steps in current CB state
    failures_last_15:   int   = 0       # failures in last WINDOW steps

    def to_feature_vector(self) -> np.ndarray:
        """9-dimensional temporal feature vector, all normalised to [0, 1]."""
        return np.array([
            # original 6 symptom features
            self.rolling_fail_rate,
            min(self.consec_failures / 25.0, 1.0),
            self.retry_succ_rate,
            (self.latency_trend + 1.0) / 2.0,           # [-1,1] → [0,1]
            min(self.time_since_success / 25.0, 1.0),
            min(self.scenario_duration  / 25.0, 1.0),
            # 3 CB feedback features
            self.cb_state / 2.0,                         # {0,1,2} → {0, 0.5, 1}
            min(self.cb_duration / 20.0, 1.0),
            min(self.failures_last_15 / CBStateMachine.WINDOW, 1.0),
        ], dtype=np.float32)


# ── Markov transitions + observation sampling ──────────────────────────────────

def next_scenario(current: ScenarioState, rng: random.Random) -> ScenarioState:
    probs   = _TRANSITION[current]
    states  = list(probs.keys())
    weights = [probs[s] for s in states]
    return rng.choices(states, weights=weights, k=1)[0]


def sample_observation(
    scenario:           ScenarioState,
    t:                  int,
    scenario_duration:  int,
    time_since_success: int,
    rng:                random.Random,
    cb:                 CBStateMachine | None = None,
) -> ScenarioObservation:
    p = _OBS_PARAMS[scenario]
    fail_rate = float(np.clip(p["base_fail_rate"] + rng.gauss(0, p["fail_noise"]), 0.0, 1.0))
    consec    = rng.randint(0, p["max_consec"])
    rsr       = float(np.clip(p["retry_succ_rate"] + rng.gauss(0, 0.05), 0.0, 1.0))
    lat_trend = float(np.clip(p["latency_dir"] + rng.gauss(0, 0.2), -1.0, 1.0))

    # CB feedback: snapshot before stepping, then advance with sampled outcome
    if cb is not None:
        cb_state, cb_dur, cb_fails = cb.snapshot()
        failed = rng.random() < fail_rate
        cb.step(failed)
    else:
        cb_state, cb_dur, cb_fails = 0, 0, 0

    return ScenarioObservation(
        scenario=scenario,
        t=t,
        rolling_fail_rate=fail_rate,
        consec_failures=consec,
        retry_succ_rate=rsr,
        latency_trend=lat_trend,
        time_since_success=time_since_success,
        scenario_duration=scenario_duration,
        cb_state=cb_state,
        cb_duration=cb_dur,
        failures_last_15=cb_fails,
    )


# ── Scenario stream generation ─────────────────────────────────────────────────

def generate_scenario_stream(
    n_steps: int,
    seed:    int              = 0,
    initial: ScenarioState    = ScenarioState.HEALTHY,
) -> List[ScenarioObservation]:
    """
    Generate n_steps consecutive scenario observations following the Markov chain.

    Each step advances the CBStateMachine so the CB state reflects the
    cumulative failure history — giving the model closed-loop feedback
    rather than just point-in-time symptom signals.
    """
    rng             = random.Random(seed)
    stream:         list[ScenarioObservation] = []
    state           = initial
    duration        = 0
    time_since_suc  = 0
    cb              = CBStateMachine()

    forced_shift_at = min(n_steps // 3, 15)

    for t in range(n_steps):
        if t == forced_shift_at and state == ScenarioState.HEALTHY:
            state    = ScenarioState.TRANSIENT_FAILURE
            duration = 0
        elif t > 0:
            new_state = next_scenario(state, rng)
            if new_state != state:
                duration = 0
                state    = new_state

        obs = sample_observation(state, t, duration, time_since_suc, rng, cb=cb)

        time_since_suc = 0 if obs.rolling_fail_rate < 0.5 else time_since_suc + 1
        duration      += 1
        stream.append(obs)

    return stream
