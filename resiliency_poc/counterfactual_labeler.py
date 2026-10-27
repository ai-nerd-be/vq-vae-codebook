"""
counterfactual_labeler.py

Oracle-free label generation via counterfactual simulation.

For each (workflow_context, scenario_observation), we try N_CANDIDATE
pre-defined policies in simulation and score their expected outcomes.
The policy with the best score becomes the ground-truth label.

No hand-written rules.  No oracle.  The training signal comes entirely
from observed simulation outcomes conditioned on the scenario state.

Scoring function (per-event):
  score = completion * 1.0
        - wasted_retries * 0.40
        - latency_units  * 0.05

The scenario state controls the simulator's success probabilities through
_SCENARIO_SUCCESS_PROB — making the optimal policy context-dependent by
construction without ever telling the model what the scenario is.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .scenario import ScenarioState
from .model.policy_heads import PolicyPrediction

# ── Failure category taxonomy ─────────────────────────────────────────────────
# 4 high-level categories that mirror expert reasoning about distributed-system
# failures.  The hierarchical decoder first predicts this category, then the
# specific recovery pattern — matching how engineers escalate through options.
#
#   NOMINAL     — system healthy; no recovery action needed
#   TRANSIENT   — intermittent fault; retry / exponential backoff resolves it
#   PERSISTENT  — sustained outage; CB open + fallback required
#   TRANSACTION — partial progress in long-running workflow; compensation/rollback

FAILURE_CATEGORY_NAMES = ["NOMINAL", "TRANSIENT", "PERSISTENT", "TRANSACTION"]
N_FAILURE_CATEGORIES   = len(FAILURE_CATEGORY_NAMES)   # 4

# Map from scenario → category (not given to model; used only for labels)
SCENARIO_TO_CATEGORY: dict[ScenarioState, int] = {
    ScenarioState.HEALTHY:           0,   # NOMINAL
    ScenarioState.TRANSIENT_FAILURE: 1,   # TRANSIENT
    ScenarioState.DEGRADED:          2,   # PERSISTENT
    ScenarioState.SERVICE_DOWN:      2,   # PERSISTENT
    ScenarioState.RECOVERING:        3,   # TRANSACTION (partial rollback / re-establish)
}


# ── Candidate policy pool ─────────────────────────────────────────────────────
# Each tuple: (retry, timeout_tier, circuit_breaker, fallback)
# Covers the full trade-off space without enumerating all 144 combinations.

CANDIDATE_POLICIES: List[PolicyPrediction] = [
    # CB open — skip straight to fallback (correct for DOWN)
    PolicyPrediction(retry=0, timeout=0, cb=2, fallback=1),  # alternate
    PolicyPrediction(retry=0, timeout=0, cb=2, fallback=2),  # cached
    PolicyPrediction(retry=0, timeout=0, cb=2, fallback=3),  # dead_letter
    # Half-open probe — correct for RECOVERING / DEGRADED
    PolicyPrediction(retry=1, timeout=1, cb=1, fallback=2),
    PolicyPrediction(retry=1, timeout=2, cb=1, fallback=2),
    # Aggressive retry — correct for TRANSIENT
    PolicyPrediction(retry=2, timeout=2, cb=0, fallback=2),
    PolicyPrediction(retry=3, timeout=2, cb=0, fallback=2),
    # Light retry — correct for HEALTHY
    PolicyPrediction(retry=1, timeout=1, cb=0, fallback=0),
    PolicyPrediction(retry=1, timeout=1, cb=0, fallback=2),
    # Threshold baseline (worst for DOWN — deliberately included as negative example)
    PolicyPrediction(retry=3, timeout=1, cb=0, fallback=0),
]

N_SIM_PER_POLICY = 150  # Monte-Carlo samples per candidate policy (higher = less label noise)


# ── Scenario-aware success probabilities ──────────────────────────────────────
# The simulator's base success probs are modulated by scenario state.
# These replace the static probs in simulator.py for temporal traces.

_SCENARIO_BASE_SUCCESS: dict[ScenarioState, float] = {
    ScenarioState.HEALTHY:           0.95,
    ScenarioState.TRANSIENT_FAILURE: 0.42,
    ScenarioState.DEGRADED:          0.05,
    ScenarioState.SERVICE_DOWN:      0.02,
    ScenarioState.RECOVERING:        0.38,
}

_TIMEOUT_BONUS  = {0: -0.15, 1: 0.00, 2: +0.18}
_FALLBACK_PROBS = {0: 0.00, 1: 0.75, 2: 0.90, 3: 1.00}


def _single_attempt(
    scenario:     ScenarioState,
    timeout_tier: int,
    attempt_idx:  int,
    rng:          random.Random,
) -> bool:
    base    = _SCENARIO_BASE_SUCCESS[scenario]
    bonus   = _TIMEOUT_BONUS.get(timeout_tier, 0.0)
    backoff = 0.07 * attempt_idx   # exponential-backoff benefit
    prob    = min(1.0, base + bonus + backoff)
    return rng.random() < prob


def _simulate_policy(
    policy:   PolicyPrediction,
    scenario: ScenarioState,
    rng:      random.Random,
) -> Tuple[float, float, float]:
    """
    Run a single Monte-Carlo trial of policy in scenario.
    Returns (completed, wasted_retries, latency_units).
    """
    # CB open: skip to fallback immediately
    if policy.cb == 2:
        fb_ok = rng.random() < _FALLBACK_PROBS[policy.fallback]
        return float(fb_ok), 0.0, 3.0

    max_attempts = min(policy.retry + 1, 2 if policy.cb == 1 else policy.retry + 1)

    succeeded = False
    attempts  = 0
    for a in range(max_attempts):
        attempts += 1
        if _single_attempt(scenario, policy.timeout, a, rng):
            succeeded = True
            break

    wasted = float(attempts > 1 and not succeeded)

    if not succeeded:
        fb_ok  = rng.random() < _FALLBACK_PROBS[policy.fallback]
        latency = float(attempts + (2 if fb_ok else 0))
        return float(fb_ok), wasted, latency

    return 1.0, wasted, float(attempts)


def _score_policy(
    policy:   PolicyPrediction,
    scenario: ScenarioState,
    rng:      random.Random,
) -> float:
    """
    Average score of policy over N_SIM_PER_POLICY Monte-Carlo trials.
    score = completion - 0.40*wasted_retries - 0.05*latency
    """
    scores = []
    for _ in range(N_SIM_PER_POLICY):
        completed, wasted, latency = _simulate_policy(policy, scenario, rng)
        scores.append(completed - 0.40 * wasted - 0.05 * latency)
    return float(np.mean(scores))


# ── Public API ─────────────────────────────────────────────────────────────────

@dataclass
class CounterfactualLabel:
    retry:    int
    timeout:  int
    cb:       int
    fallback: int
    category: int           # failure category (NOMINAL/TRANSIENT/PERSISTENT/TRANSACTION)
    scenario: ScenarioState
    score:    float         # score of the winning policy (diagnostic use only)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.retry, self.timeout, self.cb, self.fallback)


def label_from_simulation(
    scenario: ScenarioState,
    seed:     int = 0,
) -> CounterfactualLabel:
    """
    Score all CANDIDATE_POLICIES in simulation under the given scenario,
    return the policy with the highest expected score as the label.

    The caller provides the scenario (inferred from the stream); the model
    never sees the scenario label — only the observable feature vector.
    """
    rng    = random.Random(seed)
    scores = [_score_policy(p, scenario, rng) for p in CANDIDATE_POLICIES]
    best_i = int(np.argmax(scores))
    best_p = CANDIDATE_POLICIES[best_i]
    return CounterfactualLabel(
        retry=best_p.retry,    timeout=best_p.timeout,
        cb=best_p.cb,          fallback=best_p.fallback,
        category=SCENARIO_TO_CATEGORY[scenario],
        scenario=scenario,     score=scores[best_i],
    )
