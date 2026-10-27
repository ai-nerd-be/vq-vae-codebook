"""
simulator.py

Mock workflow execution simulator.

Given a list of FailureEvents (a trace) and a policy function, runs each
failure event through the policy, simulates the probabilistic outcome, and
returns an ExecutionResult.

Three policy modes
──────────────────
  threshold_policy   : static heuristic — always retry ≤3, medium timeout,
                       closed CB, no fallback.
  mlp_policy         : wraps a trained MLPResiliencyModel.
  vq_policy          : wraps a trained VQResiliencyModel.

Outcome model
─────────────
  For each failure event the simulator attempts (retry+1) executions.
  Each attempt succeeds with probability p(failure_mode, timeout, attempt_num).
  If all attempts fail the fallback is applied.

  Fallback outcome probabilities:
    none          → failure  (workflow dies)
    alternate_node → 0.75
    cached_response → 0.90
    dead_letter   → always "handled" (partial completion)

  A retry is counted as *wasted* if it was performed and all attempts failed.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

from .trace_generator import FailureEvent
from .model.policy_heads import PolicyPrediction

# ── Success probabilities per failure mode ────────────────────────────────────

# p(single attempt succeeds | failure_mode, timeout_tier, attempt_index)
# timeout_tier: 0=short, 1=medium, 2=long

_BASE_SUCCESS = {
    "no_failure":      1.00,
    "auth_error":      0.04,   # credentials don't self-heal
    "crypto_error":    0.04,
    "bad_input":       0.08,   # data issue rarely self-heals
    "dependency_down": 0.30,
    "rate_limit":      0.45,   # improves with each attempt (backoff)
    "timeout":         0.55,
}

_TIMEOUT_BONUS = {0: -0.15, 1: 0.0, 2: +0.20}   # short / medium / long

def _attempt_success_prob(
    failure_mode: str,
    timeout_tier: int,
    attempt: int,
    rng: random.Random,
) -> bool:
    base = _BASE_SUCCESS.get(failure_mode, 0.5)
    bonus = _TIMEOUT_BONUS.get(timeout_tier, 0.0)
    # Backoff effect: each retry slightly improves rate_limit / timeout odds
    backoff = 0.08 * attempt if failure_mode in ("rate_limit", "timeout") else 0.0
    prob = min(1.0, base + bonus + backoff)
    return rng.random() < prob


_FALLBACK_SUCCESS = {
    0: 0.00,   # none — fail
    1: 0.75,   # alternate_node
    2: 0.90,   # cached_response
    3: 1.00,   # dead_letter (always "handled", marks partial)
}


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class EventOutcome:
    node_name:      str
    failure_mode:   str
    policy:         PolicyPrediction
    attempts:       int      # total attempts made (0 if CB immediately opened)
    succeeded:      bool
    via_fallback:   bool
    was_wasted:     bool     # retried but all failed before fallback
    is_partial:     bool     # completed via dead_letter (partial success)
    latency_units:  int      # proxy for time (1 unit per attempt, 3 for CB open)


@dataclass
class ExecutionResult:
    workflow_name:    str
    events:           List[EventOutcome]
    completed:        bool    # True if no event fully failed
    partial:          bool    # True if any event used dead_letter
    wasted_retries:   int
    total_attempts:   int
    total_latency:    int
    invalid_policies: int    # policy contradictions predicted

    @property
    def success_rate(self) -> float:
        return 1.0 if self.completed else 0.0


# ── Threshold baseline policy ─────────────────────────────────────────────────

def threshold_policy(event: FailureEvent) -> PolicyPrediction:
    """
    Hardcoded heuristic: always retry up to 3 times, medium timeout,
    CB always closed, no fallback.  Context-blind.
    """
    if event.failure_mode == "no_failure":
        return PolicyPrediction(0, 1, 0, 0)
    return PolicyPrediction(retry=3, timeout=1, cb=0, fallback=0)


# ── Simulator ─────────────────────────────────────────────────────────────────

class WorkflowSimulator:
    def __init__(self, seed: int = 99):
        self.rng = random.Random(seed)

    def _run_event(self, event: FailureEvent, policy: PolicyPrediction) -> EventOutcome:
        if event.failure_mode == "no_failure":
            return EventOutcome(
                node_name=event.node_name, failure_mode=event.failure_mode,
                policy=policy, attempts=1, succeeded=True,
                via_fallback=False, was_wasted=False, is_partial=False,
                latency_units=1,
            )

        invalid_policy = not policy.is_valid()

        # Circuit breaker open: skip to fallback immediately
        if policy.cb == 2:
            fb_prob = _FALLBACK_SUCCESS[policy.fallback]
            fb_ok   = self.rng.random() < fb_prob
            return EventOutcome(
                node_name=event.node_name, failure_mode=event.failure_mode,
                policy=policy, attempts=0, succeeded=fb_ok,
                via_fallback=True, was_wasted=False,
                is_partial=(policy.fallback == 3 and fb_ok),
                latency_units=3,  # CB resolution overhead
            )

        # Circuit breaker half-open: cap at 1 probe attempt
        max_attempts = min(policy.retry + 1, 2 if policy.cb == 1 else policy.retry + 1)

        succeeded    = False
        attempts     = 0
        for a in range(max_attempts):
            attempts += 1
            if _attempt_success_prob(event.failure_mode, policy.timeout, a, self.rng):
                succeeded = True
                break

        was_wasted = (attempts > 1 and not succeeded)

        if not succeeded:
            fb_prob = _FALLBACK_SUCCESS[policy.fallback]
            succeeded = self.rng.random() < fb_prob
            is_partial = (policy.fallback == 3 and succeeded)
            via_fallback = True
        else:
            is_partial   = False
            via_fallback = False

        return EventOutcome(
            node_name=event.node_name, failure_mode=event.failure_mode,
            policy=policy, attempts=attempts, succeeded=succeeded,
            via_fallback=via_fallback, was_wasted=was_wasted,
            is_partial=is_partial,
            latency_units=attempts + (2 if via_fallback else 0),
        )

    def run_trace(
        self,
        events: List[FailureEvent],
        policy_fn: Callable[[FailureEvent], PolicyPrediction],
    ) -> ExecutionResult:
        outcomes:         List[EventOutcome] = []
        wasted_retries:   int = 0
        total_attempts:   int = 0
        total_latency:    int = 0
        invalid_policies: int = 0
        completed:        bool = True
        partial:          bool = False

        for event in events:
            policy  = policy_fn(event)
            outcome = self._run_event(event, policy)
            outcomes.append(outcome)

            if not policy.is_valid():
                invalid_policies += 1
            if outcome.was_wasted:
                wasted_retries += outcome.attempts - 1
            total_attempts += outcome.attempts
            total_latency  += outcome.latency_units
            if not outcome.succeeded:
                completed = False
                break   # workflow dies at first unrecovered failure
            if outcome.is_partial:
                partial = True

        return ExecutionResult(
            workflow_name=events[0].workflow_name if events else "",
            events=outcomes,
            completed=completed,
            partial=partial,
            wasted_retries=wasted_retries,
            total_attempts=total_attempts,
            total_latency=total_latency,
            invalid_policies=invalid_policies,
        )
