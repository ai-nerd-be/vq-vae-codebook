"""
label_oracle.py

Rule-based oracle that assigns the "ground truth" 4-dimensional resiliency
policy for a given failure event.  This simulates expert annotation and is
the training signal for both the MLP and VQ-VAE models.

Policy dimensions
─────────────────
  retry          : int  ∈ {0, 1, 2, 3}
  timeout_tier   : int  ∈ {0=short, 1=medium, 2=long}
  circuit_breaker: int  ∈ {0=closed, 1=half_open, 2=open}
  fallback       : int  ∈ {0=none, 1=alternate_node, 2=cached_response, 3=dead_letter}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

# ── Policy dimension labels (for display) ────────────────────────────────────

RETRY_LABELS         = ["0", "1", "2", "3"]
TIMEOUT_LABELS       = ["short(5s)", "medium(30s)", "long(120s)"]
CIRCUIT_BREAKER_LABELS = ["closed", "half_open", "open"]
FALLBACK_LABELS      = ["none", "alternate_node", "cached_response", "dead_letter"]

NUM_RETRY_CLASSES = 4
NUM_TIMEOUT_CLASSES = 3
NUM_CB_CLASSES = 3
NUM_FALLBACK_CLASSES = 4

POLICY_DIMS = (NUM_RETRY_CLASSES, NUM_TIMEOUT_CLASSES, NUM_CB_CLASSES, NUM_FALLBACK_CLASSES)

# ── Policy validity rules ─────────────────────────────────────────────────────
# A policy is contradictory if any of these hold:
#   CB open  AND retry > 0          (open circuit should not send requests)
#   CB half_open AND retry > 1      (half-open = one probe request only)
#   CB open  AND fallback = none    (open circuit with no fallback = dead end)

def is_valid_policy(retry: int, timeout: int, cb: int, fallback: int) -> bool:
    if cb == 2 and retry > 0:          # open + retry
        return False
    if cb == 1 and retry > 1:          # half-open + aggressive retry
        return False
    if cb == 2 and fallback == 0:      # open + no fallback
        return False
    return True


@dataclass(frozen=True)
class PolicyVector:
    retry:           int   # 0..3
    timeout_tier:    int   # 0=short, 1=medium, 2=long
    circuit_breaker: int   # 0=closed, 1=half_open, 2=open
    fallback:        int   # 0=none, 1=alternate, 2=cached, 3=dead_letter

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.retry, self.timeout_tier, self.circuit_breaker, self.fallback)

    def is_valid(self) -> bool:
        return is_valid_policy(*self.as_tuple())

    def describe(self) -> str:
        return (
            f"retry={RETRY_LABELS[self.retry]}, "
            f"timeout={TIMEOUT_LABELS[self.timeout_tier]}, "
            f"CB={CIRCUIT_BREAKER_LABELS[self.circuit_breaker]}, "
            f"fallback={FALLBACK_LABELS[self.fallback]}"
        )


# ── Oracle rules ─────────────────────────────────────────────────────────────

# (failure_mode, node_type_name) → PolicyVector
# More specific rules override less specific ones (checked first).
_SPECIFIC_RULES: Dict[Tuple[str, str], PolicyVector] = {
    # LLM rate limits: be patient, use cache if available
    ("rate_limit", "lmChatGoogleGemini"): PolicyVector(2, 2, 1, 2),
    # Auth failure on external email/calendar: never retry, open CB, dead-letter
    ("auth_error", "gmailTrigger"):        PolicyVector(0, 0, 2, 3),
    ("auth_error", "gmail"):               PolicyVector(0, 0, 2, 3),
    ("auth_error", "googleCalendar"):      PolicyVector(0, 0, 2, 3),
    # Webhook crypto mismatch: reject immediately, open CB
    ("crypto_error", "crypto"):            PolicyVector(0, 0, 2, 3),
    ("crypto_error", "webhook"):           PolicyVector(0, 0, 2, 3),
    # LLM timeout: long wait, half-open CB, fall back to cached answer
    ("timeout", "lmChatGoogleGemini"):     PolicyVector(2, 2, 1, 2),
    # Calendar dependency down: try alternate path
    ("dependency_down", "googleCalendar"): PolicyVector(1, 2, 2, 1),
}

# failure_mode → default PolicyVector (used when no specific rule matches)
_DEFAULT_RULES: Dict[str, PolicyVector] = {
    "no_failure":      PolicyVector(0, 0, 0, 0),  # nothing to do
    "timeout":         PolicyVector(2, 2, 1, 2),  # retry with long timeout, cache
    "rate_limit":      PolicyVector(1, 1, 1, 2),  # one retry, medium, cache
    "auth_error":      PolicyVector(0, 0, 2, 3),  # never retry, open, dead-letter
    "dependency_down": PolicyVector(1, 2, 2, 1),  # one retry, long, open CB, alternate
    "bad_input":       PolicyVector(0, 0, 0, 3),  # don't retry bad data, dead-letter
    "crypto_error":    PolicyVector(0, 0, 2, 3),  # wrong secret, open CB, dead-letter
}

_FALLBACK_POLICY = PolicyVector(1, 1, 0, 0)  # safe default if key missing


def get_policy(
    failure_mode: str,
    node_type_name: str,
    retry_count: int = 0,
    is_external: bool = False,
) -> PolicyVector:
    """
    Return the oracle policy for a failure event.

    retry_count  : how many times this node has already been retried
    is_external  : whether the node calls an external service
    """
    # 1. Specific override
    key = (failure_mode, node_type_name)
    if key in _SPECIFIC_RULES:
        base = _SPECIFIC_RULES[key]
    else:
        base = _DEFAULT_RULES.get(failure_mode, _FALLBACK_POLICY)

    # 2. Saturate retries: if already retried, reduce further retries
    adjusted_retry = max(0, base.retry - retry_count)

    # 3. If retry count is high and still failing, open the CB
    cb = base.circuit_breaker
    if retry_count >= 2 and failure_mode not in ("no_failure",):
        cb = max(cb, 1)  # at least half-open after 2 failures
    if retry_count >= 3:
        cb = 2  # open after 3 failures

    # 4. Non-external nodes rarely need long timeouts
    timeout = base.timeout_tier
    if not is_external and timeout == 2:
        timeout = 1

    # 5. Ensure validity: open CB must not have retry > 0
    if cb == 2:
        adjusted_retry = 0
        if base.fallback == 0:
            fallback = 3  # force dead-letter if fallback was none
        else:
            fallback = base.fallback
    else:
        fallback = base.fallback

    return PolicyVector(
        retry=adjusted_retry,
        timeout_tier=timeout,
        circuit_breaker=cb,
        fallback=fallback,
    )


if __name__ == "__main__":
    scenarios = [
        ("timeout",         "lmChatGoogleGemini", 0, True),
        ("rate_limit",      "lmChatGoogleGemini", 1, True),
        ("auth_error",      "gmail",              0, True),
        ("crypto_error",    "crypto",             0, False),
        ("timeout",         "code",               2, False),
        ("dependency_down", "googleCalendar",     0, True),
        ("bad_input",       "textClassifier",     0, False),
        ("no_failure",      "if",                 0, False),
    ]
    print(f"{'Failure':<18} {'Node':<22} {'Retries':>7}  Policy")
    print("-" * 75)
    for fm, nt, rc, ext in scenarios:
        p = get_policy(fm, nt, rc, ext)
        valid = "OK" if p.is_valid() else "INVALID"
        print(f"{fm:<18} {nt:<22} {rc:>7}  {p.describe()}  [{valid}]")
