"""
eval_trace_replay.py  --  Resilience policy evaluation on real DSB X-Trace data

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m eval_trace_replay

What this does
──────────────
Replays the real DeathStarBench X-Trace event stream chronologically and asks:
"For each real failure event, would this resilience policy have recovered it?"

Three policy variants are compared:
  1. No-policy   — no retry, no CB, no fallback (baseline)
  2. LLM-policy  — Ollama (llama3.1:latest) annotations per service
  3. ORS-policy  — DSB-trained ORS-VQ model annotations per service

Recovery logic (for each failure event at service S, timestamp t)
──────────────────────────────────────────────────────────────────
Step 1  CB pre-check
  The CB state machine for S is driven by real failure history up to t.
  If the recommended CB policy is aggressive and the CB is already OPEN
  at time t, the call would have been short-circuited → FALLBACK fired
  immediately (saving the retry overhead).

Step 2  Retry look-ahead
  If the CB did not intercept, check whether the service recovered
  within the retry budget:
    timeout_ms  = {short=5 000, medium=30 000, long=120 000}[timeout_tier]
    retry_count = {none=0, 1=1, 3=3, 5=5}[retry_cls]
    window_ms   = timeout_ms × retry_count
  Look ahead in the REAL trace for service S in (t, t+window_ms].
  If any real event for S in that window is a SUCCESS → RETRY_RECOVERED.
  If all real events in that window are also failures → RETRY_EXHAUSTED.

  Honest assumption: the next real call to S in the window is treated as a
  proxy for "would a retry have found the service healthy?"  This is an
  approximation; the next call is from a different request, not the same
  request retrying.  But it is grounded in what the service was actually
  doing at that time.

Step 3  Fallback
  If retries exhausted and fallback_type > 0 → FALLBACK_RECOVERED.
  If no fallback → LOST.

CB aggressiveness (how the CB machines are configured per policy)
─────────────────────────────────────────────────────────────────
  ORS cb_policy=0 (closed)    → CB disabled, threshold=1.01 (never trips)
  ORS cb_policy=1 (half_open) → Standard CB, OPEN_THRESHOLD=0.60, COOLDOWN=5
  ORS cb_policy=2 (open)      → Aggressive CB, OPEN_THRESHOLD=0.30, COOLDOWN=2
"""

from __future__ import annotations
import json, re, time, urllib.request, urllib.error
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from resiliency_poc.dsb_loader   import DSBLoader, ServiceEvent, FOLDER_SCENARIO
from resiliency_poc.scenario     import ScenarioState, SCENARIO_NAMES, CB_CLOSED, CB_HALF_OPEN, CB_OPEN
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM

from resiliency_poc.dsb_workflow_loader import load_dsb_workflows
from hardening.engine  import WorkflowHardeningEngine, _TYPE_TO_SCENARIO
from hardening.models  import ResilienceAnnotation

# ── Patch DSB service names into engine type→scenario map ─────────────────────
from resiliency_poc.scenario import ScenarioState as _SC
_TYPE_TO_SCENARIO.update({
    "nginx-thrift":             _SC.HEALTHY,
    "ComposePostService":       _SC.TRANSIENT_FAILURE,
    "PostStorageService":       _SC.DEGRADED,
    "HomeTimelineService":      _SC.DEGRADED,
    "UserTimelineService":      _SC.DEGRADED,
    "WriteHomeTimelineService": _SC.DEGRADED,
    "UserService":              _SC.DEGRADED,
    "SocialGraphService":       _SC.TRANSIENT_FAILURE,
    "UniqueIdService":          _SC.HEALTHY,
    "TextService":              _SC.HEALTHY,
    "MediaService":             _SC.TRANSIENT_FAILURE,
    "UserMentionService":       _SC.HEALTHY,
    "UrlShortenService":        _SC.HEALTHY,
})

# ── Config ─────────────────────────────────────────────────────────────────────

SEED          = 42
CODEBOOK_SIZE = 16
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.1:latest"

# Timeout tiers in milliseconds
TIMEOUT_MS    = {0: 5_000, 1: 30_000, 2: 120_000}
# Retry count from retry class index
RETRY_COUNT   = {0: 0, 1: 1, 2: 3, 3: 5}

# Fallback names for display
FALLBACK_NAMES = {0: "none", 1: "alternate", 2: "cached", 3: "dead_letter"}


# ── Outcome categories ─────────────────────────────────────────────────────────

CB_INTERCEPTED    = "CB_intercepted"   # CB was open → short-circuited to fallback
RETRY_RECOVERED   = "retry_recovered"  # retry look-ahead found a success
RETRY_FALLBACK    = "retry_fallback"   # retries exhausted → fallback fired
LOST              = "lost"             # retries exhausted, no fallback


# ── Per-service policy ─────────────────────────────────────────────────────────

@dataclass
class ServicePolicy:
    """Resilience policy for one service."""
    service:      str
    retry:        int   # 0=none, 1=1, 2=3, 3=5
    timeout_tier: int   # 0=5s, 1=30s, 2=120s
    cb_policy:    int   # 0=closed, 1=half_open, 2=open
    fallback_type: int  # 0=none, 1=alternate, 2=cached, 3=dead_letter

    def describe(self) -> str:
        r = RETRY_COUNT[self.retry]
        t = TIMEOUT_MS[self.timeout_tier] // 1000
        cb = ["closed", "half_open", "open"][self.cb_policy]
        fb = FALLBACK_NAMES[self.fallback_type]
        parts = [f"CB({cb})"]
        if r > 0:
            parts.append(f"retry({r}×{t}s)")
        if self.fallback_type > 0:
            parts.append(f"fallback({fb})")
        return " + ".join(parts) if parts else "no-policy"


# ── CB state machine (configurable aggressiveness) ─────────────────────────────

class TraceCBMachine:
    """
    CB machine driven by real trace events.
    Aggressiveness is controlled by open_threshold and cooldown:
      cb_policy=0 (closed)    → threshold=1.01, never trips
      cb_policy=1 (half_open) → threshold=0.60, cooldown=5
      cb_policy=2 (open)      → threshold=0.30, cooldown=2
    """
    WINDOW = 15  # event window for failure rate

    _THRESHOLDS = {0: 1.01, 1: 0.60, 2: 0.30}
    _COOLDOWNS  = {0:  999, 1:    2, 2:    5}  # open=longer cooldown (stays open longer)

    def __init__(self, cb_policy: int) -> None:
        self.threshold = self._THRESHOLDS[cb_policy]
        self.cooldown  = self._COOLDOWNS[cb_policy]
        self.state     = CB_CLOSED
        self.duration  = 0
        self._history: List[int] = []

    def is_open(self) -> bool:
        return self.state == CB_OPEN

    def is_half_open(self) -> bool:
        return self.state == CB_HALF_OPEN

    def step(self, failed: bool) -> None:
        """Advance CB given the REAL outcome of the current event."""
        self._history.append(int(failed))
        if len(self._history) > self.WINDOW:
            self._history.pop(0)

        fail_rate = sum(self._history) / max(len(self._history), 1)

        if self.state == CB_CLOSED:
            if fail_rate >= self.threshold:
                self.state    = CB_OPEN
                self.duration = 0
            else:
                self.duration += 1

        elif self.state == CB_OPEN:
            if self.duration >= self.cooldown:
                self.state    = CB_HALF_OPEN
                self.duration = 0
            else:
                self.duration += 1

        elif self.state == CB_HALF_OPEN:
            if not failed:
                self.state    = CB_CLOSED
                self.duration = 0
            else:
                self.state    = CB_OPEN
                self.duration = 0


# ── Policy providers ───────────────────────────────────────────────────────────

def _no_policy(services: List[str]) -> Dict[str, ServicePolicy]:
    """No resilience constructs at all."""
    return {s: ServicePolicy(s, retry=0, timeout_tier=0, cb_policy=0, fallback_type=0)
            for s in services}


def _ors_policy(services: List[str], ors_model: ORSVQModel) -> Dict[str, ServicePolicy]:
    """
    Get ORS-recommended policy per service by running the hardening engine
    on all DSB workflow graphs and collecting annotations.
    Uses the last annotation seen for each service (across all workflows).
    """
    engine = WorkflowHardeningEngine(ors_model=ors_model, seed=SEED)
    graphs  = load_dsb_workflows()

    ann_map: Dict[str, ResilienceAnnotation] = {}
    for g in graphs:
        hw = engine.analyze(g)
        for node_name, hn in hw.nodes.items():
            ann_map[node_name] = hn.annotation

    policies = {}
    for svc in services:
        ann = ann_map.get(svc)
        if ann:
            policies[svc] = ServicePolicy(
                service      = svc,
                retry        = ann.retry,
                timeout_tier = ann.timeout_tier,
                cb_policy    = ann.cb_policy,
                fallback_type= ann.fallback_type,
            )
        else:
            # Service not in any DSB topology — use conservative default
            policies[svc] = ServicePolicy(svc, retry=1, timeout_tier=1,
                                          cb_policy=1, fallback_type=2)
    return policies


def _ollama_llm_policy(services: List[str]) -> Dict[str, ServicePolicy]:
    """
    Ask Ollama for resilience policy per service.
    Uses the compose_post topology for context (covers all 13 DSB services).
    Falls back to a conservative default if Ollama is unavailable.
    """
    from resiliency_poc.dsb_workflow_loader import _compose_post_graph

    graph = _compose_post_graph()

    edges = []
    for src, dsts in graph.adj.items():
        for dst in dsts:
            edges.append(f"  {src} -> {dst}")

    nodes_desc = []
    for name, info in graph.nodes.items():
        role = "external/storage" if info.is_external else "internal service"
        nodes_desc.append(
            f"  - {name}  (depth={info.depth}, role={role}, is_crypto={info.is_crypto})"
        )

    prompt = f"""You are a distributed systems resilience expert.

Below is the DeathStarBench socialNetwork microservice call graph.
For each service, recommend the best resilience policy.

CALL GRAPH (caller -> callee):
{chr(10).join(edges)}

SERVICE NODES:
{chr(10).join(nodes_desc)}

POLICY OPTIONS:
- retry:    none | 1 | 3 | 5
- timeout:  short (5s) | medium (30s) | long (120s)
- cb:       closed (no CB) | half_open | open (aggressive CB)
- fallback: none | cached_response | dead_letter_queue

GUIDELINES:
- Entry nodes (depth=0): light retry, half_open CB
- Orchestrator (depth=1, fan-out): open CB + dead_letter
- Storage/external (is_external=True): open CB + dead_letter
- Auth/crypto (is_crypto=True): open CB + dead_letter
- Leaf compute (depth>=2, internal): minimal retry, closed CB

Respond with ONLY valid JSON — no markdown, no explanation.
Format: {{service_name: {{"retry": "...", "timeout": "...", "cb": "...", "fallback": "..."}}}}

JSON:
"""

    CB_MAP       = {"closed": 0, "half_open": 1, "half-open": 1, "open": 2}
    TIMEOUT_MAP  = {"short": 0, "5s": 0, "medium": 1, "30s": 1, "long": 2, "120s": 2}
    FALLBACK_MAP = {"none": 0, "alternate": 1, "cached_response": 2, "cached": 2,
                    "dead_letter_queue": 3, "dead_letter": 3}

    raw_policies = {}
    try:
        body = json.dumps({
            "model": OLLAMA_MODEL, "prompt": prompt,
            "stream": False, "options": {"temperature": 0.0, "seed": 42},
        }).encode()
        req = urllib.request.Request(
            OLLAMA_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data   = json.loads(resp.read())
            text   = data.get("response", "")
            match  = re.search(r'\{[\s\S]*\}', text)
            if match:
                raw_policies = json.loads(match.group())
                print(f"  [LLM] Got policies for {len(raw_policies)} services from Ollama.")
            else:
                print("  [LLM] No JSON found in response — using conservative defaults.")
    except Exception as e:
        print(f"  [LLM] Ollama error ({e}) — using conservative defaults.")

    policies = {}
    for svc in services:
        entry = raw_policies.get(svc) or next(
            (v for k, v in raw_policies.items() if k.lower() == svc.lower()), None
        )
        if entry and isinstance(entry, dict):
            r_str = str(entry.get("retry",   "1")).lower()
            if r_str in ("none", "0"):  r_cls = 0
            elif r_str == "1":          r_cls = 1
            elif r_str == "3":          r_cls = 2
            else:                       r_cls = 3
            t_cls = TIMEOUT_MAP.get(str(entry.get("timeout", "medium")).lower(), 1)
            c_cls = CB_MAP.get(str(entry.get("cb", "half_open")).lower(), 1)
            f_cls = FALLBACK_MAP.get(str(entry.get("fallback", "cached_response")).lower(), 2)
            policies[svc] = ServicePolicy(svc, retry=r_cls, timeout_tier=t_cls,
                                          cb_policy=c_cls, fallback_type=f_cls)
        else:
            policies[svc] = ServicePolicy(svc, retry=1, timeout_tier=1,
                                          cb_policy=1, fallback_type=2)

    return policies


# ── Trace replay engine ────────────────────────────────────────────────────────

@dataclass
class EventOutcome:
    service:   str
    folder:    str
    scenario:  ScenarioState
    failure_mode: str
    outcome:   str    # CB_INTERCEPTED / RETRY_RECOVERED / RETRY_FALLBACK / LOST
    had_cb_interception: bool
    had_retry: bool
    had_fallback: bool


def _build_lookahead_index(
    events: List[ServiceEvent],
) -> Dict[str, List[Tuple[int, bool]]]:
    """
    Build per-service index of (trace_ts, is_success) pairs
    for fast look-ahead during replay.
    """
    idx: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
    for ev in events:
        idx[ev.service].append((ev.trace_ts, ev.failure_mode == "no_failure"))
    return idx


def _retry_would_recover(
    service:      str,
    fail_ts:      int,
    retry_cls:    int,
    timeout_tier: int,
    lookahead:    Dict[str, List[Tuple[int, bool]]],
) -> bool:
    """
    Look ahead in the real trace for service `service` after timestamp `fail_ts`.
    Returns True if any real call to that service within the retry window succeeded.

    Window = timeout_ms × retry_count (we search for success within that total window).
    If retry_count is 0, we don't retry → always False.
    """
    n_retries = RETRY_COUNT[retry_cls]
    if n_retries == 0:
        return False

    window_ms = TIMEOUT_MS[timeout_tier] * n_retries

    # Binary-search-like scan: find events for this service within (fail_ts, fail_ts+window]
    service_events = lookahead.get(service, [])
    for ts, is_success in service_events:
        if ts <= fail_ts:
            continue
        if ts > fail_ts + window_ms:
            break
        if is_success:
            return True   # at least one retry would have succeeded

    return False


def _replay_policy(
    events:    List[ServiceEvent],
    policies:  Dict[str, ServicePolicy],
    lookahead: Dict[str, List[Tuple[int, bool]]],
) -> List[EventOutcome]:
    """
    Chronological replay of all failure events against a set of per-service policies.
    Returns one EventOutcome per failure event.
    """
    # One CB machine per service (driven by real trace outcomes)
    cb_machines: Dict[str, TraceCBMachine] = {}

    outcomes: List[EventOutcome] = []

    for ev in events:
        svc    = ev.service
        policy = policies.get(svc)
        if policy is None:
            policy = ServicePolicy(svc, 0, 0, 0, 0)

        # Initialise CB machine for this service if first time we see it
        if svc not in cb_machines:
            cb_machines[svc] = TraceCBMachine(policy.cb_policy)

        cb = cb_machines[svc]
        is_failure = ev.failure_mode != "no_failure"

        if is_failure:
            # ── Step 1: CB pre-check ───────────────────────────────────────────
            if cb.is_open() and policy.cb_policy >= CB_HALF_OPEN:
                # CB is open and policy enables it → intercept, go to fallback
                outcome_str = CB_INTERCEPTED
                outcomes.append(EventOutcome(
                    service=svc, folder=ev.folder, scenario=ev.scenario,
                    failure_mode=ev.failure_mode, outcome=outcome_str,
                    had_cb_interception=True, had_retry=False,
                    had_fallback=(policy.fallback_type > 0),
                ))
                # CB still sees this real failure (the service did fail)
                cb.step(True)
                continue

            # ── Step 2: Retry look-ahead ───────────────────────────────────────
            recovered_by_retry = _retry_would_recover(
                svc, ev.trace_ts, policy.retry, policy.timeout_tier, lookahead
            )
            if recovered_by_retry:
                outcome_str = RETRY_RECOVERED
                outcomes.append(EventOutcome(
                    service=svc, folder=ev.folder, scenario=ev.scenario,
                    failure_mode=ev.failure_mode, outcome=outcome_str,
                    had_cb_interception=False, had_retry=True,
                    had_fallback=False,
                ))
            elif policy.fallback_type > 0:
                # ── Step 3: Fallback ───────────────────────────────────────────
                outcome_str = RETRY_FALLBACK
                outcomes.append(EventOutcome(
                    service=svc, folder=ev.folder, scenario=ev.scenario,
                    failure_mode=ev.failure_mode, outcome=outcome_str,
                    had_cb_interception=False, had_retry=(policy.retry > 0),
                    had_fallback=True,
                ))
            else:
                outcome_str = LOST
                outcomes.append(EventOutcome(
                    service=svc, folder=ev.folder, scenario=ev.scenario,
                    failure_mode=ev.failure_mode, outcome=outcome_str,
                    had_cb_interception=False, had_retry=(policy.retry > 0),
                    had_fallback=False,
                ))

        # Always advance the CB machine with the real outcome
        cb.step(is_failure)

    return outcomes


# ── Printing ───────────────────────────────────────────────────────────────────

def _recovery_rate(outcomes: List[EventOutcome]) -> float:
    if not outcomes:
        return 0.0
    recovered = sum(1 for o in outcomes if o.outcome != LOST)
    return recovered / len(outcomes) * 100


def _print_policy_table(
    label:    str,
    policies: Dict[str, ServicePolicy],
) -> None:
    W = 105
    print("\n" + "=" * W)
    print(f"  POLICIES: {label}")
    print("=" * W)
    print(f"  {'Service':<32}  {'CB':>10}  {'Retry':>8}  {'Timeout':>10}  {'Fallback':>12}")
    print("  " + "-" * (W - 2))
    for svc, p in sorted(policies.items()):
        cb_name  = ["closed", "half_open", "open"][p.cb_policy]
        retry_n  = RETRY_COUNT[p.retry]
        timeout  = f"{TIMEOUT_MS[p.timeout_tier]//1000}s"
        fb_name  = FALLBACK_NAMES[p.fallback_type]
        print(f"  {svc:<32}  {cb_name:>10}  {retry_n:>8}  {timeout:>10}  {fb_name:>12}")
    print("=" * W)


def _print_scenario_breakdown(
    variant_name: str,
    outcomes: List[EventOutcome],
) -> None:
    W = 110
    print("\n" + "=" * W)
    print(f"  REPLAY RESULTS: {variant_name}")
    print(f"  (each row = one real DSB failure event evaluated against the policy)")
    print("=" * W)
    print(f"  {'Folder (scenario)':<38}  {'Failures':>9}  {'Recovered%':>11}  "
          f"{'CB%':>6}  {'Retry%':>7}  {'Fallback%':>10}  {'Lost%':>7}")
    print("  " + "-" * (W - 2))

    by_folder: Dict[str, List[EventOutcome]] = defaultdict(list)
    for o in outcomes:
        by_folder[o.folder].append(o)

    total_all = recovered_all = 0
    for folder in sorted(by_folder.keys()):
        evs     = by_folder[folder]
        n       = len(evs)
        cb_n    = sum(1 for o in evs if o.outcome == CB_INTERCEPTED)
        ret_n   = sum(1 for o in evs if o.outcome == RETRY_RECOVERED)
        fb_n    = sum(1 for o in evs if o.outcome == RETRY_FALLBACK)
        lost_n  = sum(1 for o in evs if o.outcome == LOST)
        rec_n   = cb_n + ret_n + fb_n
        scenario = SCENARIO_NAMES[int(evs[0].scenario)] if evs else "?"

        total_all    += n
        recovered_all += rec_n

        print(f"  {folder:<38}  {n:>9}  {rec_n/n*100:>10.1f}%  "
              f"{cb_n/n*100:>5.1f}%  {ret_n/n*100:>6.1f}%  "
              f"{fb_n/n*100:>9.1f}%  {lost_n/n*100:>6.1f}%")

    print("  " + "-" * (W - 2))
    overall = recovered_all / total_all * 100 if total_all else 0.0
    print(f"  {'OVERALL':<38}  {total_all:>9}  {overall:>10.1f}%")
    print("=" * W)


def _print_service_breakdown(
    variant_name: str,
    outcomes: List[EventOutcome],
) -> None:
    W = 100
    print(f"\n  Service-level recovery rates  [{variant_name}]")
    print("  " + "-" * (W - 2))
    print(f"  {'Service':<32}  {'Events':>7}  {'Recovered%':>11}  {'CB%':>6}  "
          f"{'Retry%':>7}  {'Fallback%':>10}  {'Lost%':>7}")
    print("  " + "-" * (W - 2))

    by_svc: Dict[str, List[EventOutcome]] = defaultdict(list)
    for o in outcomes:
        by_svc[o.service].append(o)

    for svc in sorted(by_svc.keys()):
        evs    = by_svc[svc]
        n      = len(evs)
        cb_n   = sum(1 for o in evs if o.outcome == CB_INTERCEPTED)
        ret_n  = sum(1 for o in evs if o.outcome == RETRY_RECOVERED)
        fb_n   = sum(1 for o in evs if o.outcome == RETRY_FALLBACK)
        lost_n = sum(1 for o in evs if o.outcome == LOST)
        rec_n  = cb_n + ret_n + fb_n
        print(f"  {svc:<32}  {n:>7}  {rec_n/n*100:>10.1f}%  {cb_n/n*100:>5.1f}%  "
              f"{ret_n/n*100:>6.1f}%  {fb_n/n*100:>9.1f}%  {lost_n/n*100:>6.1f}%")
    print("  " + "-" * (W - 2))


def _print_three_way_summary(
    no_outcomes:  List[EventOutcome],
    llm_outcomes: List[EventOutcome],
    ors_outcomes: List[EventOutcome],
) -> None:
    W = 115
    print("\n" + "=" * W)
    print("  THREE-WAY SUMMARY  —  Real DSB X-Trace event replay")
    print(f"  Assumption: next real call to a service within retry window is a proxy")
    print(f"  for whether a retry would have succeeded.")
    print("=" * W)
    print(f"  {'Folder':<38}  {'No-policy':>10}  {'LLM':>10}  {'ORS':>10}  "
          f"{'LLM gain':>10}  {'ORS gain':>10}  {'ORS>LLM?':>10}")
    print("  " + "-" * (W - 2))

    by_folder_no  = defaultdict(list)
    by_folder_llm = defaultdict(list)
    by_folder_ors = defaultdict(list)
    for o in no_outcomes:  by_folder_no[o.folder].append(o)
    for o in llm_outcomes: by_folder_llm[o.folder].append(o)
    for o in ors_outcomes: by_folder_ors[o.folder].append(o)

    ors_wins = llm_wins = ties = 0
    for folder in sorted(by_folder_no.keys()):
        n_o   = by_folder_no[folder];   r_no  = _recovery_rate(n_o)
        l_o   = by_folder_llm[folder];  r_llm = _recovery_rate(l_o)
        o_o   = by_folder_ors[folder];  r_ors = _recovery_rate(o_o)
        llm_g = r_llm - r_no
        ors_g = r_ors - r_no
        if r_ors > r_llm + 0.5:
            verdict = "ORS wins"; ors_wins += 1
        elif r_llm > r_ors + 0.5:
            verdict = "LLM wins"; llm_wins += 1
        else:
            verdict = "tie";      ties += 1
        print(f"  {folder:<38}  {r_no:>9.1f}%  {r_llm:>9.1f}%  {r_ors:>9.1f}%  "
              f"{llm_g:>+9.1f}%  {ors_g:>+9.1f}%  {verdict:>10}")

    print("  " + "-" * (W - 2))
    print(f"  {'OVERALL':<38}  "
          f"{_recovery_rate(no_outcomes):>9.1f}%  "
          f"{_recovery_rate(llm_outcomes):>9.1f}%  "
          f"{_recovery_rate(ors_outcomes):>9.1f}%")
    print()
    print(f"  ORS wins: {ors_wins}  |  LLM wins: {llm_wins}  |  Ties: {ties}")
    print("=" * W)


def _print_failure_mode_breakdown(
    no_outcomes:  List[EventOutcome],
    llm_outcomes: List[EventOutcome],
    ors_outcomes: List[EventOutcome],
) -> None:
    """Per failure mode: how well does each policy recover?"""
    W = 105
    print("\n" + "=" * W)
    print("  RECOVERY BY FAILURE MODE")
    print("=" * W)
    print(f"  {'Failure mode':<20}  {'Events':>7}  {'No-policy':>10}  "
          f"{'LLM':>10}  {'ORS':>10}")
    print("  " + "-" * (W - 2))

    by_fm_no  = defaultdict(list)
    by_fm_llm = defaultdict(list)
    by_fm_ors = defaultdict(list)
    for o in no_outcomes:  by_fm_no[o.failure_mode].append(o)
    for o in llm_outcomes: by_fm_llm[o.failure_mode].append(o)
    for o in ors_outcomes: by_fm_ors[o.failure_mode].append(o)

    for fm in sorted(by_fm_no.keys()):
        n  = len(by_fm_no[fm])
        print(f"  {fm:<20}  {n:>7}  "
              f"{_recovery_rate(by_fm_no[fm]):>9.1f}%  "
              f"{_recovery_rate(by_fm_llm.get(fm, [])):>9.1f}%  "
              f"{_recovery_rate(by_fm_ors.get(fm, [])):>9.1f}%")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── Load raw DSB service events ───────────────────────────────────────────
    print("\n[1] Loading real DSB X-Trace service events ...")
    loader = DSBLoader(seed=SEED)
    all_events: List[ServiceEvent] = loader._scan_service_events()
    # Keep only failure events for evaluation (healthy events are baseline successes)
    failure_events = [e for e in all_events if e.failure_mode != "no_failure"]
    all_services   = sorted(set(e.service for e in all_events))

    print(f"  Total events (all folders):   {len(all_events)}")
    print(f"  Failure events (for replay):  {len(failure_events)}")
    print(f"  Services seen:                {len(all_services)}")

    # ── Build look-ahead index ────────────────────────────────────────────────
    print("\n[2] Building service look-ahead index ...")
    lookahead = _build_lookahead_index(all_events)
    for svc, pairs in sorted(lookahead.items()):
        n_ok = sum(1 for _, ok in pairs if ok)
        print(f"  {svc:<32}  {len(pairs):>5} events  ({n_ok} success, "
              f"{len(pairs)-n_ok} failure)")

    # ── Load ORS model ────────────────────────────────────────────────────────
    print("\n[3] Loading DSB-trained ORS model ...")
    ors_model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run resiliency_poc.ors_dsb_poc first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors_model.load_state_dict(d["state_dict"])
    ors_model.eval()

    # ── Get policies ──────────────────────────────────────────────────────────
    print("\n[4] Computing per-service policies ...")

    print("  [4a] No-policy (baseline) ...")
    policies_no  = _no_policy(all_services)

    print("  [4b] ORS policy (DSB-trained model) ...")
    policies_ors = _ors_policy(all_services, ors_model)

    print("  [4c] LLM policy (Ollama) ...")
    policies_llm = _ollama_llm_policy(all_services)

    # Print policy tables
    _print_policy_table("No-policy (baseline)", policies_no)
    _print_policy_table(f"LLM policy ({OLLAMA_MODEL})", policies_llm)
    _print_policy_table("ORS policy (DSB-trained ORS-VQ)", policies_ors)

    # ── Replay traces ─────────────────────────────────────────────────────────
    print("\n[5] Replaying failure events chronologically ...")
    print(f"  Events replayed: {len(failure_events)} real DSB failures")

    no_outcomes  = _replay_policy(failure_events, policies_no,  lookahead)
    llm_outcomes = _replay_policy(failure_events, policies_llm, lookahead)
    ors_outcomes = _replay_policy(failure_events, policies_ors, lookahead)

    # ── Results ───────────────────────────────────────────────────────────────
    _print_scenario_breakdown("No-policy (baseline)",         no_outcomes)
    _print_scenario_breakdown(f"LLM-policy ({OLLAMA_MODEL})", llm_outcomes)
    _print_scenario_breakdown("ORS-policy (DSB-trained)",     ors_outcomes)

    print("\n  --- Service-level breakdown ---")
    _print_service_breakdown("No-policy",  no_outcomes)
    _print_service_breakdown("LLM-policy", llm_outcomes)
    _print_service_breakdown("ORS-policy", ors_outcomes)

    _print_failure_mode_breakdown(no_outcomes, llm_outcomes, ors_outcomes)
    _print_three_way_summary(no_outcomes, llm_outcomes, ors_outcomes)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print()
    print("  -- Interpretation note --")
    print("  'retry_recovered': next real call to the service within the retry window")
    print("  was a success in the trace.  This is a proxy for 'retry would have worked'")
    print("  — not a guaranteed counterfactual.  The next call may be from a different")
    print("  request, but it tells us whether the service was healthy enough to")
    print("  respond at that time.")
    print("  'CB_intercepted': CB was already open (driven by prior real failures),")
    print("  so the call would have gone straight to fallback without hitting the service.")
    print("  " + "-" * 72)


if __name__ == "__main__":
    main()
