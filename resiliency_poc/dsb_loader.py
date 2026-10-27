"""
dsb_loader.py

DeathStarBench X-Trace dataset loader for the oracle-free temporal VQ pipeline.

Converts X-Trace JSON traces (causal event DAGs) into TemporalFailureEvent
records compatible with TemporalVQResiliencyModel — using the same 44-dim
feature vector and counterfactual simulation labels as the synthetic pipeline.

Format
──────
Each trace file: [ { "id": "<hex>", "reports": [ <span>, ... ] } ]
Each span: EventID, ParentEventID[], Host, ProcessName, Label, Timestamp, Cycles

Pipeline
──────────
1. Load traces from each folder, tag with ScenarioState from folder name
2. Per trace: reconstruct call DAG, extract failing service(s), failure mode
3. Sort all traces by Timestamp → compute 50-trace rolling temporal signals
4. Build 44-dim feature vector (38 static + 6 temporal)
5. Assign counterfactual simulation labels (same as synthetic Phase 4)

Folders used
────────────
  HEALTHY  : follow, unfollow, register, write_home_timeline   (~9 400 traces)
  DOWN     : compose_broken                                      (~1 054 traces)
  TRANSIENT: read_home_timeline_broken_pipe,
             read_user_timeline_broken_pipe                      (~1 758 traces)
  DEGRADED : register_broken                                     (~  500 traces)
  Total usable                                                   (~12 712 traces)
"""

from __future__ import annotations
import json, os, random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scenario import (
    ScenarioState, SCENARIO_OBS_DIM,
    ScenarioObservation,
)
from .counterfactual_labeler import label_from_simulation, SCENARIO_TO_CATEGORY
from .temporal_trace_generator import TEMPORAL_FEATURE_DIM, TemporalFailureEvent

# ── Dataset root ───────────────────────────────────────────────────────────────

DSB_ROOT = Path(
    r"C:\Users\agarw\OneDrive\Documents\research\codebookvae"
    r"\deathstarbench_traces-master\deathstarbench_traces-master"
    r"\socialNetwork\x-traces"
)

# ── Service vocabulary (13 DSB microservices → indices 0-12) ──────────────────

DSB_SERVICE_VOCAB: Dict[str, int] = {
    "nginx-thrift":             0,
    "UserService":              1,
    "ComposePostService":       2,
    "MediaService":             3,
    "UniqueIdService":          4,
    "HomeTimelineService":      5,
    "PostStorageService":       6,
    "SocialGraphService":       7,
    "WriteHomeTimelineService": 8,
    "TextService":              9,
    "UserMentionService":      10,
    "UserTimelineService":     11,
    "UrlShortenService":       12,
}
N_SERVICES = 13   # matches FEATURE_DIM slot [0:13] and [13:26]

# ── Folder → ScenarioState (ground-truth for evaluation, never fed to model) ──

FOLDER_SCENARIO: Dict[str, ScenarioState] = {
    "follow":                          ScenarioState.HEALTHY,
    "unfollow":                        ScenarioState.HEALTHY,
    "register":                        ScenarioState.HEALTHY,
    "write_home_timeline":             ScenarioState.HEALTHY,
    "home-timeline-read":              ScenarioState.HEALTHY,
    "compose_broken":                  ScenarioState.SERVICE_DOWN,
    "read_home_timeline_broken_pipe":  ScenarioState.TRANSIENT_FAILURE,
    "read_user_timeline_broken_pipe":  ScenarioState.TRANSIENT_FAILURE,
    "register_broken":                 ScenarioState.DEGRADED,
}

# ── Failure mode mapping (Label text → 7-class index) ─────────────────────────
# Mirrors trace_generator.py failure mode indices

FAILURE_MODES = [
    "no_failure", "auth_error", "crypto_error",
    "bad_input", "dependency_down", "rate_limit", "timeout",
]
FM_IDX = {fm: i for i, fm in enumerate(FAILURE_MODES)}

def _classify_failure(label: str) -> str:
    l = label.lower()
    if not label or "start" in l or "complete" in l or "processing" in l:
        return "no_failure"
    if "timeout" in l or "pop timeout" in l:
        return "timeout"
    if "broken pipe" in l or "failed to connect" in l or "failed to read" in l \
            or "failed to get" in l or "no more data" in l \
            or "failed to upload" in l or "failed to set" in l:
        return "dependency_down"
    if "already existed" in l or "invalid method" in l or "internal server error" in l:
        return "bad_input"
    if "auth" in l or "unauthorized" in l:
        return "auth_error"
    return "no_failure"


# ── Raw trace parsing ──────────────────────────────────────────────────────────

@dataclass
class ServiceEvent:
    """One failing service within one request trace."""
    trace_id:     str
    folder:       str
    scenario:     ScenarioState
    service:      str       # ProcessName or Host
    caller:       str       # immediate parent service in call DAG
    failure_mode: str
    depth:        int       # distance from nginx-thrift (0 = nginx)
    trace_ts:     int       # min Timestamp across all spans in this trace (ms)
    trace_dur:    int       # max - min Timestamp (ms)
    is_external:  bool      # True for nginx-thrift


def _service_name(span: dict) -> str:
    return span.get("ProcessName") or span.get("Host") or "unknown"


def _parse_trace(path: Path, folder: str, scenario: ScenarioState) -> List[ServiceEvent]:
    """
    Parse one X-Trace JSON file → zero or more ServiceEvent records.

    For healthy traces:  one ServiceEvent with failure_mode="no_failure"
                         for the outermost service (nginx-thrift if present).
    For failure traces:  one ServiceEvent per span whose Label indicates failure.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return []

    if not data or "reports" not in data[0]:
        return []

    reports = data[0]["reports"]
    if not reports:
        return []

    trace_id = data[0].get("id", path.stem)

    # Build event-id → service map and parent map
    ev_to_svc: Dict[str, str] = {}
    ev_parent: Dict[str, List[str]] = {}
    for r in reports:
        eid = r.get("EventID", "")
        svc = _service_name(r)
        ev_to_svc[eid] = svc
        ev_parent[eid] = r.get("ParentEventID") or []

    # Compute BFS depth per service from the root (nginx-thrift or first span)
    # Root = span with no parent in this trace
    all_eids = set(ev_to_svc.keys())
    roots = [eid for eid, parents in ev_parent.items()
             if not any(p in all_eids for p in parents)]
    root_svc = ev_to_svc[roots[0]] if roots else "nginx-thrift"

    # BFS to assign depth per event-id
    depth_of: Dict[str, int] = {}
    queue: deque = deque()
    for r in roots:
        depth_of[r] = 0
        queue.append(r)
    # Build children map for BFS
    children: Dict[str, List[str]] = defaultdict(list)
    for eid, parents in ev_parent.items():
        for p in parents:
            if p in all_eids:
                children[p].append(eid)
    while queue:
        eid = queue.popleft()
        for child in children[eid]:
            if child not in depth_of:
                depth_of[child] = depth_of[eid] + 1
                queue.append(child)

    ts_vals = [r["Timestamp"] for r in reports if "Timestamp" in r]
    trace_ts  = min(ts_vals) if ts_vals else 0
    trace_dur = (max(ts_vals) - min(ts_vals)) if len(ts_vals) > 1 else 0

    events: List[ServiceEvent] = []

    if scenario == ScenarioState.HEALTHY:
        # Emit one healthy event for the root service
        depth = 0
        svc = root_svc
        caller_candidates = [
            ev_to_svc.get(p, "unknown")
            for r in reports if _service_name(r) == svc
            for p in (r.get("ParentEventID") or [])
            if p in ev_to_svc and ev_to_svc[p] != svc
        ]
        caller = caller_candidates[0] if caller_candidates else "none"
        events.append(ServiceEvent(
            trace_id=trace_id, folder=folder, scenario=scenario,
            service=svc, caller=caller, failure_mode="no_failure",
            depth=0, trace_ts=trace_ts, trace_dur=trace_dur,
            is_external=(svc == "nginx-thrift"),
        ))
        return events

    # For failure traces: find spans with error labels
    seen_svcs: set = set()
    for r in reports:
        label = r.get("Label", "")
        fm = _classify_failure(label)
        if fm == "no_failure":
            continue
        svc = _service_name(r)
        if svc in seen_svcs:
            continue   # one event per service per trace
        seen_svcs.add(svc)

        eid = r.get("EventID", "")
        d = depth_of.get(eid, 0)

        # Find caller: first parent that belongs to a different service
        caller = "unknown"
        for p in (r.get("ParentEventID") or []):
            p_svc = ev_to_svc.get(p, "")
            if p_svc and p_svc != svc:
                caller = p_svc
                break

        events.append(ServiceEvent(
            trace_id=trace_id, folder=folder, scenario=scenario,
            service=svc, caller=caller, failure_mode=fm,
            depth=d, trace_ts=trace_ts, trace_dur=trace_dur,
            is_external=(svc == "nginx-thrift"),
        ))

    return events


# ── Rolling temporal signals ───────────────────────────────────────────────────

ROLLING_WINDOW = 50   # traces

def _build_temporal_obs(
    events_sorted: List[ServiceEvent],
    idx:           int,
    cb_machines:   dict,
) -> ScenarioObservation:
    """
    Compute 9 temporal features for events_sorted[idx].

    The first 6 are symptom signals derived from the preceding ROLLING_WINDOW
    same-service events.  The last 3 are CB state feedback: current CB state,
    how long it has been in that state, and failures in the last CB window.
    The CB machine is advanced per event so state is cumulative across the stream.
    """
    from .scenario import CBStateMachine, CB_CLOSED, CB_HALF_OPEN, CB_OPEN

    ev  = events_sorted[idx]
    svc = ev.service

    # Per-service CB machine (persisted across calls via cb_machines dict)
    if svc not in cb_machines:
        cb_machines[svc] = CBStateMachine()
    cb = cb_machines[svc]

    # Snapshot CB state BEFORE advancing (these are the features the model sees)
    cb_state, cb_dur, cb_fails = cb.snapshot()

    # Advance CB with the outcome of this event
    failed = ev.failure_mode != "no_failure"
    cb.step(failed)

    # Collect same-service history for symptom features
    history: List[ServiceEvent] = []
    for j in range(max(0, idx - ROLLING_WINDOW), idx):
        if events_sorted[j].service == svc:
            history.append(events_sorted[j])

    n = len(history)
    if n == 0:
        fail_rate    = 0.0 if ev.scenario == ScenarioState.HEALTHY else 0.5
        consec_fails = 0
        retry_succ   = 1.0 if ev.scenario == ScenarioState.HEALTHY else 0.5
        lat_trend    = 0.0
        t_since_suc  = 0
        dur          = 1
    else:
        n_failed  = sum(1 for h in history if h.failure_mode != "no_failure")
        fail_rate = n_failed / n

        consec_fails = 0
        for h in reversed(history):
            if h.failure_mode != "no_failure":
                consec_fails += 1
            else:
                break

        retry_succ = 1.0 - fail_rate

        durs = [h.trace_dur for h in history[-10:]]
        if len(durs) >= 2:
            slope = (durs[-1] - durs[0]) / max(max(durs), 1)
            lat_trend = float(np.clip(slope, -1.0, 1.0))
        else:
            lat_trend = 0.0

        t_since_suc = 0
        for h in reversed(history):
            if h.failure_mode == "no_failure":
                break
            t_since_suc += 1

        dur = 1
        current_scenario = ev.scenario
        for h in reversed(history):
            if h.scenario == current_scenario:
                dur += 1
            else:
                break

    return ScenarioObservation(
        scenario=ev.scenario, t=idx,
        rolling_fail_rate=float(np.clip(fail_rate, 0, 1)),
        consec_failures=consec_fails,
        retry_succ_rate=float(np.clip(retry_succ, 0, 1)),
        latency_trend=lat_trend,
        time_since_success=float(t_since_suc),
        scenario_duration=dur,
        cb_state=cb_state,
        cb_duration=cb_dur,
        failures_last_15=cb_fails,
    )


# ── Static feature vector (38-dim, mirrors trace_generator.py layout) ─────────

def _static_features(ev: ServiceEvent, max_depth: int, load_bucket: int) -> np.ndarray:
    svc_idx    = DSB_SERVICE_VOCAB.get(ev.service, N_SERVICES - 1)
    caller_idx = DSB_SERVICE_VOCAB.get(ev.caller, N_SERVICES - 1)
    fm_idx     = FM_IDX.get(ev.failure_mode, 0)

    curr_oh = np.zeros(N_SERVICES, dtype=np.float32)
    curr_oh[svc_idx] = 1.0

    next_oh = np.zeros(N_SERVICES, dtype=np.float32)
    next_oh[caller_idx] = 1.0

    fm_oh = np.zeros(7, dtype=np.float32)
    fm_oh[fm_idx] = 1.0

    depth_norm = ev.depth / max(max_depth, 1)
    position   = depth_norm   # proxy: deeper = later in workflow
    retry_count = 0.0         # DeathStarBench does not retry by default
    load_f     = load_bucket / 3.0

    return np.concatenate([
        curr_oh, next_oh, fm_oh,
        [position, depth_norm, retry_count, float(ev.is_external), load_f],
    ]).astype(np.float32)   # shape (38,)


# ── Main loader class ──────────────────────────────────────────────────────────

def _stratified_sample(
    events:           List[ServiceEvent],
    n:                int,
    rng:              random.Random,
    healthy_frac:     float = 0.40,
) -> List[ServiceEvent]:
    """
    Minority-boosted stratified sample.

    HEALTHY is capped at `healthy_frac * n` to prevent it from crowding
    out rare scenarios.  The remaining budget is distributed proportionally
    among non-HEALTHY classes (capped by available data).
    """
    by_sc: Dict[ScenarioState, List] = defaultdict(list)
    for e in events:
        by_sc[e.scenario].append(e)

    # HEALTHY budget
    healthy_budget = min(int(n * healthy_frac), len(by_sc.get(ScenarioState.HEALTHY, [])))
    healthy_sample = rng.sample(by_sc[ScenarioState.HEALTHY], healthy_budget) \
                     if healthy_budget > 0 else []

    # Minority budget — proportional by available count, capped by available data
    minority_classes = {s: evs for s, evs in by_sc.items() if s != ScenarioState.HEALTHY}
    minority_total   = sum(len(v) for v in minority_classes.values())
    minority_budget  = n - healthy_budget
    out = list(healthy_sample)
    for s, evs in minority_classes.items():
        if minority_total == 0:
            break
        k = min(round(len(evs) / minority_total * minority_budget), len(evs))
        out.extend(rng.sample(evs, k))

    out.sort(key=lambda e: e.trace_ts)
    return out[:n]


def _print_split_distribution(label: str, events: List[ServiceEvent]) -> None:
    from collections import Counter
    from .scenario import SCENARIO_NAMES
    counts = Counter(e.scenario for e in events)
    parts = "  ".join(
        f"{SCENARIO_NAMES[s]}={counts[s]}" for s in ScenarioState if counts[s] > 0
    )
    print(f"        {label}: {len(events)} events  [{parts}]")


class DSBLoader:
    """
    Loads DeathStarBench X-Trace traces and converts them to
    TemporalFailureEvent records for the oracle-free Phase 4 model.

    Usage
    ─────
        loader = DSBLoader()
        train_events, test_events = loader.load_dataset(n_train=2500, n_test=500)
    """

    def __init__(self, root: Path = DSB_ROOT, seed: int = 42):
        self.root = root
        self.seed = seed

    # ── Step 1: scan all trace files ──────────────────────────────────────────

    def _scan_service_events(self) -> List[ServiceEvent]:
        all_events: List[ServiceEvent] = []

        for folder, scenario in FOLDER_SCENARIO.items():
            folder_path = self.root / folder
            if not folder_path.exists():
                print(f"  [DSB] Warning: folder not found: {folder_path}")
                continue

            json_files = sorted(
                p for p in folder_path.iterdir()
                if p.suffix == ".json" and p.is_file()
            )
            loaded = 0
            for fpath in json_files:
                evs = _parse_trace(fpath, folder, scenario)
                all_events.extend(evs)
                loaded += 1

            n_folder = len([e for e in all_events if e.folder == folder])
            print(f"  [DSB] {folder:40s} {loaded:5d} traces -> {n_folder:5d} events")

        # Sort by timestamp so rolling window is chronologically meaningful
        all_events.sort(key=lambda e: e.trace_ts)
        return all_events

    # ── Step 2: convert to TemporalFailureEvent ───────────────────────────────

    def _to_temporal_events(
        self,
        service_events: List[ServiceEvent],
        labeling_seed:  int = 0,
    ) -> List[TemporalFailureEvent]:
        max_depth = max((e.depth for e in service_events), default=1)

        # Compute a rough load bucket from trace volume per 100-trace window
        n = len(service_events)
        temporal_events: List[TemporalFailureEvent] = []
        cb_machines: dict = {}   # per-service CB state machines, persisted across events

        for idx, ev in enumerate(service_events):
            load_bucket = min(3, idx * 3 // max(n - 1, 1))
            static_f    = _static_features(ev, max_depth, load_bucket)
            temporal_obs = _build_temporal_obs(service_events, idx, cb_machines)

            label = label_from_simulation(
                ev.scenario,
                seed=labeling_seed + idx,
            )

            temporal_events.append(TemporalFailureEvent(
                workflow_name    = f"DSB/{ev.folder}",
                node_name        = ev.service,
                node_type        = ev.service,
                t                = idx,
                scenario         = ev.scenario,
                static_features  = static_f,
                scenario_obs     = temporal_obs,
                label_retry      = label.retry,
                label_timeout    = label.timeout,
                label_cb         = label.cb,
                label_fallback   = label.fallback,
                label_category   = SCENARIO_TO_CATEGORY[ev.scenario],
            ))

        return temporal_events

    # ── Public API ────────────────────────────────────────────────────────────

    def load_dataset(
        self,
        n_train:  int = 2500,
        n_test:   int = 500,
        verbose:  bool = True,
    ) -> Tuple[List[TemporalFailureEvent], List[TemporalFailureEvent]]:
        """
        Returns (train_events, test_events).

        Uses STRATIFIED splitting by scenario so that every scenario class
        appears in both train and test in the same proportions.

        Why not chronological? The DSB experiment folders were collected at
        different wall-clock times (compose_broken is 2 days later than
        follow/register), so a timestamp-based split puts entire scenario
        classes exclusively into train or test — making evaluation meaningless.
        Stratified split preserves scenario balance in both splits.
        """
        if verbose:
            print(f"\n[DSB] Scanning traces from {self.root} ...")
        service_events = self._scan_service_events()

        if verbose:
            self._print_scenario_distribution(service_events)

        rng = random.Random(self.seed)

        # ── Stratified split ───────────────────────────────────────────────────
        # Group by scenario, shuffle within group, take 80/20 per group
        by_scenario: Dict[ScenarioState, List] = defaultdict(list)
        for ev in service_events:
            by_scenario[ev.scenario].append(ev)

        train_raw: List = []
        test_raw:  List = []
        for s, evs in by_scenario.items():
            rng.shuffle(evs)
            split = max(1, int(len(evs) * 0.8))
            train_raw.extend(evs[:split])
            test_raw.extend(evs[split:])

        # Re-sort within each split by timestamp (for rolling temporal features)
        train_raw.sort(key=lambda e: e.trace_ts)
        test_raw.sort(key=lambda e: e.trace_ts)

        # Sub-sample proportionally if over budget
        if len(train_raw) > n_train:
            # Maintain proportions across scenarios when downsampling
            train_raw = _stratified_sample(train_raw, n_train, rng)
        if len(test_raw) > n_test:
            test_raw = _stratified_sample(test_raw, n_test, rng)

        if verbose:
            print(f"\n[DSB] Stratified split: {len(train_raw)} train / {len(test_raw)} test")
            _print_split_distribution("train", train_raw)
            _print_split_distribution("test",  test_raw)
            print(f"[DSB] Converting to TemporalFailureEvent ...")

        train_events = self._to_temporal_events(train_raw, labeling_seed=self.seed)
        test_events  = self._to_temporal_events(test_raw,  labeling_seed=self.seed + 9999)

        if verbose:
            print(f"[DSB] Done. feature_dim={TEMPORAL_FEATURE_DIM}")

        return train_events, test_events

    def _print_scenario_distribution(self, events: List[ServiceEvent]) -> None:
        from collections import Counter
        from .scenario import SCENARIO_NAMES
        counts = Counter(e.scenario for e in events)
        total  = len(events)
        print(f"  [DSB] Total service events: {total}")
        for s in ScenarioState:
            pct = counts[s] / total * 100 if total else 0
            print(f"        {SCENARIO_NAMES[s]:<20} {counts[s]:>6}  ({pct:.1f}%)")
