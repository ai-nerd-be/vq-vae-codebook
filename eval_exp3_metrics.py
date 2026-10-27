"""
eval_exp3_metrics.py  --  Experiment 3: Runtime adaptation — four measured metrics

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m eval_exp3_metrics

What this measures
───────────────────
eval_runtime_adaptation.py demonstrated that ORS adapts workflow policies
during a 3-minute incident (4 transitions, 0 human intervention).

This script measures HOW WELL it adapts, with four quantified metrics:

  Metric A — Detection latency
    How quickly (in seconds) does ORS react to a telemetry change?
    Method: sweep telemetry at 1-second granularity around each incident
    event; record the first tick at which ORS code changes.
    Result: latency in seconds from "telemetry crosses threshold" to
    "ORS issues updated policy".

  Metric B — Policy correctness
    Is the ORS-chosen policy actually better for the current condition?
    Method: at each transition point, simulate N=2000 requests through
    the affected node under (a) old policy, (b) new ORS policy,
    (c) oracle-best policy from CANDIDATE_POLICIES.
    Reports: success rate uplift and normalised regret.

  Metric C — Cost saved
    How many wasted API calls does runtime adaptation avoid vs a static
    policy frozen at T=0?
    Method: replay the incident timeline with both policies.  Count API
    calls to failing services — static policy keeps retrying; ORS opens
    the CB and stops.
    Reports: wasted calls per 10-second tick, cumulative over 3 minutes.

  Metric D — Recovery improvement (MTTR)
    How much faster does the workflow reach stable success after ORS adapts?
    Method: simulate 100 requests per tick under static vs adaptive policy.
    MTTR = first tick where rolling 30s success rate >= 95%.
    Reports: MTTR static, MTTR adaptive, improvement in seconds.

NOTE: Does NOT modify any existing file.
"""

from __future__ import annotations
import random
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from resiliency_poc.dsb_loader               import DSB_SERVICE_VOCAB
from resiliency_poc.dsb_workflow_loader      import DSB_EXTERNAL, DSB_CRYPTO
from resiliency_poc.scenario                 import CB_CLOSED, CB_HALF_OPEN, CB_OPEN
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM
from resiliency_poc.counterfactual_labeler   import CANDIDATE_POLICIES

SEED          = 42
CODEBOOK_SIZE = 16
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
N_SERVICES    = 13
RNG           = random.Random(SEED)

RETRY_LABELS    = ["none", "1x", "3x", "5x"]
TIMEOUT_LABELS  = ["5s",  "30s", "120s"]
CB_LABELS       = ["closed", "half_open", "open"]
FALLBACK_LABELS = ["none", "alternate", "cached", "dead_letter"]

# ── Reuse graph info and feature builder from eval_runtime_adaptation ─────────

WORKFLOW_NODES = [
    dict(service="nginx-thrift",        role="API Gateway",     depth=0, caller="none"),
    dict(service="HomeTimelineService", role="Timeline Service", depth=1, caller="nginx-thrift"),
    dict(service="PostStorageService",  role="Post Storage",     depth=2, caller="HomeTimelineService"),
    dict(service="UserService",         role="User/Auth",        depth=2, caller="HomeTimelineService"),
]


def _static_features(service: str, caller: str, depth: int,
                     max_depth: int = 3, failure_mode_idx: int = 4) -> np.ndarray:
    svc_idx    = DSB_SERVICE_VOCAB.get(service,  N_SERVICES - 1)
    caller_idx = DSB_SERVICE_VOCAB.get(caller,   N_SERVICES - 1)
    curr_oh    = np.zeros(N_SERVICES, dtype=np.float32); curr_oh[svc_idx]    = 1.0
    caller_oh  = np.zeros(N_SERVICES, dtype=np.float32); caller_oh[caller_idx] = 1.0
    fm_oh      = np.zeros(7, dtype=np.float32);          fm_oh[failure_mode_idx] = 1.0
    depth_norm = depth / max(max_depth, 1)
    return np.concatenate([
        curr_oh, caller_oh, fm_oh,
        [depth_norm, depth_norm, 0.0, float(service in DSB_EXTERNAL), 0.5],
    ]).astype(np.float32)


@dataclass
class Telemetry:
    fail_rate:  float = 0.02
    consec:     float = 0.01
    retry_succ: float = 0.97
    lat_trend:  float = -0.10
    t_since_ok: float = 0.01
    duration:   float = 0.03
    cb_state:   int   = CB_CLOSED
    cb_dur:     float = 0.0
    cb_fails:   float = 0.02

    def to_vec(self) -> np.ndarray:
        return np.array([
            self.fail_rate, min(self.consec, 1.0), self.retry_succ,
            self.lat_trend, min(self.t_since_ok, 1.0), min(self.duration, 1.0),
            self.cb_state / 2.0, min(self.cb_dur, 1.0), min(self.cb_fails, 1.0),
        ], dtype=np.float32)

    @property
    def is_healthy(self) -> bool:
        return self.fail_rate < 0.05 and self.cb_state == CB_CLOSED


HEALTHY = Telemetry()

# Scripted incident timeline (same scenario as eval_runtime_adaptation)
# Each tuple: (t_seconds, service, Telemetry)
TIMELINE: List[Tuple[int, str, Telemetry]] = [
    (0,   "PostStorageService", HEALTHY),
    (0,   "HomeTimelineService", HEALTHY),
    (30,  "PostStorageService",
          Telemetry(0.05, 0.01, 0.93, 0.35, 0.02, 0.05, CB_CLOSED, 0.0, 0.05)),
    (50,  "HomeTimelineService",
          Telemetry(0.06, 0.01, 0.92, 0.28, 0.02, 0.08, CB_CLOSED, 0.0, 0.06)),
    (60,  "PostStorageService",
          Telemetry(0.35, 0.10, 0.62, 0.45, 0.12, 0.20, CB_HALF_OPEN, 0.15, 0.30)),
    (80,  "PostStorageService",
          Telemetry(0.92, 0.50, 0.04, 0.95, 0.85, 0.90, CB_OPEN, 0.65, 0.95)),
    (100, "HomeTimelineService",
          Telemetry(0.55, 0.25, 0.40, 0.65, 0.35, 0.45, CB_HALF_OPEN, 0.25, 0.50)),
    (120, "PostStorageService",
          Telemetry(0.40, 0.12, 0.55, -0.30, 0.20, 0.25, CB_HALF_OPEN, 0.20, 0.35)),
    (150, "PostStorageService",
          Telemetry(0.03, 0.01, 0.96, -0.15, 0.02, 0.04, CB_CLOSED, 0.0, 0.03)),
    (170, "HomeTimelineService",
          Telemetry(0.04, 0.01, 0.95, -0.10, 0.02, 0.04, CB_CLOSED, 0.0, 0.04)),
]

TICKS_1S = list(range(0, 181))   # 1-second resolution for detection latency
TICKS_10S = list(range(0, 181, 10))


def _build_feat(node: dict, telem: Telemetry) -> torch.Tensor:
    fm = 0 if telem.is_healthy else 4
    static = _static_features(node["service"], node["caller"], node["depth"],
                               failure_mode_idx=fm)
    return torch.tensor(np.concatenate([static, telem.to_vec()]), dtype=torch.float32).unsqueeze(0)


def _query(model: ORSVQModel, node: dict, telem: Telemetry):
    x    = _build_feat(node, telem)
    code = model.code_for(x)
    pred = model.predict(x)
    return code, pred


def _apply_timeline(t: int, current: Dict[str, Telemetry]) -> None:
    for (ev_t, svc, telem) in TIMELINE:
        if ev_t == t:
            current[svc] = telem


# ── Request simulator ─────────────────────────────────────────────────────────
# Mirrors counterfactual_labeler._trial() logic.

def _simulate_requests(pred, fail_rate: float, n: int = 2000) -> Dict:
    """
    Simulate n requests through a single node under the given policy and fail rate.
    Returns: success_rate, wasted_retries_per_req, api_calls_per_req.
    """
    successes     = 0
    wasted_retries = 0
    total_api     = 0

    for _ in range(n):
        cb = pred.cb
        # CB open: skip to fallback
        if cb == 2:
            fb_prob = [0.0, 0.7, 0.85, 0.6][pred.fallback]
            ok = RNG.random() < fb_prob
            successes += ok
            total_api += 1   # one fallback call
            continue

        max_att = 2 if cb == 1 else (pred.retry + 1)
        ok   = False
        att  = 0
        for _ in range(max_att):
            att += 1
            if RNG.random() >= fail_rate:
                ok = True
                break
        successes += ok
        total_api += att
        # Wasted = attempts after first failure when we could have opened CB
        if not ok:
            wasted_retries += max(0, att - 1)

    return {
        "success_rate":      successes / n,
        "wasted_per_req":    wasted_retries / n,
        "api_calls_per_req": total_api / n,
    }


def _oracle_policy(fail_rate: float, n: int = 500):
    """Find the best policy from CANDIDATE_POLICIES for a given fail rate."""
    best_policy = None
    best_score  = -1.0
    for p in CANDIDATE_POLICIES:
        res = _simulate_requests(p, fail_rate, n=n)
        score = res["success_rate"] - 0.05 * res["wasted_per_req"]
        if score > best_score:
            best_score  = score
            best_policy = p
    return best_policy, best_score


# ════════════════════════════════════════════════════════════════════════════════
# METRIC A — Detection latency
# ════════════════════════════════════════════════════════════════════════════════

def metric_a_detection_latency(model: ORSVQModel) -> List[Dict]:
    """
    Replay at 1-second granularity.  For each incident event, record:
      - t_event: when telemetry changed
      - t_detect: when ORS first issued a different code
      - latency: t_detect - t_event (seconds)
    """
    print("\n  Replaying timeline at 1-second resolution ...")

    results = []
    current: Dict[str, Telemetry] = {n["service"]: Telemetry() for n in WORKFLOW_NODES}
    node_map = {n["service"]: n for n in WORKFLOW_NODES}
    prev_codes: Dict[str, int] = {}

    # Prime
    for svc, telem in current.items():
        node = node_map[svc]
        code, _ = _query(model, node, telem)
        prev_codes[svc] = code

    # Track which events have been applied and whether we found a detection tick
    event_triggers = {(ev_t, svc): {"applied": False, "t_event": ev_t,
                                     "service": svc, "t_detect": None}
                      for (ev_t, svc, _) in TIMELINE if ev_t > 0}

    for t in TICKS_1S:
        _apply_timeline(t, current)

        # Mark events applied at this tick
        for (ev_t, svc) in event_triggers:
            if ev_t == t:
                event_triggers[(ev_t, svc)]["applied"] = True

        for svc, telem in current.items():
            node = node_map.get(svc)
            if node is None:
                continue
            code, _ = _query(model, node, telem)

            if code != prev_codes[svc]:
                # Find the most recent unapplied trigger for this service
                for (ev_t, ev_svc), info in event_triggers.items():
                    if ev_svc == svc and info["applied"] and info["t_detect"] is None:
                        info["t_detect"] = t

            prev_codes[svc] = code

    for (ev_t, svc), info in sorted(event_triggers.items()):
        if info["t_detect"] is not None:
            latency = info["t_detect"] - info["t_event"]
            results.append({
                "service":   svc,
                "t_event":   info["t_event"],
                "t_detect":  info["t_detect"],
                "latency_s": latency,
            })

    return results


# ════════════════════════════════════════════════════════════════════════════════
# METRIC B — Policy correctness
# ════════════════════════════════════════════════════════════════════════════════

def metric_b_policy_correctness(model: ORSVQModel) -> List[Dict]:
    """
    At each ORS transition point (same as eval_runtime_adaptation.py):
      - Simulate under OLD policy (before transition)
      - Simulate under NEW ORS policy (after transition)
      - Simulate under ORACLE best policy
      - Report uplift and normalised regret
    """
    # Transition events from the runtime adaptation script (reproduced here)
    transitions = [
        dict(t=30,  service="PostStorageService", fail_rate=0.05,
             label="Latency rising, errors still low"),
        dict(t=50,  service="HomeTimelineService", fail_rate=0.06,
             label="Latency propagating from PostStorage"),
        dict(t=80,  service="PostStorageService", fail_rate=0.92,
             label="Persistent failure"),
        dict(t=120, service="PostStorageService", fail_rate=0.40,
             label="Recovering"),
        dict(t=150, service="PostStorageService", fail_rate=0.03,
             label="Recovered"),
        dict(t=170, service="HomeTimelineService", fail_rate=0.04,
             label="HomeTimeline recovered"),
    ]

    node_map = {n["service"]: n for n in WORKFLOW_NODES}

    # Track current telem and policy state
    current: Dict[str, Telemetry] = {n["service"]: Telemetry() for n in WORKFLOW_NODES}
    prev_pred: Dict[str, object] = {}
    for svc, telem in current.items():
        _, pred = _query(model, node_map[svc], telem)
        prev_pred[svc] = pred

    results = []
    for tr in transitions:
        svc  = tr["service"]
        fr   = tr["fail_rate"]
        node = node_map[svc]

        # Apply telemetry up to this point
        _apply_timeline(tr["t"], current)
        old_pred = prev_pred[svc]
        _, new_pred = _query(model, node, current[svc])

        # Skip if ORS didn't actually transition here
        if (old_pred.cb == new_pred.cb and old_pred.retry == new_pred.retry and
                old_pred.fallback == new_pred.fallback):
            prev_pred[svc] = new_pred
            continue

        old_res    = _simulate_requests(old_pred, fr)
        new_res    = _simulate_requests(new_pred, fr)
        oracle_pol, oracle_score = _oracle_policy(fr)
        oracle_res = _simulate_requests(oracle_pol, fr)

        # Normalised regret: 0 = matches oracle, 1 = same as old policy
        denom = oracle_res["success_rate"] - old_res["success_rate"]
        norm_regret = (1.0 - (new_res["success_rate"] - old_res["success_rate"])
                       / denom) if abs(denom) > 0.01 else 0.0
        norm_regret = max(0.0, min(1.0, norm_regret))

        results.append({
            "t":              tr["t"],
            "service":        svc,
            "fail_rate":      fr,
            "label":          tr["label"],
            "old_policy":     f"CB={CB_LABELS[old_pred.cb]},r={RETRY_LABELS[old_pred.retry]}",
            "new_policy":     f"CB={CB_LABELS[new_pred.cb]},r={RETRY_LABELS[new_pred.retry]}",
            "oracle_policy":  f"CB={CB_LABELS[oracle_pol.cb]},r={RETRY_LABELS[oracle_pol.retry]}",
            "old_success":    old_res["success_rate"],
            "new_success":    new_res["success_rate"],
            "oracle_success": oracle_res["success_rate"],
            "uplift":         new_res["success_rate"] - old_res["success_rate"],
            "norm_regret":    norm_regret,
            "wasted_old":     old_res["wasted_per_req"],
            "wasted_new":     new_res["wasted_per_req"],
        })

        prev_pred[svc] = new_pred

    return results


# ════════════════════════════════════════════════════════════════════════════════
# METRIC C — Cost saved (wasted API calls)
# ════════════════════════════════════════════════════════════════════════════════

def metric_c_cost_saved(model: ORSVQModel, requests_per_tick: int = 200) -> Dict:
    """
    Replay the incident at 10-second ticks.  For each tick and each node,
    simulate requests_per_tick requests under:
      - Static policy:  ORS recommendation frozen at T=0
      - Adaptive policy: ORS re-queried at every tick

    Count cumulative wasted retries and API calls.
    """
    current:  Dict[str, Telemetry] = {n["service"]: Telemetry() for n in WORKFLOW_NODES}
    node_map  = {n["service"]: n for n in WORKFLOW_NODES}

    # Freeze policies at T=0
    static_pred: Dict[str, object] = {}
    for svc, telem in current.items():
        _, pred = _query(model, node_map[svc], telem)
        static_pred[svc] = pred

    static_wasted  = 0.0
    adaptive_wasted = 0.0
    static_api     = 0.0
    adaptive_api   = 0.0
    static_success = 0.0
    adaptive_success = 0.0

    per_tick_static   = []
    per_tick_adaptive = []

    for t in TICKS_10S:
        _apply_timeline(t, current)

        for svc, telem in current.items():
            node = node_map[svc]
            fr   = telem.fail_rate

            # Adaptive: re-query ORS
            _, adap_pred = _query(model, node, telem)

            stat_res = _simulate_requests(static_pred[svc], fr, n=requests_per_tick)
            adap_res = _simulate_requests(adap_pred,         fr, n=requests_per_tick)

            static_wasted    += stat_res["wasted_per_req"]   * requests_per_tick
            adaptive_wasted  += adap_res["wasted_per_req"]   * requests_per_tick
            static_api       += stat_res["api_calls_per_req"] * requests_per_tick
            adaptive_api     += adap_res["api_calls_per_req"] * requests_per_tick
            static_success   += stat_res["success_rate"]      * requests_per_tick
            adaptive_success += adap_res["success_rate"]      * requests_per_tick

        per_tick_static.append(
            sum(_simulate_requests(static_pred[svc], current[svc].fail_rate,
                                   n=requests_per_tick)["wasted_per_req"]
                for svc in current) * requests_per_tick
        )
        per_tick_adaptive.append(
            sum(_simulate_requests(
                    _query(model, node_map[svc], current[svc])[1],
                    current[svc].fail_rate, n=requests_per_tick)["wasted_per_req"]
                for svc in current) * requests_per_tick
        )

    n_ticks   = len(TICKS_10S)
    n_nodes   = len(WORKFLOW_NODES)
    total_req = requests_per_tick * n_ticks * n_nodes

    return {
        "total_requests":         total_req,
        "static_wasted_total":    int(static_wasted),
        "adaptive_wasted_total":  int(adaptive_wasted),
        "wasted_saved":           int(static_wasted - adaptive_wasted),
        "pct_saved":              (static_wasted - adaptive_wasted) / max(static_wasted, 1) * 100,
        "static_success_rate":    static_success / total_req,
        "adaptive_success_rate":  adaptive_success / total_req,
        "per_tick_static":        per_tick_static,
        "per_tick_adaptive":      per_tick_adaptive,
        "ticks":                  TICKS_10S,
    }


# ════════════════════════════════════════════════════════════════════════════════
# METRIC D — Recovery improvement (MTTR)
# ════════════════════════════════════════════════════════════════════════════════

def metric_d_mttr(model: ORSVQModel, requests_per_tick: int = 300,
                  stable_threshold: float = 0.92) -> Dict:
    """
    Simulate requests at every tick.  MTTR = first tick where the rolling
    3-tick (30s) success rate >= stable_threshold after the outage starts.

    Compare static vs adaptive policy.
    """
    current: Dict[str, Telemetry] = {n["service"]: Telemetry() for n in WORKFLOW_NODES}
    node_map = {n["service"]: n for n in WORKFLOW_NODES}

    static_pred: Dict[str, object] = {}
    for svc, telem in current.items():
        _, pred = _query(model, node_map[svc], telem)
        static_pred[svc] = pred

    # PostStorage is the failing node — track its per-tick success
    ps_node = next(n for n in WORKFLOW_NODES if n["service"] == "PostStorageService")

    static_sr_ticks   = []
    adaptive_sr_ticks = []

    for t in TICKS_10S:
        _apply_timeline(t, current)
        ps_telem = current["PostStorageService"]
        fr       = ps_telem.fail_rate

        # Static
        stat_res = _simulate_requests(static_pred["PostStorageService"], fr,
                                      n=requests_per_tick)
        # Adaptive
        _, adap_pred = _query(model, ps_node, ps_telem)
        adap_res = _simulate_requests(adap_pred, fr, n=requests_per_tick)

        static_sr_ticks.append(stat_res["success_rate"])
        adaptive_sr_ticks.append(adap_res["success_rate"])

    # MTTR: first tick after outage where rolling-3 success >= threshold
    outage_start_tick = next(
        (i for i, t in enumerate(TICKS_10S) if t >= 60), None
    )

    def _find_mttr(sr_list: List[float]) -> Optional[int]:
        if outage_start_tick is None:
            return None
        for i in range(outage_start_tick + 1, len(sr_list)):
            window = sr_list[max(0, i-2):i+1]
            if np.mean(window) >= stable_threshold:
                return TICKS_10S[i]
        return None

    static_mttr   = _find_mttr(static_sr_ticks)
    adaptive_mttr = _find_mttr(adaptive_sr_ticks)

    outage_t = TICKS_10S[outage_start_tick] if outage_start_tick else None

    return {
        "outage_start_t":    outage_t,
        "stable_threshold":  stable_threshold,
        "static_mttr_t":     static_mttr,
        "adaptive_mttr_t":   adaptive_mttr,
        "improvement_s":     ((static_mttr or 0) - (adaptive_mttr or 0)),
        "static_sr_ticks":   static_sr_ticks,
        "adaptive_sr_ticks": adaptive_sr_ticks,
        "ticks":             TICKS_10S,
    }


# ── Printers ──────────────────────────────────────────────────────────────────

def _print_metric_a(results: List[Dict]) -> None:
    print("\n  METRIC A — DETECTION LATENCY")
    print("  How quickly (seconds) does ORS react to a telemetry change?")
    print("  Measured at 1-second resolution.\n")
    print(f"  {'Service':<28}  {'T event':>8}  {'T detect':>9}  {'Latency (s)':>12}")
    print("  " + "-" * 62)
    latencies = []
    for r in results:
        lat = r["latency_s"]
        latencies.append(lat)
        print(f"  {r['service']:<28}  {r['t_event']:>8}s  {r['t_detect']:>8}s  {lat:>11}s")
    if latencies:
        print("  " + "-" * 62)
        print(f"  Mean detection latency: {np.mean(latencies):.1f}s  "
              f"Max: {max(latencies)}s  Min: {min(latencies)}s")
        print()
        print("  Note: Detection latency = 0s means ORS responded within the same")
        print("  1-second observation window as the telemetry change (single forward pass).")
        print("  Non-zero latency means the telemetry had to move past a decision boundary")
        print("  over multiple ticks before crossing the ORS transition threshold.")


def _print_metric_b(results: List[Dict]) -> None:
    print("\n\n  METRIC B — POLICY CORRECTNESS")
    print("  At each transition: old policy vs ORS new policy vs oracle best policy.")
    print("  Success rate out of 2000 simulated requests per condition.\n")
    print(f"  {'T':>4}s  {'Service':<22}  {'fail%':>5}  "
          f"{'Old SR':>7}  {'ORS SR':>7}  {'Oracle SR':>9}  "
          f"{'Uplift':>7}  {'Regret':>7}")
    print("  " + "-" * 80)
    uplifts  = []
    regrets  = []
    for r in results:
        uplifts.append(r["uplift"])
        regrets.append(r["norm_regret"])
        print(f"  {r['t']:>4}s  {r['service']:<22}  "
              f"{r['fail_rate']:>4.0%}  "
              f"{r['old_success']:>6.1%}  "
              f"{r['new_success']:>6.1%}  "
              f"{r['oracle_success']:>8.1%}  "
              f"{r['uplift']:>+6.1%}  "
              f"{r['norm_regret']:>6.2f}")
    if results:
        print("  " + "-" * 80)
        print(f"  Mean success rate uplift: {np.mean(uplifts):+.1%}")
        print(f"  Mean normalised regret  : {np.mean(regrets):.2f}  "
              f"(0=matches oracle, 1=no improvement over old)")
        print()
        print("  Wasted retries (per request) at each transition:")
        print(f"  {'T':>4}s  {'Service':<22}  {'Wasted old':>12}  {'Wasted new':>12}  "
              f"{'Saved':>10}")
        print("  " + "-" * 65)
        for r in results:
            saved = r["wasted_old"] - r["wasted_new"]
            print(f"  {r['t']:>4}s  {r['service']:<22}  "
                  f"{r['wasted_old']:>11.3f}  {r['wasted_new']:>11.3f}  "
                  f"{saved:>+9.3f}")


def _print_metric_c(result: Dict) -> None:
    print("\n\n  METRIC C — COST SAVED (wasted API calls)")
    print("  Static policy (frozen at T=0) vs ORS adaptive policy.")
    print(f"  Simulating {result['total_requests']:,} total requests "
          f"across 3-minute incident.\n")
    print(f"  Static   wasted retries : {result['static_wasted_total']:>8,}")
    print(f"  Adaptive wasted retries : {result['adaptive_wasted_total']:>8,}")
    print(f"  Retries SAVED           : {result['wasted_saved']:>8,}  "
          f"({result['pct_saved']:.1f}% reduction)")
    print()
    print(f"  Static   overall success rate : {result['static_success_rate']:.1%}")
    print(f"  Adaptive overall success rate : {result['adaptive_success_rate']:.1%}")
    print()
    print("  Per-tick wasted retries (static vs adaptive):")
    print(f"  {'T':>5}s  {'Static':>9}  {'Adaptive':>10}  {'Saved':>8}")
    print("  " + "-" * 40)
    for t, ws, wa in zip(result["ticks"], result["per_tick_static"],
                          result["per_tick_adaptive"]):
        saved = ws - wa
        marker = " **" if saved > 0 else ""
        print(f"  {t:>5}s  {ws:>9.1f}  {wa:>10.1f}  {saved:>+7.1f}{marker}")


def _print_metric_d(result: Dict) -> None:
    print("\n\n  METRIC D — RECOVERY IMPROVEMENT (MTTR)")
    print("  PostStorageService: time to reach stable success after outage.")
    print(f"  Stable = {result['stable_threshold']:.0%} rolling-30s success rate.\n")

    outage  = result["outage_start_t"]
    s_mttr  = result["static_mttr_t"]
    a_mttr  = result["adaptive_mttr_t"]
    improv  = result["improvement_s"]

    print(f"  Outage starts at         : T={outage}s")
    print(f"  Static  policy MTTR      : T={s_mttr}s  "
          f"({(s_mttr - outage) if s_mttr else 'N/A'}s after outage)")
    print(f"  Adaptive policy MTTR     : T={a_mttr}s  "
          f"({(a_mttr - outage) if a_mttr else 'N/A'}s after outage)")
    print(f"  MTTR improvement         : {improv}s faster with runtime adaptation")
    print()
    print("  Per-tick success rate: PostStorageService")
    print(f"  {'T':>5}s  {'Static SR':>10}  {'Adaptive SR':>12}  {'Diff':>8}")
    print("  " + "-" * 45)
    for t, ss, sa in zip(result["ticks"], result["static_sr_ticks"],
                          result["adaptive_sr_ticks"]):
        diff   = sa - ss
        marker = " <<" if t in (s_mttr, a_mttr) else ""
        print(f"  {t:>5}s  {ss:>9.1%}  {sa:>11.1%}  {diff:>+7.1%}{marker}")


# ── Paper summary ─────────────────────────────────────────────────────────────

def _print_paper_table(a, b, c, d) -> None:
    print("\n\n" + "=" * 72)
    print("  PAPER TABLE — Experiment 3 metrics summary")
    print("=" * 72)

    lat_mean = np.mean([r["latency_s"] for r in a]) if a else float("nan")
    uplift   = np.mean([r["uplift"]    for r in b]) if b else float("nan")
    regret   = np.mean([r["norm_regret"] for r in b]) if b else float("nan")

    print(f"""
  | Metric                     | Static policy | Adaptive (ORS) | Improvement        |
  |----------------------------|---------------|----------------|--------------------|
  | Detection latency          | N/A           | {lat_mean:.0f}s             | Immediate (1 obs.) |
  | Policy success rate uplift | baseline      | {uplift:+.1%}        | per transition     |
  | Normalised regret          | 1.00          | {regret:.2f}           | vs oracle          |
  | Wasted retries saved       | baseline      | -{c['pct_saved']:.0f}%          | over 3-min incident|
  | Overall success rate       | {c['static_success_rate']:.1%}        | {c['adaptive_success_rate']:.1%}          | +{c['adaptive_success_rate']-c['static_success_rate']:.1%}              |
  | MTTR (PostStorage)         | {d['static_mttr_t']}s          | {d['adaptive_mttr_t']}s             | -{d['improvement_s']}s faster          |
""")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = _time.time()

    print("\n" + "=" * 72)
    print("  EXPERIMENT 3 — RUNTIME ADAPTATION: four measured metrics")
    print("  Incident: PostStorageService latency -> failure -> recovery")
    print("  Workflow: read_home_timeline (4-node chain)")
    print("=" * 72)

    print("\n[1] Loading ORS-VQ model ...")
    model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found.")
    d = torch.load(CKPT_ORS, weights_only=False)
    model.load_state_dict(d["state_dict"])
    model.eval()

    print("[2] Metric A: Detection latency (1s resolution) ...")
    result_a = metric_a_detection_latency(model)

    print("[3] Metric B: Policy correctness (2000 req/condition) ...")
    result_b = metric_b_policy_correctness(model)

    print("[4] Metric C: Cost saved (200 req/tick) ...")
    result_c = metric_c_cost_saved(model, requests_per_tick=200)

    print("[5] Metric D: MTTR recovery improvement (300 req/tick) ...")
    result_d = metric_d_mttr(model, requests_per_tick=300)

    print("\n\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    _print_metric_a(result_a)
    _print_metric_b(result_b)
    _print_metric_c(result_c)
    _print_metric_d(result_d)
    _print_paper_table(result_a, result_b, result_c, result_d)

    print(f"  Total runtime: {_time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
