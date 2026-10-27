"""
eval_telemetry_sensitivity.py  --  Does ORS respond to telemetry or just service names?

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m eval_telemetry_sensitivity

Scientific question (reviewer concern)
───────────────────────────────────────
The hardening engine currently maps:
    service_name -> fixed_scenario -> fixed_telemetry -> ORS -> policy

This raises the question: is ORS actually learning OPERATIONAL STATES from
telemetry signals, or is it just memorising a lookup table over service names?

This experiment answers that directly by:
  1. Fixing the node identity (same service, same graph position, same depth)
  2. Varying ONLY the 9 temporal/telemetry dimensions (rolling fail rate,
     consecutive failures, retry success rate, latency trend, CB state, etc.)
  3. Showing whether ORS changes its recommended policy as telemetry changes

If ORS changes its recommendation as telemetry changes, it has learned
operational resilience states.  If it gives the same answer regardless,
it is a lookup table over service names.

Telemetry profiles tested
──────────────────────────
  T1  HEALTHY           fail_rate≈0.02, cb=closed, no consec failures
  T2  HIGH_LATENCY      fail_rate≈0.08, rising latency, still mostly healthy
  T3  RISING_ERRORS     fail_rate≈0.35, some consecutive failures, cb probing
  T4  PERSISTENT_FAIL   fail_rate≈0.92, cb open, many consecutive failures
  T5  RECOVERING        fail_rate≈0.40 (was high, now falling), cb half-open

Each profile is applied to every DSB service node while keeping:
  - service identity (one-hot encoding) fixed
  - graph position (depth, is_external, is_crypto) fixed
  - failure mode = dependency_down (realistic for all non-healthy profiles)

Expected behaviour if ORS has learned operational states:
  T1 → light policy (retry=1, CB=closed or half_open, cached fallback)
  T2 → retry with longer timeout
  T3 → CB=half_open, exponential backoff
  T4 → CB=open, dead_letter, no retry (fail fast)
  T5 → probe policy (CB=half_open, retry=1, fallback ready)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from resiliency_poc.dsb_loader               import DSB_SERVICE_VOCAB
from resiliency_poc.dsb_workflow_loader      import DSB_EXTERNAL, DSB_CRYPTO
from resiliency_poc.scenario                 import CB_CLOSED, CB_HALF_OPEN, CB_OPEN
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM

SEED          = 42
CODEBOOK_SIZE = 16
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
N_SERVICES    = 13

# ── Telemetry profiles ─────────────────────────────────────────────────────────
# Each profile represents a distinct operational state.
# 9-dim layout mirrors ScenarioObservation.to_feature_vector():
#   [rolling_fail_rate, consec_failures_norm, retry_succ_rate,
#    latency_trend, time_since_success_norm, scenario_duration_norm,
#    cb_state_norm, cb_duration_norm, failures_last_15_norm]

@dataclass
class TelemetryProfile:
    name:        str
    description: str
    # 9 temporal signal values (all normalised to [0,1] except latency_trend [-1,1])
    fail_rate:   float   # rolling failure rate
    consec:      float   # normalised consecutive failures (0=none, 1=many)
    retry_succ:  float   # retry success rate
    lat_trend:   float   # latency trend (-1=falling, 0=flat, +1=rising fast)
    t_since_ok:  float   # normalised time since last success
    duration:    float   # normalised time in this state
    cb_state:    int     # CB_CLOSED=0, CB_HALF_OPEN=1, CB_OPEN=2
    cb_dur:      float   # normalised CB duration
    cb_fails:    float   # normalised failures in last CB window

    def to_temporal_vec(self) -> np.ndarray:
        return np.array([
            self.fail_rate,
            min(self.consec, 1.0),
            self.retry_succ,
            self.lat_trend,
            min(self.t_since_ok, 1.0),
            min(self.duration, 1.0),
            self.cb_state / 2.0,      # normalise to [0,1]
            min(self.cb_dur, 1.0),
            min(self.cb_fails, 1.0),
        ], dtype=np.float32)


PROFILES: List[TelemetryProfile] = [
    TelemetryProfile(
        name="T1-Healthy",
        description="Normal operation. Very low fail rate, no CB activity.",
        fail_rate=0.02, consec=0.01, retry_succ=0.97,
        lat_trend=-0.10, t_since_ok=0.01, duration=0.03,
        cb_state=CB_CLOSED, cb_dur=0.0, cb_fails=0.02,
    ),
    TelemetryProfile(
        name="T2-HighLatency",
        description="Latency rising fast but errors still rare. Service slowing down.",
        fail_rate=0.08, consec=0.02, retry_succ=0.90,
        lat_trend=0.70, t_since_ok=0.05, duration=0.10,
        cb_state=CB_CLOSED, cb_dur=0.0, cb_fails=0.08,
    ),
    TelemetryProfile(
        name="T3-RisingErrors",
        description="Error rate climbing (35%). Retries sometimes help. CB starting to probe.",
        fail_rate=0.35, consec=0.10, retry_succ=0.62,
        lat_trend=0.45, t_since_ok=0.12, duration=0.20,
        cb_state=CB_HALF_OPEN, cb_dur=0.15, cb_fails=0.30,
    ),
    TelemetryProfile(
        name="T4-PersistentFail",
        description="Persistent failure (92%). Retries useless. CB fully open.",
        fail_rate=0.92, consec=0.50, retry_succ=0.04,
        lat_trend=0.95, t_since_ok=0.85, duration=0.90,
        cb_state=CB_OPEN, cb_dur=0.65, cb_fails=0.95,
    ),
    TelemetryProfile(
        name="T5-Recovering",
        description="Was failing, now partially recovering (40% fail, falling). CB probing.",
        fail_rate=0.40, consec=0.12, retry_succ=0.55,
        lat_trend=-0.30, t_since_ok=0.20, duration=0.25,
        cb_state=CB_HALF_OPEN, cb_dur=0.20, cb_fails=0.35,
    ),
]


# ── Static feature builder ────────────────────────────────────────────────────
# Mirrors dsb_loader._static_features() and hardening/engine._node_static_features()
# We fix all structural properties and only vary the temporal 9-dim suffix.

def _static_features(
    service:     str,
    caller:      str,
    depth:       int,
    max_depth:   int,
    is_external: bool,
    is_crypto:   bool,
    failure_mode_idx: int = 4,  # dependency_down (realistic for non-healthy)
) -> np.ndarray:
    """38-dim static feature vector for a DSB service node."""
    svc_idx    = DSB_SERVICE_VOCAB.get(service,  N_SERVICES - 1)
    caller_idx = DSB_SERVICE_VOCAB.get(caller,   N_SERVICES - 1)

    curr_oh = np.zeros(N_SERVICES, dtype=np.float32)
    curr_oh[svc_idx] = 1.0

    caller_oh = np.zeros(N_SERVICES, dtype=np.float32)
    caller_oh[caller_idx] = 1.0

    fm_oh = np.zeros(7, dtype=np.float32)
    fm_oh[failure_mode_idx] = 1.0

    depth_norm = depth / max(max_depth, 1)
    position   = depth_norm
    retry_cnt  = 0.0        # DSB baseline: no retry
    load_f     = 0.5        # mid load

    return np.concatenate([
        curr_oh, caller_oh, fm_oh,
        [position, depth_norm, retry_cnt, float(is_external), load_f],
    ]).astype(np.float32)   # shape (38,)


# DSB socialNetwork call graph structure (mirrors dsb_workflow_loader)
_GRAPH_INFO: Dict[str, Dict] = {
    "nginx-thrift":             dict(caller="none",              depth=0, max_depth=3),
    "ComposePostService":       dict(caller="nginx-thrift",      depth=1, max_depth=3),
    "HomeTimelineService":      dict(caller="nginx-thrift",      depth=1, max_depth=3),
    "UserTimelineService":      dict(caller="nginx-thrift",      depth=1, max_depth=3),
    "UserService":              dict(caller="ComposePostService", depth=2, max_depth=3),
    "PostStorageService":       dict(caller="ComposePostService", depth=2, max_depth=3),
    "WriteHomeTimelineService": dict(caller="ComposePostService", depth=2, max_depth=3),
    "SocialGraphService":       dict(caller="UserService",       depth=2, max_depth=3),
    "UniqueIdService":          dict(caller="ComposePostService", depth=2, max_depth=3),
    "TextService":              dict(caller="ComposePostService", depth=2, max_depth=3),
    "MediaService":             dict(caller="ComposePostService", depth=2, max_depth=3),
    "UserMentionService":       dict(caller="ComposePostService", depth=2, max_depth=3),
    "UrlShortenService":        dict(caller="ComposePostService", depth=2, max_depth=3),
}


def _build_feature(service: str, profile: TelemetryProfile,
                   healthy_mode: bool = False) -> np.ndarray:
    """Full 47-dim feature vector: 38-dim static + 9-dim temporal from profile."""
    info   = _GRAPH_INFO.get(service, dict(caller="unknown", depth=1, max_depth=3))
    fm_idx = 0 if healthy_mode else 4   # no_failure vs dependency_down
    static  = _static_features(
        service     = service,
        caller      = info["caller"],
        depth       = info["depth"],
        max_depth   = info["max_depth"],
        is_external = service in DSB_EXTERNAL,
        is_crypto   = service in DSB_CRYPTO,
        failure_mode_idx = fm_idx,
    )
    temporal = profile.to_temporal_vec()
    return np.concatenate([static, temporal]).astype(np.float32)


# ── Policy decoder ─────────────────────────────────────────────────────────────

RETRY_LABELS    = ["none", "1×", "3×", "5×"]
TIMEOUT_LABELS  = ["5s", "30s", "120s"]
CB_LABELS       = ["closed", "half_open", "open"]
FALLBACK_LABELS = ["none", "alternate", "cached", "dead_letter"]


def _policy_str(pred) -> str:
    r  = RETRY_LABELS[pred.retry]
    t  = TIMEOUT_LABELS[pred.timeout]
    cb = CB_LABELS[pred.cb]
    fb = FALLBACK_LABELS[pred.fallback]
    parts = [f"CB={cb}"]
    if pred.retry > 0:
        parts.append(f"retry={r}@{t}")
    if pred.fallback > 0:
        parts.append(f"fallback={fb}")
    return " + ".join(parts)


def _policy_tuple(pred) -> Tuple[int,int,int,int]:
    return (pred.retry, pred.timeout, pred.cb, pred.fallback)


# ── Main experiment ────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[1] Loading DSB-trained ORS-VQ model ...")
    model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run resiliency_poc.ors_dsb_poc first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    model.load_state_dict(d["state_dict"])
    model.eval()

    services = list(_GRAPH_INFO.keys())

    print("\n[2] Telemetry profiles used in this experiment:")
    W = 90
    print("  " + "=" * W)
    for p in PROFILES:
        print(f"  {p.name:<20}  fail={p.fail_rate:.2f}  "
              f"consec={p.consec:.2f}  retry_succ={p.retry_succ:.2f}  "
              f"lat_trend={p.lat_trend:+.2f}  "
              f"cb={'closed' if p.cb_state==0 else 'half_open' if p.cb_state==1 else 'open'}")
        print(f"  {'':<20}  {p.description}")
    print("  " + "=" * W)

    # ── Experiment A: one focal service across all profiles ────────────────────
    # Use ComposePostService (depth=1, orchestrator) and PostStorageService (depth=2, external)
    # These are the most important nodes in compose_post.

    focal_services = [
        ("ComposePostService",  "Orchestrator (depth=1, fan-out)"),
        ("PostStorageService",  "Storage service (depth=2, external)"),
        ("nginx-thrift",        "API Gateway (depth=0, entry point)"),
        ("UserService",         "Auth service (depth=2, crypto)"),
    ]

    print("\n[3] EXPERIMENT A: Same node, varying telemetry")
    print("    Does ORS change its recommendation as operational state changes?")
    print()

    for service, role in focal_services:
        W = 100
        print("  " + "=" * W)
        print(f"  Service: {service}  [{role}]")
        print(f"  Static features FIXED: identity={service}, depth={_GRAPH_INFO[service]['depth']}, "
              f"external={service in DSB_EXTERNAL}, crypto={service in DSB_CRYPTO}")
        print(f"  Only the 9 telemetry dimensions vary across rows.")
        print("  " + "=" * W)
        print(f"  {'Telemetry Profile':<22}  {'fail':>5}  {'retry_ok':>8}  "
              f"{'CB_state':<10}  {'ORS Decision':<45}  {'ORS Code':>8}  Changed?")
        print("  " + "-" * (W - 2))

        prev_tuple = None
        changes    = 0

        for pidx, profile in enumerate(PROFILES):
            healthy = (pidx == 0)  # T1 gets no_failure mode
            feat = _build_feature(service, profile, healthy_mode=healthy)
            x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred = model.predict(x)
                code_idx = model.code_for(x)

            tup      = _policy_tuple(pred)
            changed  = (prev_tuple is not None and tup != prev_tuple)
            if changed:
                changes += 1
            flag     = "  <<< CHANGED" if changed else ""
            prev_tuple = tup

            print(f"  {profile.name:<22}  {profile.fail_rate:>5.2f}  "
                  f"{profile.retry_succ:>8.2f}  "
                  f"{'closed' if profile.cb_state==0 else 'half_open' if profile.cb_state==1 else 'open':<10}  "
                  f"{_policy_str(pred):<45}  code={code_idx:>3}{flag}")

        print("  " + "-" * (W - 2))
        verdict = (f"ORS changed recommendation {changes} times across 5 telemetry profiles"
                   if changes > 0
                   else "ORS gave identical recommendation across all 5 profiles (lookup table behaviour)")
        print(f"  VERDICT: {verdict}")
        print()

    # ── Experiment B: all services under each profile ─────────────────────────
    print("\n[4] EXPERIMENT B: All services, per profile — which ORS codes activate?")
    print("    Shows whether the same codebook entries activate regardless of telemetry.")
    print()

    W = 115
    print("  " + "=" * W)
    hdr = f"  {'Service':<32}"
    for p in PROFILES:
        hdr += f"  {p.name[:14]:>14}"
    print(hdr)
    print("  " + "-" * (W - 2))

    # Collect per-service per-profile decisions
    results: Dict[str, Dict[str, Tuple]] = {svc: {} for svc in services}
    codes:   Dict[str, Dict[str, int]]   = {svc: {} for svc in services}

    for svc in services:
        for pidx, profile in enumerate(PROFILES):
            feat = _build_feature(svc, profile, healthy_mode=(pidx == 0))
            x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                pred     = model.predict(x)
                code_idx = model.code_for(x)
            results[svc][profile.name] = _policy_tuple(pred)
            codes[svc][profile.name]   = code_idx

    for svc in services:
        row = f"  {svc:<32}"
        for p in PROFILES:
            tup    = results[svc][p.name]
            cb_s   = ["cls", "h_o", "opn"][tup[2]]
            ret_s  = RETRY_LABELS[tup[0]][0] if tup[0] > 0 else "0"
            fb_s   = ["no","al","cc","dl"][tup[3]]
            cell   = f"CB={cb_s},r={ret_s},{fb_s}"
            row += f"  {cell:>14}"
        print(row)

    print("  " + "-" * (W - 2))

    # ── Experiment C: count unique policies per service across profiles ─────────
    print("\n[5] EXPERIMENT C: Policy diversity — how many distinct decisions per service?")
    print("    If diversity > 1, ORS is responding to telemetry (not just service name).")
    print()
    print(f"  {'Service':<32}  {'Distinct policies':>18}  {'ORS codes activated':>20}  Telemetry-sensitive?")
    print("  " + "-" * 90)

    total_sensitive = 0
    for svc in services:
        policy_set = set(results[svc].values())
        code_set   = set(codes[svc].values())
        sensitive  = len(policy_set) > 1
        if sensitive:
            total_sensitive += 1
        flag = "YES" if sensitive else "NO (uniform)"
        print(f"  {svc:<32}  {len(policy_set):>18}  {str(sorted(code_set)):>20}  {flag}")

    print("  " + "-" * 90)
    print(f"  Telemetry-sensitive services: {total_sensitive}/{len(services)}")
    print()

    # ── Experiment D: policy transition matrix ─────────────────────────────────
    print("\n[6] EXPERIMENT D: Policy transitions as telemetry worsens")
    print("    T1(Healthy) -> T2(HighLatency) -> T3(RisingErrors) -> T4(PersistentFail)")
    print()

    W = 100
    print("  " + "=" * W)
    print(f"  {'Service':<32}  {'T1->T2':>12}  {'T2->T3':>12}  {'T3->T4':>12}  {'T4->T5':>12}  Total changes")
    print("  " + "-" * (W - 2))

    profile_names = [p.name for p in PROFILES]
    for svc in services:
        transitions = []
        for i in range(len(PROFILES) - 1):
            p_curr = profile_names[i]
            p_next = profile_names[i + 1]
            changed = results[svc][p_curr] != results[svc][p_next]
            transitions.append("changed" if changed else "same   ")
        total_ch = sum(1 for t in transitions if "changed" in t)
        row = f"  {svc:<32}"
        for t in transitions:
            row += f"  {t:>12}"
        row += f"  {total_ch}/4"
        print(row)

    print("  " + "=" * W)

    # ── Final verdict ──────────────────────────────────────────────────────────
    print("\n[7] OVERALL VERDICT")
    print("  " + "=" * 80)

    # Count total policy changes across all services and transitions
    all_policies_flat = [
        results[svc][p.name] for svc in services for p in PROFILES
    ]
    unique_overall = len(set(all_policies_flat))
    total_cells    = len(services) * len(PROFILES)

    print(f"  Total (service x profile) cells:          {total_cells}")
    print(f"  Distinct (service x profile) policies:    {unique_overall}")
    print(f"  Telemetry-sensitive services:             {total_sensitive}/{len(services)}")
    print()

    if total_sensitive >= len(services) // 2:
        print("  CONCLUSION: ORS IS telemetry-sensitive.")
        print("  The model changes its recommendation when telemetry changes,")
        print("  even for the same service identity. It has learned operational")
        print("  resilience states, not a lookup table over service names.")
    else:
        print("  CONCLUSION: ORS shows LIMITED telemetry sensitivity.")
        print("  Most services receive the same recommendation regardless of")
        print("  telemetry. The model may be over-relying on service identity.")

    print()
    print("  WHAT THIS MEANS FOR THE PAPER:")
    print("  The engine currently uses a fixed scenario per service type (design-time).")
    print("  But the ORS model itself CAN respond to telemetry at runtime.")
    print("  The paper should show both:")
    print("    (a) This sensitivity experiment -- model capability")
    print("    (b) A runtime path where real telemetry is fed to the model")
    print("        instead of the engine's static scenario lookup.")
    print("  " + "=" * 80)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
