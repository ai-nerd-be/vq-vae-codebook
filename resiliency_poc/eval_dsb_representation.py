"""
eval_dsb_representation.py  --  Representation quality experiments on real DSB data

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.eval_dsb_representation

Same experiments as eval_representation.py (E5-E10), but on real DeathStarBench
socialNetwork X-Trace traces instead of synthetic n8n data.

  E5  Codebook utilization
  E6  Policy consistency
  E7  Telemetry variance
  E8  Robustness under noise
  E9  Transfer: train on HEALTHY DSB workloads only, test on FAILURE workloads
  E10 ORS as retrieval memory

All experiment functions are reused from eval_representation.py.
"""

from __future__ import annotations
import math, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .dsb_loader               import DSBLoader, FOLDER_SCENARIO
from .temporal_trace_generator import TemporalFailureEvent, TEMPORAL_FEATURE_DIM
from .scenario                 import SCENARIO_NAMES, ScenarioState
from .label_oracle             import RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS
from .model.ors_model          import ORSVQModel

# Reuse all experiment functions from eval_representation
from .eval_representation import (
    _minority_oversample, _tensors, _collect_codes, _train_ors,
    _e5_codebook_utilization,
    _e6_policy_consistency,
    _e7_telemetry_variance,
    _e8_robustness,
    _e10_retrieval,
    TELEMETRY_FEATURES, CODEBOOK_SIZE, EPOCHS, BATCH_SIZE, LR,
)

SEED    = 42
N_TRAIN = 3_000
N_TEST  =   600

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CKPT_DIR        = Path(__file__).parent / "checkpoints"
CKPT_ORS        = CKPT_DIR / "ors_dsb_vq.pt"
CKPT_WFA_DSB    = CKPT_DIR / "ors_dsb_healthy_only.pt"


# ── E9 DSB variant ─────────────────────────────────────────────────────────────
# Train on HEALTHY workload folders only, test on FAILURE folders.
# Measures how well ORS codes learned on healthy traffic generalise to
# failure scenarios — zero-shot transfer from normal to anomalous operation.

def _e9_dsb_transfer(train_events: List[TemporalFailureEvent],
                     test_events:  List[TemporalFailureEvent]) -> None:
    """
    E9 for DSB:
      Group A (train): only HEALTHY-scenario events
      Group B (test):  only FAILURE-scenario events (TRANSIENT/DEGRADED/DOWN)
    Compare WF-A-only model vs full joint model on the failure test set.
    """
    HEALTHY_FOLDERS  = {"follow", "unfollow", "register", "write_home_timeline", "home-timeline-read"}

    def _folder(ev: TemporalFailureEvent) -> str:
        return ev.workflow_name.split("/")[-1] if "/" in ev.workflow_name else ev.workflow_name

    healthy_train = [ev for ev in train_events if ev.scenario == ScenarioState.HEALTHY]
    failure_test  = [ev for ev in test_events  if ev.scenario != ScenarioState.HEALTHY]

    print(f"\n[E9] CROSS-WORKLOAD TRANSFER (DSB)")
    print(f"  Train: HEALTHY workload events only  ({len(healthy_train)} events)")
    print(f"  Test:  FAILURE workload events only  ({len(failure_test)} events)")
    if not failure_test:
        print("  No failure events in test set — skipping.")
        return

    # Oversample healthy training set (already healthy-only, no oversample needed)
    X_h, Y_h = _tensors(healthy_train, noise=0.05)
    loader_h  = DataLoader(TensorDataset(X_h, Y_h), BATCH_SIZE, shuffle=True)

    # Healthy-only model
    ors_h = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if CKPT_WFA_DSB.exists():
        d = torch.load(CKPT_WFA_DSB, weights_only=False)
        ors_h.load_state_dict(d["state_dict"])
        ors_h.eval()
        print(f"  Loaded healthy-only model from {CKPT_WFA_DSB.name}")
    else:
        print(f"  Training healthy-only ORS model ({EPOCHS} epochs) ...")
        _train_ors(ors_h, loader_h, EPOCHS, "ORS-HEALTHY")
        torch.save({"state_dict": ors_h.state_dict()}, CKPT_WFA_DSB)

    # Full joint model (already trained on all DSB scenarios)
    ors_joint = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run ors_dsb_poc.py first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors_joint.load_state_dict(d["state_dict"])
    ors_joint.eval()

    def _f1_exact(model, evs):
        from sklearn.metrics import f1_score
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for ev in evs:
                x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                pred = model.predict(x)
                y_true.append([ev.label_retry, ev.label_timeout, ev.label_cb, ev.label_fallback])
                y_pred.append([pred.retry, pred.timeout, pred.cb, pred.fallback])
        yt = np.array(y_true); yp = np.array(y_pred)
        per_dim = [f1_score(yt[:, i], yp[:, i], average="macro", zero_division=0) for i in range(4)]
        return np.mean(per_dim), np.all(yt == yp, axis=1).mean()

    f1_h, ex_h = _f1_exact(ors_h,     failure_test)
    f1_j, ex_j = _f1_exact(ors_joint, failure_test)

    codes_train = set(_collect_codes(ors_h, healthy_train).tolist())
    codes_test  = set(_collect_codes(ors_h, failure_test).tolist())
    shared      = codes_train & codes_test
    reuse_pct   = len(shared) / max(len(codes_train), 1)

    W = 90
    print("\n" + "=" * W)
    print("  E9: CROSS-WORKLOAD TRANSFER RESULTS  (DSB)")
    print(f"  Healthy-only train: {len(healthy_train)} events")
    print(f"  Failure-only test:  {len(failure_test)} events")
    print("=" * W)
    print(f"  {'Model':<28}  {'Avg F1':>8}  {'Exact match':>12}")
    print("  " + "-" * (W - 2))
    print(f"  {'ORS-VQ (healthy-only train)':<28}  {f1_h:>7.1%}  {ex_h:>11.1%}")
    print(f"  {'ORS-VQ (joint DSB train)':<28}  {f1_j:>7.1%}  {ex_j:>11.1%}")
    print("  " + "-" * (W - 2))
    print(f"  Codes seen in healthy training:  {sorted(codes_train)}")
    print(f"  Codes activated on failure test: {sorted(codes_test)}")
    print(f"  Shared codes: {sorted(shared)}  ({reuse_pct:.0%} of healthy codes reused)")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[Setup] Loading DeathStarBench X-Trace dataset ...")
    dsb = DSBLoader(seed=SEED)
    train_raw, test_events = dsb.load_dataset(n_train=N_TRAIN, n_test=N_TEST, verbose=True)
    train_events = _minority_oversample(train_raw)
    print(f"  Train (oversampled): {len(train_events)}  |  Test: {len(test_events)}")

    # Load ORS-VQ trained on DSB
    print("\n[Setup] Loading ORS-VQ checkpoint (ors_dsb_vq.pt) ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run ors_dsb_poc.py first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors.load_state_dict(d["state_dict"])
    ors.eval()

    # ── E5 ───────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e5_codebook_utilization(ors, test_events)

    # ── E6 ───────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e6_policy_consistency(ors, test_events)

    # ── E7 ───────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e7_telemetry_variance(ors, test_events)

    # ── E8 ───────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e8_robustness(ors, test_events)

    # ── E9 ───────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e9_dsb_transfer(train_events, test_events)

    # ── E10 ──────────────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("\n[E10] ORS as retrieval memory ...")
    _e10_retrieval(ors, train_events, test_events)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
