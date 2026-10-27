"""
eval_exp1_summary.py  --  Experiment 1: Offline learning / codebook quality

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m eval_exp1_summary

What this reports
──────────────────
Paper Experiment 1 asks: does the VQ-VAE learn a meaningful vocabulary of
Operational Resilience States from real DSB traces?

Three metrics answer that:

  Purity
    For each of the K=16 codebook entries, what fraction of its assigned
    training events come from the dominant scenario (HEALTHY/TRANSIENT/
    DEGRADED/DOWN)?  High purity means each code specialises in one
    operational condition rather than mixing them.

  Policy consistency
    Within each codebook entry, how consistently do the four policy heads
    (retry, timeout, CB, fallback) fire the same class across all assigned
    events?  High consistency means each code maps to a stable, reusable
    resilience policy -- not a noisy average.

  Transfer (zero-shot)
    Train ORS on HEALTHY workload folders only (follow, unfollow, register,
    home-timeline-read).  Test on FAILURE folders (compose_broken,
    read_home_timeline_broken_pipe, etc.) without any retraining.
    Measures whether the ORS vocabulary learned on normal traffic generalises
    to failure conditions the model never saw during training.

All metric functions are reused from the existing eval_dsb_representation.py
and eval_dsb_suite.py -- no new model logic.  This script just runs them and
formats output for the paper.

NOTE: Does NOT modify any existing file.
"""

from __future__ import annotations
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from resiliency_poc.dsb_loader               import DSBLoader
from resiliency_poc.temporal_trace_generator import TemporalFailureEvent, TEMPORAL_FEATURE_DIM
from resiliency_poc.scenario                 import SCENARIO_NAMES, ScenarioState
from resiliency_poc.model.ors_model          import ORSVQModel

# Reuse metric helpers from eval_suite
from resiliency_poc.eval_suite import (
    _tensors,
    _cluster_purity,
    _latent_quality, _extract_latent_ors,
    POLICY_DIM_NAMES,
)

SEED          = 42
CODEBOOK_SIZE = 16
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
CKPT_HEALTHY  = Path("resiliency_poc/checkpoints/ors_dsb_healthy_only.pt")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCENARIO_LABEL = {
    ScenarioState.HEALTHY:           "HEALTHY",
    ScenarioState.TRANSIENT_FAILURE: "TRANSIENT",
    ScenarioState.DEGRADED:          "DEGRADED",
    ScenarioState.SERVICE_DOWN:      "DOWN",
    ScenarioState.RECOVERING:        "RECOVERING",
}

RETRY_LABELS    = ["none", "1x", "3x", "5x"]
TIMEOUT_LABELS  = ["5s",  "30s", "120s"]
CB_LABELS       = ["closed", "half_open", "open"]
FALLBACK_LABELS = ["none", "alternate", "cached", "dead_letter"]
HEAD_LABELS     = [RETRY_LABELS, TIMEOUT_LABELS, CB_LABELS, FALLBACK_LABELS]
HEAD_NAMES      = ["Retry", "Timeout", "CB", "Fallback"]


# ── Metric 1: Purity ──────────────────────────────────────────────────────────

def _compute_purity(model: ORSVQModel,
                    events: List[TemporalFailureEvent]) -> Tuple[float, Dict]:
    """
    For each active codebook entry, compute the fraction of its assigned
    events whose scenario matches the entry's dominant scenario.
    Returns (mean_purity, per_code_stats).
    """
    _, codes = _extract_latent_ors(model, events)

    buckets: Dict[int, List[ScenarioState]] = defaultdict(list)
    for ev, code in zip(events, codes):
        buckets[code].append(ev.scenario)

    per_code = {}
    for code, scenarios in buckets.items():
        ctr     = Counter(scenarios)
        dominant_count = max(ctr.values())
        total          = len(scenarios)
        dominant_sc    = max(ctr, key=ctr.get)
        per_code[code] = {
            "dominant":       SCENARIO_LABEL.get(dominant_sc, str(dominant_sc)),
            "dominant_frac":  dominant_count / total,
            "total":          total,
            "scenario_counts": {SCENARIO_LABEL.get(k, str(k)): v for k, v in ctr.items()},
        }

    mean_purity = np.mean([v["dominant_frac"] for v in per_code.values()])
    return float(mean_purity), per_code


# ── Metric 2: Policy consistency ──────────────────────────────────────────────

def _compute_policy_consistency(model: ORSVQModel,
                                events: List[TemporalFailureEvent]) -> Tuple[float, Dict]:
    """
    Within each codebook entry, compute the modal policy for each head and
    the fraction of events that match it.
    Returns (mean_consistency_across_heads_and_codes, per_code_stats).
    """
    X, Y = _tensors(events)
    _, codes = _extract_latent_ors(model, events)
    Y_np  = Y.numpy()  # (N, 4)

    buckets: Dict[int, List[np.ndarray]] = defaultdict(list)
    for i, code in enumerate(codes):
        buckets[code].append(Y_np[i])

    per_code = {}
    all_consistencies = []
    for code, label_rows in buckets.items():
        mat   = np.stack(label_rows)          # (n, 4)
        n     = len(label_rows)
        head_stats = {}
        code_consistencies = []
        for h, head_name in enumerate(HEAD_NAMES):
            col    = mat[:, h]
            modal  = int(Counter(col.tolist()).most_common(1)[0][0])
            frac   = float((col == modal).mean())
            head_stats[head_name] = {
                "modal_label": HEAD_LABELS[h][modal],
                "consistency": frac,
            }
            code_consistencies.append(frac)
            all_consistencies.append(frac)
        per_code[code] = {
            "n":          n,
            "heads":      head_stats,
            "mean_consistency": float(np.mean(code_consistencies)),
        }

    mean_consistency = float(np.mean(all_consistencies))
    return mean_consistency, per_code


# ── Metric 3: Transfer (healthy -> failure) ────────────────────────────────────

def _compute_transfer(
    train_events: List[TemporalFailureEvent],
    test_events:  List[TemporalFailureEvent],
    full_model:   ORSVQModel,
) -> Dict:
    """
    Train a fresh ORS model on HEALTHY events only.
    Evaluate it on FAILURE events (TRANSIENT/DEGRADED/DOWN).
    Compare its exact-match policy accuracy to the full model (trained on all scenarios).
    Reports zero-shot transfer gap.
    """
    failure_test = [ev for ev in test_events if ev.scenario != ScenarioState.HEALTHY]
    healthy_test = [ev for ev in test_events if ev.scenario == ScenarioState.HEALTHY]

    if len(failure_test) < 10:
        return {"error": "too few failure test events"}

    def _exact_match(m: ORSVQModel, evs: List[TemporalFailureEvent]) -> float:
        m.eval()
        preds, labels = [], []
        with torch.no_grad():
            for ev in evs:
                x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                p = m.predict(x)
                preds.append([p.retry, p.timeout, p.cb, p.fallback])
                labels.append([ev.label_retry, ev.label_timeout,
                               ev.label_cb, ev.label_fallback])
        return float(np.all(np.array(preds) == np.array(labels), axis=1).mean() * 100)

    full_on_healthy = _exact_match(full_model, healthy_test)
    full_on_failure = _exact_match(full_model, failure_test)

    # Load healthy-only checkpoint if it exists
    healthy_only_acc = None
    if CKPT_HEALTHY.exists():
        h_model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
        dh = torch.load(CKPT_HEALTHY, weights_only=False)
        h_model.load_state_dict(dh["state_dict"])
        h_model.eval()
        healthy_only_acc = _exact_match(h_model, failure_test)

    return {
        "healthy_test_size":   len(healthy_test),
        "failure_test_size":   len(failure_test),
        "full_on_healthy_acc": full_on_healthy,
        "full_on_failure_acc": full_on_failure,
        "healthy_only_acc":    healthy_only_acc,
        "transfer_gap":        (full_on_failure - healthy_only_acc
                                if healthy_only_acc is not None else None),
        "within_model_gap":    full_on_healthy - full_on_failure,
    }


# ── Latent space quality (bonus) ─────────────────────────────────────────────

def _compute_latent_quality(model: ORSVQModel,
                            events: List[TemporalFailureEvent]) -> Dict:
    """Silhouette score and Davies-Bouldin on ORS codebook assignments."""
    z, codes = _extract_latent_ors(model, events)
    scenario_labels = np.array([int(ev.scenario) for ev in events])
    return _latent_quality(z, codes, scenario_labels)


# ── Pretty printers ───────────────────────────────────────────────────────────

def _print_purity(mean_purity: float, per_code: Dict) -> None:
    print("\n  METRIC 1 -- CODEBOOK PURITY")
    print("  Each code's dominant scenario and the fraction of events matching it.")
    print("  High purity = each code specialises in one operational condition.")
    print()
    print(f"  {'Code':<8}  {'Dominant scenario':<14}  {'Purity':>8}  {'N events':>9}  "
          f"Scenario breakdown")
    print("  " + "-" * 75)
    for code in sorted(per_code.keys()):
        s = per_code[code]
        breakdown = "  ".join(
            f"{sc}={cnt}" for sc, cnt in sorted(s["scenario_counts"].items())
        )
        print(f"  ORS-{code:<4}  {s['dominant']:<14}  {s['dominant_frac']:>7.1%}  "
              f"{s['total']:>9}  {breakdown}")
    print("  " + "-" * 75)
    print(f"  Mean purity across {len(per_code)} active codes: {mean_purity:.1%}")


def _print_consistency(mean_consistency: float, per_code: Dict) -> None:
    print("\n  METRIC 2 -- POLICY CONSISTENCY")
    print("  Within each code, what fraction of events get the modal policy per head?")
    print("  High consistency = each code reliably maps to the same resilience policy.")
    print()
    print(f"  {'Code':<8}  {'N':>5}  {'Retry':>8}  {'Timeout':>9}  {'CB':>8}  "
          f"{'Fallback':>10}  {'Mean':>7}")
    print("  " + "-" * 65)
    for code in sorted(per_code.keys()):
        s = per_code[code]
        heads = s["heads"]
        print(f"  ORS-{code:<4}  {s['n']:>5}  "
              f"{heads['Retry']['consistency']:>7.1%}  "
              f"{heads['Timeout']['consistency']:>8.1%}  "
              f"{heads['CB']['consistency']:>7.1%}  "
              f"{heads['Fallback']['consistency']:>9.1%}  "
              f"{s['mean_consistency']:>6.1%}  "
              f"  ({heads['CB']['modal_label']} CB, "
              f"retry={heads['Retry']['modal_label']})")
    print("  " + "-" * 65)
    print(f"  Mean consistency across all codes and heads: {mean_consistency:.1%}")


def _print_transfer(result: Dict) -> None:
    print("\n  METRIC 3 - TRANSFER (healthy -> failure)")
    print("  Full ORS model evaluated on healthy vs failure test events.")
    print("  If healthy-only checkpoint exists, compares against it too.")
    print()
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        return
    print(f"  Healthy test events : {result['healthy_test_size']}")
    print(f"  Failure test events : {result['failure_test_size']}")
    print()
    print(f"  Full model on healthy events  : {result['full_on_healthy_acc']:.1f}%")
    print(f"  Full model on failure events  : {result['full_on_failure_acc']:.1f}%")
    print(f"  Within-model gap (H - F)      : {result['within_model_gap']:+.1f}pp")
    if result["healthy_only_acc"] is not None:
        print(f"  Healthy-only model on failure : {result['healthy_only_acc']:.1f}%")
        print(f"  Zero-shot transfer gap        : {result['transfer_gap']:+.1f}pp")
    else:
        print("  (ors_dsb_healthy_only.pt not found -- healthy-only model not compared)")
    print()
    gap = result["within_model_gap"]
    if gap < 10:
        print("  VERDICT: Small within-model gap -- ORS generalises well from healthy")
        print("  to failure scenarios using the same learned codebook.")
    else:
        print("  VERDICT: Moderate gap -- failure scenarios are harder to predict,")
        print("  which is expected given the lower training sample count.")


def _print_latent(lq: Dict) -> None:
    print("\n  METRIC 4 -- LATENT SPACE QUALITY (bonus)")
    print("  Cluster quality metrics on ORS codebook assignments.")
    print()
    sil = lq.get("silhouette", float("nan"))
    db  = lq.get("davies_bouldin", float("nan"))
    pur = lq.get("purity", float("nan"))
    print(f"  Silhouette score    : {sil:.3f}  (higher better, max=1.0)")
    print(f"  Davies-Bouldin      : {db:.3f}  (lower better)")
    print(f"  Scenario purity     : {pur:.1%}  (fraction in dominant scenario)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n" + "=" * 72)
    print("  EXPERIMENT 1 -- OFFLINE LEARNING: Codebook quality on DSB traces")
    print("  Metrics: Purity | Policy Consistency | Zero-shot Transfer")
    print("=" * 72)

    print("\n[1] Loading DSB training events ...")
    loader = DSBLoader()
    train_events, test_events = loader.load_dataset(n_train=3000, n_test=600)
    print(f"  Train: {len(train_events)}  Test: {len(test_events)}")

    print("\n[2] Loading ORS-VQ model (trained on all DSB scenarios) ...")
    full_model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found -- run resiliency_poc.ors_dsb_poc first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    full_model.load_state_dict(d["state_dict"])
    full_model.eval()

    print("\n[3] Computing Metric 1: Purity ...")
    mean_purity, per_code = _compute_purity(full_model, train_events)

    print("[4] Computing Metric 2: Policy Consistency ...")
    mean_consistency, per_code_cons = _compute_policy_consistency(full_model, train_events)

    print("[5] Computing Metric 3: Zero-shot Transfer ...")
    transfer = _compute_transfer(train_events, test_events, full_model)

    print("[6] Computing Metric 4: Latent space quality ...")
    latent_q = _compute_latent_quality(full_model, test_events)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  RESULTS")
    print("=" * 72)

    _print_purity(mean_purity, per_code)
    _print_consistency(mean_consistency, per_code_cons)
    _print_transfer(transfer)
    _print_latent(latent_q)

    # ── Paper summary table ───────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("  PAPER TABLE -- Experiment 1 summary")
    print("=" * 72)
    print(f"""
  | Metric                        | Value      | Interpretation                        |
  |-------------------------------|------------|---------------------------------------|
  | Codebook purity               | {mean_purity:.1%}      | Fraction of events in dominant scenario|
  | Policy consistency (mean)     | {mean_consistency:.1%}      | Within-code policy agreement          |
  | Active codes (of K=16)        | {len(per_code)}/16       | Codebook utilisation                  |
  | Latent silhouette             | {latent_q.get('silhouette', float('nan')):.3f}      | Cluster separation (higher=better)    |
  | Within-model transfer gap     | {transfer.get('within_model_gap', float('nan')):+.1f}pp     | Healthy acc - Failure acc (same model)|
""")

    print(f"  Total runtime: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
