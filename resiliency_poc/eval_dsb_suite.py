"""
eval_dsb_suite.py  --  Baseline comparison on real DeathStarBench data

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.eval_dsb_suite

Same experiments as eval_suite.py (E1-E4), but:
  - Training and test data come from real DSB socialNetwork X-Trace traces
  - ORS-VQ loaded from ors_dsb_vq.pt  (trained on DSB, not synthetic n8n)
  - MLP-47 and Cont-AE retrained on DSB data
  - E3 cross-transfer uses DSB workload folders as "workflows"
    (HEALTHY workloads: follow, register vs TRANSIENT: broken_pipe folders)

All model code and metric functions are reused from eval_suite.py.
"""

from __future__ import annotations
import random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.tree        import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb

from .dsb_loader               import DSBLoader
from .temporal_trace_generator import TemporalFailureEvent, TEMPORAL_FEATURE_DIM
from .scenario                 import SCENARIO_NAMES, ScenarioState
from .label_oracle             import (
    RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS,
    get_policy, NUM_RETRY_CLASSES, NUM_TIMEOUT_CLASSES, NUM_CB_CLASSES, NUM_FALLBACK_CLASSES,
)
from .model.ors_model import ORSVQModel

# Import all reusable pieces from eval_suite
from .eval_suite import (
    ContinuousAEModel, MLP47Model, ThresholdBaseline,
    _minority_oversample, _tensors, _numpy_XY,
    _train_neural, _load_neural, _save_neural,
    _predict_all_neural, _predict_threshold,
    _policy_metrics, _print_e1,
    _extract_latent_ors, _extract_latent_neural,
    _latent_quality, _kmeans_clusters, _cluster_purity,
    _code_purity_per_entry, _print_e2,
    _print_e4_ablation,
    POLICY_DIM_NAMES,
)

# ── Config ──────────────────────────────────────────────────────────────────────

SEED = 42
N_TRAIN       = 3_000
N_TEST        =   600
BATCH_SIZE    =    64
LR            =  8e-4
EPOCHS        =    50
CODEBOOK_SIZE =    16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

CKPT_DIR  = Path(__file__).parent / "checkpoints"
CKPT_ORS  = CKPT_DIR / "ors_dsb_vq.pt"
CKPT_MLP  = CKPT_DIR / "eval_dsb_mlp47.pt"
CKPT_AE   = CKPT_DIR / "eval_dsb_cont_ae.pt"


# ── E3: Cross-workload transfer (DSB) ─────────────────────────────────────────
# For DSB, "workflows" = two groups of DSB folders by scenario type:
#   Group A: HEALTHY workloads (follow, register, write_home_timeline)
#   Group B: TRANSIENT workloads (read_home_timeline_broken_pipe, read_user_timeline_broken_pipe)
# We compare which ORS codes fire for the same scenario across the two groups.

def _print_e3_dsb(model: ORSVQModel, test_events: List[TemporalFailureEvent]) -> None:
    """
    Split test events by DSB workload group and show cross-group code agreement.
    Group A = HEALTHY workloads, Group B = TRANSIENT/DOWN workloads.
    """
    W = 110
    print("\n" + "=" * W)
    print("  E3: CROSS-WORKLOAD ORS TRANSFER  (real DSB data)")
    print("  Group A = HEALTHY workloads (follow, register, write_home_timeline)")
    print("  Group B = FAILURE workloads (compose_broken, broken_pipe, register_broken)")
    print("  Claim: same failure scenario => same dominant ORS code across workload groups.")
    print("=" * W)

    HEALTHY_FOLDERS  = {"follow", "unfollow", "register", "write_home_timeline", "home-timeline-read"}
    FAILURE_FOLDERS  = {"compose_broken", "read_home_timeline_broken_pipe",
                        "read_user_timeline_broken_pipe", "register_broken"}

    def _folder(ev: TemporalFailureEvent) -> str:
        # workflow_name is "DSB/<folder>"
        return ev.workflow_name.split("/")[-1] if "/" in ev.workflow_name else ev.workflow_name

    group_a = [ev for ev in test_events if _folder(ev) in HEALTHY_FOLDERS]
    group_b = [ev for ev in test_events if _folder(ev) in FAILURE_FOLDERS]

    def _code_map(events):
        by_sc: Dict[str, Counter] = defaultdict(Counter)
        for ev in events:
            x  = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            c  = model.code_for(x)
            sc = SCENARIO_NAMES[int(ev.scenario)]
            by_sc[sc][c] += 1
        return by_sc

    model.eval()
    with torch.no_grad():
        map_a = _code_map(group_a)
        map_b = _code_map(group_b)

    col = 30
    print(f"  {'Scenario':<14}  {'Group A (healthy wl)':>{col}}  {'Group B (failure wl)':>{col}}  Result")
    print("  " + "-" * (W - 2))

    same = total = 0
    for sc in SCENARIO_NAMES:
        ca = map_a.get(sc)
        cb = map_b.get(sc)
        if not ca or not cb:
            continue
        dom_a, n_a, tot_a = ca.most_common(1)[0][0], ca.most_common(1)[0][1], sum(ca.values())
        dom_b, n_b, tot_b = cb.most_common(1)[0][0], cb.most_common(1)[0][1], sum(cb.values())
        agree  = dom_a == dom_b
        result = "[SAME]" if agree else "[diff]"
        print(f"  {sc:<14}  code={dom_a} ({n_a}/{tot_a}){'':<{col-16}}  "
              f"code={dom_b} ({n_b}/{tot_b}){'':<{col-16}}  {result}")
        same  += int(agree)
        total += 1

    print("  " + "-" * (W - 2))
    if total:
        print(f"  Transfer agreement: {same}/{total} scenarios ({same/total:.0%})")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── 1. Load real DSB data ─────────────────────────────────────────────────
    print("\n[1] Loading DeathStarBench X-Trace dataset ...")
    dsb = DSBLoader(seed=SEED)
    train_raw, test_events = dsb.load_dataset(n_train=N_TRAIN, n_test=N_TEST, verbose=True)

    train_events = _minority_oversample(train_raw)
    sc_dist = Counter(SCENARIO_NAMES[int(ev.scenario)] for ev in train_events)
    print(f"\n    Train (oversampled): {len(train_events)}  |  Test: {len(test_events)}")
    for sc, n in sorted(sc_dist.items()):
        print(f"      {sc:<20} {n}")

    X_tr,    Y_tr    = _tensors(train_events, noise=0.05)
    X_te,    Y_te    = _tensors(test_events)
    X_tr_np, Y_tr_np = _numpy_XY(train_events)
    X_te_np, Y_te_np = _numpy_XY(test_events)
    sc_labels_te = np.array([int(ev.scenario) for ev in test_events])

    loader = DataLoader(TensorDataset(X_tr, Y_tr), BATCH_SIZE, shuffle=True)

    # ── 2. Train / load models ────────────────────────────────────────────────
    print("\n[2] Training / loading models ...")

    threshold = ThresholdBaseline()

    print("  Decision Tree ...")
    dt = MultiOutputClassifier(DecisionTreeClassifier(max_depth=12, random_state=SEED))
    dt.fit(X_tr_np, Y_tr_np)

    print("  XGBoost (4 heads) ...")
    n_classes_per_dim = [NUM_RETRY_CLASSES, NUM_TIMEOUT_CLASSES, NUM_CB_CLASSES, NUM_FALLBACK_CLASSES]
    X_pad = np.zeros((max(n_classes_per_dim), X_tr_np.shape[1]), dtype=np.float32)
    xgb_models = []
    for i, (dim, nc) in enumerate(zip(POLICY_DIM_NAMES, n_classes_per_dim)):
        y_pad = np.arange(nc, dtype=np.int32)
        X_aug = np.vstack([X_tr_np, X_pad[:nc]])
        y_aug = np.concatenate([Y_tr_np[:, i], y_pad])
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=SEED, verbosity=0, eval_metric="mlogloss",
                                 num_class=nc)
        clf.fit(X_aug, y_aug)
        xgb_models.append(clf)

    print(f"  MLP-47 ({EPOCHS} epochs) ...")
    mlp47 = MLP47Model()
    if CKPT_MLP.exists():
        _load_neural(mlp47, CKPT_MLP)
    else:
        _train_neural(mlp47, loader, EPOCHS, "MLP-47")
        _save_neural(mlp47, CKPT_MLP, epochs=EPOCHS)

    print(f"  Continuous AE ({EPOCHS} epochs) ...")
    cont_ae = ContinuousAEModel()
    if CKPT_AE.exists():
        _load_neural(cont_ae, CKPT_AE)
    else:
        _train_neural(cont_ae, loader, EPOCHS, "Cont-AE")
        _save_neural(cont_ae, CKPT_AE, epochs=EPOCHS)

    print(f"  ORS-VQ (load ors_dsb_vq.pt) ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run ors_dsb_poc.py first.")
    _load_neural(ors, CKPT_ORS)

    # ── E1: Policy prediction ─────────────────────────────────────────────────
    print("\n[E1] Policy prediction metrics ...")

    xgb_preds = np.column_stack([clf.predict(X_te_np) for clf in xgb_models])
    dt_preds  = np.array(dt.predict(X_te_np))

    e1_rows = [
        ("Threshold",     _policy_metrics(*_predict_threshold(threshold, test_events))),
        ("Decision Tree", _policy_metrics(Y_te_np, dt_preds)),
        ("XGBoost",       _policy_metrics(Y_te_np, xgb_preds)),
        ("MLP-47",        _policy_metrics(*_predict_all_neural(mlp47, test_events))),
        ("Cont-AE",       _policy_metrics(*_predict_all_neural(cont_ae, test_events))),
        ("ORS-VQ",        _policy_metrics(*_predict_all_neural(ors, test_events))),
    ]
    _print_e1(e1_rows)

    # ── E2: Latent space quality ──────────────────────────────────────────────
    print("\n[E2] Latent space quality metrics ...")
    z_mlp  = _extract_latent_neural(mlp47, test_events)
    z_ae   = _extract_latent_neural(cont_ae, test_events)
    z_ors, ors_codes = _extract_latent_ors(ors, test_events)

    km_mlp = _kmeans_clusters(z_mlp, n=5)
    km_ae  = _kmeans_clusters(z_ae,  n=5)

    e2_rows = [
        ("MLP-47",  _latent_quality(z_mlp, km_mlp,    sc_labels_te), False),
        ("Cont-AE", _latent_quality(z_ae,  km_ae,     sc_labels_te), False),
        ("ORS-VQ",  _latent_quality(z_ors, ors_codes, sc_labels_te), True),
    ]
    _print_e2(e2_rows)

    print("\n  ORS-VQ per-code scenario purity (DSB test set):")
    purity_rows = _code_purity_per_entry(ors_codes, sc_labels_te)
    for r in purity_rows:
        bar = "#" * int(r["purity"] * 20)
        print(f"    code {r['code']:>2}  n={r['count']:>4}  {r['scenario']:<12}  "
              f"{r['purity']:.0%}  |{bar:<20}|")

    # ── E3: Cross-workload transfer ───────────────────────────────────────────
    print("\n[E3] Cross-workload ORS transfer (DSB workload groups) ...")
    _print_e3_dsb(ors, test_events)

    # ── E4: Ablation ─────────────────────────────────────────────────────────
    _print_e4_ablation(e1_rows, e2_rows)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
