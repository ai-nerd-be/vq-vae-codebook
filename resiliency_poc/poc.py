"""
poc.py  --  VQ-VAE Workflow Resiliency PoC  (4-phase pipeline)

Run (train everything from scratch):
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.poc

Run (skip training, load saved checkpoints):
    py -m resiliency_poc.poc --load

Pipeline
--------
  Phases 1-3  use the SYNTHETIC oracle (rule-based labels) — kept as ablation
              baselines so the paper can show why oracle-free is better.
  Phase  4    uses ORACLE-FREE counterfactual-simulation labels.
              This is the main paper contribution.

Checkpoints are saved to:
    resiliency_poc/checkpoints/
      mlp.pt       phase1_mlp_state + metadata
      vq_ema.pt    phase2_shared_vq_state + metadata
      fact_vq.pt   phase3_factorized_vq_state + metadata
      temp_vq.pt   phase4_temporal_vq_state + metadata  ← oracle-free

Dependencies:  torch  numpy
"""

from __future__ import annotations
import argparse, os, random, time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .workflow_loader import load_both_workflows
from .trace_generator import TraceGenerator, FailureEvent, FEATURE_DIM
from .label_oracle     import get_policy
from .model.vqvae_model import (
    MLPResiliencyModel, VQResiliencyModel, FactorizedVQResiliencyModel,
    TemporalVQResiliencyModel,
)
from .evaluator import (
    evaluate, codebook_profile, factorized_codebook_profile,
    evaluate_dsb, evaluate_all_on_temporal, evaluate_all_on_dsb,
)
from .temporal_trace_generator import (
    TemporalTraceGenerator, TemporalFailureEvent, TEMPORAL_FEATURE_DIM,
)
from .scenario import SCENARIO_NAMES, ScenarioState
from .dsb_loader import DSBLoader
from .counterfactual_labeler import FAILURE_CATEGORY_NAMES, N_FAILURE_CATEGORIES

# ── Reproducibility ────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_TRAIN_TRACES = 3_000
N_TEST_TRACES  =   600
BATCH_SIZE     =    64
LR             =  1e-3

MLP_EPOCHS     =  10
VQ_EPOCHS      =  20
FACT_EPOCHS    =  20

CODEBOOK_SIZE  =  16
VQ_BETA        =  0.25
EMA_DECAY      =  0.99
RESTART_THRESH =  1.0

TEMP_EPOCHS    =  40
N_TRAIN_TEMP   = 3_000
N_TEST_TEMP    =   600

# ── Checkpoint paths ───────────────────────────────────────────────────────────

CKPT_DIR = Path(__file__).parent / "checkpoints"

def _ckpt(name: str) -> Path:
    CKPT_DIR.mkdir(exist_ok=True)
    return CKPT_DIR / name


def _save(model: nn.Module, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    print(f"      Saved  {path.name}")


def _load(model: nn.Module, path: Path) -> dict:
    data = torch.load(path, weights_only=False)
    model.load_state_dict(data["state_dict"])
    model.eval()
    print(f"      Loaded {path.name}")
    return {k: v for k, v in data.items() if k != "state_dict"}


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _label_events(events: List[FailureEvent]) -> List[FailureEvent]:
    # Oracle labels — used ONLY for Phases 1-3 (ablation baselines).
    # Phase 4 uses counterfactual simulation labels instead.
    for e in events:
        p = get_policy(e.failure_mode, e.node_type, e.retry_count, e.is_external)
        e.label_retry    = p.retry
        e.label_timeout  = p.timeout_tier
        e.label_cb       = p.circuit_breaker
        e.label_fallback = p.fallback
    return events


def _to_tensors(events: List[FailureEvent]) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor(
        np.stack([e.to_feature_vector() for e in events]), dtype=torch.float32
    )
    Y = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback] for e in events],
        dtype=torch.long,
    )
    return X, Y


# ── Training loops ─────────────────────────────────────────────────────────────

def _step(opt: torch.optim.Optimizer, loss: torch.Tensor) -> float:
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def _train_vq_generic(model: nn.Module, loader: DataLoader,
                      epochs: int, label: str) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = total_task = total_vq = 0.0
        for x_batch, y_batch in loader:
            loss, info = model.loss(x_batch, y_batch)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            total_task += info["task_loss"]
            total_vq   += info["vq_loss"]
        n = len(loader)
        if epoch % 4 == 0 or epoch == epochs:
            print(
                f"  [{label}] epoch {epoch:>3}/{epochs}  "
                f"loss={total_loss/n:.4f}  task={total_task/n:.4f}  "
                f"vq={total_vq/n:.4f}  perplexity={info['perplexity']:.2f}"
            )


def _head_accuracies(model: nn.Module, X: torch.Tensor, Y: torch.Tensor,
                     name: str) -> dict:
    """Print per-head accuracy and return a dict of float values (0-100)."""
    model.eval()
    with torch.no_grad():
        r_l, t_l, c_l, f_l = model.get_logits(X)
    dim_names = ["retry", "timeout", "CB", "fallback"]
    vals = {
        lab: (lg.argmax(1) == Y[:, i]).float().mean().item() * 100
        for i, (lab, lg) in enumerate(zip(dim_names, [r_l, t_l, c_l, f_l]))
    }
    accs = [f"{k}={v:.1f}%" for k, v in vals.items()]
    # Category accuracy for temporal models (5th label column)
    if Y.shape[1] > 4 and hasattr(model, "category_accuracy"):
        cat_acc = model.category_accuracy(X, Y[:, 4]) * 100
        vals["category"] = cat_acc
        accs.append(f"category={cat_acc:.1f}%")
    print(f"  {name} test accuracy: " + "  ".join(accs))
    return vals


# ── Counterfactual policy accuracy ────────────────────────────────────────────

def _cf_accuracy(
    model:      nn.Module,
    events:     list,          # List[TemporalFailureEvent]
    use_static: bool = False,  # True = 38-dim static features (Phases 1-3 on synthetic)
) -> dict:
    """
    Evaluate model predictions against simulator-optimal counterfactual labels.
    Same ground truth for all models — the only fair cross-model accuracy metric.

    Returns dict: retry, CB, fallback, exact_match (all three correct), timeout.
    """
    model.eval()
    n = len(events)
    retry_ok = cb_ok = fallback_ok = timeout_ok = exact_ok = 0

    with torch.no_grad():
        for ev in events:
            if use_static:
                feats = ev.static_features
            else:
                feats = ev.to_feature_vector()
            x    = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)

            r_ok = int(pred.retry   == ev.label_retry)
            t_ok = int(pred.timeout == ev.label_timeout)
            c_ok = int(pred.cb      == ev.label_cb)
            f_ok = int(pred.fallback == ev.label_fallback)

            retry_ok   += r_ok
            timeout_ok += t_ok
            cb_ok      += c_ok
            fallback_ok += f_ok
            exact_ok   += int(r_ok and c_ok and f_ok)   # retry+CB+fallback all correct

    return {
        "cf_retry":   retry_ok   / n * 100,
        "cf_timeout": timeout_ok / n * 100,
        "cf_cb":      cb_ok      / n * 100,
        "cf_fallback":fallback_ok / n * 100,
        "cf_exact":   exact_ok   / n * 100,   # full pattern match
    }


def _print_cf_table(rows: list) -> None:
    """
    Print counterfactual accuracy table.
    rows: list of (name, cf_dict) tuples.
    """
    W = 90
    print("\n" + "=" * W)
    print("  COUNTERFACTUAL POLICY ACCURACY  (same ground truth for all models)")
    print("  Ground truth = simulator-optimal policy from counterfactual labeling.")
    print("=" * W)
    hdr = f"  {'Model':<38}  {'Retry':>6}  {'Timeout':>7}  {'CB':>6}  {'Fallback':>8}  {'Exact match':>11}"
    print(hdr)
    print("  " + "-" * (W - 2))
    for name, cf in rows:
        print(
            f"  {name:<38}  "
            f"{cf['cf_retry']:>5.1f}%  "
            f"{cf['cf_timeout']:>6.1f}%  "
            f"{cf['cf_cb']:>5.1f}%  "
            f"{cf['cf_fallback']:>7.1f}%  "
            f"{cf['cf_exact']:>10.1f}%"
        )
    print("=" * W)
    print("  Exact match = retry AND CB AND fallback all match the simulator-optimal policy.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(load_checkpoints: bool = False) -> None:
    t0 = time.time()

    # ── 1. Load workflows ──────────────────────────────────────────────────────
    print("\n[1] Loading workflow graphs ...")
    graphs = load_both_workflows()
    if not graphs:
        raise RuntimeError("No workflows found. Check paths in workflow_loader.py.")
    for g in graphs:
        print(f"    {g.name:52s} nodes={len(g.nodes)} paths={g.path_count()}")

    # ── 2. Oracle-labelled dataset (Phases 1-3 ablation baselines) ─────────────
    print(f"\n[2] Generating oracle-labelled traces (Phases 1-3 ablation) ...")
    print(f"    NOTE: Phases 1-3 use hand-written oracle rules — they are")
    print(f"          ablation baselines. Phase 4 is the oracle-free contribution.")
    train_events = _label_events(TraceGenerator(graphs, SEED).generate_dataset(N_TRAIN_TRACES))
    test_events  = _label_events(TraceGenerator(graphs, SEED + 1).generate_dataset(N_TEST_TRACES))
    fail_train   = [e for e in train_events if e.failure_mode != "no_failure"]
    fail_test    = [e for e in test_events  if e.failure_mode != "no_failure"]
    print(f"    train: {len(train_events)} events ({len(fail_train)} failures)")
    print(f"    test:  {len(test_events)} events ({len(fail_test)} failures)")
    X_train, Y_train = _to_tensors(fail_train)
    X_test,  Y_test  = _to_tensors(fail_test)
    loader = DataLoader(TensorDataset(X_train, Y_train), BATCH_SIZE, shuffle=True)

    # ── 3. Phase 1 — MLP baseline ──────────────────────────────────────────────
    print(f"\n[3] Phase 1: MLP baseline (oracle labels, {MLP_EPOCHS} epochs) ...")
    mlp = MLPResiliencyModel(FEATURE_DIM)
    if load_checkpoints and _ckpt("mlp.pt").exists():
        _load(mlp, _ckpt("mlp.pt"))
    else:
        opt = torch.optim.Adam(mlp.parameters(), lr=LR)
        mlp.train()
        for epoch in range(1, MLP_EPOCHS + 1):
            total = sum(_step(opt, mlp.loss(x, y)) for x, y in loader)
            if epoch % 2 == 0 or epoch == MLP_EPOCHS:
                print(f"  [MLP] epoch {epoch:>3}/{MLP_EPOCHS}  loss={total/len(loader):.4f}")
        _save(mlp, _ckpt("mlp.pt"), epochs=MLP_EPOCHS, seed=SEED)
    acc_mlp = _head_accuracies(mlp, X_test, Y_test, "MLP")

    # ── 4. Phase 2 — Shared VQ-EMA ────────────────────────────────────────────
    print(f"\n[4] Phase 2: Shared VQ-EMA (oracle labels, {VQ_EPOCHS} epochs) ...")
    vq = VQResiliencyModel(FEATURE_DIM, codebook_size=CODEBOOK_SIZE,
                           vq_beta=VQ_BETA, ema_decay=EMA_DECAY,
                           restart_threshold=RESTART_THRESH)
    if load_checkpoints and _ckpt("vq_ema.pt").exists():
        _load(vq, _ckpt("vq_ema.pt"))
    else:
        vq.load_encoder_from_mlp(mlp)
        _train_vq_generic(vq, loader, VQ_EPOCHS, "VQ-EMA")
        _save(vq, _ckpt("vq_ema.pt"), epochs=VQ_EPOCHS, seed=SEED)
    acc_vq = _head_accuracies(vq, X_test, Y_test, "VQ-EMA")

    # ── 5. Phase 3 — Factorized VQ ────────────────────────────────────────────
    print(f"\n[5] Phase 3: Factorized VQ (oracle labels, {FACT_EPOCHS} epochs) ...")
    fact = FactorizedVQResiliencyModel(FEATURE_DIM, vq_beta=VQ_BETA,
                                       ema_decay=EMA_DECAY,
                                       restart_threshold=RESTART_THRESH)
    if load_checkpoints and _ckpt("fact_vq.pt").exists():
        _load(fact, _ckpt("fact_vq.pt"))
    else:
        fact.load_encoder_from_mlp(mlp)
        _train_vq_generic(fact, loader, FACT_EPOCHS, "Fact-VQ")
        _save(fact, _ckpt("fact_vq.pt"), epochs=FACT_EPOCHS, seed=SEED)
    acc_fact = _head_accuracies(fact, X_test, Y_test, "Fact-VQ")

    # ── 6. Codebook analysis (Phases 2-3) ──────────────────────────────────────
    print("[6] Codebook prototype analysis (oracle-labelled) ...")
    codebook_profile(vq, fail_test)
    factorized_codebook_profile(fact, fail_test)

    # ── 8. Phase 4 — Temporal VQ (oracle-free, main contribution) ─────────────
    print(f"\n[8] Phase 4: Temporal VQ — ORACLE-FREE (main contribution) ...")
    print(f"    Labels: counterfactual simulation, NOT hand-written rules.")
    print(f"    Input:  44-dim (38 static workflow + 6 temporal scenario observables)")
    print(f"    Generating {N_TRAIN_TEMP} train + {N_TEST_TEMP} test temporal events ...")

    temp_gen_train = TemporalTraceGenerator(graphs, seed=SEED)
    temp_gen_test  = TemporalTraceGenerator(graphs, seed=SEED + 99)
    temp_train     = temp_gen_train.generate_dataset(N_TRAIN_TEMP)
    temp_test      = temp_gen_test.generate_dataset(N_TEST_TEMP)
    temp_gen_train.scenario_distribution(temp_train)

    X_tt = torch.tensor(np.stack([e.to_feature_vector() for e in temp_train]), dtype=torch.float32)
    Y_tt = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback, e.label_category]
         for e in temp_train], dtype=torch.long)
    X_te = torch.tensor(np.stack([e.to_feature_vector() for e in temp_test]),  dtype=torch.float32)
    Y_te = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback, e.label_category]
         for e in temp_test],  dtype=torch.long)
    temp_loader = DataLoader(TensorDataset(X_tt, Y_tt), BATCH_SIZE, shuffle=True)

    temp_vq = TemporalVQResiliencyModel(
        feature_dim=TEMPORAL_FEATURE_DIM,
        vq_beta=VQ_BETA, ema_decay=EMA_DECAY, restart_threshold=RESTART_THRESH,
    )
    if load_checkpoints and _ckpt("temp_vq.pt").exists():
        _load(temp_vq, _ckpt("temp_vq.pt"))
    else:
        _train_vq_generic(temp_vq, temp_loader, TEMP_EPOCHS, "Temp-VQ")
        _save(temp_vq, _ckpt("temp_vq.pt"), epochs=TEMP_EPOCHS, seed=SEED,
              feature_dim=TEMPORAL_FEATURE_DIM)
    acc_temp = _head_accuracies(temp_vq, X_te, Y_te, "Temporal VQ (oracle-free)")

    _temporal_codebook_profile(temp_vq, temp_test)

    # ── Apples-to-apples simulation: all models on the same temporal test set ──
    print(f"\n[8b] Simulation: all models on shared synthetic test set ...")
    shared_report = evaluate_all_on_temporal(temp_test, mlp, vq, fact, temp_vq)
    shared_report.print()

    # ── Counterfactual accuracy: Option A (synthetic, same CF ground truth) ───
    print(f"\n[8c] Counterfactual policy accuracy (Option A, synthetic test set) ...")
    cf_a_rows = [
        ("MLP (Phase 1, oracle)",          _cf_accuracy(mlp,    temp_test, use_static=True)),
        ("VQ-EMA (Phase 2, oracle)",       _cf_accuracy(vq,     temp_test, use_static=True)),
        ("Fact-VQ (Phase 3, oracle)",      _cf_accuracy(fact,   temp_test, use_static=True)),
        ("Temp-VQ (Phase 4, oracle-free)", _cf_accuracy(temp_vq, temp_test, use_static=False)),
    ]
    _print_cf_table(cf_a_rows)

    # ── 9. Phase 4b — Temporal VQ on REAL DeathStarBench traces ───────────────
    print(f"\n[9] Phase 4b: Temporal VQ — DeathStarBench REAL traces ...")
    print(f"    Same architecture as Phase 4, trained on real microservice traces.")

    dsb = DSBLoader(seed=SEED)
    dsb_train, dsb_test = dsb.load_dataset(n_train=4000, n_test=500)

    X_dsb_train = torch.tensor(
        np.stack([e.to_feature_vector() for e in dsb_train]), dtype=torch.float32
    )
    Y_dsb_train = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback, e.label_category]
         for e in dsb_train], dtype=torch.long
    )
    X_dsb_test = torch.tensor(
        np.stack([e.to_feature_vector() for e in dsb_test]), dtype=torch.float32
    )
    Y_dsb_test = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback, e.label_category]
         for e in dsb_test], dtype=torch.long
    )
    dsb_loader = DataLoader(TensorDataset(X_dsb_train, Y_dsb_train), BATCH_SIZE, shuffle=True)

    dsb_vq = TemporalVQResiliencyModel(
        feature_dim=TEMPORAL_FEATURE_DIM,
        vq_beta=VQ_BETA, ema_decay=EMA_DECAY, restart_threshold=RESTART_THRESH,
    )
    if load_checkpoints and _ckpt("dsb_temp_vq.pt").exists():
        _load(dsb_vq, _ckpt("dsb_temp_vq.pt"))
    else:
        _train_vq_generic(dsb_vq, dsb_loader, TEMP_EPOCHS, "DSB-VQ")
        _save(dsb_vq, _ckpt("dsb_temp_vq.pt"), epochs=TEMP_EPOCHS, seed=SEED,
              source="deathstarbench")
    acc_dsb = _head_accuracies(dsb_vq, X_dsb_test, Y_dsb_test, "DSB Temporal VQ (real traces)")

    print("\n  Simulation evaluation on DSB test set ...")
    dsb_report = evaluate_dsb(dsb_test, dsb_vq, seed=7)
    dsb_report.print()

    _category_breakdown(dsb_vq, dsb_test)
    _cb_deep_dive(dsb_vq, dsb_test)
    _degraded_diagnosis(dsb_vq, dsb_test)
    _temporal_codebook_profile(dsb_vq, dsb_test)

    # ── 10. Option B: all models trained + tested on DSB ──────────────────────
    print(f"\n[10] Option B: all models on DSB data (apples-to-apples on real traces) ...")
    print(f"     Phases 1-3 retrained on DSB with oracle labels, 47-dim input.")

    # Add oracle labels to DSB events (same hand-written rules as Phases 1-3)
    for e in dsb_train + dsb_test:
        fe = e.to_failure_event()
        p = get_policy(fe.failure_mode, fe.node_type, fe.retry_count, fe.is_external)
        e.oracle_retry    = p.retry
        e.oracle_timeout  = p.timeout_tier
        e.oracle_cb       = p.circuit_breaker
        e.oracle_fallback = p.fallback

    X_dsb_tr47 = torch.tensor(
        np.stack([e.to_feature_vector() for e in dsb_train]), dtype=torch.float32
    )
    Y_dsb_oracle = torch.tensor(
        [[e.oracle_retry, e.oracle_timeout, e.oracle_cb, e.oracle_fallback]
         for e in dsb_train], dtype=torch.long
    )
    X_dsb_te47 = torch.tensor(
        np.stack([e.to_feature_vector() for e in dsb_test]), dtype=torch.float32
    )
    Y_dsb_oracle_te = torch.tensor(
        [[e.oracle_retry, e.oracle_timeout, e.oracle_cb, e.oracle_fallback]
         for e in dsb_test], dtype=torch.long
    )
    dsb_oracle_loader = DataLoader(
        TensorDataset(X_dsb_tr47, Y_dsb_oracle), BATCH_SIZE, shuffle=True
    )

    # Phase 1 retrained on DSB (47-dim, oracle labels)
    dsb_mlp = MLPResiliencyModel(TEMPORAL_FEATURE_DIM)
    if load_checkpoints and _ckpt("dsb_mlp.pt").exists():
        _load(dsb_mlp, _ckpt("dsb_mlp.pt"))
    else:
        opt = torch.optim.Adam(dsb_mlp.parameters(), lr=LR)
        dsb_mlp.train()
        for epoch in range(1, MLP_EPOCHS + 1):
            total = sum(_step(opt, dsb_mlp.loss(x, y)) for x, y in dsb_oracle_loader)
            if epoch % 2 == 0 or epoch == MLP_EPOCHS:
                print(f"  [DSB-MLP] epoch {epoch:>3}/{MLP_EPOCHS}  loss={total/len(dsb_oracle_loader):.4f}")
        _save(dsb_mlp, _ckpt("dsb_mlp.pt"), epochs=MLP_EPOCHS)
    acc_dsb_mlp = _head_accuracies(dsb_mlp, X_dsb_te47, Y_dsb_oracle_te, "DSB-MLP (Phase 1, oracle)")

    # Phase 2 retrained on DSB (47-dim, oracle labels)
    dsb_vq2 = VQResiliencyModel(TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE,
                                vq_beta=VQ_BETA, ema_decay=EMA_DECAY,
                                restart_threshold=RESTART_THRESH)
    if load_checkpoints and _ckpt("dsb_vq_ema.pt").exists():
        _load(dsb_vq2, _ckpt("dsb_vq_ema.pt"))
    else:
        dsb_vq2.load_encoder_from_mlp(dsb_mlp)
        _train_vq_generic(dsb_vq2, dsb_oracle_loader, VQ_EPOCHS, "DSB-VQ-EMA")
        _save(dsb_vq2, _ckpt("dsb_vq_ema.pt"), epochs=VQ_EPOCHS)
    acc_dsb_vq2 = _head_accuracies(dsb_vq2, X_dsb_te47, Y_dsb_oracle_te, "DSB-VQ-EMA (Phase 2, oracle)")

    # Phase 3 retrained on DSB (47-dim, oracle labels)
    dsb_fact = FactorizedVQResiliencyModel(TEMPORAL_FEATURE_DIM, vq_beta=VQ_BETA,
                                           ema_decay=EMA_DECAY,
                                           restart_threshold=RESTART_THRESH)
    if load_checkpoints and _ckpt("dsb_fact_vq.pt").exists():
        _load(dsb_fact, _ckpt("dsb_fact_vq.pt"))
    else:
        dsb_fact.load_encoder_from_mlp(dsb_mlp)
        _train_vq_generic(dsb_fact, dsb_oracle_loader, FACT_EPOCHS, "DSB-Fact-VQ")
        _save(dsb_fact, _ckpt("dsb_fact_vq.pt"), epochs=FACT_EPOCHS)
    acc_dsb_fact = _head_accuracies(dsb_fact, X_dsb_te47, Y_dsb_oracle_te, "DSB-Fact-VQ (Phase 3, oracle)")

    print("\n  Simulation: all models on same DSB test set ...")
    dsb_all_report = evaluate_all_on_dsb(dsb_test, dsb_mlp, dsb_vq2, dsb_fact, dsb_vq)
    dsb_all_report.print()

    # ── Counterfactual accuracy: Option B (DSB, same CF ground truth) ─────────
    print(f"\n[10b] Counterfactual policy accuracy (Option B, DSB test set) ...")
    cf_b_rows = [
        ("MLP (Phase 1, oracle, DSB)",     _cf_accuracy(dsb_mlp,  dsb_test, use_static=False)),
        ("VQ-EMA (Phase 2, oracle, DSB)",  _cf_accuracy(dsb_vq2,  dsb_test, use_static=False)),
        ("Fact-VQ (Phase 3, oracle, DSB)", _cf_accuracy(dsb_fact, dsb_test, use_static=False)),
        ("DSB-VQ (Phase 4b, oracle-free)", _cf_accuracy(dsb_vq,   dsb_test, use_static=False)),
    ]
    _print_cf_table(cf_b_rows)

    # ── Unified summary across all phases ──────────────────────────────────────
    # shared_report    = Option A: all models on same synthetic temporal test set
    # dsb_all_report   = Option B: all models retrained+tested on real DSB data
    _print_unified_summary(
        phase_accs={
            "MLP (Phase 1, oracle)":           acc_mlp,
            "VQ-EMA (Phase 2, oracle)":        acc_vq,
            "Fact-VQ (Phase 3, oracle)":       acc_fact,
            "Temp-VQ (Phase 4, oracle-free)":  acc_temp,
            "DSB-VQ (Phase 4b, real traces)":  acc_dsb,
            # Option B (DSB-retrained, oracle labels)
            "DSB-MLP (Ph1, oracle, DSB)":      acc_dsb_mlp,
            "DSB-VQ-EMA (Ph2, oracle, DSB)":   acc_dsb_vq2,
            "DSB-Fact (Ph3, oracle, DSB)":     acc_dsb_fact,
        },
        sim_reports={
            # Option A — synthetic shared test set
            "MLP (Phase 1, oracle)":           shared_report.metrics[1],
            "VQ-EMA (Phase 2, oracle)":        shared_report.metrics[2],
            "Fact-VQ (Phase 3, oracle)":       shared_report.metrics[3],
            "Temp-VQ (Phase 4, oracle-free)":  shared_report.metrics[4],
            # Option B — same DSB test set for all
            "DSB-VQ (Phase 4b, real traces)":  dsb_all_report.metrics[4],
            "DSB-MLP (Ph1, oracle, DSB)":      dsb_all_report.metrics[1],
            "DSB-VQ-EMA (Ph2, oracle, DSB)":   dsb_all_report.metrics[2],
            "DSB-Fact (Ph3, oracle, DSB)":     dsb_all_report.metrics[3],
        },
        baseline_sim=shared_report.metrics[0],
        dsb_baseline_sim=dsb_all_report.metrics[0],
    )

    print(f"\nDone in {time.time()-t0:.1f}s")
    print(f"Checkpoints saved to: {CKPT_DIR}")


# ── Category breakdown ────────────────────────────────────────────────────────

def _category_breakdown(
    model:  TemporalVQResiliencyModel,
    events: List[TemporalFailureEvent],
) -> None:
    """
    Show per-category accuracy and which codebook entries the model
    assigns to each failure category.  Demonstrates that the hierarchical
    decoder's Level-1 output maps cleanly onto the codebook archetypes.
    """
    from collections import defaultdict, Counter
    import torch

    model.eval()
    cat_correct   = defaultdict(int)
    cat_total     = defaultdict(int)
    cat_predicted = defaultdict(Counter)   # true_cat → predicted_cat counts
    cat_codes     = defaultdict(list)      # true_cat → list of retry-dim codes

    for ev in events:
        x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        pred = model.predict(x)
        true_cat = ev.label_category
        cat_total[true_cat]    += 1
        cat_predicted[true_cat][pred.category] += 1
        if pred.category == true_cat:
            cat_correct[true_cat] += 1
        codes = model.codes_for(x)
        cat_codes[true_cat].append(codes[0])   # retry dim is most discriminative

    print("\n-- Failure Category Breakdown (hierarchical decoder Level 1) ----------")
    print(f"  {'Category':<14} {'N':>5}  {'Acc':>7}  {'Dominant codebook entries'}")
    print("  " + "-" * 65)
    for cat_idx, name in enumerate(FAILURE_CATEGORY_NAMES):
        n = cat_total[cat_idx]
        if n == 0:
            print(f"  {name:<14} {'0':>5}  {'  N/A':>7}  -")
            continue
        acc     = cat_correct[cat_idx] / n * 100
        entries = Counter(cat_codes[cat_idx]).most_common(3)
        code_str = "  ".join(f"e{c}({cnt})" for c, cnt in entries)
        print(f"  {name:<14} {n:>5}  {acc:>6.1f}%  {code_str}")

    print("\n  Category confusion (rows=true, cols=predicted):")
    all_cats = sorted(set(
        list(cat_total.keys()) +
        [p for c in cat_predicted.values() for p in c.keys()]
    ))
    col_hdr = "".join(f"{FAILURE_CATEGORY_NAMES[c]:>14}" for c in all_cats)
    print(f"  {'True / Pred':<14}{col_hdr}")
    for true_c in all_cats:
        col_vals = "".join(f"{cat_predicted[true_c][p]:>14}" for p in all_cats)
        print(f"  {FAILURE_CATEGORY_NAMES[true_c]:<14}{col_vals}")


# ── CB accuracy deep-dive ─────────────────────────────────────────────────────

def _cb_deep_dive(
    model:  TemporalVQResiliencyModel,
    events: List[TemporalFailureEvent],
) -> None:
    """
    Diagnose why CB accuracy < 100%.

    For every scenario, show:
      1. Label distribution  — what the counterfactual labeler assigned
      2. Prediction distribution — what the model predicts
      3. Error count and dominant error direction
      4. Whether errors stem from label ambiguity or model error
    """
    from collections import defaultdict, Counter
    from .scenario import ScenarioState, SCENARIO_NAMES
    from .label_oracle import CIRCUIT_BREAKER_LABELS
    import torch

    CB_NAMES = CIRCUIT_BREAKER_LABELS   # ['0', '1', '2'] or ['closed','half_open','open']

    # Collect per-scenario CB labels and predictions
    by_sc: dict = defaultdict(list)
    for ev in events:
        x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        pred = model.predict(x)
        by_sc[ev.scenario].append((ev.label_cb, pred.cb))

    print("\n" + "=" * 90)
    print("  CB ACCURACY DEEP-DIVE")
    print("=" * 90)

    total_wrong = 0
    error_log   = []   # (scenario_name, true_cb, pred_cb, count)

    for sc in ScenarioState:
        rows = by_sc[sc]
        if not rows:
            continue
        n = len(rows)
        true_dist = Counter(r[0] for r in rows)
        pred_dist = Counter(r[1] for r in rows)
        wrong     = [(t, p) for t, p in rows if t != p]
        n_wrong   = len(wrong)
        total_wrong += n_wrong

        print(f"\n  Scenario: {SCENARIO_NAMES[sc]}  (N={n}, CB errors={n_wrong})")

        # Label distribution
        label_str = "  ".join(
            f"{CB_NAMES[k]}={v}({v/n*100:.0f}%)"
            for k, v in sorted(true_dist.items())
        )
        print(f"    Labels assigned by counterfactual sim:  {label_str}")

        # Prediction distribution
        pred_str = "  ".join(
            f"{CB_NAMES[k]}={v}({v/n*100:.0f}%)"
            for k, v in sorted(pred_dist.items())
        )
        print(f"    Model predictions:                      {pred_str}")

        if n_wrong == 0:
            print(f"    No errors.")
            continue

        # Error pattern
        err_counter = Counter(wrong)
        print(f"    Error breakdown (true->pred):")
        for (t, p), cnt in err_counter.most_common():
            pct = cnt / n * 100
            # Root cause: if label is mixed for this scenario → label ambiguity
            label_mix = len(true_dist) > 1
            cause = "label ambiguity (sim assigns multiple CB values for this scenario)" \
                    if label_mix else "model confusion (labels are consistent)"
            print(f"      {CB_NAMES[t]:>9} -> {CB_NAMES[p]:<9}  x{cnt}  ({pct:.1f}%)  [{cause}]")
            error_log.append((SCENARIO_NAMES[sc], CB_NAMES[t], CB_NAMES[p], cnt))

    print(f"\n  Total CB errors: {total_wrong} / {len(events)}  "
          f"({total_wrong/len(events)*100:.1f}% error rate, "
          f"{(1 - total_wrong/len(events))*100:.1f}% accuracy)")

    # Summary: label ambiguity vs model error
    print("\n  Root cause summary:")
    print("  +" + "-" * 63 + "+")
    for sc, true_cb, pred_cb, cnt in error_log:
        print(f"  |  {sc:<12}  {true_cb:>9} -> {pred_cb:<9}  x{cnt:<3}  ", end="")
        sc_rows   = by_sc[[s for s in ScenarioState if SCENARIO_NAMES[s] == sc][0]]
        true_vals = set(r[0] for r in sc_rows)
        if len(true_vals) > 1:
            print("label ambiguity     |")
        else:
            print("model misclassifies |")
    print("  +" + "-" * 63 + "+")


# ── Unified model comparison ───────────────────────────────────────────────────

def _print_unified_summary(
    phase_accs,
    sim_reports,
    baseline_sim,
    dsb_baseline_sim=None,
):
    W = 112
    print("\n" + "=" * W)
    print("  UNIFIED MODEL COMPARISON  (all phases)")
    print("=" * W)

    hdr = (
        f"  {'Model':<36}  {'Retry':>6}  {'CB':>6}  {'Fallback':>8}  "
        f"{'Category':>9}  {'Completion':>11}  {'Wasted%':>8}  {'Latency':>8}  Labels"
    )
    print(hdr)

    def _row(name, accs, sim, lbl):
        retry    = f"{accs.get('retry',    0):.0f}%"
        cb       = f"{accs.get('CB',       0):.0f}%"
        fallback = f"{accs.get('fallback', 0):.0f}%"
        category = f"{accs.get('category', 0):.0f}%" if "category" in accs else "  -"
        if sim is not None:
            compl   = f"{sim.completion_rate*100:.1f}%"
            wasted  = f"{sim.wasted_retry_rate*100:.1f}%"
            latency = f"{sim.avg_latency:.2f}"
        else:
            compl = wasted = latency = "  -"
        print(
            f"  {name:<36}  {retry:>6}  {cb:>6}  {fallback:>8}  "
            f"{category:>9}  {compl:>11}  {wasted:>8}  {latency:>8}  {lbl}"
        )

    def _baseline_row(sim, note):
        print(f"  {'Threshold (baseline)':<36}  {'  -':>6}  {'  -':>6}  {'  -':>8}  "
              f"{'  -':>9}  {sim.completion_rate*100:>10.1f}%  "
              f"{sim.wasted_retry_rate*100:>7.1f}%  "
              f"{sim.avg_latency:>8.2f}  {note}")

    lbl_map = {
        "MLP (Phase 1, oracle)":           "oracle",
        "VQ-EMA (Phase 2, oracle)":        "oracle",
        "Fact-VQ (Phase 3, oracle)":       "oracle",
        "Temp-VQ (Phase 4, oracle-free)":  "counterfactual",
        "DSB-VQ (Phase 4b, real traces)":  "counterfactual (real)",
        "DSB-MLP (Ph1, oracle, DSB)":      "oracle (DSB)",
        "DSB-VQ-EMA (Ph2, oracle, DSB)":   "oracle (DSB)",
        "DSB-Fact (Ph3, oracle, DSB)":     "oracle (DSB)",
    }

    print("\n  -- Option A: all models on same SYNTHETIC temporal test set --")
    print("  " + "-" * (W - 2))
    _baseline_row(baseline_sim, "rule-based")
    for name in ["MLP (Phase 1, oracle)", "VQ-EMA (Phase 2, oracle)",
                 "Fact-VQ (Phase 3, oracle)", "Temp-VQ (Phase 4, oracle-free)"]:
        _row(name, phase_accs.get(name, {}), sim_reports.get(name), lbl_map[name])

    if dsb_baseline_sim is not None:
        print("\n  -- Option B: all models retrained + tested on same REAL (DSB) test set --")
        print("  " + "-" * (W - 2))
        _baseline_row(dsb_baseline_sim, "rule-based")
        for name in ["DSB-MLP (Ph1, oracle, DSB)", "DSB-VQ-EMA (Ph2, oracle, DSB)",
                     "DSB-Fact (Ph3, oracle, DSB)", "DSB-VQ (Phase 4b, real traces)"]:
            _row(name, phase_accs.get(name, {}), sim_reports.get(name), lbl_map[name])

    print("=" * W)
    print("  Category = failure-category head accuracy (NOMINAL/TRANSIENT/PERSISTENT/TRANSACTION)")
    print("  Option A: Phases 1-3 accuracy vs oracle labels; Phase 4 vs counterfactual labels.")
    print("  Option B: all retrained on DSB; Phases 1-3 vs oracle; Phase 4b vs counterfactual.")

# ── DEGRADED scenario diagnosis ────────────────────────────────────────────────

def _degraded_diagnosis(
    model:  TemporalVQResiliencyModel,
    events: List[TemporalFailureEvent],
) -> None:
    from collections import Counter, defaultdict
    from .scenario import ScenarioState, SCENARIO_NAMES
    from .label_oracle import CIRCUIT_BREAKER_LABELS, RETRY_LABELS
    import torch

    deg_events = [e for e in events if e.scenario == ScenarioState.DEGRADED]
    if not deg_events:
        print("\n  [DEGRADED diag] No DEGRADED events in test set.")
        return

    print(f"\n-- DEGRADED Diagnosis ({len(deg_events)} test events) ----------------")

    # 1. Label distribution
    cb_label_counts    = Counter(e.label_cb    for e in deg_events)
    retry_label_counts = Counter(e.label_retry for e in deg_events)
    print("  Label distribution (ground truth):")
    print(f"    CB:    { {CIRCUIT_BREAKER_LABELS[k]: v for k, v in sorted(cb_label_counts.items())} }")
    print(f"    Retry: { {RETRY_LABELS[k]: v for k, v in sorted(retry_label_counts.items())} }")

    # 2. Prediction confusion matrix for CB
    model.eval()
    cb_confusion: dict = defaultdict(Counter)   # true → predicted counts
    retry_confusion: dict = defaultdict(Counter)
    feat_vecs = []
    for ev in deg_events:
        x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        pred = model.predict(x)
        cb_confusion[ev.label_cb][pred.cb] += 1
        retry_confusion[ev.label_retry][pred.retry] += 1
        feat_vecs.append(ev.scenario_obs.to_feature_vector())

    print("\n  CB confusion (rows=true, cols=predicted):")
    all_cb = sorted(set(list(cb_confusion.keys()) + [p for c in cb_confusion.values() for p in c]))
    col_hdr = "".join(f"{CIRCUIT_BREAKER_LABELS[k]:>12}" for k in all_cb)
    print(f"  {'True / Pred':<14}{col_hdr}")
    for true_k in all_cb:
        col_vals = "".join(f"{cb_confusion[true_k][p]:>12}" for p in all_cb)
        print(f"  {CIRCUIT_BREAKER_LABELS[true_k]:<14}{col_vals}")

    print("\n  Retry confusion (rows=true, cols=predicted):")
    all_r = sorted(set(list(retry_confusion.keys()) + [p for c in retry_confusion.values() for p in c]))
    col_hdr = "".join(f"{RETRY_LABELS[k]:>10}" for k in all_r)
    print(f"  {'True / Pred':<14}{col_hdr}")
    for true_k in all_r:
        col_vals = "".join(f"{retry_confusion[true_k][p]:>10}" for p in all_r)
        print(f"  {RETRY_LABELS[true_k]:<14}{col_vals}")

    # 3. Scenario observable stats for DEGRADED vs TRANSIENT (feature overlap check)
    import numpy as np
    scenario_counts = Counter(e.scenario for e in events)
    print(f"\n  Scenarios in events: { {int(k): v for k, v in scenario_counts.items()} }")
    trans_events = [e for e in events if int(e.scenario) == int(ScenarioState.TRANSIENT_FAILURE)]
    obs_names = ["roll_fail", "consec_fail", "retry_succ", "lat_trend", "t_since_ok", "sc_dur"]
    print(f"\n  Scenario observable means (feature overlap check):")
    print(f"  {'Feature':<14}  {'DEGRADED':>10}  {'TRANSIENT':>10}  {'delta':>8}")
    if deg_events and trans_events:
        deg_obs  = np.mean([e.scenario_obs.to_feature_vector() for e in deg_events],  axis=0)
        tran_obs = np.mean([e.scenario_obs.to_feature_vector() for e in trans_events], axis=0)
        for name, d, t in zip(obs_names, deg_obs, tran_obs):
            print(f"  {name:<14}  {d:>10.3f}  {t:>10.3f}  {abs(d-t):>8.3f}")
    else:
        print(f"  (deg={len(deg_events)}, trans={len(trans_events)} — skipped)")


# ── Temporal codebook analysis ─────────────────────────────────────────────────

def _temporal_codebook_profile(
    model:  TemporalVQResiliencyModel,
    events: List[TemporalFailureEvent],
) -> None:
    from collections import Counter, defaultdict
    from .label_oracle import RETRY_LABELS, CIRCUIT_BREAKER_LABELS

    entry_scenarios: dict = defaultdict(list)
    entry_policies:  dict = defaultdict(list)

    model.eval()
    for ev in events:
        x     = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        codes = model.codes_for(x)
        pred  = model.predict(x)
        entry_scenarios[codes[0]].append(SCENARIO_NAMES[ev.scenario])
        entry_policies[codes[0]].append(
            (RETRY_LABELS[pred.retry], CIRCUIT_BREAKER_LABELS[pred.cb])
        )

    print("\n-- Temporal VQ Codebook (retry dim) — Scenario Archetype Analysis ------")
    print(f"  {'Entry':<6} {'Count':>6}  {'Dominant scenario':<20} {'Purity':>7}  Dominant policy")
    print("  " + "-" * 72)
    for k in sorted(entry_scenarios.keys()):
        sc_list = entry_scenarios[k]
        sc_cnts = Counter(sc_list)
        top_sc, top_cnt = sc_cnts.most_common(1)[0]
        purity  = top_cnt / len(sc_list) * 100
        top_pol, _ = Counter(entry_policies[k]).most_common(1)[0]
        print(f"  e{k:<5} {len(sc_list):>6}  {top_sc:<20} {purity:>6.1f}%  "
              f"retry={top_pol[0]} cb={top_pol[1]}")
    print()
    perps    = model.codebooks.perplexities()
    dims     = model.codebooks.policy_dims
    perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(perps, dims))
    print(f"  Codebook perplexities (retry/timeout/CB/fallback): {perp_str}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Load saved checkpoints instead of retraining")
    args = parser.parse_args()
    main(load_checkpoints=args.load)
