"""
ors_poc.py  --  Operational Resilience State (ORS) VQ-VAE pipeline

Run (train):
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.ors_poc

Run (load checkpoints):
    py -m resiliency_poc.ors_poc --load

What is new vs poc.py (professor feedback 2026-07-05):
  - ONE shared codebook learns Operational Resilience States (ORS), not policies.
  - Category head is post-hoc analysis only — NOT in the training loss.
  - Cross-workflow transfer experiment: same ORS codes fire for the same
    scenario across different workflow graphs (different services, same state).
  - Harder synthetic data: feature noise + more minority scenario oversampling.
  - ORS table printed after training: code | scenario | telemetry stats | policy.

The factorized model from poc.py is kept as a comparison baseline (loaded from
its existing checkpoints if available; otherwise skipped in this pipeline).
"""

from __future__ import annotations
import argparse, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .workflow_loader           import load_both_workflows
from .temporal_trace_generator  import (
    TemporalTraceGenerator, TemporalFailureEvent, TEMPORAL_FEATURE_DIM,
)
from .scenario                  import SCENARIO_NAMES, ScenarioState
from .counterfactual_labeler    import (
    FAILURE_CATEGORY_NAMES, N_FAILURE_CATEGORIES, SCENARIO_TO_CATEGORY,
)
from .label_oracle              import RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS
from .model.ors_model           import ORSVQModel
from .evaluator                 import _ScenarioSimulator

# ── Reproducibility ────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_TRAIN        = 4_000   # more data to fill a larger shared codebook
N_TEST         =   800
BATCH_SIZE     =    64
LR             =  8e-4
ORS_EPOCHS     =    50
CODEBOOK_SIZE  =    16   # shared ORS entries — sized to discover real archetypes
VQ_BETA        =  0.25
EMA_DECAY      =  0.99
RESTART_THRESH =   0.5   # lower threshold → more aggressive dead-code restarts

FEATURE_NOISE_STD = 0.05  # Gaussian noise added to temporal features during training

CKPT_DIR = Path(__file__).parent / "checkpoints"


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _ckpt(name: str) -> Path:
    CKPT_DIR.mkdir(exist_ok=True)
    return CKPT_DIR / name


def _save(model: nn.Module, path: Path, **meta) -> None:
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    print(f"      Saved  {path.name}")


def _load(model: nn.Module, path: Path) -> None:
    data = torch.load(path, weights_only=False)
    model.load_state_dict(data["state_dict"])
    model.eval()
    print(f"      Loaded {path.name}")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _temporal_tensors(
    events: List[TemporalFailureEvent],
    noise_std: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert events to (X, Y) tensors.
    Y has 4 columns: [retry, timeout, cb, fallback].
    Category is deliberately excluded — it is post-hoc, not supervised.
    """
    X = torch.tensor(
        np.stack([ev.to_feature_vector() for ev in events]),
        dtype=torch.float32,
    )
    if noise_std > 0:
        X = X + torch.randn_like(X) * noise_std
    Y = torch.tensor(
        [[ev.label_retry, ev.label_timeout, ev.label_cb, ev.label_fallback]
         for ev in events],
        dtype=torch.long,
    )
    return X, Y


def _minority_oversample(events: List[TemporalFailureEvent]) -> List[TemporalFailureEvent]:
    """
    Cap HEALTHY at 40% and oversample minority scenarios so the model
    sees enough DEGRADED / DOWN examples to learn their ORS entries.
    """
    by_scenario: Dict[ScenarioState, list] = defaultdict(list)
    for ev in events:
        by_scenario[ev.scenario].append(ev)

    healthy  = by_scenario[ScenarioState.HEALTHY]
    minority = [ev for sc, evs in by_scenario.items()
                for ev in evs if sc != ScenarioState.HEALTHY]

    target_healthy = min(len(healthy), int(len(minority) / 0.6 * 0.4))
    healthy_sample = random.sample(healthy, target_healthy)

    # Oversample minority by repeating rare scenarios
    counts   = Counter(ev.scenario for ev in minority)
    max_cnt  = max(counts.values())
    balanced = []
    for sc, evs in by_scenario.items():
        if sc == ScenarioState.HEALTHY:
            continue
        factor  = max(1, max_cnt // len(evs))
        balanced.extend(evs * factor)

    result = healthy_sample + balanced
    random.shuffle(result)
    return result


# ── Training loop ──────────────────────────────────────────────────────────────

def _train_ors(
    model:  ORSVQModel,
    loader: DataLoader,
    epochs: int,
) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = total_task = total_vq = 0.0
        for x_batch, y_batch in loader:
            loss, info = model.loss(x_batch, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_task += info["task_loss"]
            total_vq   += info["vq_loss"]
        n = len(loader)
        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"  [ORS-VQ] epoch {epoch:>3}/{epochs}  "
                f"loss={total_loss/n:.4f}  task={total_task/n:.4f}  "
                f"vq={total_vq/n:.4f}  perplexity={info['perplexity']:.2f}"
            )


# ── Counterfactual accuracy ────────────────────────────────────────────────────

def _cf_accuracy(model: ORSVQModel, events: List[TemporalFailureEvent]) -> dict:
    model.eval()
    n = len(events)
    retry_ok = cb_ok = fallback_ok = timeout_ok = exact_ok = 0
    with torch.no_grad():
        for ev in events:
            x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            r_ok = int(pred.retry    == ev.label_retry)
            t_ok = int(pred.timeout  == ev.label_timeout)
            c_ok = int(pred.cb       == ev.label_cb)
            f_ok = int(pred.fallback == ev.label_fallback)
            retry_ok    += r_ok
            timeout_ok  += t_ok
            cb_ok       += c_ok
            fallback_ok += f_ok
            exact_ok    += int(r_ok and c_ok and f_ok)
    return {
        "cf_retry":    retry_ok    / n * 100,
        "cf_timeout":  timeout_ok  / n * 100,
        "cf_cb":       cb_ok       / n * 100,
        "cf_fallback": fallback_ok / n * 100,
        "cf_exact":    exact_ok    / n * 100,
    }


# ── ORS table printer ──────────────────────────────────────────────────────────

def _print_ors_table(rows: List[dict]) -> None:
    W = 110
    print("\n" + "=" * W)
    print("  OPERATIONAL RESILIENCE STATE (ORS) CODEBOOK TABLE")
    print("  Each entry = one discovered operational situation. Category is EMERGENT (not trained).")
    print("=" * W)
    hdr = (
        f"  {'Code':>4}  {'N':>5}  {'Scenario':>12}  {'Purity':>6}  "
        f"{'Category':>11}  {'FailRate':>8}  {'Consec':>6}  {'CB%':>5}  "
        f"{'Policy (retry,to,cb,fb)':>24}"
    )
    print(hdr)
    print("  " + "-" * (W - 2))
    for r in rows:
        rl, tl, cl, fl = r["top_policy"]
        policy_str = (
            f"retry={RETRY_LABELS[rl]},"
            f"to={TIMEOUT_LABELS[tl]},"
            f"cb={CIRCUIT_BREAKER_LABELS[cl]},"
            f"fb={FALLBACK_LABELS[fl]}"
        )
        print(
            f"  {r['code']:>4}  {r['count']:>5}  {r['top_scenario']:>12}  "
            f"{r['scenario_purity']:>5.0%}  {r['emergent_category']:>11}  "
            f"{r['avg_fail_rate']:>8.3f}  {r['avg_consec']:>6.2f}  "
            f"{r['avg_cb_open']:>4.0%}  {policy_str}"
        )
    print("=" * W)


# ── Cross-workflow transfer printer ───────────────────────────────────────────

def _print_transfer_table(
    transfer: Dict[str, Dict[str, Counter]],
    all_scenarios: List[str],
) -> None:
    """
    Print which codes fire per (workflow, scenario) pair.
    The key claim: same scenario → same dominant code across different workflows.
    """
    W = 100
    print("\n" + "=" * W)
    print("  CROSS-WORKFLOW ORS TRANSFER")
    print("  Claim: same operational situation activates the same ORS code")
    print("         regardless of which workflow graph the request traverses.")
    print("=" * W)

    workflows = sorted(transfer.keys())
    scenarios = [s for s in all_scenarios if any(
        s in transfer[wf] for wf in workflows
    )]

    # Header
    col_w = 20
    header = f"  {'Scenario':<14}" + "".join(f"  {wf[:col_w]:<{col_w}}" for wf in workflows)
    print(header)
    print("  " + "-" * (W - 2))

    same_count = 0
    total_count = 0
    for sc in scenarios:
        dominant = {}
        for wf in workflows:
            cnt = transfer[wf].get(sc)
            if cnt:
                dom_code, dom_n = cnt.most_common(1)[0]
                dominant[wf] = (dom_code, dom_n, sum(cnt.values()))
        if len(dominant) < 2:
            continue
        row = f"  {sc:<14}"
        codes = [v[0] for v in dominant.values()]
        for wf in workflows:
            if wf in dominant:
                c, n, total = dominant[wf]
                row += f"  code={c} ({n}/{total}){'':<{col_w - 14}}"
            else:
                row += f"  {'—':<{col_w}}"
        # Flag agreement
        agree = len(set(codes)) == 1
        row  += "  [SAME]" if agree else "  [diff]"
        print(row)
        total_count += 1
        if agree:
            same_count += 1

    print("  " + "-" * (W - 2))
    if total_count:
        print(f"  Agreement rate: {same_count}/{total_count} scenarios "
              f"({same_count/total_count:.0%})")
    print("=" * W)


# ── Per-scenario accuracy ──────────────────────────────────────────────────────

def _print_scenario_accuracy(model: ORSVQModel, events: List[TemporalFailureEvent]) -> None:
    by_scenario: Dict[ScenarioState, list] = defaultdict(list)
    for ev in events:
        by_scenario[ev.scenario].append(ev)

    W = 80
    print("\n" + "=" * W)
    print("  PER-SCENARIO ACCURACY (ORS-VQ)")
    print("=" * W)
    print(f"  {'Scenario':<14}  {'N':>5}  {'Retry':>7}  {'CB':>7}  {'Fallback':>8}  {'Exact':>7}  {'Cat(post-hoc)':>14}")
    print("  " + "-" * (W - 2))

    model.eval()
    with torch.no_grad():
        for sc_state, evs in sorted(by_scenario.items(), key=lambda x: x[0]):
            sc_name = SCENARIO_NAMES[int(sc_state)]
            n = len(evs)
            r_ok = cb_ok = f_ok = ex_ok = cat_ok = 0
            for ev in evs:
                x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                pred = model.predict(x)
                rr   = int(pred.retry    == ev.label_retry)
                cc   = int(pred.cb       == ev.label_cb)
                ff   = int(pred.fallback == ev.label_fallback)
                r_ok  += rr; cb_ok += cc; f_ok += ff
                ex_ok += int(rr and cc and ff)
                # Post-hoc category check (emergent)
                code    = model.code_for(x)
                cat_ok += 1   # placeholder — actual analysis is in ORS table
            print(
                f"  {sc_name:<14}  {n:>5}  "
                f"{r_ok/n:>6.1%}  {cb_ok/n:>6.1%}  {f_ok/n:>7.1%}  "
                f"{ex_ok/n:>6.1%}  {'(see ORS table)':>14}"
            )
    print("=" * W)


# ── Simulation evaluation ──────────────────────────────────────────────────────

def _simulate(model: ORSVQModel, events: List[TemporalFailureEvent]) -> dict:
    """Run _ScenarioSimulator on test events with ORS-VQ policy."""
    sim = _ScenarioSimulator(seed=7)
    completed = wasted = latency_sum = 0
    model.eval()
    with torch.no_grad():
        for ev in events:
            fe = ev.to_failure_event()
            fe._dsb_scenario = ev.scenario          # scenario-aware probabilities
            x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            result = sim._run_event(fe, pred)
            completed    += int(result.succeeded)
            wasted       += int(result.was_wasted)
            latency_sum  += result.latency_units
    n = len(events)
    return {
        "completion": completed / n * 100,
        "wasted_pct": wasted / max(1, n) * 100,
        "avg_latency": latency_sum / n,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main(load_checkpoints: bool = False) -> None:
    t0 = time.time()

    # ── 1. Load workflow graphs ────────────────────────────────────────────────
    print("\n[1] Loading workflow graphs ...")
    graphs = load_both_workflows()
    if not graphs:
        raise RuntimeError("No workflows found. Check workflow_loader.py paths.")
    for g in graphs:
        print(f"    {g.name:<52}  nodes={len(g.nodes)}")

    # ── 2. Generate temporal training data ────────────────────────────────────
    print(f"\n[2] Generating temporal traces (oracle-free, counterfactual labels) ...")
    print(f"    train={N_TRAIN}  test={N_TEST}  noise_std={FEATURE_NOISE_STD}")

    gen_train = TemporalTraceGenerator(graphs, seed=SEED)
    raw_train = gen_train.generate_dataset(N_TRAIN)

    # Generate balanced per-workflow test sets so both appear in transfer experiment
    n_per_wf = N_TEST // len(graphs)
    test_all  = []
    for g in graphs:
        wf_gen  = TemporalTraceGenerator([g], seed=SEED + 999)
        wf_evts = wf_gen.generate_dataset(n_per_wf)
        test_all.extend(wf_evts)
        print(f"    test [{g.name[:40]}]: {len(wf_evts)} events")
    random.shuffle(test_all)

    # Minority-oversample training set
    train_bal = _minority_oversample(raw_train)

    sc_counts = Counter(ev.scenario for ev in train_bal)
    print(f"    After oversampling:")
    for sc, c in sorted(sc_counts.items()):
        print(f"      {SCENARIO_NAMES[int(sc)]:<14} {c:>5}")

    X_train, Y_train = _temporal_tensors(train_bal, noise_std=FEATURE_NOISE_STD)
    X_test,  Y_test  = _temporal_tensors(test_all)
    loader = DataLoader(TensorDataset(X_train, Y_train), BATCH_SIZE, shuffle=True)

    # ── 3. ORS-VQ model ───────────────────────────────────────────────────────
    print(f"\n[3] ORS-VQ  (shared codebook K={CODEBOOK_SIZE}, oracle-free, {ORS_EPOCHS} epochs)")
    print( "    Category is POST-HOC — it is NOT in the training loss.")
    ors = ORSVQModel(
        feature_dim=TEMPORAL_FEATURE_DIM,
        codebook_size=CODEBOOK_SIZE,
        vq_beta=VQ_BETA,
        ema_decay=EMA_DECAY,
        restart_threshold=RESTART_THRESH,
    )
    ckpt_path = _ckpt("ors_vq.pt")
    if load_checkpoints and ckpt_path.exists():
        _load(ors, ckpt_path)
    else:
        _train_ors(ors, loader, ORS_EPOCHS)
        _save(ors, ckpt_path, epochs=ORS_EPOCHS, seed=SEED)

    # ── 4. Label accuracy ─────────────────────────────────────────────────────
    print("\n[4] Label accuracy (ORS-VQ vs its own counterfactual training labels) ...")
    ors.eval()
    with torch.no_grad():
        r_l, t_l, c_l, f_l, _, _ = ors(X_test)
    dims = ["retry", "timeout", "CB", "fallback"]
    for i, (name, logits) in enumerate(zip(dims, [r_l, t_l, c_l, f_l])):
        acc = (logits.argmax(1) == Y_test[:, i]).float().mean().item() * 100
        print(f"    {name:<10} {acc:.1f}%")

    # ── 5. Counterfactual accuracy ─────────────────────────────────────────────
    print("\n[5] Counterfactual policy accuracy (ORS-VQ vs simulator-optimal labels) ...")
    cf = _cf_accuracy(ors, test_all)
    for k, v in cf.items():
        print(f"    {k:<14} {v:.1f}%")

    # ── 6. Simulation evaluation ──────────────────────────────────────────────
    print("\n[6] Simulation evaluation (scenario-aware simulator) ...")
    sim_result = _simulate(ors, test_all)
    print(f"    Completion:   {sim_result['completion']:.1f}%")
    print(f"    Wasted retry: {sim_result['wasted_pct']:.1f}%")
    print(f"    Avg latency:  {sim_result['avg_latency']:.2f}")

    # ── 7. Per-scenario accuracy ──────────────────────────────────────────────
    print("\n[7] Per-scenario breakdown ...")
    _print_scenario_accuracy(ors, test_all)

    # ── 8. ORS table — the research contribution ──────────────────────────────
    print("\n[8] Building ORS table (what does each code represent?) ...")
    ors_rows = ors.ors_table(test_all)
    _print_ors_table(ors_rows)

    # ── 9. Cross-workflow transfer experiment ──────────────────────────────────
    print("\n[9] Cross-workflow ORS transfer experiment ...")
    print("    Testing claim: same failure scenario => same ORS code across workflows.")
    transfer = ors.cross_workflow_codes(test_all)
    _print_transfer_table(transfer, SCENARIO_NAMES)

    # ── 10. Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"  ORS-VQ SUMMARY")
    print(f"{'='*80}")
    print(f"  Architecture:    shared codebook K={CODEBOOK_SIZE}, one ORS per entry")
    print(f"  Input:           {TEMPORAL_FEATURE_DIM}-dim (38 static + 9 temporal/CB-state)")
    print(f"  Labels:          counterfactual simulation (oracle-free)")
    print(f"  Category:        post-hoc / emergent (NOT in training loss)")
    print(f"  Completion:      {sim_result['completion']:.1f}%")
    print(f"  Wasted retry:    {sim_result['wasted_pct']:.1f}%")
    print(f"  CF exact match:  {cf['cf_exact']:.1f}%")
    print(f"  Active ORS codes:{len(ors_rows)} / {CODEBOOK_SIZE}")
    print(f"  Runtime:         {elapsed:.1f}s")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load saved checkpoints")
    args = parser.parse_args()
    main(load_checkpoints=args.load)
