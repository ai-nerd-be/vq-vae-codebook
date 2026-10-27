"""
hardening_poc.py  --  Workflow Hardening Engine demo

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m hardening_poc

Pipeline:
    LLM-generated workflow (n8n JSON)
        |
        v
    WorkflowHardeningEngine
        |  For each node:
        |    1. Assign risk scenario from node type + graph position
        |    2. Build synthetic 47-dim telemetry feature vector
        |    3. ORS-VQ model -> ORS code + resilience policy
        |    4. Record ResilienceAnnotation
        v
    HardenedWorkflow  (original + per-node constructs)
        |
        v
    WorkflowFailureSimulator
        |  Compare original vs hardened under:
        |    HEALTHY / TRANSIENT / DEGRADED / DOWN scenarios
        |  Metrics: completion, API cost, wasted calls, MTTR, availability
        v
    Results table
"""

from __future__ import annotations
import random, time
from pathlib import Path

import torch

from resiliency_poc.workflow_loader    import load_both_workflows
from resiliency_poc.scenario           import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model    import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM

from hardening.engine    import WorkflowHardeningEngine
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics

SEED          = 42
CODEBOOK_SIZE = 16
N_TRIALS      = 500
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_vq.pt")

random.seed(SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_ors() -> ORSVQModel:
    model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_vq.pt not found — run ors_poc.py first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model


def _print_comparison_table(
    workflow_name: str,
    orig_rows:     list,
    hard_rows:     list,
) -> None:
    W = 115
    print("\n" + "=" * W)
    print(f"  WORKFLOW: {workflow_name}")
    print(f"  Comparison: Original (no resilience) vs ORS-Hardened")
    print("=" * W)
    hdr = (f"  {'Scenario':<12}  {'Model':<10}  {'Complete%':>9}  "
           f"{'API calls':>9}  {'Wasted':>7}  {'Fallback%':>9}  "
           f"{'DataLoss%':>9}  {'Steps':>6}")
    print(hdr)
    print("  " + "-" * (W - 2))

    scenarios = [SCENARIO_NAMES[int(s)] for s in [
        ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN,
    ]]

    orig_by_sc = {m.scenario: m for m in orig_rows}
    hard_by_sc = {m.scenario: m for m in hard_rows}

    for sc in scenarios:
        o = orig_by_sc.get(sc)
        h = hard_by_sc.get(sc)
        if o:
            print(f"  {sc:<12}  {'Original':<10}  {o.completion_pct:>8.1f}%  "
                  f"{o.avg_api_calls:>9.2f}  {o.avg_wasted:>7.2f}  "
                  f"{o.fallback_pct:>8.1f}%  {o.data_loss_pct:>8.1f}%  "
                  f"{o.avg_steps:>6.2f}")
        if h:
            delta_complete = (h.completion_pct - o.completion_pct) if o else 0
            delta_tag = f"  (+{delta_complete:.1f}%)" if delta_complete > 0 else ""
            print(f"  {'':<12}  {'ORS-Hard':<10}  {h.completion_pct:>8.1f}%{delta_tag:<12}"
                  f"{h.avg_api_calls:>9.2f}  {h.avg_wasted:>7.2f}  "
                  f"{h.fallback_pct:>8.1f}%  {h.data_loss_pct:>8.1f}%  "
                  f"{h.avg_steps:>6.2f}")
        print()
    print("=" * W)


def _print_cost_savings(workflow_name: str, orig_rows: list, hard_rows: list) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"  COST SAVINGS SUMMARY: {workflow_name}")
    print(f"  {N_TRIALS} trials per scenario")
    print("=" * W)

    scenarios = [SCENARIO_NAMES[int(s)] for s in [
        ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN,
    ]]
    orig_by_sc = {m.scenario: m for m in orig_rows}
    hard_by_sc = {m.scenario: m for m in hard_rows}

    total_saved_calls = 0.0
    for sc in scenarios:
        o = orig_by_sc.get(sc)
        h = hard_by_sc.get(sc)
        if not o or not h:
            continue
        saved_calls   = (o.avg_api_calls - h.avg_api_calls) * N_TRIALS
        saved_wasted  = (o.avg_wasted    - h.avg_wasted)    * N_TRIALS
        gained_compl  = (h.completion_pct - o.completion_pct)
        total_saved_calls += max(saved_calls, 0)

        print(f"  {sc:<14}:")
        print(f"    Completion gain:   +{gained_compl:+.1f}%")
        print(f"    API calls saved:   {saved_calls:+.0f}  ({o.avg_api_calls:.2f} -> {h.avg_api_calls:.2f} per run)")
        print(f"    Wasted calls saved:{saved_wasted:+.0f}  ({o.avg_wasted:.2f} -> {h.avg_wasted:.2f} per run)")

    print(f"\n  Total primary API calls saved across all scenarios: ~{total_saved_calls:.0f}")
    print("=" * W)


def _print_ors_compiler_view(hw) -> None:
    """Show the ORS as a 'compiler pass' — what got inserted where."""
    W = 100
    print("\n" + "=" * W)
    print(f"  ORS COMPILER VIEW: {hw.original.name}")
    print("  Each node shows: BEFORE (raw) -> AFTER (hardened by ORS)")
    print("=" * W)

    visited = set()
    queue   = list(hw.original.entry_nodes)
    order   = []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        order.append(n)
        queue.extend(hw.original.adj.get(n, []))

    for node_name in order:
        hn  = hw.nodes.get(node_name)
        if not hn:
            continue
        ann  = hn.annotation
        node = hn.node
        ext  = " [ext]" if node.is_external else "      "
        before = node.type_name
        after  = ann.describe()
        ors    = f"ORS-{ann.ors_code}"
        risk   = ann.risk_scenario
        changed = "*" if not ann.is_trivial else " "
        print(f"  {changed} {node_name:<30}{ext}  {before:<22} -> {after}")
        if not ann.is_trivial:
            print(f"      [{ors} | risk={risk}]")
    print(f"\n  (* = node received resilience constructs)")
    print(f"  Total nodes:              {len(hw.nodes)}")
    print(f"  Nodes hardened:           {len(hw.hardened_nodes)}")
    print(f"  Total constructs added:   {hw.total_added_constructs}")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[1] Loading ORS-VQ model ...")
    ors = _load_ors()
    print(f"    Codebook size: {CODEBOOK_SIZE}  |  Input: {TEMPORAL_FEATURE_DIM}-dim")

    print("\n[2] Loading LLM-generated workflow graphs ...")
    graphs = load_both_workflows()
    for g in graphs:
        print(f"    {g.name:<60} nodes={len(g.nodes)}")

    engine = WorkflowHardeningEngine(ors_model=ors, seed=SEED)
    sim    = WorkflowFailureSimulator(seed=SEED)

    eval_scenarios = [
        ScenarioState.HEALTHY,
        ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED,
        ScenarioState.SERVICE_DOWN,
    ]

    for graph in graphs:
        print(f"\n{'='*80}")
        print(f"  Hardening: {graph.name}")
        print(f"{'='*80}")

        # ── Harden ────────────────────────────────────────────────────────────
        print("\n[3] Running ORS hardening analysis ...")
        hw = engine.analyze(graph)

        # ── Compiler view ──────────────────────────────────────────────────────
        _print_ors_compiler_view(hw)

        # ── Full diff ─────────────────────────────────────────────────────────
        print("\n[4] Hardened workflow node-by-node diff ...")
        hw.print_diff()

        # ── Simulate ──────────────────────────────────────────────────────────
        print(f"\n[5] Failure injection simulation ({N_TRIALS} trials per scenario) ...")
        orig_rows, hard_rows = [], []
        for scenario in eval_scenarios:
            sc_name = SCENARIO_NAMES[int(scenario)]
            orig_m, hard_m = evaluate_scenario(sim, graph, hw, scenario, N_TRIALS)
            orig_rows.append(orig_m)
            hard_rows.append(hard_m)
            print(f"    {sc_name:<14} done  "
                  f"original={orig_m.completion_pct:.1f}%  "
                  f"hardened={hard_m.completion_pct:.1f}%")

        # ── Results ───────────────────────────────────────────────────────────
        _print_comparison_table(graph.name, orig_rows, hard_rows)
        _print_cost_savings(graph.name, orig_rows, hard_rows)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
