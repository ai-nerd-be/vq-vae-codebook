"""
hardening_poc_dsb.py  --  Workflow Hardening Engine using DSB-trained ORS model

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m hardening_poc_dsb

Identical pipeline to hardening_poc.py, but loads ors_dsb_vq.pt —
the ORS model trained on real DeathStarBench socialNetwork X-Trace traces
(~21,578 real service events) instead of synthetic n8n traces.

This means the resilience policies assigned to each workflow node are grounded
in real microservice failure data, not simulated patterns.

Pipeline:
    LLM-generated workflow (n8n JSON)
        |
        v
    WorkflowHardeningEngine
        |  For each node:
        |    1. Assign risk scenario from node type + graph position
        |    2. Build synthetic 47-dim telemetry feature vector
        |    3. DSB-trained ORS-VQ model -> ORS code + resilience policy
        |    4. Record ResilienceAnnotation
        v
    HardenedWorkflow  (original + per-node constructs)
        |
        v
    WorkflowFailureSimulator
        |  Compare original vs hardened under:
        |    HEALTHY / TRANSIENT / DEGRADED / DOWN scenarios
        |  500 Monte-Carlo trials per scenario
        v
    Results table + cost savings
"""

from __future__ import annotations
import random, time
from pathlib import Path

import torch

from resiliency_poc.workflow_loader           import load_both_workflows
from resiliency_poc.scenario                  import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model           import ORSVQModel
from resiliency_poc.temporal_trace_generator  import TEMPORAL_FEATURE_DIM

from hardening.engine    import WorkflowHardeningEngine
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario

SEED          = 42
CODEBOOK_SIZE = 16
N_TRIALS      = 500
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")   # DSB-trained

random.seed(SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_ors() -> ORSVQModel:
    model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError(
            "ors_dsb_vq.pt not found — run resiliency_poc.ors_dsb_poc first."
        )
    d = torch.load(CKPT_ORS, weights_only=False)
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model


def _print_comparison_table(workflow_name: str, orig_rows: list, hard_rows: list) -> None:
    W = 115
    print("\n" + "=" * W)
    print(f"  WORKFLOW: {workflow_name}")
    print(f"  Comparison: Original (no resilience) vs ORS-Hardened  [DSB-trained model]")
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
        if h and o:
            delta = h.completion_pct - o.completion_pct
            delta_tag = f"  (+{delta:.1f}%)" if delta > 0 else ""
            print(f"  {'':<12}  {'ORS-Hard':<10}  {h.completion_pct:>8.1f}%{delta_tag:<12}"
                  f"{h.avg_api_calls:>9.2f}  {h.avg_wasted:>7.2f}  "
                  f"{h.fallback_pct:>8.1f}%  {h.data_loss_pct:>8.1f}%  "
                  f"{h.avg_steps:>6.2f}")
        print()
    print("=" * W)


def _print_cost_savings(workflow_name: str, orig_rows: list, hard_rows: list) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"  COST SAVINGS SUMMARY: {workflow_name}  [DSB-trained model]")
    print(f"  {N_TRIALS} trials per scenario")
    print("=" * W)

    scenarios = [SCENARIO_NAMES[int(s)] for s in [
        ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN,
    ]]
    orig_by_sc = {m.scenario: m for m in orig_rows}
    hard_by_sc = {m.scenario: m for m in hard_rows}

    total_saved = 0.0
    for sc in scenarios:
        o = orig_by_sc.get(sc)
        h = hard_by_sc.get(sc)
        if not o or not h:
            continue
        saved_calls  = (o.avg_api_calls - h.avg_api_calls) * N_TRIALS
        saved_wasted = (o.avg_wasted    - h.avg_wasted)    * N_TRIALS
        gained       = h.completion_pct - o.completion_pct
        total_saved += max(saved_calls, 0)

        print(f"  {sc:<14}:")
        print(f"    Completion gain:    +{gained:.1f}%")
        print(f"    API calls saved:   {saved_calls:+.0f}  ({o.avg_api_calls:.2f} -> {h.avg_api_calls:.2f} per run)")
        print(f"    Wasted calls saved:{saved_wasted:+.0f}  ({o.avg_wasted:.2f} -> {h.avg_wasted:.2f} per run)")

    print(f"\n  Total primary API calls saved across all scenarios: ~{total_saved:.0f}")
    print("=" * W)


def _print_ors_compiler_view(hw) -> None:
    W = 100
    print("\n" + "=" * W)
    print(f"  ORS COMPILER VIEW: {hw.original.name}  [DSB-trained ORS]")
    print("  Each node shows: BEFORE (raw) -> AFTER (hardened by DSB-trained ORS)")
    print("=" * W)

    visited, queue, order = set(), list(hw.original.entry_nodes), []
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
        changed = "*" if not ann.is_trivial else " "
        print(f"  {changed} {node_name:<30}{ext}  {node.type_name:<22} -> {ann.describe()}")
        if not ann.is_trivial:
            print(f"      [ORS-{ann.ors_code} | risk={ann.risk_scenario}]")

    print(f"\n  (* = node received resilience constructs)")
    print(f"  Total nodes:              {len(hw.nodes)}")
    print(f"  Nodes hardened:           {len(hw.hardened_nodes)}")
    print(f"  Total constructs added:   {hw.total_added_constructs}")
    print("=" * W)


def _print_ors_diff_vs_synthetic(hw_dsb, hw_syn, graph_name: str) -> None:
    """
    Compare DSB-trained vs synthetic-trained ORS annotations node by node.
    Shows where the two models disagree on resilience policy.
    """
    W = 115
    print("\n" + "=" * W)
    print(f"  ORS POLICY COMPARISON: {graph_name}")
    print("  DSB-trained model vs Synthetic-trained model — where do they disagree?")
    print("=" * W)
    print(f"  {'Node':<32}  {'DSB-ORS policy':<40}  {'Syn-ORS policy':<40}  Match?")
    print("  " + "-" * (W - 2))

    differ = same = 0
    for node_name in hw_dsb.nodes:
        hn_d = hw_dsb.nodes.get(node_name)
        hn_s = hw_syn.nodes.get(node_name) if hw_syn else None
        if not hn_d:
            continue
        ann_d = hn_d.annotation
        ann_s = hn_s.annotation if hn_s else None

        pol_d = ann_d.describe()
        pol_s = ann_s.describe() if ann_s else "N/A"
        match = (ann_d.retry == ann_s.retry and
                 ann_d.cb_policy == ann_s.cb_policy and
                 ann_d.fallback_type == ann_s.fallback_type) if ann_s else False
        flag  = "  [SAME]" if match else "  [DIFF]"
        print(f"  {node_name:<32}  {pol_d:<40}  {pol_s:<40}  {flag}")
        if match:
            same += 1
        else:
            differ += 1

    print("  " + "-" * (W - 2))
    total = same + differ
    print(f"  Agreement: {same}/{total} nodes ({same/total:.0%})  |  "
          f"Disagreement: {differ}/{total} nodes ({differ/total:.0%})")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[1] Loading DSB-trained ORS-VQ model ...")
    ors_dsb = _load_ors()
    print(f"    Checkpoint: {CKPT_ORS}")
    print(f"    Codebook size: {CODEBOOK_SIZE}  |  Input: {TEMPORAL_FEATURE_DIM}-dim")
    print(f"    Trained on: real DeathStarBench socialNetwork X-Trace (~21k events)")

    # Also load synthetic model for comparison if available
    syn_ckpt = Path("resiliency_poc/checkpoints/ors_vq.pt")
    ors_syn  = None
    if syn_ckpt.exists():
        ors_syn = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
        d = torch.load(syn_ckpt, weights_only=False)
        ors_syn.load_state_dict(d["state_dict"])
        ors_syn.eval()
        print(f"    Also loaded synthetic model ({syn_ckpt.name}) for comparison.")

    print("\n[2] Loading LLM-generated workflow graphs ...")
    graphs = load_both_workflows()
    for g in graphs:
        print(f"    {g.name:<60} nodes={len(g.nodes)}")

    engine_dsb = WorkflowHardeningEngine(ors_model=ors_dsb, seed=SEED)
    engine_syn = WorkflowHardeningEngine(ors_model=ors_syn, seed=SEED) if ors_syn else None
    sim        = WorkflowFailureSimulator(seed=SEED)

    eval_scenarios = [
        ScenarioState.HEALTHY,
        ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED,
        ScenarioState.SERVICE_DOWN,
    ]

    for graph in graphs:
        print(f"\n{'='*80}")
        print(f"  Hardening: {graph.name}  [DSB-trained ORS]")
        print(f"{'='*80}")

        # ── Harden with DSB model ──────────────────────────────────────────────
        print("\n[3] Running ORS hardening analysis (DSB model) ...")
        hw_dsb = engine_dsb.analyze(graph)
        hw_syn = engine_syn.analyze(graph) if engine_syn else None

        # ── Compiler view ──────────────────────────────────────────────────────
        _print_ors_compiler_view(hw_dsb)

        # ── Diff view ──────────────────────────────────────────────────────────
        print("\n[4] Hardened workflow node-by-node diff ...")
        hw_dsb.print_diff()

        # ── Compare DSB vs synthetic policies ─────────────────────────────────
        if hw_syn:
            print("\n[4b] DSB-trained vs Synthetic-trained ORS policy comparison ...")
            _print_ors_diff_vs_synthetic(hw_dsb, hw_syn, graph.name)

        # ── Simulate ──────────────────────────────────────────────────────────
        print(f"\n[5] Failure injection simulation ({N_TRIALS} trials per scenario) ...")
        orig_rows, hard_rows = [], []
        for scenario in eval_scenarios:
            sc_name = SCENARIO_NAMES[int(scenario)]
            orig_m, hard_m = evaluate_scenario(sim, graph, hw_dsb, scenario, N_TRIALS)
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


if __name__ == "__main__":
    main()
