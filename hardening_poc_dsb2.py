"""
hardening_poc_dsb2.py  --  Workflow Hardening on real DSB microservice graphs

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m hardening_poc_dsb2

Uses:
  - ORS model trained on real DSB X-Trace data  (ors_dsb_vq.pt)
  - Workflow graphs built from real DSB socialNetwork call topologies
    (not n8n workflow JSON files)

DSB workloads hardened:
  1. compose_post          (healthy fan-out, 9 services)
  2. read_home_timeline    (healthy read, 4 services)
  3. compose_broken        (SERVICE_DOWN — same topology, PostStorage down)
  4. read_home_broken_pipe (TRANSIENT — broken pipe failures)

For each workload the engine:
  1. Assigns risk scenario per node from DSB service type + graph position
  2. Builds synthetic 47-dim telemetry matching that scenario
  3. Runs DSB-trained ORS-VQ -> ORS code + resilience policy
  4. Simulates original vs hardened under HEALTHY/TRANSIENT/DEGRADED/DOWN (500 trials each)
"""

from __future__ import annotations
import random, time
from pathlib import Path

import torch

from resiliency_poc.dsb_workflow_loader       import load_dsb_healthy_and_failure
from resiliency_poc.scenario                  import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model           import ORSVQModel
from resiliency_poc.temporal_trace_generator  import TEMPORAL_FEATURE_DIM
from resiliency_poc.scenario                  import (
    CB_CLOSED, CB_HALF_OPEN, CB_OPEN,
)

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO, _DEEP_EXTERNAL_SCENARIO
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario
from hardening.models    import FALLBACK_NAMES

SEED          = 42
CODEBOOK_SIZE = 16
N_TRIALS      = 500
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")

random.seed(SEED)

# ── Extend the engine's node-type -> scenario map with DSB service names ───────
# The hardening engine maps node type_name -> ScenarioState.
# DSB nodes have type_name = service name, so we add those here.

_DSB_TYPE_TO_SCENARIO = {
    # API gateway — entry point, healthy
    "nginx-thrift":             ScenarioState.HEALTHY,
    # Orchestrator services — can have transient fan-out failures
    "ComposePostService":       ScenarioState.TRANSIENT_FAILURE,
    # Storage services — more failure-prone, treated as degraded
    "PostStorageService":       ScenarioState.DEGRADED,
    "HomeTimelineService":      ScenarioState.DEGRADED,
    "UserTimelineService":      ScenarioState.DEGRADED,
    "WriteHomeTimelineService": ScenarioState.DEGRADED,
    # Auth / user service — crypto-based, can degrade
    "UserService":              ScenarioState.DEGRADED,
    # Social graph — external store calls
    "SocialGraphService":       ScenarioState.TRANSIENT_FAILURE,
    # Leaf processing services — mostly healthy, internal compute
    "UniqueIdService":          ScenarioState.HEALTHY,
    "TextService":              ScenarioState.HEALTHY,
    "MediaService":             ScenarioState.TRANSIENT_FAILURE,
    "UserMentionService":       ScenarioState.HEALTHY,
    "UrlShortenService":        ScenarioState.HEALTHY,
}

# Patch into the engine's lookup at runtime
_TYPE_TO_SCENARIO.update(_DSB_TYPE_TO_SCENARIO)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_ors() -> ORSVQModel:
    model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run resiliency_poc.ors_dsb_poc first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    model.load_state_dict(d["state_dict"])
    model.eval()
    return model


def _print_compiler_view(hw) -> None:
    W = 105
    print("\n" + "=" * W)
    print(f"  ORS COMPILER VIEW: {hw.original.name}")
    print("  DSB microservice call graph hardened by DSB-trained ORS model")
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
        hn = hw.nodes.get(node_name)
        if not hn:
            continue
        ann  = hn.annotation
        node = hn.node
        ext  = "[ext]" if node.is_external else "     "
        changed = "*" if not ann.is_trivial else " "
        print(f"  {changed} {node_name:<30} {ext}  depth={node.depth}  "
              f"risk={ann.risk_scenario:<10}  ORS-{ann.ors_code:<2}  => {ann.describe()}")

    print(f"\n  Total nodes: {len(hw.nodes)}  |  "
          f"Hardened: {len(hw.hardened_nodes)}  |  "
          f"Constructs added: {hw.total_added_constructs}")
    print("=" * W)


def _print_comparison_table(workflow_name: str, orig_rows: list, hard_rows: list) -> None:
    W = 115
    print("\n" + "=" * W)
    print(f"  WORKFLOW: {workflow_name}")
    print(f"  Comparison: Original DSB microservice chain vs ORS-Hardened  [DSB-trained model]")
    print("=" * W)
    print(f"  {'Scenario':<12}  {'Model':<10}  {'Complete%':>9}  "
          f"{'API calls':>9}  {'Wasted':>7}  {'Fallback%':>9}  "
          f"{'DataLoss%':>9}  {'Steps':>6}")
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
                  f"{o.fallback_pct:>8.1f}%  {o.data_loss_pct:>8.1f}%  {o.avg_steps:>6.2f}")
        if h and o:
            delta = h.completion_pct - o.completion_pct
            tag   = f"  (+{delta:.1f}%)" if delta > 0 else ""
            print(f"  {'':<12}  {'ORS-Hard':<10}  {h.completion_pct:>8.1f}%{tag:<12}"
                  f"{h.avg_api_calls:>9.2f}  {h.avg_wasted:>7.2f}  "
                  f"{h.fallback_pct:>8.1f}%  {h.data_loss_pct:>8.1f}%  {h.avg_steps:>6.2f}")
        print()
    print("=" * W)


def _print_cost_savings(workflow_name: str, orig_rows: list, hard_rows: list) -> None:
    W = 80
    print("\n" + "=" * W)
    print(f"  COST SAVINGS: {workflow_name}")
    print(f"  {N_TRIALS} trials per scenario  |  DSB-trained ORS model")
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
        saved  = (o.avg_api_calls - h.avg_api_calls) * N_TRIALS
        wasted = (o.avg_wasted    - h.avg_wasted)    * N_TRIALS
        gained = h.completion_pct - o.completion_pct
        total_saved += max(saved, 0)
        print(f"  {sc:<14}: gain=+{gained:.1f}%  "
              f"calls={saved:+.0f}  wasted={wasted:+.0f}")

    print(f"\n  Total API calls saved (healthy scenario): ~{total_saved:.0f}")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[1] Loading DSB-trained ORS-VQ model ...")
    ors = _load_ors()
    print(f"    Checkpoint: {CKPT_ORS}")
    print(f"    Trained on: real DSB socialNetwork X-Trace (~21k service events)")

    print("\n[2] Loading DSB microservice call graph topologies ...")
    graphs = load_dsb_healthy_and_failure()
    for g in graphs:
        n_edges = sum(len(v) for v in g.adj.values())
        print(f"    {g.name:<50}  nodes={len(g.nodes)}  edges={n_edges}")

    engine = WorkflowHardeningEngine(ors_model=ors, seed=SEED)
    sim    = WorkflowFailureSimulator(seed=SEED)

    eval_scenarios = [
        ScenarioState.HEALTHY,
        ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED,
        ScenarioState.SERVICE_DOWN,
    ]

    all_results = {}

    for graph in graphs:
        print(f"\n{'='*80}")
        print(f"  Hardening: {graph.name}")
        print(f"{'='*80}")

        print("\n[3] ORS hardening analysis ...")
        hw = engine.analyze(graph)
        _print_compiler_view(hw)

        print("\n[4] Hardened workflow diff ...")
        hw.print_diff()

        print(f"\n[5] Failure injection simulation ({N_TRIALS} trials per scenario) ...")
        orig_rows, hard_rows = [], []
        for scenario in eval_scenarios:
            sc_name = SCENARIO_NAMES[int(scenario)]
            orig_m, hard_m = evaluate_scenario(sim, graph, hw, scenario, N_TRIALS)
            orig_rows.append(orig_m)
            hard_rows.append(hard_m)
            print(f"    {sc_name:<14}  original={orig_m.completion_pct:.1f}%  "
                  f"hardened={hard_m.completion_pct:.1f}%")

        _print_comparison_table(graph.name, orig_rows, hard_rows)
        _print_cost_savings(graph.name, orig_rows, hard_rows)
        all_results[graph.name] = (orig_rows, hard_rows)

    # ── Cross-workload summary ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  CROSS-WORKLOAD HARDENING SUMMARY  (DSB microservice graphs)")
    print(f"{'='*80}")
    print(f"  {'Workflow':<50}  {'Scenario':<12}  {'Original':>9}  {'Hardened':>9}  {'Gain':>6}")
    print("  " + "-" * 100)

    for gname, (orig_rows, hard_rows) in all_results.items():
        orig_by_sc = {m.scenario: m for m in orig_rows}
        hard_by_sc = {m.scenario: m for m in hard_rows}
        for sc in [SCENARIO_NAMES[int(s)] for s in [
                ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
                ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN]]:
            o = orig_by_sc.get(sc)
            h = hard_by_sc.get(sc)
            if o and h:
                gain = h.completion_pct - o.completion_pct
                print(f"  {gname[:50]:<50}  {sc:<12}  "
                      f"{o.completion_pct:>8.1f}%  {h.completion_pct:>8.1f}%  "
                      f"{gain:>+5.1f}%")
        print()

    elapsed = time.time() - t0
    print(f"  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
