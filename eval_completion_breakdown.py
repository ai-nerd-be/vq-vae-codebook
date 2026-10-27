"""
eval_completion_breakdown.py  --  Outcome category breakdown per method

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m eval_completion_breakdown

What this does
--------------
Re-runs the same Monte Carlo simulation as hardening_comparison.py but
instead of aggregating into a single Completion % it breaks each trial
into one of five mutually-exclusive outcome categories:

  CLEAN       -- completed, no fallback triggered, no retries wasted
  RETRY-SAVED -- completed, retries fired but eventual primary success
                 (wasted_calls > 0, fallback_used=False, data_loss=False)
  FALLBACK    -- completed via fallback path (cached / alternate route)
                 (fallback_used=True, data_loss=False)
  DEAD-LETTER -- workflow "completed" but data routed to dead-letter queue
                 (data_loss=True) — counted separately from real completion
  ABORTED     -- workflow failed outright (completed=False)

This is the key question: does "Temporal completes more" mean it's
actually recovering cleanly, or is it hiding failures behind dead-letter?
ORS may appear to "complete less" overall while actually producing more
*useful* completions (CLEAN + RETRY-SAVED + FALLBACK without data loss).

The script prints:
  1. Outcome breakdown table  (rows=method, cols=category %)
  2. ORS vs Temporal side-by-side spotlight
  3. "True completion" = CLEAN + RETRY-SAVED + FALLBACK  (no data loss)

Does NOT modify any existing file.
"""

from __future__ import annotations
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.scenario                 import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM
from resiliency_poc.workflow_loader          import WorkflowGraph

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO
from hardening.simulator import WorkflowFailureSimulator, RunResult
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow

# patch DSB service names (same as hardening_comparison.py)
from resiliency_poc.scenario import ScenarioState as _SC
_TYPE_TO_SCENARIO.update({
    "nginx-thrift":             _SC.HEALTHY,
    "ComposePostService":       _SC.TRANSIENT_FAILURE,
    "PostStorageService":       _SC.DEGRADED,
    "HomeTimelineService":      _SC.DEGRADED,
    "UserTimelineService":      _SC.DEGRADED,
    "WriteHomeTimelineService": _SC.DEGRADED,
    "UserService":              _SC.DEGRADED,
    "SocialGraphService":       _SC.TRANSIENT_FAILURE,
    "UniqueIdService":          _SC.HEALTHY,
    "TextService":              _SC.HEALTHY,
    "MediaService":             _SC.TRANSIENT_FAILURE,
    "UserMentionService":       _SC.HEALTHY,
    "UrlShortenService":        _SC.HEALTHY,
})

SEED          = 42
CODEBOOK_SIZE = 16
N_TRIALS      = 500
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")

EVAL_SCENARIOS = [
    ScenarioState.HEALTHY,
    ScenarioState.TRANSIENT_FAILURE,
    ScenarioState.DEGRADED,
    ScenarioState.SERVICE_DOWN,
]


# ── Outcome categories ─────────────────────────────────────────────────────────

OUTCOMES = ["CLEAN", "RETRY-SAVED", "FALLBACK", "DEAD-LETTER", "ABORTED"]


def categorize(r: RunResult) -> str:
    """Map a single RunResult to one of the five outcome categories."""
    if not r.completed:
        return "ABORTED"
    if r.data_loss:
        return "DEAD-LETTER"
    if r.fallback_used:
        return "FALLBACK"
    if r.wasted_calls > 0:
        return "RETRY-SAVED"
    return "CLEAN"


@dataclass
class OutcomeSummary:
    method:        str
    scenario:      str
    n:             int
    counts:        Dict[str, int] = field(default_factory=dict)
    total_wasted:  float = 0.0   # sum of wasted_calls across all trials
    total_api:     float = 0.0   # sum of api_calls across all trials

    def pct(self, cat: str) -> float:
        return self.counts.get(cat, 0) / self.n * 100

    def true_completion_pct(self) -> float:
        """CLEAN + RETRY-SAVED + FALLBACK  (excludes DEAD-LETTER and ABORTED)."""
        useful = self.counts.get("CLEAN", 0) + self.counts.get("RETRY-SAVED", 0) + \
                 self.counts.get("FALLBACK", 0)
        return useful / self.n * 100

    def nominal_completion_pct(self) -> float:
        """As reported by hardening_comparison: includes DEAD-LETTER."""
        return (self.n - self.counts.get("ABORTED", 0)) / self.n * 100

    def avg_wasted(self) -> float:
        return self.total_wasted / self.n

    def avg_api(self) -> float:
        return self.total_api / self.n


# ── Helpers to build hardened workflows (mirrors hardening_comparison.py) ──────

def _ann(retry, timeout_tier, cb, fallback, risk="") -> ResilienceAnnotation:
    return ResilienceAnnotation(
        ors_code=-1, retry=retry, timeout_tier=timeout_tier,
        cb_policy=cb, fallback_type=fallback, risk_scenario=risk,
    )


def _make_hw(graph: WorkflowGraph, ann_fn) -> HardenedWorkflow:
    nodes = {}
    for name, info in graph.nodes.items():
        nodes[name] = HardenedNode(node=info, annotation=ann_fn(name, info))
    return HardenedWorkflow(original=graph, nodes=nodes)


def manual_uniform_hw(graph):
    return _make_hw(graph, lambda n, i: _ann(2, 1, 1, 2, "MANUAL"))

def rule_based_hw(graph):
    def _rule(name, info):
        d = info.depth
        nc = len(graph.adj.get(name, []))
        if d == 0:               return _ann(1, 2, 1, 2, "RULE_GW")
        if d == 1 and nc >= 3:   return _ann(0, 1, 2, 3, "RULE_ORCH")
        if info.is_external or info.is_crypto:
                                  return _ann(0, 2, 2, 3, "RULE_EXT")
        if d >= 2 and nc == 0:   return _ann(2, 0, 0, 2, "RULE_LEAF")
        return _ann(1, 1, 1, 2, "RULE_DEF")
    return _make_hw(graph, _rule)

def temporal_style_hw(graph):
    # retry=3, timeout=30s, NO circuit breaker, fallback=dead_letter
    return _make_hw(graph, lambda n, i: _ann(2, 1, 0, 3, "TEMPORAL"))

def ors_hw(graph, engine):
    return engine.analyze(graph)


# ── Simulation ─────────────────────────────────────────────────────────────────

def run_trials(
    sim:      WorkflowFailureSimulator,
    graph:    WorkflowGraph,
    hw:       HardenedWorkflow,
    scenario: ScenarioState,
    method:   str,
    is_original: bool = False,
) -> OutcomeSummary:
    results = []
    for t in range(N_TRIALS):
        if is_original:
            r = sim.original_run(graph, scenario, t)
        else:
            r = sim.hardened_run(hw, scenario, t)
        results.append(r)

    counts = Counter(categorize(r) for r in results)
    sc_name = SCENARIO_NAMES[int(scenario)]
    return OutcomeSummary(
        method=method, scenario=sc_name, n=N_TRIALS,
        counts=dict(counts),
        total_wasted=sum(r.wasted_calls for r in results),
        total_api=sum(r.api_calls for r in results),
    )


# ── Printing ───────────────────────────────────────────────────────────────────

METHOD_ORDER = [
    "Original",
    "Manual-Uniform",
    "Rule-Based",
    "Temporal-style",
    "ORS-Hardened",
]

CAT_WIDTH = 13


def _print_breakdown_table(
    summaries: Dict[str, Dict[str, OutcomeSummary]],  # [method][scenario]
    graph_name: str,
) -> None:
    cat_hdr = "".join(f"  {c:>{CAT_WIDTH}}" for c in OUTCOMES)
    print(f"\n  Outcome breakdown -- {graph_name}")
    print(f"  {'Method':<18}  {'Scenario':<14}{cat_hdr}  {'TrueCompl%':>11}  {'NomCompl%':>10}  {'AvgWasted':>10}  {'AvgAPI':>7}")
    print("  " + "-" * 150)
    for sc in [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS]:
        for m in METHOD_ORDER:
            s = summaries.get(m, {}).get(sc)
            if not s:
                continue
            cats = "".join(f"  {s.pct(c):>{CAT_WIDTH-1}.1f}%" for c in OUTCOMES)
            print(f"  {m:<18}  {sc:<14}{cats}  {s.true_completion_pct():>10.1f}%  {s.nominal_completion_pct():>9.1f}%  {s.avg_wasted():>10.2f}  {s.avg_api():>7.2f}")
        print()


def _print_ors_vs_temporal(
    all_summaries: Dict[str, Dict[str, Dict[str, OutcomeSummary]]],
) -> None:
    W = 110
    print("\n" + "=" * W)
    print("  ORS-Hardened vs Temporal-style  --  OUTCOME BREAKDOWN SPOTLIGHT")
    print("  'True completion' = CLEAN + RETRY-SAVED + FALLBACK  (no data loss)")
    print("  'Nominal'         = everything except ABORTED  (includes DEAD-LETTER)")
    print("=" * W)

    hdr = (f"  {'Graph':<30}  {'Scenario':<14}  {'Method':<16}"
           + "".join(f"  {c:>{CAT_WIDTH}}" for c in OUTCOMES)
           + f"  {'TrueCompl%':>11}  {'NomCompl%':>10}  {'AvgWasted':>10}  {'AvgAPI':>7}")
    print(hdr)
    print("  " + "-" * 160)

    for graph_name, sc_dict in all_summaries.items():
        short = graph_name.split("/")[-1][:30]
        for sc in [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS]:
            for m in ["ORS-Hardened", "Temporal-style"]:
                s = sc_dict.get(sc, {}).get(m)
                if not s:
                    continue
                cats = "".join(f"  {s.pct(c):>{CAT_WIDTH-1}.1f}%" for c in OUTCOMES)
                print(f"  {short:<30}  {sc:<14}  {m:<16}{cats}"
                      f"  {s.true_completion_pct():>10.1f}%  {s.nominal_completion_pct():>9.1f}%"
                      f"  {s.avg_wasted():>10.2f}  {s.avg_api():>7.2f}")
            print()
    print("=" * W)


def _print_aggregate_table(
    all_summaries: Dict[str, Dict[str, Dict[str, OutcomeSummary]]],
) -> None:
    """One row per method, averaged across all graphs and scenarios."""
    W = 100
    print("\n" + "=" * W)
    print("  AGGREGATE -- averaged across all graphs x all scenarios  (500 trials each)")
    print("=" * W)
    cat_hdr = "".join(f"  {c:>{CAT_WIDTH}}" for c in OUTCOMES)
    print(f"  {'Method':<18}{cat_hdr}  {'TrueCompl%':>11}  {'NomCompl%':>10}  {'AvgWasted':>10}  {'AvgAPI':>7}")
    print("  " + "-" * (W + 10))

    for m in METHOD_ORDER:
        cat_totals: Dict[str, float] = {c: 0.0 for c in OUTCOMES}
        true_total   = 0.0
        nom_total    = 0.0
        wasted_total = 0.0
        api_total    = 0.0
        n_cells      = 0
        for sc_dict in all_summaries.values():
            for sc_name, m_dict in sc_dict.items():
                s = m_dict.get(m)
                if not s:
                    continue
                for c in OUTCOMES:
                    cat_totals[c] += s.pct(c)
                true_total   += s.true_completion_pct()
                nom_total    += s.nominal_completion_pct()
                wasted_total += s.avg_wasted()
                api_total    += s.avg_api()
                n_cells      += 1
        if n_cells == 0:
            continue
        cats = "".join(f"  {cat_totals[c]/n_cells:>{CAT_WIDTH-1}.1f}%" for c in OUTCOMES)
        print(f"  {m:<18}{cats}  {true_total/n_cells:>10.1f}%  {nom_total/n_cells:>9.1f}%"
              f"  {wasted_total/n_cells:>10.2f}  {api_total/n_cells:>7.2f}")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("  COMPLETION OUTCOME BREAKDOWN  --  What does 'completed' actually mean?")
    print("=" * 80)

    print("\n[1] Loading DSB-trained ORS-VQ model ...")
    ors_model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors_model.load_state_dict(d["state_dict"])
    ors_model.eval()

    print("[2] Loading DSB workflow graphs ...")
    graphs = load_dsb_healthy_and_failure()
    for g in graphs:
        print(f"    {g.name}  nodes={len(g.nodes)}")

    engine = WorkflowHardeningEngine(ors_model=ors_model, seed=SEED)
    sim    = WorkflowFailureSimulator(seed=SEED)

    # all_summaries[graph_name][scenario_name][method_name] = OutcomeSummary
    all_summaries: Dict[str, Dict[str, Dict[str, OutcomeSummary]]] = {}

    for graph in graphs:
        gname = graph.name
        print(f"\n  Graph: {gname}")
        all_summaries[gname] = {SCENARIO_NAMES[int(s)]: {} for s in EVAL_SCENARIOS}

        hw_manual   = manual_uniform_hw(graph)
        hw_rule     = rule_based_hw(graph)
        hw_temporal = temporal_style_hw(graph)
        hw_ors      = ors_hw(graph, engine)

        variants = {
            "Original":       (hw_manual, True),   # original_run ignores hw
            "Manual-Uniform": (hw_manual, False),
            "Rule-Based":     (hw_rule,   False),
            "Temporal-style": (hw_temporal, False),
            "ORS-Hardened":   (hw_ors,   False),
        }

        for scenario in EVAL_SCENARIOS:
            sc_name = SCENARIO_NAMES[int(scenario)]
            for method, (hw, is_orig) in variants.items():
                summary = run_trials(sim, graph, hw, scenario, method, is_orig)
                all_summaries[gname][sc_name][method] = summary

            print(f"    {sc_name:<14}  " + "  ".join(
                f"{m[:7]}:true={all_summaries[gname][sc_name][m].true_completion_pct():.0f}%"
                f"/nom={all_summaries[gname][sc_name][m].nominal_completion_pct():.0f}%"
                for m in ["Temporal-style", "ORS-Hardened"]
            ))

        _print_breakdown_table(all_summaries[gname], gname)

    _print_ors_vs_temporal(all_summaries)
    _print_aggregate_table(all_summaries)

    print(f"\n  Total runtime: {time.time()-t0:.1f}s")
    print()
    print("  Key:")
    print("  CLEAN       = success with no retries and no fallback")
    print("  RETRY-SAVED = primary call eventually succeeded after >= 1 failed retry")
    print("  FALLBACK    = primary exhausted; cached/alternate path saved the call")
    print("  DEAD-LETTER = workflow 'completed' but data routed to dead-letter queue")
    print("                (Temporal counts this as completion; ORS counts it as data loss)")
    print("  ABORTED     = hard failure, no recovery path succeeded")


if __name__ == "__main__":
    main()
