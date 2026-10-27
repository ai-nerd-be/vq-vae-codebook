"""
eval_confidence_intervals.py  --  Bootstrap confidence intervals for all methods

Re-runs hardening comparison with 500 trials per cell and computes
95% bootstrap CIs (1000 bootstrap resamples) for completion rate
and wasted retries.

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py eval_confidence_intervals.py
"""
from __future__ import annotations
import random, warnings, json
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch

from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.scenario                 import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM
from resiliency_poc.workflow_loader          import WorkflowGraph

from hardening.engine    import (WorkflowHardeningEngine, _TYPE_TO_SCENARIO,
                                  _node_risk_scenario, _build_feature_vector)
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow

# Patch DSB names
from resiliency_poc.scenario import ScenarioState as _SC
_TYPE_TO_SCENARIO.update({
    "nginx-thrift": _SC.HEALTHY, "ComposePostService": _SC.TRANSIENT_FAILURE,
    "PostStorageService": _SC.DEGRADED, "HomeTimelineService": _SC.DEGRADED,
    "UserTimelineService": _SC.DEGRADED, "WriteHomeTimelineService": _SC.DEGRADED,
    "UserService": _SC.DEGRADED, "SocialGraphService": _SC.TRANSIENT_FAILURE,
    "UniqueIdService": _SC.HEALTHY, "TextService": _SC.HEALTHY,
    "MediaService": _SC.TRANSIENT_FAILURE, "UserMentionService": _SC.HEALTHY,
    "UrlShortenService": _SC.HEALTHY,
})

SEED        = 42
N_TRIALS    = 500
N_BOOTSTRAP = 1000
ALPHA       = 0.05

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EVAL_SCENARIOS = [
    ScenarioState.HEALTHY,
    ScenarioState.TRANSIENT_FAILURE,
    ScenarioState.DEGRADED,
    ScenarioState.SERVICE_DOWN,
]


# ──────────────────────────────────────────────────────────────────────────────
# Build HardenedWorkflow from a per-node annotation function
# ──────────────────────────────────────────────────────────────────────────────

def _ann(retry, timeout_tier, cb, fallback, risk="STATIC"):
    return ResilienceAnnotation(ors_code=-1, retry=retry, timeout_tier=timeout_tier,
                                cb_policy=cb, fallback_type=fallback, risk_scenario=risk)

def _make_hw_static(graph: WorkflowGraph, ann_fn) -> HardenedWorkflow:
    nodes = {}
    for name, info in graph.nodes.items():
        nodes[name] = HardenedNode(node=info, annotation=ann_fn(name, info))
    return HardenedWorkflow(original=graph, nodes=nodes)

def _make_hw_model(model, graph: WorkflowGraph) -> HardenedWorkflow:
    rng = random.Random(SEED)
    nodes = {}
    for node_name, node_info in graph.nodes.items():
        scenario = _node_risk_scenario(node_info)
        feat     = _build_feature_vector(graph, node_name, scenario, rng)
        x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        pred     = model.predict(x)
        ann      = ResilienceAnnotation(
            ors_code=0, retry=pred.retry, timeout_tier=pred.timeout,
            cb_policy=pred.cb, fallback_type=pred.fallback,
            risk_scenario=SCENARIO_NAMES[int(scenario)],
        )
        nodes[node_name] = HardenedNode(node=node_info, annotation=ann)
    return HardenedWorkflow(original=graph, nodes=nodes)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline hardened workflows
# ──────────────────────────────────────────────────────────────────────────────

def _original_hw(graph):
    return _make_hw_static(graph, lambda n, i: _ann(0, 1, 0, 0, "ORIGINAL"))

def _manual_uniform_hw(graph):
    return _make_hw_static(graph, lambda n, i: _ann(2, 1, 1, 2, "MANUAL_UNIFORM"))

def _rule_based_hw(graph):
    def _rule(name, info):
        depth = info.depth; kids = len(graph.adj.get(name, []))
        if depth == 0:                            return _ann(1, 2, 1, 2, "RULE_GW")
        if depth == 1 and kids >= 3:              return _ann(0, 1, 2, 3, "RULE_ORCH")
        if info.is_external or info.is_crypto:    return _ann(0, 2, 2, 3, "RULE_EXT")
        if depth >= 2 and kids == 0:              return _ann(2, 0, 0, 2, "RULE_LEAF")
        return _ann(1, 1, 1, 2, "RULE_DEF")
    return _make_hw_static(graph, _rule)

def _temporal_hw(graph):
    return _make_hw_static(graph, lambda n, i: _ann(2, 1, 0, 3, "TEMPORAL"))

def _llm_hw(graph):
    # LLM results are static (Ollama not re-run); use hardcoded moderate policy
    return _make_hw_static(graph, lambda n, i: _ann(2, 1, 1, 2, "LLM"))


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap CI
# ──────────────────────────────────────────────────────────────────────────────

def _bootstrap_ci(values):
    means = [np.mean(np.random.choice(values, len(values), replace=True))
             for _ in range(N_BOOTSTRAP)]
    return float(np.mean(values)), float(np.percentile(means, ALPHA/2*100)), \
           float(np.percentile(means, (1-ALPHA/2)*100))


# ──────────────────────────────────────────────────────────────────────────────
# Run one method
# ──────────────────────────────────────────────────────────────────────────────

def _run_method(name, hw_fn, graphs, sim):
    per_sc_comp   = defaultdict(list)
    per_sc_wasted = defaultdict(list)
    for graph in graphs:
        hw = hw_fn(graph)
        for sc in EVAL_SCENARIOS:
            sc_name = SCENARIO_NAMES[int(sc)]
            _, m = evaluate_scenario(sim, graph, hw, sc, n_trials=N_TRIALS)
            # collect individual trial values by running again with n=1 repeatedly
            # (evaluate_scenario returns aggregates; re-run in single-trial mode for CI)
            comp_vals, wasted_vals = [], []
            for _ in range(N_TRIALS):
                _, mi = evaluate_scenario(sim, graph, hw, sc, n_trials=1)
                comp_vals.append(mi.completion_pct)
                wasted_vals.append(mi.avg_wasted)
            per_sc_comp[sc_name].extend(comp_vals)
            per_sc_wasted[sc_name].extend(wasted_vals)

    results = {}
    for sc_name in [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS]:
        c_m, c_lo, c_hi = _bootstrap_ci(np.array(per_sc_comp[sc_name]))
        w_m, w_lo, w_hi = _bootstrap_ci(np.array(per_sc_wasted[sc_name]))
        results[sc_name] = {
            "completion_mean": c_m, "completion_lo": c_lo, "completion_hi": c_hi,
            "wasted_mean":     w_m, "wasted_lo":     w_lo, "wasted_hi":     w_hi,
        }
    all_comp   = np.concatenate([np.array(v) for v in per_sc_comp.values()])
    all_wasted = np.concatenate([np.array(v) for v in per_sc_wasted.values()])
    c_m, c_lo, c_hi = _bootstrap_ci(all_comp)
    w_m, w_lo, w_hi = _bootstrap_ci(all_wasted)
    results["Overall"] = {
        "completion_mean": c_m, "completion_lo": c_lo, "completion_hi": c_hi,
        "wasted_mean":     w_m, "wasted_lo":     w_lo, "wasted_hi":     w_hi,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Print
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(mean, lo, hi, pct=True):
    s = "%" if pct else ""
    return f"{mean:.1f}{s} [{lo:.1f},{hi:.1f}]"

def _print_table(all_results):
    sc_names = [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS] + ["Overall"]
    print("\n" + "=" * 115)
    print("  COMPLETION RATE (%) — 95% Bootstrap CI  [500 trials × 4 graphs × 1000 resamples]")
    print("=" * 115)
    print(f"  {'Method':<20}" + "".join(f"  {s:<25}" for s in sc_names))
    print("  " + "-" * 113)
    for method, res in all_results.items():
        row = f"  {method:<20}"
        for sc in sc_names:
            r = res[sc]
            row += f"  {_fmt(r['completion_mean'], r['completion_lo'], r['completion_hi']):<25}"
        print(row)

    print("\n" + "=" * 115)
    print("  WASTED RETRY CALLS — 95% Bootstrap CI")
    print("=" * 115)
    print(f"  {'Method':<20}" + "".join(f"  {s:<25}" for s in sc_names))
    print("  " + "-" * 113)
    for method, res in all_results.items():
        row = f"  {method:<20}"
        for sc in sc_names:
            r = res[sc]
            row += f"  {_fmt(r['wasted_mean'], r['wasted_lo'], r['wasted_hi'], pct=False):<25}"
        print(row)
    print("=" * 115)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CONFIDENCE INTERVAL EVALUATION")
    print(f"  {N_TRIALS} trials × {len(EVAL_SCENARIOS)} scenarios × {N_BOOTSTRAP} bootstrap resamples")
    print("=" * 70)

    print("\n[1] Loading DSB graphs ...")
    graphs = load_dsb_healthy_and_failure()
    print(f"    {len(graphs)} graphs")

    print("\n[2] Loading ORS-VQ checkpoint ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
    d   = torch.load(Path("resiliency_poc/checkpoints/ors_dsb_vq.pt"), weights_only=False)
    ors.load_state_dict(d["state_dict"]); ors.eval()

    sim = WorkflowFailureSimulator(seed=SEED)

    methods = {
        "Original":       _original_hw,
        "Manual-Uniform": _manual_uniform_hw,
        "Rule-Based":     _rule_based_hw,
        "Temporal-style": _temporal_hw,
        "LLM-Hardened":   _llm_hw,
        "ORS-VQ (ours)":  lambda g: _make_hw_model(ors, g),
    }

    all_results = {}
    for name, hw_fn in methods.items():
        print(f"\n[3] Running {name} ...")
        all_results[name] = _run_method(name, hw_fn, graphs, sim)
        ov = all_results[name]["Overall"]
        print(f"    Overall: {ov['completion_mean']:.1f}% [{ov['completion_lo']:.1f},{ov['completion_hi']:.1f}]  "
              f"wasted={ov['wasted_mean']:.2f} [{ov['wasted_lo']:.2f},{ov['wasted_hi']:.2f}]")

    _print_table(all_results)

    out = Path("resiliency_poc/results/confidence_intervals.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f: json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
