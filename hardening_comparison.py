"""
hardening_comparison.py  --  Full baseline comparison for the reviewer table

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m hardening_comparison

Dataset
───────
  Workflow graphs : 4 real DSB socialNetwork call topologies
                    (compose_post, read_home_timeline, compose_broken,
                     read_home_broken_pipe)
  ORS model       : ors_dsb_vq.pt  (trained on ~21k real X-Trace events)
  Simulation      : Monte Carlo, 500 trials per scenario per graph

Six methods compared
────────────────────
  1. Original          -- no resilience constructs
  2. Manual-Uniform    -- retry=3, timeout=30s, CB=half_open, fallback=cached
                          applied identically to every node (what a developer
                          does without trace data or graph awareness)
  3. Rule-Based        -- depth/type heuristics, no learning:
                          gateway → CB=half_open, retry=1
                          orchestrator (fan-out) → CB=open, fallback=dead_letter
                          storage/external → CB=open, fallback=dead_letter
                          leaf compute → CB=closed, retry=3
  4. Temporal-style    -- documented Temporal/Camunda/Step-Functions defaults:
                          retry=3, timeout=30s, CB=none (these engines have no
                          built-in CB), fallback=dead_letter (saga compensation)
                          uniform across all nodes, no adaptation
  5. LLM-Hardened      -- Ollama (llama3.1:latest) annotates each node
  6. ORS-Hardened      -- DSB-trained ORS-VQ, learned from real trace data

Key claim being tested
──────────────────────
  ORS is the only method that is BOTH graph-aware AND data-driven.
  It learns from what actually failed in production traces, not from
  documentation defaults or topology heuristics alone.

Metrics
───────
  Completion %   : fraction of Monte Carlo trials that completed successfully
  MTTR (steps)   : avg steps to first completion (proxy for recovery speed)
  API calls/run  : avg total API calls per trial (efficiency)
  Wasted calls   : avg failed calls that were retried and still failed
  Fallback %     : fraction of trials that used fallback path
"""

from __future__ import annotations
import json, re, time, urllib.request, urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.scenario                 import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM
from resiliency_poc.workflow_loader          import WorkflowGraph

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow

# ── Patch DSB service names ────────────────────────────────────────────────────
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

# ── Config ─────────────────────────────────────────────────────────────────────

SEED          = 42
CODEBOOK_SIZE = 16
N_TRIALS      = 500
CKPT_ORS      = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "llama3.1:latest"

EVAL_SCENARIOS = [
    ScenarioState.HEALTHY,
    ScenarioState.TRANSIENT_FAILURE,
    ScenarioState.DEGRADED,
    ScenarioState.SERVICE_DOWN,
]


# ── Helpers to build HardenedWorkflow from a flat annotation dict ──────────────

def _make_hw(graph: WorkflowGraph,
             ann_fn) -> HardenedWorkflow:
    """
    Build a HardenedWorkflow by calling ann_fn(node_name, node_info) → ResilienceAnnotation
    for every node in the graph.
    """
    nodes = {}
    for name, info in graph.nodes.items():
        ann = ann_fn(name, info)
        nodes[name] = HardenedNode(node=info, annotation=ann)
    return HardenedWorkflow(original=graph, nodes=nodes)


def _ann(retry: int, timeout_tier: int, cb: int, fallback: int,
         risk: str = "RULE") -> ResilienceAnnotation:
    return ResilienceAnnotation(
        ors_code=-1, retry=retry, timeout_tier=timeout_tier,
        cb_policy=cb, fallback_type=fallback, risk_scenario=risk,
    )


# ── Baseline 1: Original (no resilience) ──────────────────────────────────────
# The simulator's evaluate_scenario() always runs original_run() on the
# unhardened graph, so this is produced automatically as the first return value.
# We don't need a separate HardenedWorkflow for it.


# ── Baseline 2: Manual-Uniform ────────────────────────────────────────────────
# What a developer does with no data: apply one sensible default to every node.
# Retry=3, timeout=30s, CB=half_open, fallback=cached.

def manual_uniform_hw(graph: WorkflowGraph) -> HardenedWorkflow:
    return _make_hw(graph, lambda name, info: _ann(
        retry=2,          # 3 retries
        timeout_tier=1,   # 30s
        cb=1,             # half_open
        fallback=2,       # cached_response
        risk="MANUAL_UNIFORM",
    ))


# ── Baseline 3: Rule-Based ────────────────────────────────────────────────────
# Heuristics based on node depth + graph position. No model, no training data.
# Mirrors what a senior engineer would write as a config script.

def rule_based_hw(graph: WorkflowGraph) -> HardenedWorkflow:
    def _rule(name: str, info) -> ResilienceAnnotation:
        depth = info.depth
        n_children = len(graph.adj.get(name, []))
        is_external = info.is_external
        is_crypto   = info.is_crypto

        if depth == 0:
            # Gateway / entry node — protect with light retry + half_open CB
            return _ann(retry=1, timeout_tier=2, cb=1, fallback=2, risk="RULE_GATEWAY")

        if depth == 1 and n_children >= 3:
            # Orchestrator with large fan-out — aggressive CB, dead-letter
            return _ann(retry=0, timeout_tier=1, cb=2, fallback=3, risk="RULE_ORCHESTRATOR")

        if is_external or is_crypto:
            # External store or auth service — open CB, no retry, dead-letter
            return _ann(retry=0, timeout_tier=2, cb=2, fallback=3, risk="RULE_EXTERNAL")

        if depth >= 2 and n_children == 0:
            # Internal leaf service — light retry, no CB (reliable compute)
            return _ann(retry=2, timeout_tier=0, cb=0, fallback=2, risk="RULE_LEAF")

        # Default: moderate
        return _ann(retry=1, timeout_tier=1, cb=1, fallback=2, risk="RULE_DEFAULT")

    return _make_hw(graph, _rule)


# ── Baseline 4: Temporal / Camunda style ──────────────────────────────────────
# Documented defaults from Temporal, Camunda, and AWS Step Functions:
#   Temporal  : maxAttempts=3, initialInterval=1s, backoffCoef=2 (docs.temporal.io)
#   Camunda   : retries=3 per job worker (docs.camunda.io)
#   Step Funcs: MaxAttempts=3, IntervalSeconds=2, BackoffRate=2 (AWS docs)
# None of these engines have a built-in circuit breaker.
# Fallback = dead_letter (closest to saga compensation / catch block).
# All nodes get the same policy — no adaptation.

def temporal_style_hw(graph: WorkflowGraph) -> HardenedWorkflow:
    return _make_hw(graph, lambda name, info: _ann(
        retry=2,          # 3 retries (Temporal/Camunda/Step Functions default)
        timeout_tier=1,   # 30s activity timeout
        cb=0,             # NO circuit breaker — none of these engines have one
        fallback=3,       # dead_letter (saga compensation / catch block)
        risk="TEMPORAL_STYLE",
    ))


# ── Baseline 5: LLM-Hardened (Ollama) ─────────────────────────────────────────

def _call_ollama(graph: WorkflowGraph) -> Optional[str]:
    edges = [f"  {s} -> {d}"
             for s, dsts in graph.adj.items() for d in dsts]
    nodes_desc = [
        f"  - {n}  (depth={i.depth}, external={i.is_external}, crypto={i.is_crypto})"
        for n, i in graph.nodes.items()
    ]
    prompt = f"""You are a distributed systems resilience expert.

Workflow: {graph.name}

CALL GRAPH:
{chr(10).join(edges)}

NODES:
{chr(10).join(nodes_desc)}

For each node recommend: retry (none/1/3/5), timeout (short/medium/long),
cb (closed/half_open/open), fallback (none/cached_response/dead_letter_queue).

Rules: gateway=half_open CB + light retry; orchestrator fan-out=open CB + dead_letter;
storage/external=open CB + dead_letter; auth/crypto=open CB + dead_letter;
leaf internal=closed CB + retry.

Respond ONLY with valid JSON:
{{service: {{"retry":"...","timeout":"...","cb":"...","fallback":"..."}}}}

JSON:
"""
    body = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt,
        "stream": False, "options": {"temperature": 0.0, "seed": 42},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read()).get("response", "")
    except Exception as e:
        print(f"  [LLM] Ollama error: {e}")
        return None


_CB_MAP      = {"closed": 0, "half_open": 1, "half-open": 1, "open": 2}
_TIMEOUT_MAP = {"short": 0, "5s": 0, "medium": 1, "30s": 1, "long": 2, "120s": 2}
_FB_MAP      = {"none": 0, "alternate": 1, "cached_response": 2, "cached": 2,
                "dead_letter_queue": 3, "dead_letter": 3}


def _parse_retry(v: str) -> int:
    v = str(v).strip().lower()
    if v in ("none", "0"): return 0
    if v == "1":            return 1
    if v == "3":            return 2
    return 3


def llm_hw(graph: WorkflowGraph) -> HardenedWorkflow:
    print(f"  [LLM] Querying Ollama for {graph.name} ...")
    raw = {}
    text = _call_ollama(graph)
    if text:
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                raw = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    def _node_ann(name: str, info) -> ResilienceAnnotation:
        entry = raw.get(name) or next(
            (v for k, v in raw.items() if k.lower() == name.lower()), None
        )
        if not entry or not isinstance(entry, dict):
            return _ann(1, 1, 1, 2, "LLM_DEFAULT")
        return _ann(
            retry        = _parse_retry(entry.get("retry", "1")),
            timeout_tier = _TIMEOUT_MAP.get(str(entry.get("timeout","medium")).lower(), 1),
            cb           = _CB_MAP.get(str(entry.get("cb","half_open")).lower(), 1),
            fallback     = _FB_MAP.get(str(entry.get("fallback","cached_response")).lower(), 2),
            risk         = "LLM",
        )

    return _make_hw(graph, _node_ann)


# ── Baseline 6: ORS-Hardened ───────────────────────────────────────────────────

def ors_hw(graph: WorkflowGraph,
           engine: WorkflowHardeningEngine) -> HardenedWorkflow:
    return engine.analyze(graph)


# ── MTTR helper ────────────────────────────────────────────────────────────────
# avg_steps in ScenarioMetrics is already tracked by the simulator.
# We expose it as MTTR (proxy: steps to completion = recovery speed).

def _mttr(m: ScenarioMetrics) -> str:
    return f"{m.avg_steps:.1f}" if m.completion_pct > 0 else "N/A"


# ── Printing ───────────────────────────────────────────────────────────────────

METHOD_ORDER = [
    "Original",
    "Manual-Uniform",
    "Rule-Based",
    "Temporal-style",
    "LLM-Hardened",
    "ORS-Hardened",
]

METHOD_FEATURES = {
    "Original":       ("No",  "No",  "No",  "No"),
    "Manual-Uniform": ("Yes", "Yes", "No",  "No"),
    "Rule-Based":     ("Yes", "Yes", "Partial", "No"),
    "Temporal-style": ("Yes", "No",  "Yes", "No"),
    "LLM-Hardened":   ("Yes", "Yes", "Yes", "No"),
    "ORS-Hardened":   ("Yes", "Yes", "Yes", "Yes"),
}


def _print_feature_table() -> None:
    W = 90
    print("\n" + "=" * W)
    print("  METHOD FEATURE COMPARISON")
    print("=" * W)
    print(f"  {'Method':<18}  {'Retry':>7}  {'CB':>7}  {'Fallback':>10}  {'Adaptive':>10}")
    print("  " + "-" * (W - 2))
    for m in METHOD_ORDER:
        r, cb, fb, ad = METHOD_FEATURES[m]
        marker = "  <<< learned from data" if m == "ORS-Hardened" else ""
        print(f"  {m:<18}  {r:>7}  {cb:>7}  {fb:>10}  {ad:>10}{marker}")
    print("=" * W)


def _print_scenario_table(
    graph_name: str,
    scenario:   str,
    results:    Dict[str, ScenarioMetrics],   # method → metrics
) -> None:
    W = 110
    print(f"\n  [{scenario}]  {graph_name}")
    print(f"  {'Method':<18}  {'Complete%':>10}  {'MTTR(steps)':>12}  "
          f"{'API/run':>8}  {'Wasted':>7}  {'Fallback%':>10}")
    print("  " + "-" * (W - 2))
    orig_pct = results["Original"].completion_pct
    for m in METHOD_ORDER:
        met = results.get(m)
        if not met:
            continue
        gain = met.completion_pct - orig_pct
        tag  = f" (+{gain:.1f}%)" if gain > 0 else ""
        print(f"  {m:<18}  {met.completion_pct:>9.1f}%{tag:<10}"
              f"  {_mttr(met):>11}  "
              f"{met.avg_api_calls:>8.2f}  {met.avg_wasted:>7.2f}  "
              f"{met.fallback_pct:>9.1f}%")
    print()


def _print_summary_table(
    all_results: Dict[str, Dict[str, Dict[str, ScenarioMetrics]]]
) -> None:
    """
    Final cross-workflow × cross-scenario summary.
    Rows: scenario × method.  Columns: one per graph.
    """
    W = 130
    graphs  = list(all_results.keys())
    short   = {g: g.split("/")[-1].split(" ")[0][:18] for g in graphs}

    print("\n" + "=" * W)
    print("  REVIEWER TABLE  —  Completion Rate (%)  across all workflows and failure scenarios")
    print(f"  {N_TRIALS} Monte Carlo trials per cell  |  ORS model trained on real DSB X-Trace")
    print("=" * W)

    hdr = f"  {'Method':<18}  {'Scenario':<12}"
    for g in graphs:
        hdr += f"  {short[g]:>20}"
    hdr += f"  {'Mean':>8}"
    print(hdr)
    print("  " + "-" * (W - 2))

    sc_names = [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS]

    for sc in sc_names:
        for midx, m in enumerate(METHOD_ORDER):
            row = f"  {m:<18}  {sc:<12}"
            vals = []
            for g in graphs:
                met = all_results[g].get(sc, {}).get(m)
                if met:
                    row += f"  {met.completion_pct:>19.1f}%"
                    vals.append(met.completion_pct)
                else:
                    row += f"  {'N/A':>20}"
            if vals:
                row += f"  {sum(vals)/len(vals):>7.1f}%"
            print(row)
        print()

    print("=" * W)


def _print_wasted_table(
    all_results: Dict[str, Dict[str, Dict[str, ScenarioMetrics]]]
) -> None:
    """Wasted API calls — shows cost of retry-heavy policies."""
    W = 130
    graphs = list(all_results.keys())
    short  = {g: g.split("/")[-1].split(" ")[0][:18] for g in graphs}

    print("\n" + "=" * W)
    print("  WASTED RETRY CALLS per run  (lower = more efficient)")
    print("=" * W)
    hdr = f"  {'Method':<18}  {'Scenario':<12}"
    for g in graphs:
        hdr += f"  {short[g]:>20}"
    print(hdr)
    print("  " + "-" * (W - 2))

    sc_names = [SCENARIO_NAMES[int(s)] for s in EVAL_SCENARIOS]
    for sc in sc_names:
        for m in METHOD_ORDER:
            row = f"  {m:<18}  {sc:<12}"
            for g in graphs:
                met = all_results[g].get(sc, {}).get(m)
                row += f"  {met.avg_wasted:>19.2f} " if met else f"  {'N/A':>20}"
            print(row)
        print()
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── Load ORS model ────────────────────────────────────────────────────────
    print("\n[1] Loading DSB-trained ORS-VQ model ...")
    ors_model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_dsb_vq.pt not found — run resiliency_poc.ors_dsb_poc first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors_model.load_state_dict(d["state_dict"])
    ors_model.eval()

    # ── Load graphs ───────────────────────────────────────────────────────────
    print("\n[2] Loading DSB microservice call graph topologies ...")
    graphs = load_dsb_healthy_and_failure()
    for g in graphs:
        n_e = sum(len(v) for v in g.adj.values())
        print(f"    {g.name:<55}  nodes={len(g.nodes)}  edges={n_e}")

    engine_ors = WorkflowHardeningEngine(ors_model=ors_model, seed=SEED)
    sim        = WorkflowFailureSimulator(seed=SEED)

    _print_feature_table()

    # all_results[graph_name][scenario_name][method_name] = ScenarioMetrics
    all_results: Dict[str, Dict[str, Dict[str, ScenarioMetrics]]] = {}

    for graph in graphs:
        gname = graph.name
        print(f"\n{'='*80}")
        print(f"  Graph: {gname}")
        print(f"{'='*80}")
        all_results[gname] = {SCENARIO_NAMES[int(s)]: {} for s in EVAL_SCENARIOS}

        # ── Build all hardened variants ────────────────────────────────────────
        print("\n  Building hardened workflow variants ...")
        hw_manual   = manual_uniform_hw(graph)
        hw_rule     = rule_based_hw(graph)
        hw_temporal = temporal_style_hw(graph)
        hw_llm      = llm_hw(graph)
        hw_ors      = ors_hw(graph, engine_ors)

        variants = {
            "Manual-Uniform": hw_manual,
            "Rule-Based":     hw_rule,
            "Temporal-style": hw_temporal,
            "LLM-Hardened":   hw_llm,
            "ORS-Hardened":   hw_ors,
        }

        # ── Print per-node policy comparison ──────────────────────────────────
        print(f"\n  Per-node policy comparison ({len(graph.nodes)} nodes)")
        W_node = 120
        print("  " + "-" * W_node)
        hdr_row = f"  {'Node':<28}  {'d':>2}  {'Manual':>14}  {'Rule':>14}  {'Temporal':>14}  {'LLM':>14}  {'ORS':>14}"
        print(hdr_row)
        print("  " + "-" * W_node)

        def _short(ann: ResilienceAnnotation) -> str:
            cb_s  = ["cls", "h_o", "opn"][ann.cb_policy]
            ret_s = str([0,1,3,5][ann.retry])
            fb_s  = ["no", "alt", "cch", "dlq"][ann.fallback_type]
            return f"CB={cb_s},r={ret_s},{fb_s}"

        visited, queue, order = set(), list(graph.entry_nodes), []
        while queue:
            n = queue.pop(0)
            if n in visited: continue
            visited.add(n); order.append(n)
            queue.extend(graph.adj.get(n, []))

        for nname in order:
            info   = graph.nodes[nname]
            row    = f"  {nname:<28}  {info.depth:>2}"
            for key in ["Manual-Uniform","Rule-Based","Temporal-style","LLM-Hardened","ORS-Hardened"]:
                ann = variants[key].nodes[nname].annotation
                row += f"  {_short(ann):>14}"
            print(row)
        print("  " + "-" * W_node)

        # ── Simulate all scenarios ─────────────────────────────────────────────
        print(f"\n  Running simulation ({N_TRIALS} trials × {len(EVAL_SCENARIOS)} scenarios) ...")

        for scenario in EVAL_SCENARIOS:
            sc_name = SCENARIO_NAMES[int(scenario)]

            # Original: use any hw variant (original_run ignores the hw part)
            orig_m, _ = evaluate_scenario(sim, graph, hw_manual, scenario, N_TRIALS)
            all_results[gname][sc_name]["Original"] = orig_m

            for method_name, hw in variants.items():
                _, hard_m = evaluate_scenario(sim, graph, hw, scenario, N_TRIALS)
                all_results[gname][sc_name][method_name] = hard_m

            print(f"    {sc_name:<14}  " +
                  "  ".join(f"{m[:8]}={all_results[gname][sc_name][m].completion_pct:.0f}%"
                             for m in METHOD_ORDER))

        # ── Per-scenario table for this graph ──────────────────────────────────
        for scenario in EVAL_SCENARIOS:
            sc_name = SCENARIO_NAMES[int(scenario)]
            _print_scenario_table(gname, sc_name, all_results[gname][sc_name])

    # ── Final summary tables ───────────────────────────────────────────────────
    _print_summary_table(all_results)
    _print_wasted_table(all_results)

    # ── Win/loss count ─────────────────────────────────────────────────────────
    print("\n  ORS vs other methods — completion rate comparison")
    print("  " + "-" * 70)
    for method in METHOD_ORDER[1:-1]:   # skip Original and ORS itself
        ors_wins = other_wins = ties = 0
        for gname, sc_dict in all_results.items():
            for sc_name, m_dict in sc_dict.items():
                ors_pct   = m_dict.get("ORS-Hardened")
                other_pct = m_dict.get(method)
                if not ors_pct or not other_pct:
                    continue
                if ors_pct.completion_pct > other_pct.completion_pct + 0.5:
                    ors_wins += 1
                elif other_pct.completion_pct > ors_pct.completion_pct + 0.5:
                    other_wins += 1
                else:
                    ties += 1
        total = ors_wins + other_wins + ties
        print(f"  ORS vs {method:<18}  "
              f"ORS wins={ors_wins}/{total}  "
              f"{method[:12]} wins={other_wins}/{total}  "
              f"ties={ties}/{total}")
    print("  " + "-" * 70)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")
    print()
    print("  Note on Temporal-style baseline:")
    print("  Temporal, Camunda, AWS Step Functions and Apache Airflow do not have")
    print("  built-in circuit breakers (CB=none above). Retry=3 and saga/catch")
    print("  compensation (fallback=dead_letter) reflect their documented defaults.")
    print("  All nodes receive the same policy — there is no adaptation based on")
    print("  observed telemetry or graph position.")


if __name__ == "__main__":
    main()
