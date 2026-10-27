"""
hardening_llm_baseline.py  --  LLM-as-annotator resilience baseline

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m hardening_llm_baseline

Three-way comparison on real DeathStarBench microservice call graphs:

  1. Original      — no resilience constructs at all
  2. LLM-Hardened  — local Ollama (llama3.1:8B) asked to annotate each node
  3. ORS-Hardened  — DSB-trained ORS-VQ model (learned from X-Trace data)

LLM strategy:
  - One prompt per workflow that describes the full call topology
  - Asks for retry / circuit-breaker / fallback policy per node in JSON
  - Parsed into the same ResilienceAnnotation format used by the ORS engine
  - Same failure-injection simulator used for all three

This is the paper's key comparison:
  "A domain-expert LLM can annotate policies — but does it match
   what a model that actually learned from failure traces recommends?"
"""

from __future__ import annotations
import json, re, time, urllib.request, urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.scenario                 import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow, FALLBACK_NAMES
from resiliency_poc.workflow_loader import WorkflowGraph

# ── Patch DSB service names into engine's type->scenario map ──────────────────
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

# Policy label mappings (mirrors label_oracle.py)
RETRY_MAP    = {"none": 0, "0": 0, "1": 1, "3": 2, "5": 3}
TIMEOUT_MAP  = {"short": 0, "5s": 0, "medium": 1, "30s": 1, "long": 2, "120s": 2}
CB_MAP       = {"closed": 0, "half_open": 1, "half-open": 1, "open": 2}
FALLBACK_MAP = {"none": 0, "alternate": 1, "cached": 2, "cached_response": 2,
                "dead_letter": 3, "dead_letter_queue": 3}


# ── Ollama client ──────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, model: str = OLLAMA_MODEL,
                     timeout: int = 120) -> Optional[str]:
    """Call local Ollama and return the generated text, or None on failure."""
    body = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "seed": 42},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except urllib.error.URLError as e:
        print(f"  [Ollama] Connection error: {e}")
        return None
    except Exception as e:
        print(f"  [Ollama] Unexpected error: {e}")
        return None


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(graph: WorkflowGraph) -> str:
    """
    Describe the DSB microservice call graph and ask the LLM to annotate
    resilience policies for each service node in structured JSON.
    """
    # Build adjacency description
    edges = []
    for src, dsts in graph.adj.items():
        for dst in dsts:
            edges.append(f"  {src} -> {dst}")
    edge_str = "\n".join(edges) if edges else "  (no edges)"

    nodes_desc = []
    for name, info in graph.nodes.items():
        role = "external storage/API" if info.is_external else "internal service"
        nodes_desc.append(
            f"  - {name}  (depth={info.depth}, role={role}, "
            f"is_crypto={info.is_crypto})"
        )
    nodes_str = "\n".join(nodes_desc)

    return f"""You are a distributed systems resilience expert.

Below is a microservice workflow from the DeathStarBench socialNetwork benchmark.
Your task: for each service node, recommend the best resilience policy based on
its role, position (depth), and whether it calls external systems or does crypto.

WORKFLOW: {graph.name}

CALL GRAPH (caller -> callee):
{edge_str}

SERVICE NODES:
{nodes_str}

RESILIENCE POLICY OPTIONS:
- retry:    none (0 retries) | 1 | 3 | 5
- timeout:  short (5s) | medium (30s) | long (120s)
- cb:       closed (no circuit breaker) | half_open | open (trip immediately)
- fallback: none | cached_response | dead_letter_queue

RULES OF THUMB:
- Entry nodes (depth=0, nginx gateway): light retry, half_open CB — protect the entry point
- Orchestrator services (depth=1): if they fan out to many children, open CB + dead_letter
- Storage/timeline services (external, depth>=2): open CB + dead_letter (can't retry storage failures)
- Auth/crypto services: open CB + dead_letter (crypto errors are not transient)
- Internal leaf services (no external calls, depth>=2): minimal retry, closed CB

Respond with ONLY valid JSON — no explanation, no markdown, no code block.
The JSON must be an object where each key is a service name and the value is:
  {{"retry": "<none|1|3|5>", "timeout": "<short|medium|long>",
   "cb": "<closed|half_open|open>", "fallback": "<none|cached_response|dead_letter_queue>"}}

JSON:
"""


# ── Response parser ────────────────────────────────────────────────────────────

def _parse_llm_response(text: str, graph: WorkflowGraph) -> Dict[str, dict]:
    """
    Extract per-node policy dict from LLM JSON response.
    Falls back to a conservative default for any node not in the response.
    """
    # Try to extract the first {...} block
    text = text.strip()
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        print("  [LLM] No JSON block found in response — using defaults for all nodes.")
        return {}
    try:
        raw = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"  [LLM] JSON parse error: {e} — using defaults for all nodes.")
        return {}

    policies = {}
    for node_name in graph.nodes:
        # Try exact match, then case-insensitive
        entry = raw.get(node_name) or next(
            (v for k, v in raw.items() if k.lower() == node_name.lower()), None
        )
        if entry and isinstance(entry, dict):
            policies[node_name] = entry
        else:
            print(f"  [LLM] No policy for '{node_name}' — using default.")
    return policies


def _policy_to_annotation(node_name: str, policy: Optional[dict],
                           graph: WorkflowGraph) -> ResilienceAnnotation:
    """
    Convert an LLM policy dict to ResilienceAnnotation.
    If policy is None or malformed, apply a safe conservative default.
    """
    node = graph.nodes[node_name]

    if not policy:
        # Conservative default: half_open CB + 1 retry + cached fallback
        return ResilienceAnnotation(
            ors_code=-1, retry=1, timeout_tier=1,
            cb_policy=1, fallback_type=2,
            risk_scenario="LLM_DEFAULT",
        )

    def _pick(val: str, mapping: dict, default: int) -> int:
        v = str(val).strip().lower()
        return mapping.get(v, default)

    retry_str   = str(policy.get("retry",   "1")).lower()
    timeout_str = str(policy.get("timeout", "medium")).lower()
    cb_str      = str(policy.get("cb",      "half_open")).lower()
    fb_str      = str(policy.get("fallback","cached_response")).lower()

    # Map retry string to class index
    if retry_str in ("none", "0"):
        retry_cls = 0
    elif retry_str == "1":
        retry_cls = 1
    elif retry_str == "3":
        retry_cls = 2
    elif retry_str in ("5", "5+"):
        retry_cls = 3
    else:
        retry_cls = 1  # default: 1 retry

    timeout_cls = TIMEOUT_MAP.get(timeout_str, 1)
    cb_cls      = CB_MAP.get(cb_str, 1)
    fb_cls      = FALLBACK_MAP.get(fb_str, 2)

    return ResilienceAnnotation(
        ors_code=-1,   # -1 = LLM, not an ORS code
        retry=retry_cls,
        timeout_tier=timeout_cls,
        cb_policy=cb_cls,
        fallback_type=fb_cls,
        risk_scenario="LLM",
    )


# ── LLM hardening ──────────────────────────────────────────────────────────────

def llm_harden(graph: WorkflowGraph, verbose: bool = True) -> HardenedWorkflow:
    """
    Ask Ollama to annotate each node with a resilience policy,
    then wrap the graph in a HardenedWorkflow.
    """
    if verbose:
        print(f"\n  Querying Ollama ({OLLAMA_MODEL}) for resilience annotations ...")

    prompt   = _build_prompt(graph)
    response = _ollama_generate(prompt)

    if response is None:
        print("  [LLM] Ollama unavailable — applying conservative defaults to all nodes.")
        policies = {}
    else:
        if verbose:
            # Show first 300 chars of raw response for debugging
            preview = response[:300].replace("\n", " ")
            print(f"  [LLM] Raw response preview: {preview}...")
        policies = _parse_llm_response(response, graph)

    hardened_nodes = {}
    for node_name, node_info in graph.nodes.items():
        policy     = policies.get(node_name)
        annotation = _policy_to_annotation(node_name, policy, graph)
        hardened_nodes[node_name] = HardenedNode(node=node_info, annotation=annotation)

    return HardenedWorkflow(original=graph, nodes=hardened_nodes)


# ── Printing helpers ───────────────────────────────────────────────────────────

def _print_llm_annotations(hw: HardenedWorkflow) -> None:
    W = 105
    print("\n" + "=" * W)
    print(f"  LLM ANNOTATIONS: {hw.original.name}")
    print(f"  Model: {OLLAMA_MODEL}  |  Strategy: describe topology, ask for JSON policy")
    print("=" * W)
    print(f"  {'Node':<32}  {'depth':>5}  {'LLM policy'}")
    print("  " + "-" * (W - 2))

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
        ann = hn.annotation
        print(f"  {node_name:<32}  {hn.node.depth:>5}  {ann.describe()}")
    print("=" * W)


def _print_three_way_comparison(
    workflow_name: str,
    orig_rows:  List[ScenarioMetrics],
    llm_rows:   List[ScenarioMetrics],
    ors_rows:   List[ScenarioMetrics],
) -> None:
    W = 130
    print("\n" + "=" * W)
    print(f"  THREE-WAY COMPARISON: {workflow_name}")
    print(f"  Original  vs  LLM-Hardened ({OLLAMA_MODEL})  vs  ORS-Hardened (DSB-trained)")
    print("=" * W)
    hdr = (f"  {'Scenario':<12}  {'Model':<14}  {'Complete%':>9}  "
           f"{'API calls':>9}  {'Wasted':>7}  {'Fallback%':>9}  {'DataLoss%':>9}")
    print(hdr)
    print("  " + "-" * (W - 2))

    scenarios = [SCENARIO_NAMES[int(s)] for s in [
        ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN,
    ]]
    orig_by_sc = {m.scenario: m for m in orig_rows}
    llm_by_sc  = {m.scenario: m for m in llm_rows}
    ors_by_sc  = {m.scenario: m for m in ors_rows}

    for sc in scenarios:
        o = orig_by_sc.get(sc)
        l = llm_by_sc.get(sc)
        r = ors_by_sc.get(sc)
        if o:
            print(f"  {sc:<12}  {'Original':<14}  {o.completion_pct:>8.1f}%  "
                  f"{o.avg_api_calls:>9.2f}  {o.avg_wasted:>7.2f}  "
                  f"{o.fallback_pct:>8.1f}%  {o.data_loss_pct:>8.1f}%")
        if l and o:
            delta = l.completion_pct - o.completion_pct
            tag   = f"(+{delta:.1f}%)" if delta > 0 else f"({delta:.1f}%)"
            print(f"  {'':<12}  {'LLM-Hardened':<14}  {l.completion_pct:>8.1f}% {tag:<10}"
                  f"{l.avg_api_calls:>9.2f}  {l.avg_wasted:>7.2f}  "
                  f"{l.fallback_pct:>8.1f}%  {l.data_loss_pct:>8.1f}%")
        if r and o:
            delta = r.completion_pct - o.completion_pct
            tag   = f"(+{delta:.1f}%)" if delta > 0 else f"({delta:.1f}%)"
            print(f"  {'':<12}  {'ORS-Hardened':<14}  {r.completion_pct:>8.1f}% {tag:<10}"
                  f"{r.avg_api_calls:>9.2f}  {r.avg_wasted:>7.2f}  "
                  f"{r.fallback_pct:>8.1f}%  {r.data_loss_pct:>8.1f}%")
        print()
    print("=" * W)


def _print_policy_diff(hw_llm: HardenedWorkflow, hw_ors: HardenedWorkflow) -> None:
    """Node-by-node: where do LLM and ORS disagree?"""
    W = 120
    print("\n" + "=" * W)
    print(f"  POLICY DISAGREEMENT: LLM vs ORS  ({hw_llm.original.name})")
    print("=" * W)
    print(f"  {'Node':<32}  {'LLM policy':<42}  {'ORS policy':<42}  Match?")
    print("  " + "-" * (W - 2))

    same = differ = 0
    for node_name in hw_llm.nodes:
        hn_l = hw_llm.nodes.get(node_name)
        hn_o = hw_ors.nodes.get(node_name)
        if not hn_l or not hn_o:
            continue
        ann_l, ann_o = hn_l.annotation, hn_o.annotation
        match = (ann_l.retry    == ann_o.retry and
                 ann_l.cb_policy == ann_o.cb_policy and
                 ann_l.fallback_type == ann_o.fallback_type)
        flag  = "[SAME]" if match else "[DIFF]"
        print(f"  {node_name:<32}  {ann_l.describe():<42}  {ann_o.describe():<42}  {flag}")
        if match: same += 1
        else:     differ += 1

    total = same + differ
    print("  " + "-" * (W - 2))
    print(f"  Agreement: {same}/{total} nodes ({same/total:.0%})  |  "
          f"Disagreement: {differ}/{total} nodes ({differ/total:.0%})")
    print("=" * W)


def _print_summary_table(results: dict) -> None:
    """Final cross-workflow summary of all three models."""
    W = 130
    print("\n" + "=" * W)
    print("  FINAL SUMMARY  — Original vs LLM vs ORS across all DSB workflows")
    print(f"  LLM: {OLLAMA_MODEL}  |  ORS: DSB-trained ORS-VQ (learned from X-Trace)")
    print("=" * W)
    print(f"  {'Workflow':<42}  {'Scenario':<12}  "
          f"{'Orig':>6}  {'LLM':>6}  {'ORS':>6}  "
          f"{'LLM gain':>9}  {'ORS gain':>9}  {'ORS > LLM?':>10}")
    print("  " + "-" * (W - 2))

    scenarios = [SCENARIO_NAMES[int(s)] for s in [
        ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN,
    ]]

    ors_wins = llm_wins = ties = 0
    for gname, (orig_rows, llm_rows, ors_rows) in results.items():
        orig_by_sc = {m.scenario: m for m in orig_rows}
        llm_by_sc  = {m.scenario: m for m in llm_rows}
        ors_by_sc  = {m.scenario: m for m in ors_rows}
        for sc in scenarios:
            o = orig_by_sc.get(sc)
            l = llm_by_sc.get(sc)
            r = ors_by_sc.get(sc)
            if not (o and l and r):
                continue
            llm_gain = l.completion_pct - o.completion_pct
            ors_gain = r.completion_pct - o.completion_pct
            if r.completion_pct > l.completion_pct + 0.5:
                verdict = "ORS wins"
                ors_wins += 1
            elif l.completion_pct > r.completion_pct + 0.5:
                verdict = "LLM wins"
                llm_wins += 1
            else:
                verdict = "tie"
                ties += 1
            print(f"  {gname[:42]:<42}  {sc:<12}  "
                  f"{o.completion_pct:>5.1f}%  {l.completion_pct:>5.1f}%  {r.completion_pct:>5.1f}%  "
                  f"{llm_gain:>+8.1f}%  {ors_gain:>+8.1f}%  {verdict:>10}")
        print()

    total = ors_wins + llm_wins + ties
    print("  " + "-" * (W - 2))
    print(f"  ORS wins: {ors_wins}/{total}  |  LLM wins: {llm_wins}/{total}  |  "
          f"Ties: {ties}/{total}")
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
    print(f"    Loaded: {CKPT_ORS}")

    # ── Load DSB graphs ───────────────────────────────────────────────────────
    print("\n[2] Loading DSB microservice call graph topologies ...")
    graphs = load_dsb_healthy_and_failure()
    for g in graphs:
        n_edges = sum(len(v) for v in g.adj.values())
        print(f"    {g.name:<50}  nodes={len(g.nodes)}  edges={n_edges}")

    engine_ors = WorkflowHardeningEngine(ors_model=ors_model, seed=SEED)
    sim        = WorkflowFailureSimulator(seed=SEED)

    eval_scenarios = [
        ScenarioState.HEALTHY,
        ScenarioState.TRANSIENT_FAILURE,
        ScenarioState.DEGRADED,
        ScenarioState.SERVICE_DOWN,
    ]

    all_results = {}

    for graph in graphs:
        print(f"\n{'='*80}")
        print(f"  Workflow: {graph.name}")
        print(f"{'='*80}")

        # ── LLM hardening ──────────────────────────────────────────────────────
        print("\n[3] LLM hardening (Ollama) ...")
        hw_llm = llm_harden(graph, verbose=True)
        _print_llm_annotations(hw_llm)

        # ── ORS hardening ──────────────────────────────────────────────────────
        print("\n[4] ORS hardening (DSB-trained model) ...")
        hw_ors = engine_ors.analyze(graph)

        # ── Policy disagreement ────────────────────────────────────────────────
        _print_policy_diff(hw_llm, hw_ors)

        # ── Simulate all three ─────────────────────────────────────────────────
        print(f"\n[5] Failure injection simulation ({N_TRIALS} trials per scenario) ...")
        orig_rows, llm_rows, ors_rows = [], [], []

        for scenario in eval_scenarios:
            sc_name = SCENARIO_NAMES[int(scenario)]

            orig_m, llm_m = evaluate_scenario(sim, graph, hw_llm, scenario, N_TRIALS)
            _,      ors_m = evaluate_scenario(sim, graph, hw_ors, scenario, N_TRIALS)

            orig_rows.append(orig_m)
            llm_rows.append(llm_m)
            ors_rows.append(ors_m)

            print(f"    {sc_name:<14}  "
                  f"original={orig_m.completion_pct:.1f}%  "
                  f"LLM={llm_m.completion_pct:.1f}%  "
                  f"ORS={ors_m.completion_pct:.1f}%")

        _print_three_way_comparison(graph.name, orig_rows, llm_rows, ors_rows)
        all_results[graph.name] = (orig_rows, llm_rows, ors_rows)

    # ── Final summary ──────────────────────────────────────────────────────────
    _print_summary_table(all_results)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
