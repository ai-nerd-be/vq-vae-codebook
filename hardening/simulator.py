"""
hardening/simulator.py

Failure injection simulator for comparing original vs hardened workflows.

Simulates workflow execution under different failure scenarios by:
  1. Walking the workflow graph node by node
  2. For each node, rolling a failure die based on scenario failure probability
  3. Applying (or not applying) resilience constructs

Metrics collected per run:
  completed       — did the workflow finish successfully?
  api_calls       — total service calls made (original + retries)
  wasted_calls    — calls made that didn't contribute to progress
  fallback_used   — were any fallback responses served?
  data_loss       — did the workflow end in dead_letter (partial result)?
  steps           — total execution steps (latency proxy)

The same random seeds are used for both original and hardened runs
so the comparison is fair (same failure events, different handling).
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from resiliency_poc.workflow_loader import WorkflowGraph
from resiliency_poc.scenario        import ScenarioState, SCENARIO_NAMES
from .models import HardenedWorkflow, ResilienceAnnotation, FALLBACK_NAMES

# ── Scenario failure probabilities (per call attempt) ─────────────────────────

_SCENARIO_FAIL_PROB: Dict[ScenarioState, float] = {
    ScenarioState.HEALTHY:           0.05,
    ScenarioState.TRANSIENT_FAILURE: 0.35,
    ScenarioState.DEGRADED:          0.70,
    ScenarioState.SERVICE_DOWN:      0.97,
    ScenarioState.RECOVERING:        0.40,
}

# Probability that a fallback call itself succeeds
_FALLBACK_SUCCESS: Dict[int, float] = {
    0: 0.00,   # none — always fails
    1: 0.85,   # alternate_node
    2: 0.95,   # cached_response
    3: 1.00,   # dead_letter_queue — always "succeeds" but flags data_loss
}


# ── Per-node execution result ─────────────────────────────────────────────────

@dataclass
class NodeResult:
    node_name:     str
    succeeded:     bool
    api_calls:     int    # primary service calls
    wasted_calls:  int    # calls that failed and weren't the last retry
    fallback_used: bool
    data_loss:     bool   # ended in dead_letter


# ── Per-workflow run result ────────────────────────────────────────────────────

@dataclass
class RunResult:
    scenario:      str
    completed:     bool
    api_calls:     int
    wasted_calls:  int
    fallback_used: bool
    data_loss:     bool
    steps:         int
    node_results:  List[NodeResult] = field(default_factory=list)


# ── Simulator ─────────────────────────────────────────────────────────────────

class WorkflowFailureSimulator:
    """
    Injects failures into a workflow execution and measures the outcome.

    original_run():  simulates the bare workflow with no resilience
    hardened_run():  simulates the hardened workflow with ORS constructs

    Both methods walk the graph in topological order (BFS from entry nodes).
    A workflow "completes" if all nodes in at least one path from entry to
    terminal succeed. For simplicity we walk all nodes and track completion
    as: no node fails without recovery.
    """

    def __init__(self, seed: int = 42):
        self.base_seed = seed

    def _rng(self, scenario: ScenarioState, trial: int) -> random.Random:
        return random.Random(self.base_seed + int(scenario) * 1000 + trial)

    def _bfs_order(self, graph: WorkflowGraph) -> List[str]:
        visited, queue, order = set(), list(graph.entry_nodes), []
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            order.append(n)
            queue.extend(graph.adj.get(n, []))
        return order

    # ── Original workflow (no resilience) ────────────────────────────────────

    def original_run(
        self,
        graph:    WorkflowGraph,
        scenario: ScenarioState,
        trial:    int = 0,
    ) -> RunResult:
        rng        = self._rng(scenario, trial)
        fail_prob  = _SCENARIO_FAIL_PROB[scenario]
        node_order = self._bfs_order(graph)

        completed    = True
        api_calls    = 0
        wasted_calls = 0
        steps        = 0
        node_results = []

        for node_name in node_order:
            api_calls += 1
            steps     += 1
            failed     = rng.random() < fail_prob

            if failed:
                # No resilience — workflow fails immediately
                completed = False
                node_results.append(NodeResult(
                    node_name=node_name, succeeded=False,
                    api_calls=1, wasted_calls=1,
                    fallback_used=False, data_loss=False,
                ))
                break   # cascade failure
            else:
                node_results.append(NodeResult(
                    node_name=node_name, succeeded=True,
                    api_calls=1, wasted_calls=0,
                    fallback_used=False, data_loss=False,
                ))

        return RunResult(
            scenario=SCENARIO_NAMES[int(scenario)],
            completed=completed,
            api_calls=api_calls,
            wasted_calls=wasted_calls,
            fallback_used=False,
            data_loss=False,
            steps=steps,
            node_results=node_results,
        )

    # ── Hardened workflow (ORS constructs applied) ────────────────────────────

    def hardened_run(
        self,
        hw:       HardenedWorkflow,
        scenario: ScenarioState,
        trial:    int = 0,
    ) -> RunResult:
        rng        = self._rng(scenario, trial)
        fail_prob  = _SCENARIO_FAIL_PROB[scenario]
        node_order = self._bfs_order(hw.original)

        completed    = True
        api_calls    = 0
        wasted_calls = 0
        fallback_used = False
        data_loss    = False
        steps        = 0
        node_results = []

        for node_name in node_order:
            hn  = hw.nodes.get(node_name)
            ann = hn.annotation if hn else None

            result = self._execute_node(
                node_name, ann, fail_prob, rng
            )
            api_calls     += result.api_calls
            wasted_calls  += result.wasted_calls
            fallback_used  = fallback_used or result.fallback_used
            data_loss      = data_loss or result.data_loss
            steps         += result.api_calls   # each call = one step

            if not result.succeeded:
                completed = False
                node_results.append(result)
                break
            node_results.append(result)

        return RunResult(
            scenario=SCENARIO_NAMES[int(scenario)],
            completed=completed,
            api_calls=api_calls,
            wasted_calls=wasted_calls,
            fallback_used=fallback_used,
            data_loss=data_loss,
            steps=steps,
            node_results=node_results,
        )

    def _execute_node(
        self,
        node_name: str,
        ann:       Optional[ResilienceAnnotation],
        fail_prob: float,
        rng:       random.Random,
    ) -> NodeResult:
        """
        Simulate one node execution with the given resilience annotation.
        """
        if ann is None:
            # No annotation — treat as no resilience
            failed = rng.random() < fail_prob
            return NodeResult(node_name=node_name, succeeded=not failed,
                              api_calls=1, wasted_calls=int(failed),
                              fallback_used=False, data_loss=False)

        # ── Circuit Breaker OPEN: skip to fallback immediately ────────────────
        if ann.cb_policy == 2:
            fb_ok   = rng.random() < _FALLBACK_SUCCESS[ann.fallback_type]
            d_loss  = ann.fallback_type == 3 and fb_ok
            return NodeResult(
                node_name=node_name, succeeded=fb_ok,
                api_calls=0,         # CB prevents any primary call
                wasted_calls=0,
                fallback_used=True,
                data_loss=d_loss,
            )

        # ── Retry loop ────────────────────────────────────────────────────────
        max_attempts = ann.retry + 1
        # CB half-open: allow only 1 probe attempt
        if ann.cb_policy == 1:
            max_attempts = 1

        # Timeout bonus: longer timeout = slightly better success per attempt
        timeout_bonus = {0: -0.05, 1: 0.0, 2: 0.05}.get(ann.timeout_tier, 0.0)
        eff_fail = max(0.0, min(1.0, fail_prob - timeout_bonus))

        succeeded    = False
        api_calls    = 0
        wasted_calls = 0

        for attempt in range(max_attempts):
            api_calls += 1
            failed     = rng.random() < eff_fail
            if not failed:
                succeeded = True
                break
            elif attempt < max_attempts - 1:
                wasted_calls += 1   # this attempt was wasted, more coming

        # ── Fallback if all retries exhausted ─────────────────────────────────
        fallback_used = False
        data_loss     = False
        if not succeeded and ann.fallback_type > 0:
            fb_ok     = rng.random() < _FALLBACK_SUCCESS[ann.fallback_type]
            succeeded  = fb_ok
            fallback_used = True
            data_loss     = ann.fallback_type == 3 and fb_ok

        return NodeResult(
            node_name=node_name, succeeded=succeeded,
            api_calls=api_calls, wasted_calls=wasted_calls,
            fallback_used=fallback_used, data_loss=data_loss,
        )


# ── Batch evaluation ──────────────────────────────────────────────────────────

@dataclass
class ScenarioMetrics:
    scenario:       str
    n_trials:       int
    completion_pct: float
    avg_api_calls:  float
    avg_wasted:     float
    fallback_pct:   float
    data_loss_pct:  float
    avg_steps:      float


def evaluate_scenario(
    sim:       WorkflowFailureSimulator,
    graph:     WorkflowGraph,
    hw:        HardenedWorkflow,
    scenario:  ScenarioState,
    n_trials:  int = 500,
) -> Tuple[ScenarioMetrics, ScenarioMetrics]:
    """Run N trials of both original and hardened under the same scenario."""

    orig_results = [sim.original_run(graph, scenario, t) for t in range(n_trials)]
    hard_results = [sim.hardened_run(hw,    scenario, t) for t in range(n_trials)]

    def _metrics(results: List[RunResult], sc: str) -> ScenarioMetrics:
        n = len(results)
        return ScenarioMetrics(
            scenario       = sc,
            n_trials       = n,
            completion_pct = sum(r.completed    for r in results) / n * 100,
            avg_api_calls  = sum(r.api_calls    for r in results) / n,
            avg_wasted     = sum(r.wasted_calls for r in results) / n,
            fallback_pct   = sum(r.fallback_used for r in results) / n * 100,
            data_loss_pct  = sum(r.data_loss    for r in results) / n * 100,
            avg_steps      = sum(r.steps        for r in results) / n,
        )

    sc_name = SCENARIO_NAMES[int(scenario)]
    return _metrics(orig_results, sc_name), _metrics(hard_results, sc_name)
