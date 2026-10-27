"""
temporal_trace_generator.py

Generates training data for the oracle-free, time-varying scenario pipeline.

Each TemporalFailureEvent combines:
  - Static workflow context (38-dim from trace_generator.py)
  - Scenario observables  ( 6-dim from scenario.py)
  ─────────────────────────────────────────────────
  TEMPORAL_FEATURE_DIM = 44

Labels come from counterfactual simulation (counterfactual_labeler.py),
not from a hand-written oracle.

Generation strategy
───────────────────
For each workflow × node, we generate a scenario stream of length STREAM_LEN
and produce one TemporalFailureEvent per time-step.  The scenario transitions
over the stream, so the dataset contains examples from every scenario state
and every transition (the hardest case for the model).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

import numpy as np

from .workflow_loader  import WorkflowGraph, load_both_workflows, NODE_TYPE_VOCAB, NodeInfo
from .trace_generator  import FailureEvent, FEATURE_DIM
from .scenario         import (
    ScenarioState, SCENARIO_OBS_DIM,
    generate_scenario_stream, ScenarioObservation,
)
from .counterfactual_labeler import label_from_simulation, CounterfactualLabel, SCENARIO_TO_CATEGORY

TEMPORAL_FEATURE_DIM = FEATURE_DIM + SCENARIO_OBS_DIM   # 38 + 9 = 47

STREAM_LEN       = 40    # time-steps per node per workflow per seed
N_SEEDS_PER_NODE = 3     # independent streams per node (different initial scenario)


# ── Data record ───────────────────────────────────────────────────────────────

@dataclass
class TemporalFailureEvent:
    """
    Enriched failure event that carries both static workflow features
    and dynamic scenario observables.  Labels are derived from simulation.
    """
    workflow_name: str
    node_name:     str
    node_type:     str
    t:             int
    scenario:      ScenarioState    # ground truth — not in feature vector

    # Static workflow features (same layout as FailureEvent.to_feature_vector)
    static_features: np.ndarray     # (38,)

    # Temporal scenario observables
    scenario_obs: ScenarioObservation

    # Labels (counterfactual simulation best policy)
    label_retry:    int
    label_timeout:  int
    label_cb:       int
    label_fallback: int
    label_category: int   # failure category: 0=NOMINAL 1=TRANSIENT 2=PERSISTENT 3=TRANSACTION

    def to_feature_vector(self) -> np.ndarray:
        """44-dim vector: [static(38) | scenario_obs(6)]."""
        return np.concatenate([
            self.static_features,
            self.scenario_obs.to_feature_vector(),
        ]).astype(np.float32)

    def to_failure_event(self) -> "FailureEvent":
        """Return a minimal FailureEvent for WorkflowSimulator compatibility."""
        ev = object.__new__(FailureEvent)
        ev.workflow_name  = self.workflow_name
        ev.node_name      = self.node_name
        ev.node_type      = self.node_type
        ev.failure_mode   = _scenario_to_failure_mode(self.scenario)
        ev.retry_count    = self.label_retry
        ev.is_external    = float(self.static_features[36]) > 0.5
        ev.label_retry    = self.label_retry
        ev.label_timeout  = self.label_timeout
        ev.label_cb       = self.label_cb
        ev.label_fallback = self.label_fallback
        return ev


def _scenario_to_failure_mode(s: ScenarioState) -> str:
    return {
        ScenarioState.HEALTHY:           "no_failure",
        ScenarioState.TRANSIENT_FAILURE: "timeout",
        ScenarioState.DEGRADED:          "dependency_down",
        ScenarioState.SERVICE_DOWN:      "dependency_down",
        ScenarioState.RECOVERING:        "rate_limit",
    }[s]


# ── Static feature vector for a workflow node ─────────────────────────────────

def _node_static_features(
    graph:     WorkflowGraph,
    node_name: str,
    rng:       random.Random,
) -> np.ndarray:
    """
    Build the 38-dim static feature vector for a node.
    Mirrors FailureEvent.to_feature_vector() layout.
    """
    node: NodeInfo = graph.nodes[node_name]
    type_idx       = node.type_idx

    # Predict next node (first successor or self for terminal nodes)
    succs = list(graph.adj.get(node_name, []))
    next_type_idx = graph.nodes[succs[0]].type_idx if succs else type_idx

    # current node one-hot [0:13]
    curr_oh = np.zeros(13, dtype=np.float32)
    curr_oh[type_idx % 13] = 1.0

    # next node one-hot [13:26]
    next_oh = np.zeros(13, dtype=np.float32)
    next_oh[next_type_idx % 13] = 1.0

    # failure mode one-hot [26:33] — zeros for static vector (scenario drives failure)
    fm_oh = np.zeros(7, dtype=np.float32)

    # Normalise depth: find max depth across all nodes in this graph
    all_depths = [n.depth for n in graph.nodes.values()]
    max_depth  = max(all_depths) if all_depths else 1
    depth_norm = node.depth / max(max_depth, 1)

    retry_count = rng.randint(0, 3) / 3.0
    load_bucket = rng.randint(0, 3) / 3.0

    return np.concatenate([
        curr_oh, next_oh, fm_oh,
        [node.position, depth_norm, retry_count, float(node.is_external), load_bucket],
    ]).astype(np.float32)   # shape (38,)


# ── Main generator ────────────────────────────────────────────────────────────

class TemporalTraceGenerator:
    """
    Generates TemporalFailureEvent records with counterfactual labels.

    For every node in every loaded workflow, the generator creates
    STREAM_LEN * N_SEEDS_PER_NODE temporal events, each covering a
    different point in a scenario stream.  The result is a rich dataset
    that includes scenario transitions and their correct resiliency labels.
    """

    def __init__(
        self,
        graphs: List[WorkflowGraph] | None = None,
        seed:   int = 0,
    ):
        self.graphs = graphs or load_both_workflows()
        self.seed   = seed

    def generate_dataset(
        self,
        n_samples: int,
        verbose:   bool = False,
    ) -> List[TemporalFailureEvent]:
        """
        Generate up to n_samples temporal events.

        Events are drawn by cycling over (workflow, node, seed) triples.
        Each triple produces STREAM_LEN events.  Generation stops once
        n_samples events have been collected.
        """
        rng    = random.Random(self.seed)
        events: List[TemporalFailureEvent] = []

        for graph in self.graphs:
            if not graph.nodes:
                continue
            node_names = list(graph.nodes.keys())

            for stream_seed_offset in range(N_SEEDS_PER_NODE):
                for node_name in node_names:
                    if len(events) >= n_samples:
                        return events[:n_samples]

                    stream_seed    = self.seed * 1000 + stream_seed_offset * 100 + hash(node_name) % 100
                    static_feats   = _node_static_features(graph, node_name, rng)
                    initial_state  = ScenarioState(stream_seed_offset % len(ScenarioState))
                    stream         = generate_scenario_stream(
                        STREAM_LEN, seed=stream_seed, initial=initial_state
                    )

                    for obs in stream:
                        if len(events) >= n_samples:
                            return events[:n_samples]

                        label = label_from_simulation(
                            obs.scenario,
                            seed=stream_seed + obs.t,
                        )

                        node_type = graph.nodes[node_name].type_name
                        ev = TemporalFailureEvent(
                            workflow_name   = graph.name,
                            node_name       = node_name,
                            node_type       = node_type,
                            t               = obs.t,
                            scenario        = obs.scenario,
                            static_features = static_feats.copy(),
                            scenario_obs    = obs,
                            label_retry     = label.retry,
                            label_timeout   = label.timeout,
                            label_cb        = label.cb,
                            label_fallback  = label.fallback,
                            label_category  = label.category,
                        )
                        events.append(ev)

                    if verbose:
                        print(f"      {graph.name}/{node_name}[seed={stream_seed_offset}]: "
                              f"{len(stream)} events")

        return events[:n_samples]

    def scenario_distribution(self, events: List[TemporalFailureEvent]) -> None:
        """Print scenario distribution for diagnostics."""
        from collections import Counter
        from .scenario import SCENARIO_NAMES
        counts = Counter(e.scenario for e in events)
        total  = len(events)
        print("      Scenario distribution:")
        for s in ScenarioState:
            pct = counts[s] / total * 100 if total else 0
            print(f"        {SCENARIO_NAMES[s]:<18} {counts[s]:>5}  ({pct:.1f}%)")
