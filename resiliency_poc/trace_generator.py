"""
trace_generator.py

Generates synthetic workflow execution traces from real graph topologies.
Each trace is a sequence of FailureEvent records — one per node visited —
capturing the context needed to build the feature vector and the oracle label.

Feature vector layout  (FEATURE_DIM = 38)
──────────────────────────────────────────
  [0  :13]  current_node_type   one-hot (13 types)
  [13 :26]  next_node_type      one-hot (13 types, zeros for terminal)
  [26 :33]  failure_mode        one-hot (7 modes)
  [33]      position_in_workflow  float [0, 1]
  [34]      depth_normalised      float [0, 1]
  [35]      retry_count_norm      float [0, 1]   (retry_count / 3)
  [36]      is_external_dep       float {0, 1}
  [37]      load_bucket           float {0.0, 0.33, 0.67, 1.0}
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .workflow_loader import WorkflowGraph, NUM_NODE_TYPES

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_DIM = 38

FAILURE_MODES = [
    "no_failure",
    "timeout",
    "rate_limit",
    "auth_error",
    "dependency_down",
    "bad_input",
    "crypto_error",
]
NUM_FAILURE_MODES = len(FAILURE_MODES)
FAILURE_IDX = {fm: i for i, fm in enumerate(FAILURE_MODES)}

# Base failure probabilities per node type
_NODE_FAIL_PROB = {
    "gmailTrigger":       0.10,
    "textClassifier":     0.05,
    "lmChatGoogleGemini": 0.25,   # LLM = highest failure surface
    "emailSend":          0.12,
    "gmail":              0.10,
    "if":                 0.02,
    "googleCalendar":     0.12,
    "splitInBatches":     0.03,
    "code":               0.06,
    "webhook":            0.15,
    "crypto":             0.08,
    "respondToWebhook":   0.08,
    "noOp":               0.01,
    "unknown":            0.05,
}

# Failure mode weights per node type
# Each row: [no_failure, timeout, rate_limit, auth_error, dep_down, bad_input, crypto_error]
_FAIL_MODE_WEIGHTS = {
    "lmChatGoogleGemini": [0.0, 0.30, 0.40, 0.05, 0.15, 0.10, 0.00],
    "gmail":              [0.0, 0.20, 0.05, 0.45, 0.20, 0.05, 0.05],
    "gmailTrigger":       [0.0, 0.20, 0.05, 0.45, 0.20, 0.05, 0.05],
    "googleCalendar":     [0.0, 0.25, 0.10, 0.30, 0.25, 0.05, 0.05],
    "emailSend":          [0.0, 0.25, 0.10, 0.30, 0.25, 0.05, 0.05],
    "webhook":            [0.0, 0.20, 0.05, 0.20, 0.15, 0.05, 0.35],
    "crypto":             [0.0, 0.05, 0.00, 0.10, 0.05, 0.10, 0.70],
    "respondToWebhook":   [0.0, 0.25, 0.00, 0.10, 0.30, 0.05, 0.30],
    "textClassifier":     [0.0, 0.20, 0.10, 0.05, 0.15, 0.45, 0.05],
    "code":               [0.0, 0.10, 0.00, 0.00, 0.05, 0.80, 0.05],
    "if":                 [0.0, 0.05, 0.00, 0.00, 0.00, 0.90, 0.05],
    "splitInBatches":     [0.0, 0.40, 0.00, 0.00, 0.30, 0.25, 0.05],
    "noOp":               [0.0, 0.20, 0.00, 0.00, 0.40, 0.35, 0.05],
}
_DEFAULT_FAIL_MODE_WEIGHTS = [0.0, 0.25, 0.15, 0.15, 0.20, 0.20, 0.05]


def _sample_failure(node_type: str, rng: random.Random) -> str:
    """Sample a failure mode for a given node type."""
    fail_prob = _NODE_FAIL_PROB.get(node_type, 0.05)
    if rng.random() >= fail_prob:
        return "no_failure"
    weights = _FAIL_MODE_WEIGHTS.get(node_type, _DEFAULT_FAIL_MODE_WEIGHTS)
    # weights[0] is for no_failure; skip it and renormalise
    conditional = weights[1:]
    total = sum(conditional)
    if total == 0:
        return "timeout"
    r = rng.random() * total
    cumulative = 0.0
    for i, w in enumerate(conditional):
        cumulative += w
        if r <= cumulative:
            return FAILURE_MODES[i + 1]
    return FAILURE_MODES[1]


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class FailureEvent:
    """Single failure event within a workflow execution."""
    workflow_name:   str
    node_name:       str
    node_type:       str
    node_type_idx:   int
    next_node_type:  str       # "" if terminal
    next_node_idx:   int       # NUM_NODE_TYPES-1 if terminal (uses "unknown" slot)
    failure_mode:    str
    failure_idx:     int
    retry_count:     int       # prior retries at this node
    position:        float     # normalised position [0,1]
    depth_norm:      float     # normalised depth [0,1]
    is_external:     bool
    load_bucket:     float     # simulated load {0.0, 0.33, 0.67, 1.0}

    # Label (set by label_oracle after construction)
    label_retry:    int = 0
    label_timeout:  int = 0
    label_cb:       int = 0
    label_fallback: int = 0

    def to_feature_vector(self) -> np.ndarray:
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        vec[self.node_type_idx] = 1.0                          # [0:13]
        vec[NUM_NODE_TYPES + self.next_node_idx] = 1.0         # [13:26]
        vec[2 * NUM_NODE_TYPES + self.failure_idx] = 1.0       # [26:33]
        vec[33] = self.position
        vec[34] = self.depth_norm
        vec[35] = min(self.retry_count / 3.0, 1.0)
        vec[36] = float(self.is_external)
        vec[37] = self.load_bucket
        return vec

    def label_tuple(self) -> Tuple[int, int, int, int]:
        return (self.label_retry, self.label_timeout, self.label_cb, self.label_fallback)


@dataclass
class Trace:
    """One complete (or partial) workflow execution."""
    workflow_name: str
    path:          List[str]          # node sequence
    events:        List[FailureEvent] # failure events along the path


# ── Generator ─────────────────────────────────────────────────────────────────

class TraceGenerator:
    def __init__(self, graphs: List[WorkflowGraph], seed: int = 42):
        self.graphs = graphs
        self.rng = random.Random(seed)
        self._validate()

    def _validate(self) -> None:
        if not self.graphs:
            raise ValueError("At least one WorkflowGraph is required.")

    def _sample_path(self, graph: WorkflowGraph) -> List[str]:
        """Sample a random path from entry to terminal through the graph."""
        if graph.all_paths:
            return self.rng.choice(graph.all_paths)
        # Fallback: simple random walk
        path = list(graph.entry_nodes[:1])
        visited = set(path)
        current = path[0]
        for _ in range(30):
            successors = [
                s for s in graph.adj.get(current, [])
                if s not in visited
            ]
            if not successors:
                break
            current = self.rng.choice(successors)
            path.append(current)
            visited.add(current)
        return path

    def _load_bucket(self) -> float:
        return self.rng.choice([0.0, 0.33, 0.67, 1.0])

    def generate_trace(self, graph: WorkflowGraph) -> Trace:
        path = self._sample_path(graph)
        events: List[FailureEvent] = []
        retry_counters: dict = {}  # node_name → retry count so far

        for i, node_name in enumerate(path):
            info = graph.nodes.get(node_name)
            if info is None:
                continue

            retry_count = retry_counters.get(node_name, 0)
            failure_mode = _sample_failure(info.type_name, self.rng)

            # Next node in path
            if i + 1 < len(path):
                next_name = path[i + 1]
                next_info = graph.nodes.get(next_name)
                next_type = next_info.type_name if next_info else "unknown"
                next_idx = next_info.type_idx if next_info else NUM_NODE_TYPES - 1
            else:
                next_type = ""
                next_idx = NUM_NODE_TYPES - 1  # terminal sentinel

            event = FailureEvent(
                workflow_name  = graph.name,
                node_name      = node_name,
                node_type      = info.type_name,
                node_type_idx  = info.type_idx,
                next_node_type = next_type,
                next_node_idx  = next_idx,
                failure_mode   = failure_mode,
                failure_idx    = FAILURE_IDX[failure_mode],
                retry_count    = retry_count,
                position       = info.position,
                depth_norm     = graph.depth_norm(node_name),
                is_external    = info.is_external,
                load_bucket    = self._load_bucket(),
            )
            events.append(event)

            # Simulate retry counter update
            if failure_mode != "no_failure":
                retry_counters[node_name] = retry_count + 1

        return Trace(workflow_name=graph.name, path=path, events=events)

    def generate_dataset(
        self,
        n_traces: int,
        workflow_weights: Optional[List[float]] = None,
    ) -> List[FailureEvent]:
        """
        Generate n_traces traces across all loaded graphs.
        Returns a flat list of FailureEvents (only failure events, not no_failure).
        Actually returns ALL events so the model sees the full context.
        """
        if workflow_weights is None:
            workflow_weights = [1.0] * len(self.graphs)

        all_events: List[FailureEvent] = []
        total_w = sum(workflow_weights)
        counts = [
            max(1, round(n_traces * w / total_w))
            for w in workflow_weights
        ]

        for graph, count in zip(self.graphs, counts):
            for _ in range(count):
                trace = self.generate_trace(graph)
                all_events.extend(trace.events)

        return all_events


if __name__ == "__main__":
    from .workflow_loader import load_both_workflows
    from .label_oracle import get_policy

    graphs = load_both_workflows()
    gen = TraceGenerator(graphs, seed=0)
    events = gen.generate_dataset(n_traces=200)

    failure_events = [e for e in events if e.failure_mode != "no_failure"]
    print(f"Total events  : {len(events)}")
    print(f"Failure events: {len(failure_events)}")
    print(f"Feature shape : {events[0].to_feature_vector().shape}")

    from collections import Counter
    mode_counts = Counter(e.failure_mode for e in events)
    print("\nFailure mode distribution:")
    for fm, cnt in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {fm:<20} {cnt}")
