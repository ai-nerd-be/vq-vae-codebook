"""
hardening/models.py

Data classes for the workflow hardening pipeline.

A HardenedWorkflow wraps the original WorkflowGraph and annotates
each node with the resilience constructs the ORS model recommends.
The constructs are represented as ResilienceWrap objects that describe
what to insert AROUND the node in the execution graph.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from resiliency_poc.workflow_loader import WorkflowGraph, NodeInfo
from resiliency_poc.label_oracle import (
    RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS,
)


# ── Resilience construct names ────────────────────────────────────────────────

TIMEOUT_SECONDS = {0: 5, 1: 30, 2: 120}   # tier → seconds

FALLBACK_NAMES = {
    0: "none",
    1: "alternate_node",
    2: "cached_response",
    3: "dead_letter_queue",
}


# ── Per-node annotation produced by the hardening engine ─────────────────────

@dataclass
class ResilienceAnnotation:
    """
    What the ORS model recommends for one workflow node.

    ors_code       — which Operational Resilience State fired
    retry          — max retry attempts (0 = no retry)
    timeout_tier   — 0=short(5s)  1=medium(30s)  2=long(120s)
    cb_policy      — 0=closed  1=half_open  2=open
    fallback_type  — 0=none  1=alternate  2=cached  3=dead_letter
    risk_scenario  — scenario used to build synthetic telemetry
    """
    ors_code:       int
    retry:          int
    timeout_tier:   int
    cb_policy:      int
    fallback_type:  int
    risk_scenario:  str   # e.g. "TRANSIENT"

    @property
    def timeout_seconds(self) -> int:
        return TIMEOUT_SECONDS[self.timeout_tier]

    @property
    def is_trivial(self) -> bool:
        """True when the policy adds no new constructs (healthy node)."""
        return self.retry <= 1 and self.cb_policy == 0 and self.fallback_type == 0

    def describe(self) -> str:
        parts = []
        if self.cb_policy == 2:
            parts.append("CircuitBreaker(open)")
        elif self.cb_policy == 1:
            parts.append("CircuitBreaker(half_open)")
        if self.retry > 0:
            parts.append(f"Retry(max={self.retry}, timeout={self.timeout_seconds}s)")
        if self.fallback_type > 0:
            parts.append(f"Fallback({FALLBACK_NAMES[self.fallback_type]})")
        return " + ".join(parts) if parts else "PassThrough"

    def added_nodes(self) -> int:
        """Rough count of logical nodes inserted around this node."""
        n = 0
        if self.retry > 0:       n += 1  # retry wrapper
        if self.cb_policy > 0:   n += 1  # CB node
        if self.fallback_type > 0: n += 1  # fallback branch
        return n


# ── Hardened workflow ─────────────────────────────────────────────────────────

@dataclass
class HardenedNode:
    node:        NodeInfo
    annotation:  ResilienceAnnotation

    def summary_line(self) -> str:
        return (
            f"  [{self.node.name:<30}]  ORS-{self.annotation.ors_code:<2}"
            f"  risk={self.annotation.risk_scenario:<10}"
            f"  => {self.annotation.describe()}"
        )


@dataclass
class HardenedWorkflow:
    original:     WorkflowGraph
    nodes:        Dict[str, HardenedNode]   # node_name -> HardenedNode

    @property
    def total_added_constructs(self) -> int:
        return sum(hn.annotation.added_nodes() for hn in self.nodes.values())

    @property
    def hardened_nodes(self) -> List[HardenedNode]:
        """Nodes that received at least one resilience construct."""
        return [hn for hn in self.nodes.values() if not hn.annotation.is_trivial]

    def print_diff(self) -> None:
        """Print a before/after view of the workflow."""
        W = 100
        print("=" * W)
        print(f"  HARDENED WORKFLOW: {self.original.name}")
        print(f"  Original nodes: {len(self.original.nodes)}"
              f"  |  Added constructs: {self.total_added_constructs}"
              f"  |  Hardened nodes: {len(self.hardened_nodes)}/{len(self.nodes)}")
        print("=" * W)

        # Walk the workflow in topological order (BFS from entry)
        visited = set()
        queue   = list(self.original.entry_nodes)
        order   = []
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            order.append(n)
            queue.extend(self.original.adj.get(n, []))

        for node_name in order:
            hn = self.nodes.get(node_name)
            if not hn:
                continue
            node = hn.node
            ann  = hn.annotation

            # Original
            ext_tag = " [external]" if node.is_external else ""
            print(f"\n  Node: {node_name}{ext_tag}  (type={node.type_name}, depth={node.depth})")
            print(f"    Before:  {node.type_name}")

            # After
            parts = []
            if ann.cb_policy == 2:
                parts.append(f"CircuitBreaker(open, fallback={FALLBACK_NAMES[ann.fallback_type]})")
            elif ann.cb_policy == 1:
                parts.append(f"CircuitBreaker(half_open)")
            if ann.retry > 0:
                parts.append(f"Retry(x{ann.retry}, timeout={ann.timeout_seconds}s)")
            parts.append(node.type_name)
            if ann.fallback_type > 0 and ann.cb_policy < 2:
                parts.append(f"Fallback({FALLBACK_NAMES[ann.fallback_type]})")

            chain = " -> ".join(parts)
            ors_tag = f"ORS-{ann.ors_code} ({ann.risk_scenario})"
            print(f"    After:   {chain}")
            print(f"    ORS:     {ors_tag}")

        print("\n" + "=" * W)
