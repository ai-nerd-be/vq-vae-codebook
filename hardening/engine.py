"""
hardening/engine.py

WorkflowHardeningEngine — analyzes an LLM-generated workflow and
inserts the right resilience constructs at each node.

Pipeline:
  WorkflowGraph
      |
      v
  Node Risk Profiler   (maps node type -> expected failure scenario)
      |
      v
  Synthetic Telemetry  (builds 47-dim feature vector from risk profile)
      |
      v
  ORS-VQ Model         (predicts Operational Resilience State + policy)
      |
      v
  Transformation Plan  (retry / CB / fallback per node)
      |
      v
  HardenedWorkflow

This runs at DESIGN TIME — before any real execution.
The engine uses the workflow graph structure and node types to estimate
which failure scenarios each node is likely to encounter, then proactively
inserts the constructs the ORS model has learned are optimal for that state.
"""

from __future__ import annotations
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from resiliency_poc.workflow_loader   import WorkflowGraph, NodeInfo, EXTERNAL_NODE_TYPES
from resiliency_poc.scenario          import (
    ScenarioState, SCENARIO_NAMES, ScenarioObservation, CBStateMachine,
    CB_CLOSED, CB_HALF_OPEN, CB_OPEN,
)
from resiliency_poc.temporal_trace_generator import _node_static_features
from resiliency_poc.model.ors_model   import ORSVQModel
from resiliency_poc.label_oracle      import RETRY_LABELS, TIMEOUT_LABELS

from .models import ResilienceAnnotation, HardenedNode, HardenedWorkflow


# ── Node risk profiles ────────────────────────────────────────────────────────
#
# At design time we don't have live telemetry, so we assign each node a
# "risk scenario" based on its type and position in the workflow.
# This represents the designer's prior knowledge about which nodes are risky.
#
# The ORS model then maps the synthetic telemetry for that scenario to the
# optimal resilience policy it learned from real/simulated execution traces.

_TYPE_TO_SCENARIO: Dict[str, ScenarioState] = {
    # Entry/trigger nodes — usually healthy, just receive events
    "gmailTrigger":       ScenarioState.HEALTHY,
    "webhook":            ScenarioState.HEALTHY,

    # External API calls — can have transient failures
    "gmail":              ScenarioState.TRANSIENT_FAILURE,
    "emailSend":          ScenarioState.TRANSIENT_FAILURE,
    "lmChatGoogleGemini": ScenarioState.TRANSIENT_FAILURE,
    "googleCalendar":     ScenarioState.TRANSIENT_FAILURE,
    "respondToWebhook":   ScenarioState.TRANSIENT_FAILURE,

    # Computation nodes — internal, rarely fail
    "code":               ScenarioState.HEALTHY,
    "transform":          ScenarioState.HEALTHY,
    "condition":          ScenarioState.HEALTHY,
    "if":                 ScenarioState.HEALTHY,
    "noOp":               ScenarioState.HEALTHY,

    # Crypto / auth — can degrade under load
    "crypto":             ScenarioState.DEGRADED,

    # Batch / queue — can stall
    "splitInBatches":     ScenarioState.TRANSIENT_FAILURE,
}

# Override: external nodes deep in the graph are under more pressure
_DEEP_EXTERNAL_SCENARIO = ScenarioState.DEGRADED
_DEEP_THRESHOLD          = 3   # depth >= this and external -> DEGRADED

# Synthetic telemetry parameters per scenario
_SCENARIO_TELEMETRY: Dict[ScenarioState, Dict] = {
    ScenarioState.HEALTHY: dict(
        fail_rate=0.05, consec=0.01, retry_succ=0.93,
        latency=0.15, time_since_ok=0.02, duration=0.05,
        cb_state=CB_CLOSED, cb_dur=0.0, cb_fails=0.10,
    ),
    ScenarioState.TRANSIENT_FAILURE: dict(
        fail_rate=0.30, consec=0.08, retry_succ=0.65,
        latency=0.55, time_since_ok=0.10, duration=0.15,
        cb_state=CB_CLOSED, cb_dur=0.0, cb_fails=0.20,
    ),
    ScenarioState.DEGRADED: dict(
        fail_rate=0.65, consec=0.22, retry_succ=0.25,
        latency=0.80, time_since_ok=0.30, duration=0.40,
        cb_state=CB_HALF_OPEN, cb_dur=0.20, cb_fails=0.55,
    ),
    ScenarioState.SERVICE_DOWN: dict(
        fail_rate=0.97, consec=0.45, retry_succ=0.02,
        latency=0.98, time_since_ok=0.80, duration=0.90,
        cb_state=CB_OPEN, cb_dur=0.60, cb_fails=0.95,
    ),
    ScenarioState.RECOVERING: dict(
        fail_rate=0.40, consec=0.12, retry_succ=0.50,
        latency=0.45, time_since_ok=0.20, duration=0.25,
        cb_state=CB_HALF_OPEN, cb_dur=0.25, cb_fails=0.35,
    ),
}


def _node_risk_scenario(node: NodeInfo) -> ScenarioState:
    """Assign a failure scenario to a node based on type and graph position."""
    base = _TYPE_TO_SCENARIO.get(node.type_name, ScenarioState.TRANSIENT_FAILURE)
    # Deep external nodes face more pressure
    if node.is_external and node.depth >= _DEEP_THRESHOLD and base != ScenarioState.HEALTHY:
        return _DEEP_EXTERNAL_SCENARIO
    return base


def _synthetic_telemetry(scenario: ScenarioState) -> np.ndarray:
    """
    Build a 9-dim temporal feature vector for the given scenario.
    Layout mirrors ScenarioObservation.to_feature_vector().
    """
    p = _SCENARIO_TELEMETRY[scenario]
    return np.array([
        p["fail_rate"],
        min(p["consec"], 1.0),
        p["retry_succ"],
        p["latency"],
        min(p["time_since_ok"], 1.0),
        min(p["duration"], 1.0),
        p["cb_state"] / 2.0,
        min(p["cb_dur"], 1.0),
        min(p["cb_fails"], 1.0),
    ], dtype=np.float32)


def _build_feature_vector(
    graph:    WorkflowGraph,
    node_name: str,
    scenario: ScenarioState,
    rng:      random.Random,
) -> np.ndarray:
    """
    Build the full 47-dim feature vector for a node under a given risk scenario.
    Static features (38-dim) from workflow structure + synthetic temporal (9-dim).
    """
    static = _node_static_features(graph, node_name, rng)   # (38,)
    temporal = _synthetic_telemetry(scenario)                 # (9,)
    return np.concatenate([static, temporal]).astype(np.float32)


# ── Hardening Engine ──────────────────────────────────────────────────────────

class WorkflowHardeningEngine:
    """
    Analyzes an LLM-generated workflow and produces a HardenedWorkflow.

    For each node:
      1. Estimate risk scenario from node type + graph position
      2. Build synthetic 47-dim telemetry feature vector
      3. Run ORS-VQ model → get ORS code + resilience policy
      4. Record the ResilienceAnnotation

    The result is a HardenedWorkflow that can be simulated, diffed, or
    serialized for downstream execution engines (n8n, LangGraph, etc.).
    """

    def __init__(self, ors_model: ORSVQModel, seed: int = 0):
        self.model = ors_model
        self.rng   = random.Random(seed)
        self.model.eval()

    def analyze(self, graph: WorkflowGraph) -> HardenedWorkflow:
        """
        Run the full hardening analysis on a workflow graph.
        Returns a HardenedWorkflow with per-node annotations.
        """
        hardened_nodes: Dict[str, HardenedNode] = {}

        for node_name, node in graph.nodes.items():
            scenario    = _node_risk_scenario(node)
            feat        = _build_feature_vector(graph, node_name, scenario, self.rng)
            annotation  = self._annotate(node, scenario, feat)
            hardened_nodes[node_name] = HardenedNode(node=node, annotation=annotation)

        return HardenedWorkflow(original=graph, nodes=hardened_nodes)

    def _annotate(
        self,
        node:     NodeInfo,
        scenario: ScenarioState,
        feat:     np.ndarray,
    ) -> ResilienceAnnotation:
        """Run the ORS model on one node's feature vector and return annotation."""
        x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            code = self.model.code_for(x)
            pred = self.model.predict(x)
        return ResilienceAnnotation(
            ors_code      = code,
            retry         = pred.retry,
            timeout_tier  = pred.timeout,
            cb_policy     = pred.cb,
            fallback_type = pred.fallback,
            risk_scenario = SCENARIO_NAMES[int(scenario)],
        )
