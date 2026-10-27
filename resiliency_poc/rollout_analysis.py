"""
rollout_analysis.py

Generates data for the temporal rollout figure:
  - Trains TemporalVQResiliencyModel quickly
  - Runs a CONTROLLED scenario stream with forced transitions:
      t=0..14   HEALTHY
      t=15..24  TRANSIENT_FAILURE
      t=25..34  SERVICE_DOWN
      t=35..44  RECOVERING
      t=45..54  HEALTHY
  - Records per time-step: true_scenario, retry_code, cb_code, predicted_policy
  - Prints JSON for the visualizer

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.rollout_analysis
"""

from __future__ import annotations
import json, random, time
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .workflow_loader import load_both_workflows
from .scenario import (
    ScenarioState, SCENARIO_NAMES, SCENARIO_OBS_DIM,
    sample_observation, ScenarioObservation, CBStateMachine,
)
from .counterfactual_labeler import label_from_simulation
from .temporal_trace_generator import (
    TemporalTraceGenerator, TEMPORAL_FEATURE_DIM, _node_static_features,
)
from .model.vqvae_model import TemporalVQResiliencyModel
from .label_oracle import RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ── Training ──────────────────────────────────────────────────────────────────

def train_temporal_model(n_samples: int = 4000, epochs: int = 40) -> TemporalVQResiliencyModel:
    print("[train] Generating temporal training data ...")
    graphs = load_both_workflows()
    gen    = TemporalTraceGenerator(graphs, seed=SEED)
    events = gen.generate_dataset(n_samples)

    X = torch.tensor(np.stack([e.to_feature_vector() for e in events]), dtype=torch.float32)
    Y = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback] for e in events],
        dtype=torch.long,
    )
    loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

    model = TemporalVQResiliencyModel(
        feature_dim=TEMPORAL_FEATURE_DIM, vq_beta=0.25, ema_decay=0.995,
        restart_threshold=0.5,
    )
    opt = torch.optim.Adam(model.parameters(), lr=8e-4)

    model.train()
    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            loss, _ = model.loss(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0:
            loss_val = loss.item()
            print(f"[train] epoch {epoch}/{epochs}  loss={loss_val:.4f}")

    model.eval()
    print("[train] done.")
    return model, graphs


# ── Controlled scenario stream ─────────────────────────────────────────────────

SCENARIO_SCHEDULE = [
    (0,  14, ScenarioState.HEALTHY),
    (15, 24, ScenarioState.TRANSIENT_FAILURE),
    (25, 34, ScenarioState.SERVICE_DOWN),
    (35, 44, ScenarioState.RECOVERING),
    (45, 54, ScenarioState.HEALTHY),
]

def _get_scenario_at(t: int) -> ScenarioState:
    for start, end, s in SCENARIO_SCHEDULE:
        if start <= t <= end:
            return s
    return ScenarioState.HEALTHY


def run_rollout(model: TemporalVQResiliencyModel, graphs, n_steps: int = 55) -> list:
    rng        = random.Random(SEED + 7)
    graph      = graphs[0]
    node_name  = list(graph.nodes.keys())[3]   # a mid-graph node
    static_f   = _node_static_features(graph, node_name, rng)

    records = []
    scenario_duration  = 0
    time_since_success = 0
    prev_scenario      = None
    cb                 = CBStateMachine()

    for t in range(n_steps):
        scenario = _get_scenario_at(t)
        if scenario != prev_scenario:
            scenario_duration  = 0
        prev_scenario = scenario

        obs = sample_observation(scenario, t, scenario_duration, time_since_success, rng, cb=cb)

        feat = np.concatenate([static_f, obs.to_feature_vector()]).astype(np.float32)
        x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            codes = model.codes_for(x)
            pred  = model.predict(x)

        time_since_success = 0 if obs.rolling_fail_rate < 0.5 else time_since_success + 1
        scenario_duration += 1

        records.append({
            "t":              t,
            "scenario":       SCENARIO_NAMES[scenario],
            "scenario_id":    int(scenario),
            "retry_code":     codes[0],
            "timeout_code":   codes[1],
            "cb_code":        codes[2],
            "fallback_code":  codes[3],
            "pred_retry":     RETRY_LABELS[pred.retry],
            "pred_cb":        CIRCUIT_BREAKER_LABELS[pred.cb],
            "pred_fallback":  FALLBACK_LABELS[pred.fallback],
            "rolling_fail":   round(obs.rolling_fail_rate, 3),
            "consec_fails":   obs.consec_failures,
            "retry_succ":     round(obs.retry_succ_rate, 3),
            "cb_state":       obs.cb_state,
            "cb_dur":         obs.cb_duration,
            "failures_last15": obs.failures_last_15,
        })

    return records


ROLLOUT_CKPT = Path(__file__).parent / "checkpoints" / "rollout_temp_vq.pt"

if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="Load saved rollout checkpoint")
    args = parser.parse_args()

    t0 = time.time()
    graphs = load_both_workflows()

    if args.load and ROLLOUT_CKPT.exists():
        model = TemporalVQResiliencyModel(feature_dim=TEMPORAL_FEATURE_DIM)
        data  = torch.load(ROLLOUT_CKPT, weights_only=False)
        model.load_state_dict(data["state_dict"])
        model.eval()
        print(f"[load] Loaded {ROLLOUT_CKPT.name}")
    else:
        model, _ = train_temporal_model(n_samples=4000, epochs=40)
        ROLLOUT_CKPT.parent.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict()}, ROLLOUT_CKPT)
        print(f"[save] Saved {ROLLOUT_CKPT.name}")

    records = run_rollout(model, graphs)
    print("\nROLLOUT_DATA_START")
    print(json.dumps(records, indent=None))
    print("ROLLOUT_DATA_END")
    print(f"\nDone in {time.time()-t0:.1f}s")
