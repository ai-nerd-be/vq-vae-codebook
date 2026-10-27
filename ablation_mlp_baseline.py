"""
ablation_mlp_baseline.py  --  Learning-based baseline: plain MLP classifier

Trains a direct MLP (no VQ bottleneck) on the same counterfactual labels
and evaluates it as a learning-based baseline against ORS-VQ.

The MLP is identical in capacity to ORS-VQ's encoder + policy heads,
but without the discrete codebook. This isolates the contribution of the
VQ bottleneck to policy consistency and downstream resilience.

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py ablation_mlp_baseline.py
"""
from __future__ import annotations
import random, warnings, json
from collections import defaultdict, Counter
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from resiliency_poc.dsb_loader               import DSBLoader
from resiliency_poc.temporal_trace_generator import TemporalFailureEvent, TEMPORAL_FEATURE_DIM
from resiliency_poc.scenario                 import ScenarioState
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.model.policy_heads       import PolicyPrediction
from resiliency_poc.workflow_loader          import WorkflowGraph
from resiliency_poc.scenario                 import SCENARIO_NAMES

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO, _node_risk_scenario, _build_feature_vector
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow

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

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

EPOCHS      = 50
BATCH_SIZE  = 64
LR          = 8e-4
N_TRIALS    = 300
N_SIMS      = 100


# ──────────────────────────────────────────────────────────────────────────────
# MLP Baseline: same encoder capacity, no VQ
# ──────────────────────────────────────────────────────────────────────────────

class MLPBaseline(nn.Module):
    """
    Direct MLP policy predictor — no discrete bottleneck.
    Architecture: Linear(47->64) -> ReLU -> LayerNorm(64) -> Linear(64->32)
    -> ReLU -> four output heads (same as ORS-VQ heads).
    Total params comparable to ORS-VQ.
    """
    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )
        self.retry    = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 4))
        self.timeout  = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 3))
        self.cb       = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 3))
        self.fallback = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        z = self.encoder(x)
        return self.retry(z), self.timeout(z), self.cb(z), self.fallback(z)

    def loss(self, x, labels):
        r_l, t_l, c_l, f_l = self.forward(x)
        loss = (F.cross_entropy(r_l, labels[:, 0])
              + F.cross_entropy(t_l, labels[:, 1])
              + F.cross_entropy(c_l, labels[:, 2])
              + F.cross_entropy(f_l, labels[:, 3]))
        return loss

    def predict(self, x):
        with torch.no_grad():
            r_l, t_l, c_l, f_l = self.forward(x)
        return PolicyPrediction(
            int(r_l.argmax(1).item()), int(t_l.argmax(1).item()),
            int(c_l.argmax(1).item()), int(f_l.argmax(1).item()))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _tensors(events):
    X = torch.tensor(np.stack([ev.to_feature_vector() for ev in events]), dtype=torch.float32)
    Y = torch.tensor([[ev.label_retry, ev.label_timeout, ev.label_cb, ev.label_fallback]
                       for ev in events], dtype=torch.long)
    return X, Y

def _oversample(events):
    by_sc = defaultdict(list)
    for ev in events: by_sc[ev.scenario].append(ev)
    healthy  = by_sc.get(ScenarioState.HEALTHY, [])
    minority = [ev for sc, evs in by_sc.items() for ev in evs if sc != ScenarioState.HEALTHY]
    target   = min(len(healthy), int(len(minority) / 0.6 * 0.4))
    healthy_s = random.sample(healthy, target) if target > 0 else []
    counts   = Counter(ev.scenario for ev in minority)
    max_cnt  = max(counts.values()) if counts else 1
    balanced = []
    for sc, evs in by_sc.items():
        if sc == ScenarioState.HEALTHY: continue
        factor = max(1, max_cnt // len(evs))
        balanced.extend(evs * factor)
    out = healthy_s + balanced
    random.shuffle(out)
    return out

def _cf_accuracy(model, events):
    n = len(events); exact_ok = retry_ok = cb_ok = fb_ok = 0
    model.eval()
    with torch.no_grad():
        for ev in events:
            x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            r_ok = int(pred.retry    == ev.label_retry)
            c_ok = int(pred.cb       == ev.label_cb)
            f_ok = int(pred.fallback == ev.label_fallback)
            retry_ok += r_ok; cb_ok += c_ok; fb_ok += f_ok
            exact_ok += int(r_ok and c_ok and f_ok)
    return {
        "cf_retry":    retry_ok / n * 100,
        "cf_cb":       cb_ok    / n * 100,
        "cf_fallback": fb_ok    / n * 100,
        "cf_exact":    exact_ok / n * 100,
    }

def _policy_consistency(model, events):
    """Measure policy consistency: given same input region, how often same policy?
    Proxy: cluster events by scenario, measure modal policy within cluster."""
    by_sc = defaultdict(list)
    for ev in events: by_sc[ev.scenario].append(ev)
    consistencies = []
    model.eval()
    with torch.no_grad():
        for sc, evs in by_sc.items():
            preds = []
            for ev in evs:
                x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                pred = model.predict(x)
                preds.append((pred.cb, pred.fallback))  # most policy-critical dims
            cb_vals  = [p[0] for p in preds]
            fb_vals  = [p[1] for p in preds]
            cb_mode  = Counter(cb_vals).most_common(1)[0][1]
            fb_mode  = Counter(fb_vals).most_common(1)[0][1]
            consistencies.append((cb_mode + fb_mode) / (2 * len(preds)) * 100)
    return float(np.mean(consistencies))

EVAL_SCENARIOS = [ScenarioState.HEALTHY, ScenarioState.TRANSIENT_FAILURE,
                  ScenarioState.DEGRADED, ScenarioState.SERVICE_DOWN]

def _make_hw(model, graph: WorkflowGraph) -> HardenedWorkflow:
    import random as _r; rng = _r.Random(SEED)
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

def _hardening_metrics(model, graphs):
    sim = WorkflowFailureSimulator(seed=SEED)
    completions, wasteds = [], []
    for graph in graphs:
        hw = _make_hw(model, graph)
        for sc in EVAL_SCENARIOS:
            _, m = evaluate_scenario(sim, graph, hw, sc, n_trials=N_TRIALS)
            completions.append(m.completion_pct)
            wasteds.append(m.avg_wasted)
    return {"mean_completion": float(np.mean(completions)),
            "mean_wasted":     float(np.mean(wasteds))}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  LEARNING-BASED BASELINE: MLP Classifier (no VQ)")
    print("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading DSB data ...")
    loader_obj = DSBLoader()
    train_events, test_events = loader_obj.load_dataset()
    train_events = _oversample(train_events)
    X_tr, Y_tr = _tensors(train_events)
    train_loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    print(f"    Train={len(train_events)}  Test={len(test_events)}")

    # ── Train MLP baseline ─────────────────────────────────────────────────────
    print("\n[2] Training MLP baseline ...")
    mlp = MLPBaseline(feature_dim=TEMPORAL_FEATURE_DIM)
    opt = torch.optim.Adam(mlp.parameters(), lr=LR)
    mlp.train()
    for epoch in range(1, EPOCHS + 1):
        tot = 0.0
        for xb, yb in train_loader:
            loss = mlp.loss(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  [MLP] epoch {epoch:>3}/{EPOCHS}  loss={tot/len(train_loader):.4f}")
    mlp.eval()

    # ── Load ORS-VQ for comparison ─────────────────────────────────────────────
    print("\n[3] Loading ORS-VQ checkpoint ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
    ckpt = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
    d = torch.load(ckpt, weights_only=False)
    ors.load_state_dict(d["state_dict"]); ors.eval()

    # ── Representation metrics ─────────────────────────────────────────────────
    print("\n[4] Evaluating representation quality ...")
    cf_mlp = _cf_accuracy(mlp, test_events)
    cf_ors = _cf_accuracy(ors, test_events)
    cons_mlp = _policy_consistency(mlp, test_events)
    cons_ors = _policy_consistency(ors, test_events)
    print(f"    MLP  CF-exact={cf_mlp['cf_exact']:.1f}%  Policy-consistency={cons_mlp:.1f}%")
    print(f"    ORS  CF-exact={cf_ors['cf_exact']:.1f}%  Policy-consistency={cons_ors:.1f}%")

    # ── Hardening metrics ──────────────────────────────────────────────────────
    print(f"\n[5] Hardening comparison ({N_TRIALS} trials × {N_SIMS} sims) ...")
    graphs = load_dsb_healthy_and_failure()

    print("    Running MLP ...")
    h_mlp = _hardening_metrics(mlp, graphs)
    print(f"    MLP: completion={h_mlp['mean_completion']:.1f}%  wasted={h_mlp['mean_wasted']:.2f}")

    print("    Running ORS-VQ ...")
    h_ors = _hardening_metrics(ors, graphs)
    print(f"    ORS: completion={h_ors['mean_completion']:.1f}%  wasted={h_ors['mean_wasted']:.2f}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  MLP BASELINE vs ORS-VQ  (learning-based comparison)")
    print("=" * 72)
    print(f"  {'Method':<16} {'CF-exact%':>10} {'Consistency%':>13} {'Completion%':>12} {'AvgWasted':>10}")
    print("  " + "-" * 64)
    print(f"  {'MLP-Classifier':<16} {cf_mlp['cf_exact']:>10.1f} {cons_mlp:>13.1f} "
          f"{h_mlp['mean_completion']:>12.1f} {h_mlp['mean_wasted']:>10.2f}")
    print(f"  {'ORS-VQ (ours)':<16} {cf_ors['cf_exact']:>10.1f} {cons_ors:>13.1f} "
          f"{h_ors['mean_completion']:>12.1f} {h_ors['mean_wasted']:>10.2f}")
    print("=" * 72)

    # ── Save ───────────────────────────────────────────────────────────────────
    out = Path("resiliency_poc/results/ablation_mlp_baseline.json")
    out.parent.mkdir(exist_ok=True)
    results = {
        "MLP": {**cf_mlp, "policy_consistency": cons_mlp, **h_mlp},
        "ORS": {**cf_ors, "policy_consistency": cons_ors, **h_ors},
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
