"""
ablation_codebooks.py  --  Ablation: EMA vs Grad vs Factorized codebook

Trains three ORS model variants, evaluates codebook quality and hardening
performance. Uses the same evaluate_scenario() pipeline as hardening_comparison.py.

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py ablation_codebooks.py
"""
from __future__ import annotations
import random, warnings, json, sys
from collections import Counter, defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from resiliency_poc.dsb_loader               import DSBLoader
from resiliency_poc.temporal_trace_generator import TemporalFailureEvent, TEMPORAL_FEATURE_DIM
from resiliency_poc.scenario                 import ScenarioState, SCENARIO_NAMES
from resiliency_poc.model.encoder            import WorkflowEncoder, LATENT_DIM
from resiliency_poc.model.codebook           import VQCodebookEMA, VQCodebookGrad, FactorizedVQCodebook
from resiliency_poc.model.policy_heads       import PolicyHeads, PolicyPrediction
from resiliency_poc.model.ors_model          import ORSVQModel
from resiliency_poc.dsb_workflow_loader      import load_dsb_healthy_and_failure
from resiliency_poc.workflow_loader          import WorkflowGraph

from hardening.engine    import WorkflowHardeningEngine, _TYPE_TO_SCENARIO, _node_risk_scenario, _build_feature_vector
from hardening.simulator import WorkflowFailureSimulator, evaluate_scenario, ScenarioMetrics
from hardening.models    import ResilienceAnnotation, HardenedNode, HardenedWorkflow

# Patch DSB service names (same as hardening_comparison.py)
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

SEED          = 42
EPOCHS        = 50
BATCH_SIZE    = 64
LR            = 8e-4
VQ_BETA       = 0.25
N_TRIALS      = 300   # per scenario per graph (faster than full 500)

EVAL_SCENARIOS = [
    ScenarioState.HEALTHY,
    ScenarioState.TRANSIENT_FAILURE,
    ScenarioState.DEGRADED,
    ScenarioState.SERVICE_DOWN,
]

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Factorized model wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FactorizedORSModel(nn.Module):
    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM):
        super().__init__()
        self.encoder  = WorkflowEncoder(feature_dim, LATENT_DIM)
        self.codebook = FactorizedVQCodebook(
            embedding_dim=LATENT_DIM, policy_dims=(4, 3, 3, 4),
            beta=VQ_BETA, decay=0.99, restart_threshold=1.0)
        self.heads    = PolicyHeads(LATENT_DIM)

    def forward(self, x):
        z_e = self.encoder(x)
        z_qs, code_idxs, vq_loss = self.codebook(z_e)
        z_q = torch.stack(z_qs, dim=0).mean(0)
        r_l, t_l, c_l, f_l = self.heads(z_q)
        return r_l, t_l, c_l, f_l, code_idxs, vq_loss

    def loss(self, x, labels):
        r_l, t_l, c_l, f_l, _, vq_loss = self.forward(x)
        task = (F.cross_entropy(r_l, labels[:,0]) + F.cross_entropy(t_l, labels[:,1])
              + F.cross_entropy(c_l, labels[:,2]) + F.cross_entropy(f_l, labels[:,3]))
        return task + vq_loss, {"task_loss": task.item(), "vq_loss": vq_loss.item()}

    def predict(self, x):
        with torch.no_grad():
            r_l, t_l, c_l, f_l, _, _ = self.forward(x)
        return PolicyPrediction(int(r_l.argmax(1).item()), int(t_l.argmax(1).item()),
                                int(c_l.argmax(1).item()), int(f_l.argmax(1).item()))


class GradORSModel(nn.Module):
    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM, codebook_size: int = 16):
        super().__init__()
        self.codebook_size = codebook_size
        self.encoder  = WorkflowEncoder(feature_dim, LATENT_DIM)
        self.codebook = VQCodebookGrad(codebook_size, LATENT_DIM, VQ_BETA)
        self.heads    = PolicyHeads(LATENT_DIM)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, code_idx, vq_loss = self.codebook(z_e)
        r_l, t_l, c_l, f_l = self.heads(z_q)
        return r_l, t_l, c_l, f_l, code_idx, vq_loss

    def loss(self, x, labels):
        r_l, t_l, c_l, f_l, _, vq_loss = self.forward(x)
        task = (F.cross_entropy(r_l, labels[:,0]) + F.cross_entropy(t_l, labels[:,1])
              + F.cross_entropy(c_l, labels[:,2]) + F.cross_entropy(f_l, labels[:,3]))
        return task + vq_loss, {"task_loss": task.item(), "vq_loss": vq_loss.item()}

    def predict(self, x):
        with torch.no_grad():
            r_l, t_l, c_l, f_l, _, _ = self.forward(x)
        return PolicyPrediction(int(r_l.argmax(1).item()), int(t_l.argmax(1).item()),
                                int(c_l.argmax(1).item()), int(f_l.argmax(1).item()))


# ──────────────────────────────────────────────────────────────────────────────
# Data helpers
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
        balanced.extend(evs * max(1, max_cnt // len(evs)))
    out = healthy_s + balanced; random.shuffle(out); return out


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def _train(model, loader, epochs, tag):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for xb, yb in loader:
            loss, _ = model.loss(xb, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 10 == 0 or ep == epochs:
            print(f"  [{tag}] epoch {ep:>3}/{epochs}  loss={tot/len(loader):.4f}")
    model.eval()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def _cf_accuracy(model, events):
    n = len(events); exact_ok = cb_ok = fb_ok = retry_ok = 0
    model.eval()
    with torch.no_grad():
        for ev in events:
            x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            r_ok = int(pred.retry    == ev.label_retry)
            c_ok = int(pred.cb       == ev.label_cb)
            f_ok = int(pred.fallback == ev.label_fallback)
            retry_ok += r_ok; cb_ok += c_ok; fb_ok += f_ok
            exact_ok += int(r_ok and c_ok and f_ok)
    return {"cf_retry": retry_ok/n*100, "cf_cb": cb_ok/n*100,
            "cf_fallback": fb_ok/n*100, "cf_exact": exact_ok/n*100}

def _codebook_metrics(model, events):
    model.eval()
    codes_all = []; sc_all = []; cb_all = []
    with torch.no_grad():
        for ev in events:
            x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            if isinstance(model, ORSVQModel):
                z_e = model.encoder(x); _, idx, _ = model.codebook(z_e)
                code = int(idx[0].item())
            elif isinstance(model, FactorizedORSModel):
                z_e = model.encoder(x); _, idxs, _ = model.codebook(z_e)
                code = int(idxs[0][0].item())
            else:
                z_e = model.encoder(x); _, idx, _ = model.codebook(z_e)
                code = int(idx[0].item())
            codes_all.append(code); sc_all.append(int(ev.scenario)); cb_all.append(pred.cb)

    codes_all = np.array(codes_all); sc_all = np.array(sc_all); cb_all = np.array(cb_all)
    code_set  = sorted(set(codes_all.tolist()))
    purities  = [Counter(sc_all[codes_all==c].tolist()).most_common(1)[0][1] / (codes_all==c).sum()
                 for c in code_set]
    consists  = [Counter(cb_all[codes_all==c].tolist()).most_common(1)[0][1] / (codes_all==c).sum()
                 for c in code_set]
    total = model.codebook_size if hasattr(model, "codebook_size") else 16
    if isinstance(model, FactorizedORSModel): total = 4
    return {"mean_purity":      float(np.mean(purities)*100),
            "mean_consistency": float(np.mean(consists)*100),
            "active_codes":     len(code_set),
            "total_codes":      total,
            "collapse_rate":    (total - len(code_set)) / total * 100}


def _make_hardened_workflow(model, graph: WorkflowGraph) -> HardenedWorkflow:
    """Build a HardenedWorkflow by querying model.predict() for each node using
    the same _node_risk_scenario + _build_feature_vector helpers as the engine."""
    import random as _rng
    rng = _rng.Random(SEED)
    nodes = {}
    for node_name, node_info in graph.nodes.items():
        scenario = _node_risk_scenario(node_info)
        feat     = _build_feature_vector(graph, node_name, scenario, rng)
        x        = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        pred     = model.predict(x)
        # code_for may not exist on all variants — use 0 as fallback
        code = 0
        if hasattr(model, "code_for"):
            with torch.no_grad(): code = model.code_for(x)
        ann = ResilienceAnnotation(
            ors_code=code, retry=pred.retry, timeout_tier=pred.timeout,
            cb_policy=pred.cb, fallback_type=pred.fallback,
            risk_scenario=SCENARIO_NAMES[int(scenario)],
        )
        nodes[node_name] = HardenedNode(node=node_info, annotation=ann)
    return HardenedWorkflow(original=graph, nodes=nodes)


def _hardening_score(model, graphs, n_trials):
    sim = WorkflowFailureSimulator(seed=SEED)
    completions, wasteds = [], []
    for graph in graphs:
        hw = _make_hardened_workflow(model, graph)
        for sc in EVAL_SCENARIOS:
            _, hard_m = evaluate_scenario(sim, graph, hw, sc, n_trials=n_trials)
            completions.append(hard_m.completion_pct)
            wasteds.append(hard_m.avg_wasted)
    return {"mean_completion": float(np.mean(completions)),
            "mean_wasted":     float(np.mean(wasteds))}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  CODEBOOK ABLATION: EMA vs Gradient vs Factorized")
    print("=" * 70)

    print("\n[1] Loading DSB data ...")
    loader = DSBLoader()
    train_events, test_events = loader.load_dataset()
    train_events = _oversample(train_events)
    X_tr, Y_tr   = _tensors(train_events)
    train_loader  = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=BATCH_SIZE, shuffle=True)
    print(f"    Train={len(train_events)}  Test={len(test_events)}")

    print("\n[2] Loading DSB graphs ...")
    graphs = load_dsb_healthy_and_failure()
    print(f"    {len(graphs)} graphs")

    results = {}

    # ── EMA (pre-trained checkpoint) ─────────────────────────────────────────
    print("\n[3a] EMA codebook ...")
    ema = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
    ckpt = Path("resiliency_poc/checkpoints/ors_dsb_vq.pt")
    if ckpt.exists():
        d = torch.load(ckpt, weights_only=False)
        ema.load_state_dict(d["state_dict"]); ema.eval()
        print("    Loaded checkpoint.")
    else:
        _train(ema, train_loader, EPOCHS, "EMA")
    cb  = _codebook_metrics(ema, test_events)
    cf  = _cf_accuracy(ema, test_events)
    results["ORS-VQ (EMA)"] = {**cb, **cf}
    print(f"    Purity={cb['mean_purity']:.1f}%  Consistency={cb['mean_consistency']:.1f}%  "
          f"Active={cb['active_codes']}/{cb['total_codes']}  CF={cf['cf_exact']:.1f}%")

    # ── Gradient codebook ─────────────────────────────────────────────────────
    print("\n[3b] Gradient codebook (VQCodebookGrad) ...")
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    grad = GradORSModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
    _train(grad, train_loader, EPOCHS, "Grad")
    cb  = _codebook_metrics(grad, test_events)
    cf  = _cf_accuracy(grad, test_events)
    results["ORS-VQ (Grad)"] = {**cb, **cf}
    print(f"    Purity={cb['mean_purity']:.1f}%  Consistency={cb['mean_consistency']:.1f}%  "
          f"Active={cb['active_codes']}/{cb['total_codes']}  CF={cf['cf_exact']:.1f}%")

    # ── Factorized codebook ───────────────────────────────────────────────────
    print("\n[3c] Factorized codebook ...")
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    fact = FactorizedORSModel(feature_dim=TEMPORAL_FEATURE_DIM)
    _train(fact, train_loader, EPOCHS, "Factorized")
    cb  = _codebook_metrics(fact, test_events)
    cf  = _cf_accuracy(fact, test_events)
    results["ORS-VQ (Factorized)"] = {**cb, **cf}
    print(f"    Purity={cb['mean_purity']:.1f}%  Consistency={cb['mean_consistency']:.1f}%  "
          f"Active={cb['active_codes']}/{cb['total_codes']}  CF={cf['cf_exact']:.1f}%")

    # ── Hardening scores ──────────────────────────────────────────────────────
    print(f"\n[4] Hardening scores ({N_TRIALS} trials) ...")
    for name, model in [("ORS-VQ (EMA)", ema), ("ORS-VQ (Grad)", grad), ("ORS-VQ (Factorized)", fact)]:
        print(f"    {name} ...")
        try:
            h = _hardening_score(model, graphs, N_TRIALS)
            results[name].update(h)
            print(f"    completion={h['mean_completion']:.1f}%  wasted={h['mean_wasted']:.2f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name]["mean_completion"] = None
            results[name]["mean_wasted"]     = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("  CODEBOOK ABLATION SUMMARY")
    print("=" * 85)
    print(f"  {'Variant':<22} {'Purity%':>8} {'Consist%':>9} {'Active':>8} {'Collapse%':>10} {'CF%':>7} {'Compl%':>8} {'Wasted':>8}")
    print("  " + "-" * 83)
    for name, r in results.items():
        c = f"{r['mean_completion']:.1f}" if r.get("mean_completion") is not None else "N/A"
        w = f"{r['mean_wasted']:.2f}"     if r.get("mean_wasted")     is not None else "N/A"
        print(f"  {name:<22} {r['mean_purity']:>8.1f} {r['mean_consistency']:>9.1f} "
              f"{r['active_codes']:>5}/{r['total_codes']:<3} {r['collapse_rate']:>8.1f}  "
              f"{r['cf_exact']:>7.1f} {c:>8} {w:>8}")
    print("=" * 85)

    out = Path("resiliency_poc/results/ablation_codebooks.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
