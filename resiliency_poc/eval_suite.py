"""
eval_suite.py  --  Systematic evaluation of ORS-VQ against baselines

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.eval_suite

Models compared (each answers a specific question):
  1. Threshold Rules    — current industry baseline
  2. Decision Tree      — interpretable non-neural baseline
  3. XGBoost            — strongest tabular baseline
  4. MLP-47             — same encoder, no VQ  (does discretisation help?)
  5. Continuous AE      — encoder + continuous bottleneck, no quantisation
                         (is the *discrete* vocabulary specifically what matters?)
  6. ORS-VQ             — shared codebook, the contribution

Evaluations:
  E1  Policy prediction accuracy (F1, precision, recall, exact match per head)
  E2  Latent space quality       (purity, silhouette, Davies-Bouldin, C-H index)
  E3  Cross-workflow ORS transfer (does code X fire for both workflows?)
  E4  Ablation summary table
"""

from __future__ import annotations
import random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.tree           import DecisionTreeClassifier
from sklearn.multioutput    import MultiOutputClassifier
from sklearn.metrics        import (
    f1_score, precision_score, recall_score, accuracy_score,
)
from sklearn.preprocessing  import LabelEncoder
from sklearn.metrics        import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import xgboost as xgb

from .workflow_loader           import load_both_workflows
from .temporal_trace_generator  import (
    TemporalTraceGenerator, TemporalFailureEvent, TEMPORAL_FEATURE_DIM,
)
from .scenario                  import SCENARIO_NAMES, ScenarioState
from .counterfactual_labeler    import SCENARIO_TO_CATEGORY, FAILURE_CATEGORY_NAMES
from .label_oracle              import (
    RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS,
    get_policy,
)
from .model.encoder             import WorkflowEncoder, LATENT_DIM
from .model.codebook            import VQCodebookEMA
from .model.policy_heads        import PolicyHeads, PolicyPrediction
from .model.ors_model           import ORSVQModel

# ── Reproducibility ────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Hyperparameters ────────────────────────────────────────────────────────────

N_TRAIN        = 4_000
N_TEST         =   800
BATCH_SIZE     =    64
LR             =  8e-4
EPOCHS         =    50
CODEBOOK_SIZE  =    16

CKPT_DIR  = Path(__file__).parent / "checkpoints"
CKPT_ORS  = CKPT_DIR / "ors_vq.pt"
CKPT_MLP  = CKPT_DIR / "eval_mlp47.pt"
CKPT_AE   = CKPT_DIR / "eval_cont_ae.pt"

POLICY_DIM_NAMES = ["retry", "timeout", "CB", "fallback"]
SCENARIO_LABELS  = [SCENARIO_NAMES[int(s)] for s in sorted(ScenarioState)]


# ─────────────────────────────────────────────────────────────────────────────
# Continuous Autoencoder  (ablation: is discreteness important?)
# Same encoder as ORS-VQ, same policy heads, but no codebook.
# ─────────────────────────────────────────────────────────────────────────────

class ContinuousAEModel(nn.Module):
    """Encoder → continuous bottleneck → policy heads.  No quantisation."""

    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = WorkflowEncoder(feature_dim, latent_dim)
        # Extra projection so the model has the same parameter count as ORS-VQ
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.LayerNorm(latent_dim),
        )
        self.heads = PolicyHeads(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple:
        z = self.bottleneck(self.encoder(x))
        r_l, t_l, c_l, f_l = self.heads(z)
        return r_l, t_l, c_l, f_l

    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple:
        r_l, t_l, c_l, f_l = self.forward(x)
        task_loss = (
            F.cross_entropy(r_l, labels[:, 0])
            + F.cross_entropy(t_l, labels[:, 1])
            + F.cross_entropy(c_l, labels[:, 2])
            + F.cross_entropy(f_l, labels[:, 3])
        )
        return task_loss, {"task_loss": task_loss.item(), "vq_loss": 0.0, "perplexity": float("nan")}

    def predict(self, x: torch.Tensor) -> PolicyPrediction:
        with torch.no_grad():
            r_l, t_l, c_l, f_l = self.forward(x)
            return PolicyPrediction(
                retry=int(r_l.argmax(1).item()),
                timeout=int(t_l.argmax(1).item()),
                cb=int(c_l.argmax(1).item()),
                fallback=int(f_l.argmax(1).item()),
            )

    def get_logits(self, x):
        return self.forward(x)

    def latent(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.bottleneck(self.encoder(x)).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# MLP-47  (same encoder as ORS-VQ, no VQ layer)
# ─────────────────────────────────────────────────────────────────────────────

class MLP47Model(nn.Module):
    """Encoder → continuous z → policy heads.  No codebook at all."""

    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = WorkflowEncoder(feature_dim, latent_dim)
        self.heads   = PolicyHeads(latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple:
        z = self.encoder(x)
        return self.heads(z)

    def loss(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple:
        r_l, t_l, c_l, f_l = self.forward(x)
        task_loss = (
            F.cross_entropy(r_l, labels[:, 0])
            + F.cross_entropy(t_l, labels[:, 1])
            + F.cross_entropy(c_l, labels[:, 2])
            + F.cross_entropy(f_l, labels[:, 3])
        )
        return task_loss, {"task_loss": task_loss.item(), "vq_loss": 0.0, "perplexity": float("nan")}

    def predict(self, x: torch.Tensor) -> PolicyPrediction:
        with torch.no_grad():
            r_l, t_l, c_l, f_l = self.forward(x)
            return PolicyPrediction(
                retry=int(r_l.argmax(1).item()),
                timeout=int(t_l.argmax(1).item()),
                cb=int(c_l.argmax(1).item()),
                fallback=int(f_l.argmax(1).item()),
            )

    def get_logits(self, x):
        return self.forward(x)

    def latent(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.encoder(x).numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Threshold baseline  (rule-based, no ML)
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdBaseline:
    """
    Replicates the existing label_oracle.get_policy heuristic.
    Uses static features only (failure mode is inferred from scenario).
    """
    def predict(self, ev: TemporalFailureEvent) -> PolicyPrediction:
        from .trace_generator import FailureEvent
        fe = ev.to_failure_event()
        p  = get_policy(fe.failure_mode, fe.node_type, fe.retry_count, fe.is_external)
        return PolicyPrediction(p.retry, p.timeout_tier, p.circuit_breaker, p.fallback)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _minority_oversample(events):
    by_sc = defaultdict(list)
    for ev in events:
        by_sc[ev.scenario].append(ev)
    minority = [ev for sc, evs in by_sc.items()
                for ev in evs if sc != ScenarioState.HEALTHY]
    healthy  = by_sc[ScenarioState.HEALTHY]
    target_h = min(len(healthy), int(len(minority) / 0.6 * 0.4))
    healthy_s = random.sample(healthy, target_h)
    counts    = Counter(ev.scenario for ev in minority)
    max_cnt   = max(counts.values())
    balanced  = []
    for sc, evs in by_sc.items():
        if sc == ScenarioState.HEALTHY:
            continue
        factor = max(1, max_cnt // len(evs))
        balanced.extend(evs * factor)
    result = healthy_s + balanced
    random.shuffle(result)
    return result


def _tensors(events, noise=0.0):
    X = torch.tensor(np.stack([e.to_feature_vector() for e in events]), dtype=torch.float32)
    if noise > 0:
        X = X + torch.randn_like(X) * noise
    Y = torch.tensor(
        [[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback] for e in events],
        dtype=torch.long,
    )
    return X, Y


def _numpy_XY(events):
    X = np.stack([e.to_feature_vector() for e in events]).astype(np.float32)
    Y = np.array([[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback]
                  for e in events], dtype=np.int32)
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_neural(model, loader, epochs, label):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for x, y in loader:
            loss, _ = model.loss(x, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == epochs:
            print(f"  [{label}] epoch {epoch:>3}/{epochs}  loss={total/len(loader):.4f}")


def _load_neural(model, path):
    data = torch.load(path, weights_only=False)
    model.load_state_dict(data["state_dict"])
    model.eval()


def _save_neural(model, path, **meta):
    torch.save({"state_dict": model.state_dict(), **meta}, path)


# ─────────────────────────────────────────────────────────────────────────────
# E1: Policy prediction metrics
# ─────────────────────────────────────────────────────────────────────────────

def _predict_all_neural(model, events):
    """Return (Y_true, Y_pred) arrays for a neural model."""
    model.eval()
    preds = []
    with torch.no_grad():
        for ev in events:
            x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            preds.append([pred.retry, pred.timeout, pred.cb, pred.fallback])
    Y_true = np.array([[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback]
                       for e in events])
    return Y_true, np.array(preds)


def _predict_threshold(baseline, events):
    preds = []
    for ev in events:
        p = baseline.predict(ev)
        preds.append([p.retry, p.timeout, p.cb, p.fallback])
    Y_true = np.array([[e.label_retry, e.label_timeout, e.label_cb, e.label_fallback]
                       for e in events])
    return Y_true, np.array(preds)


def _policy_metrics(Y_true, Y_pred) -> dict:
    """Per-dimension F1 (macro) + exact match rate."""
    metrics = {}
    for i, dim in enumerate(POLICY_DIM_NAMES):
        yt, yp = Y_true[:, i], Y_pred[:, i]
        metrics[f"{dim}_acc"]  = accuracy_score(yt, yp) * 100
        metrics[f"{dim}_f1"]   = f1_score(yt, yp, average="macro", zero_division=0) * 100
        metrics[f"{dim}_prec"] = precision_score(yt, yp, average="macro", zero_division=0) * 100
        metrics[f"{dim}_rec"]  = recall_score(yt, yp, average="macro", zero_division=0) * 100
    exact = np.all(Y_true == Y_pred, axis=1).mean() * 100
    metrics["exact_match"] = exact
    return metrics


def _print_e1(rows: List[Tuple[str, dict]]) -> None:
    W = 115
    print("\n" + "=" * W)
    print("  E1: POLICY PREDICTION  (macro F1 per dimension + exact match)")
    print("  Ground truth = counterfactual simulator-optimal labels")
    print("=" * W)
    hdr = (f"  {'Model':<22}  {'Retry F1':>8}  {'Timeout F1':>10}  "
           f"{'CB F1':>6}  {'Fallback F1':>11}  {'Exact%':>7}  {'Avg F1':>7}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for name, m in rows:
        avg_f1 = np.mean([m[f"{d}_f1"] for d in POLICY_DIM_NAMES])
        print(
            f"  {name:<22}  "
            f"{m['retry_f1']:>7.1f}%  {m['timeout_f1']:>9.1f}%  "
            f"{m['CB_f1']:>5.1f}%  {m['fallback_f1']:>10.1f}%  "
            f"{m['exact_match']:>6.1f}%  {avg_f1:>6.1f}%"
        )
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# E2: Latent space quality
# ─────────────────────────────────────────────────────────────────────────────

def _extract_latent_ors(model: ORSVQModel, events) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (z_e, code_indices) — pre-quantisation encoder outputs + code assignments."""
    model.eval()
    z_es, codes = [], []
    with torch.no_grad():
        for ev in events:
            x   = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            z_e = model.encoder(x)
            _, idx, _ = model.codebook(z_e)
            z_es.append(z_e.squeeze(0).numpy())
            codes.append(int(idx.item()))
    return np.array(z_es), np.array(codes)


def _extract_latent_neural(model, events) -> np.ndarray:
    """Returns encoder/bottleneck output for MLP47 or ContinuousAE."""
    model.eval()
    zs = []
    with torch.no_grad():
        for ev in events:
            x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            zs.append(model.latent(x).squeeze(0))
    return np.array(zs)


def _cluster_purity(z: np.ndarray, scenario_labels: np.ndarray,
                    cluster_labels: np.ndarray) -> float:
    """
    For each cluster, find its dominant scenario.
    Purity = fraction of points in their cluster's dominant scenario.
    """
    total = correct = 0
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() == 0:
            continue
        dominant = Counter(scenario_labels[mask].tolist()).most_common(1)[0][1]
        correct += dominant
        total   += mask.sum()
    return correct / total if total > 0 else 0.0


def _code_purity_per_entry(codes: np.ndarray, scenario_labels: np.ndarray) -> List[dict]:
    """Per-code breakdown: dominant scenario + purity."""
    rows = []
    for c in sorted(np.unique(codes)):
        mask = codes == c
        sc_counts = Counter(scenario_labels[mask].tolist())
        top_sc, top_n = sc_counts.most_common(1)[0]
        rows.append({
            "code":     c,
            "count":    int(mask.sum()),
            "scenario": SCENARIO_NAMES[top_sc],
            "purity":   top_n / mask.sum(),
        })
    return rows


def _latent_quality(z: np.ndarray, cluster_labels: np.ndarray,
                    scenario_labels: np.ndarray) -> dict:
    """Compute clustering quality metrics."""
    n_clusters = len(np.unique(cluster_labels))
    metrics = {"n_clusters": n_clusters}

    if n_clusters < 2 or n_clusters >= len(z):
        metrics.update({"silhouette": float("nan"), "davies_bouldin": float("nan"),
                        "calinski_harabasz": float("nan"), "purity": float("nan")})
        return metrics

    # Sub-sample for speed if large
    if len(z) > 2000:
        idx = np.random.choice(len(z), 2000, replace=False)
        z_s, c_s, sc_s = z[idx], cluster_labels[idx], scenario_labels[idx]
    else:
        z_s, c_s, sc_s = z, cluster_labels, scenario_labels

    try:
        metrics["silhouette"]         = silhouette_score(z_s, c_s)
        metrics["davies_bouldin"]     = davies_bouldin_score(z_s, c_s)
        metrics["calinski_harabasz"]  = calinski_harabasz_score(z_s, c_s)
        metrics["purity"]             = _cluster_purity(z_s, sc_s, c_s)
    except Exception:
        metrics.update({"silhouette": float("nan"), "davies_bouldin": float("nan"),
                        "calinski_harabasz": float("nan"), "purity": float("nan")})
    return metrics


def _kmeans_clusters(z: np.ndarray, n: int = 5) -> np.ndarray:
    """KMeans cluster assignment for continuous latent models."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n, random_state=SEED, n_init=10)
    return km.fit_predict(z)


def _print_e2(rows: List[Tuple[str, dict, bool]]) -> None:
    """rows: (name, metrics_dict, has_codebook)"""
    W = 110
    print("\n" + "=" * W)
    print("  E2: LATENT SPACE QUALITY")
    print("  Scenario labels used as ground-truth clusters (5 scenarios).")
    print("  ORS-VQ uses codebook assignments; others use KMeans(k=5) on encoder output.")
    print("  Silhouette: higher better | Davies-Bouldin: lower better | C-H: higher better")
    print("=" * W)
    hdr = (f"  {'Model':<22}  {'Clusters':>8}  {'Purity':>7}  "
           f"{'Silhouette':>10}  {'D-B Index':>10}  {'C-H Index':>10}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for name, m, has_cb in rows:
        cb_tag = " (codebook)" if has_cb else " (KMeans-5)"
        sil  = f"{m['silhouette']:.3f}"   if not np.isnan(m.get('silhouette', float('nan')))   else "  N/A"
        db   = f"{m['davies_bouldin']:.3f}" if not np.isnan(m.get('davies_bouldin', float('nan'))) else "  N/A"
        ch   = f"{m['calinski_harabasz']:.1f}" if not np.isnan(m.get('calinski_harabasz', float('nan'))) else "  N/A"
        pur  = f"{m['purity']:.1%}"        if not np.isnan(m.get('purity', float('nan')))        else "  N/A"
        print(f"  {name:<22}  {str(m['n_clusters']) + cb_tag:>16}  {pur:>7}  "
              f"{sil:>10}  {db:>10}  {ch:>10}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# E3: Cross-workflow transfer (ORS-VQ)
# ─────────────────────────────────────────────────────────────────────────────

def _print_e3(model: ORSVQModel, test_wf1: list, test_wf2: list) -> None:
    W = 100
    print("\n" + "=" * W)
    print("  E3: CROSS-WORKFLOW ORS TRANSFER")
    print("  Same failure scenario => same dominant ORS code across different workflow graphs.")
    print("=" * W)

    wf_names = [test_wf1[0].workflow_name, test_wf2[0].workflow_name] if test_wf1 and test_wf2 else ["WF1", "WF2"]

    def _code_map(events):
        by_sc = defaultdict(Counter)
        for ev in events:
            x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            c = model.code_for(x)
            sc = SCENARIO_NAMES[int(ev.scenario)]
            by_sc[sc][c] += 1
        return by_sc

    model.eval()
    with torch.no_grad():
        map1 = _code_map(test_wf1)
        map2 = _code_map(test_wf2)

    col = 26
    print(f"  {'Scenario':<14}  {wf_names[0][:col]:<{col}}  {wf_names[1][:col]:<{col}}  Result")
    print("  " + "-" * (W - 2))

    same = total = 0
    for sc in SCENARIO_NAMES:
        c1 = map1.get(sc)
        c2 = map2.get(sc)
        if not c1 or not c2:
            continue
        dom1, n1, tot1 = c1.most_common(1)[0][0], c1.most_common(1)[0][1], sum(c1.values())
        dom2, n2, tot2 = c2.most_common(1)[0][0], c2.most_common(1)[0][1], sum(c2.values())
        agree  = dom1 == dom2
        result = "[SAME]" if agree else "[diff]"
        print(f"  {sc:<14}  code={dom1} ({n1}/{tot1}){'':<{col-14}}  "
              f"code={dom2} ({n2}/{tot2}){'':<{col-14}}  {result}")
        same  += int(agree)
        total += 1

    print("  " + "-" * (W - 2))
    if total:
        print(f"  Transfer agreement: {same}/{total} scenarios ({same/total:.0%})")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# E4: Ablation summary
# ─────────────────────────────────────────────────────────────────────────────

def _print_e4_ablation(e1_rows, e2_rows) -> None:
    W = 110
    print("\n" + "=" * W)
    print("  E4: ABLATION — what does each architectural choice buy?")
    print("=" * W)
    print(f"  {'Model':<22}  {'Avg F1':>7}  {'Exact%':>7}  {'Purity':>7}  "
          f"{'Silhouette':>10}  {'D-B':>7}  Key addition")
    print("  " + "-" * (W - 2))

    e1  = {name: m   for name, m      in e1_rows}
    e2m = {name: (m, hcb) for name, m, hcb in e2_rows}

    order = [
        ("Threshold",    "No learning — rule-based heuristic"),
        ("Decision Tree","Interpretable non-neural, no latent space"),
        ("XGBoost",      "Strong tabular baseline, no latent space"),
        ("MLP-47",       "Neural encoder, continuous z, no quantisation"),
        ("Cont-AE",      "+ bottleneck projection (same param count as ORS-VQ)"),
        ("ORS-VQ",       "+ discrete shared codebook -> ORS vocabulary"),
    ]
    for name, description in order:
        m1 = e1.get(name, {})
        m2, _ = e2m.get(name, ({}, False))
        avg_f1 = np.mean([m1.get(f"{d}_f1", float("nan")) for d in POLICY_DIM_NAMES]) if m1 else float("nan")
        exact  = m1.get("exact_match", float("nan"))
        purity = m2.get("purity", float("nan"))
        sil    = m2.get("silhouette", float("nan"))
        db     = m2.get("davies_bouldin", float("nan"))

        def _fmt(v): return f"{v:.1f}" if not np.isnan(v) else " N/A"
        def _fmtp(v): return f"{v:.1%}" if not np.isnan(v) else " N/A"
        def _fmts(v): return f"{v:.3f}" if not np.isnan(v) else " N/A"

        print(f"  {name:<22}  {_fmt(avg_f1):>6}%  {_fmt(exact):>6}%  "
              f"{_fmtp(purity):>7}  {_fmts(sil):>10}  {_fmts(db):>7}  {description}")
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("\n[1] Loading workflows and generating dataset ...")
    graphs = load_both_workflows()
    for g in graphs:
        print(f"    {g.name[:55]:<55} nodes={len(g.nodes)}")

    # Training set — mixed, oversampled
    raw_train = TemporalTraceGenerator(graphs, seed=SEED).generate_dataset(N_TRAIN)
    train_bal = _minority_oversample(raw_train)

    # Test set — balanced per workflow (for transfer experiment)
    n_per_wf  = N_TEST // len(graphs)
    test_wf1  = TemporalTraceGenerator([graphs[0]], seed=SEED + 999).generate_dataset(n_per_wf)
    test_wf2  = TemporalTraceGenerator([graphs[1]], seed=SEED + 999).generate_dataset(n_per_wf)
    test_all  = test_wf1 + test_wf2
    random.shuffle(test_all)

    sc_dist = Counter(SCENARIO_NAMES[int(ev.scenario)] for ev in train_bal)
    print(f"    Train (oversampled): {len(train_bal)}  |  Test: {len(test_all)}")
    for sc, n in sorted(sc_dist.items()):
        print(f"      {sc:<14} {n}")

    X_tr, Y_tr = _tensors(train_bal, noise=0.05)
    X_te, Y_te = _tensors(test_all)
    X_tr_np, Y_tr_np = _numpy_XY(train_bal)
    X_te_np, Y_te_np = _numpy_XY(test_all)
    sc_labels_te = np.array([int(ev.scenario) for ev in test_all])

    loader = DataLoader(TensorDataset(X_tr, Y_tr), BATCH_SIZE, shuffle=True)

    # ── 2. Train / load models ────────────────────────────────────────────────
    print("\n[2] Training / loading models ...")

    # Threshold (no training)
    threshold = ThresholdBaseline()

    # Decision Tree
    print("  Decision Tree ...")
    dt = MultiOutputClassifier(DecisionTreeClassifier(max_depth=12, random_state=SEED))
    dt.fit(X_tr_np, Y_tr_np)

    # XGBoost — one classifier per policy head
    # Pad training set with one dummy row per class to guarantee all classes present
    print("  XGBoost (4 heads) ...")
    from .label_oracle import NUM_RETRY_CLASSES, NUM_TIMEOUT_CLASSES, NUM_CB_CLASSES, NUM_FALLBACK_CLASSES
    n_classes_per_dim = [NUM_RETRY_CLASSES, NUM_TIMEOUT_CLASSES, NUM_CB_CLASSES, NUM_FALLBACK_CLASSES]
    X_pad = np.zeros((max(n_classes_per_dim), X_tr_np.shape[1]), dtype=np.float32)
    xgb_models = []
    for i, (dim, nc) in enumerate(zip(POLICY_DIM_NAMES, n_classes_per_dim)):
        y_pad = np.arange(nc, dtype=np.int32)
        X_aug = np.vstack([X_tr_np, X_pad[:nc]])
        y_aug = np.concatenate([Y_tr_np[:, i], y_pad])
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 random_state=SEED, verbosity=0, eval_metric="mlogloss",
                                 num_class=nc)
        clf.fit(X_aug, y_aug)
        xgb_models.append(clf)

    # MLP-47
    print(f"  MLP-47 ({EPOCHS} epochs) ...")
    mlp47 = MLP47Model()
    if CKPT_MLP.exists():
        _load_neural(mlp47, CKPT_MLP)
    else:
        _train_neural(mlp47, loader, EPOCHS, "MLP-47")
        _save_neural(mlp47, CKPT_MLP, epochs=EPOCHS)

    # Continuous AE
    print(f"  Continuous AE ({EPOCHS} epochs) ...")
    cont_ae = ContinuousAEModel()
    if CKPT_AE.exists():
        _load_neural(cont_ae, CKPT_AE)
    else:
        _train_neural(cont_ae, loader, EPOCHS, "Cont-AE")
        _save_neural(cont_ae, CKPT_AE, epochs=EPOCHS)

    # ORS-VQ (shared codebook)
    print(f"  ORS-VQ (load from checkpoint) ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if CKPT_ORS.exists():
        _load_neural(ors, CKPT_ORS)
    else:
        raise RuntimeError("ors_vq.pt not found — run ors_poc.py first.")

    # ── E1: Policy prediction ─────────────────────────────────────────────────
    print("\n[E1] Policy prediction metrics ...")

    # XGB predictions
    xgb_preds = np.column_stack([clf.predict(X_te_np) for clf in xgb_models])
    # DT predictions
    dt_preds  = np.array(dt.predict(X_te_np))

    e1_rows = [
        ("Threshold",    _policy_metrics(*_predict_threshold(threshold, test_all))),
        ("Decision Tree",_policy_metrics(Y_te_np, dt_preds)),
        ("XGBoost",      _policy_metrics(Y_te_np, xgb_preds)),
        ("MLP-47",       _policy_metrics(*_predict_all_neural(mlp47, test_all))),
        ("Cont-AE",      _policy_metrics(*_predict_all_neural(cont_ae, test_all))),
        ("ORS-VQ",       _policy_metrics(*_predict_all_neural(ors, test_all))),
    ]
    _print_e1(e1_rows)

    # ── E2: Latent space quality ──────────────────────────────────────────────
    print("\n[E2] Latent space quality metrics ...")
    print("    Extracting latent vectors (this may take a moment) ...")

    z_mlp  = _extract_latent_neural(mlp47, test_all)
    z_ae   = _extract_latent_neural(cont_ae, test_all)
    z_ors, ors_codes = _extract_latent_ors(ors, test_all)

    km_mlp = _kmeans_clusters(z_mlp, n=5)
    km_ae  = _kmeans_clusters(z_ae,  n=5)

    e2_rows = [
        ("MLP-47",  _latent_quality(z_mlp,  km_mlp,    sc_labels_te), False),
        ("Cont-AE", _latent_quality(z_ae,   km_ae,     sc_labels_te), False),
        ("ORS-VQ",  _latent_quality(z_ors,  ors_codes, sc_labels_te), True),
    ]
    _print_e2(e2_rows)

    # Per-code purity for ORS-VQ
    print("\n  ORS-VQ per-code scenario purity:")
    purity_rows = _code_purity_per_entry(ors_codes, sc_labels_te)
    for r in purity_rows:
        bar = "#" * int(r["purity"] * 20)
        print(f"    code {r['code']:>2}  n={r['count']:>4}  {r['scenario']:<12}  "
              f"{r['purity']:.0%}  |{bar:<20}|")

    # ── E3: Cross-workflow transfer ───────────────────────────────────────────
    print("\n[E3] Cross-workflow ORS transfer ...")
    _print_e3(ors, test_wf1, test_wf2)

    # ── E4: Ablation summary ──────────────────────────────────────────────────
    _print_e4_ablation(e1_rows, e2_rows)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
