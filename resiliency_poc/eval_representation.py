"""
eval_representation.py  --  Representation quality experiments for ORS-VQ

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m resiliency_poc.eval_representation

Implements the experiments suggested by professor feedback:

  E5  Codebook utilization   — usage counts, entropy, dead codes
  E6  Policy consistency     — does every code map to one policy?
  E7  Telemetry variance     — is every code one operational regime?
  E8  Robustness             — ORS stability under feature noise
  E9  Transfer               — train WF-A only, test WF-B only
  E10 ORS as retrieval       — retrieve similar historical situations by code
"""

from __future__ import annotations
import math, random, time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .workflow_loader           import load_both_workflows
from .temporal_trace_generator  import (
    TemporalTraceGenerator, TemporalFailureEvent, TEMPORAL_FEATURE_DIM,
)
from .scenario                  import SCENARIO_NAMES, ScenarioState
from .counterfactual_labeler    import FAILURE_CATEGORY_NAMES
from .label_oracle              import (
    RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS,
)
from .model.ors_model           import ORSVQModel

# ── Config ─────────────────────────────────────────────────────────────────────

SEED          = 42
N_TRAIN       = 4_000
N_TEST        =   800
BATCH_SIZE    =    64
LR            =  8e-4
EPOCHS        =    50
CODEBOOK_SIZE =    16

CKPT_DIR  = Path(__file__).parent / "checkpoints"
CKPT_ORS  = CKPT_DIR / "ors_vq.pt"
CKPT_WFA  = CKPT_DIR / "ors_wfa_only.pt"   # model trained on WF-A only

# Temporal feature indices (static features are 0..37; temporal start at 38)
FEAT_FAIL_RATE   = 38   # rolling failure rate
FEAT_CONSEC      = 39   # consecutive failures (normalised)
FEAT_RETRY_SUCC  = 40   # retry success rate
FEAT_LATENCY     = 41   # latency trend
FEAT_CB_STATE    = 44   # CB state (0=closed, 0.5=half, 1=open)
FEAT_CB_FAILS15  = 46   # failures in last 15 steps

TELEMETRY_FEATURES = {
    "fail_rate":   FEAT_FAIL_RATE,
    "consec_fail": FEAT_CONSEC,
    "retry_succ":  FEAT_RETRY_SUCC,
    "latency_dir": FEAT_LATENCY,
    "cb_state":    FEAT_CB_STATE,
    "cb_fails_15": FEAT_CB_FAILS15,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Data helpers ───────────────────────────────────────────────────────────────

def _minority_oversample(events):
    by_sc = defaultdict(list)
    for ev in events:
        by_sc[ev.scenario].append(ev)
    minority = [ev for sc, evs in by_sc.items()
                for ev in evs if sc != ScenarioState.HEALTHY]
    healthy  = by_sc[ScenarioState.HEALTHY]
    target_h = min(len(healthy), int(len(minority) / 0.6 * 0.4))
    balanced = []
    for sc, evs in by_sc.items():
        if sc == ScenarioState.HEALTHY:
            continue
        counts  = Counter(ev.scenario for ev in minority)
        max_cnt = max(counts.values())
        factor  = max(1, max_cnt // len(evs))
        balanced.extend(evs * factor)
    result = random.sample(healthy, target_h) + balanced
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


def _collect_codes(model: ORSVQModel, events) -> np.ndarray:
    model.eval()
    codes = []
    with torch.no_grad():
        for ev in events:
            x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            codes.append(model.code_for(x))
    return np.array(codes)


def _train_ors(model, loader, epochs, label="ORS"):
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


# ── E5: Codebook utilization ───────────────────────────────────────────────────

def _e5_codebook_utilization(model: ORSVQModel, events) -> None:
    codes = _collect_codes(model, events)
    K     = model.codebook_size
    n     = len(codes)

    counts     = Counter(codes.tolist())
    dead       = [c for c in range(K) if counts.get(c, 0) == 0]
    probs      = np.array([counts.get(c, 0) / n for c in range(K)])
    entropy    = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
    max_entr   = math.log(K)
    norm_entr  = entropy / max_entr        # 1.0 = perfectly uniform
    avg_occ    = n / (K - len(dead))

    W = 90
    print("\n" + "=" * W)
    print("  E5: CODEBOOK UTILIZATION")
    print(f"  K={K}  |  Dead codes: {len(dead)}  |  "
          f"Usage entropy: {entropy:.3f} / {max_entr:.3f} (normalised: {norm_entr:.2%})")
    print(f"  Avg occupancy (active codes): {avg_occ:.1f} samples/code")
    print("=" * W)
    bar_scale = 30 / max(max(counts.values()), 1)
    for c in range(K):
        cnt = counts.get(c, 0)
        bar = "#" * int(cnt * bar_scale)
        pct = cnt / n * 100
        sc_label = ""
        if cnt > 0:
            mask      = codes == c
            sc_dom    = Counter(int(ev.scenario) for ev, m in zip(events, mask) if m)
            top_sc    = sc_dom.most_common(1)[0][0] if sc_dom else -1
            sc_label  = f"  [{SCENARIO_NAMES[top_sc]}]" if top_sc >= 0 else ""
        print(f"  code {c:>2}  {bar:<30}  {cnt:>5} ({pct:>5.1f}%){sc_label}")
    if dead:
        print(f"\n  Dead codes: {dead}")
    print("=" * W)


# ── E6: Policy consistency ─────────────────────────────────────────────────────

def _e6_policy_consistency(model: ORSVQModel, events) -> None:
    codes = _collect_codes(model, events)
    model.eval()

    # Predict policy for every event
    all_preds = []
    with torch.no_grad():
        for ev in events:
            x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            all_preds.append((pred.retry, pred.timeout, pred.cb, pred.fallback))

    W = 110
    print("\n" + "=" * W)
    print("  E6: POLICY CONSISTENCY PER ORS CODE")
    print("  Consistency = fraction of samples in a code that get the *identical* policy tuple.")
    print("  High consistency => each code = one operational regime with one recovery action.")
    print("=" * W)
    print(f"  {'Code':>4}  {'N':>5}  {'Consistency':>11}  {'Dominant policy'}")
    print("  " + "-" * (W - 2))

    total_consistent = total_n = 0
    for c in sorted(set(codes)):
        idxs  = [i for i, code in enumerate(codes) if code == c]
        preds = [all_preds[i] for i in idxs]
        top, top_n = Counter(preds).most_common(1)[0]
        consistency = top_n / len(preds)
        r, t, cb, f = top
        pol = (f"retry={RETRY_LABELS[r]}, to={TIMEOUT_LABELS[t]}, "
               f"cb={CIRCUIT_BREAKER_LABELS[cb]}, fb={FALLBACK_LABELS[f]}")
        bar = "#" * int(consistency * 20)
        print(f"  {c:>4}  {len(preds):>5}  {consistency:>10.1%}  |{bar:<20}|  {pol}")
        total_consistent += top_n
        total_n          += len(preds)

    overall = total_consistent / total_n if total_n else 0
    print("  " + "-" * (W - 2))
    print(f"  Overall policy consistency: {overall:.1%}  "
          f"({total_consistent}/{total_n} samples match their code's dominant policy)")
    print("=" * W)


# ── E7: Telemetry variance per code ───────────────────────────────────────────

def _e7_telemetry_variance(model: ORSVQModel, events) -> None:
    codes = _collect_codes(model, events)
    feats = np.stack([ev.to_feature_vector() for ev in events])

    feat_names = list(TELEMETRY_FEATURES.keys())
    feat_idxs  = list(TELEMETRY_FEATURES.values())

    W = 110
    print("\n" + "=" * W)
    print("  E7: TELEMETRY VARIANCE PER ORS CODE")
    print("  Low variance within a code => it captures one operational regime.")
    print("  Values shown as mean ± std.")
    print("=" * W)
    hdr = f"  {'Code':>4}  {'N':>5}  " + "  ".join(f"{n:>14}" for n in feat_names)
    print(hdr)
    print("  " + "-" * (W - 2))

    all_means = []
    for c in sorted(set(codes)):
        mask = codes == c
        if mask.sum() == 0:
            continue
        subset = feats[mask][:, feat_idxs]
        means  = subset.mean(0)
        stds   = subset.std(0)
        all_means.append(means)
        cols = "  ".join(f"{m:>5.2f}±{s:>4.2f}" for m, s in zip(means, stds))
        print(f"  {c:>4}  {mask.sum():>5}  {cols}")

    # Across-code variance (how different are the codes from each other?)
    if len(all_means) > 1:
        across = np.std(all_means, axis=0)
        print("  " + "-" * (W - 2))
        print(f"  {'cross-code':>10}         " + "  ".join(f"{'std='+f'{v:.2f}':>14}" for v in across))
    print("=" * W)


# ── E8: Robustness (ORS stability under noise) ─────────────────────────────────

def _e8_robustness(model: ORSVQModel, events) -> None:
    clean_codes = _collect_codes(model, events)

    W = 80
    print("\n" + "=" * W)
    print("  E8: ORS ROBUSTNESS UNDER FEATURE NOISE")
    print("  Gaussian noise added to feature vector. Stability = same code assigned.")
    print("=" * W)
    print(f"  {'Noise std':>10}  {'Noise %':>8}  {'Stability':>10}  Bar")
    print("  " + "-" * (W - 2))

    noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
    model.eval()
    for noise in noise_levels:
        if noise == 0.0:
            stability = 1.0
        else:
            noisy_codes = []
            with torch.no_grad():
                for ev in events:
                    x = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                    x = x + torch.randn_like(x) * noise
                    noisy_codes.append(model.code_for(x))
            stability = np.mean(np.array(noisy_codes) == clean_codes)

        # Noise as % of mean feature magnitude
        bar = "#" * int(stability * 30)
        print(f"  {noise:>10.2f}  {noise*100:>7.0f}%  {stability:>9.1%}  |{bar:<30}|")
    print("=" * W)


# ── E9: Transfer — train WF-A only, test WF-B only ─────────────────────────────

def _e9_transfer(graphs) -> None:
    if len(graphs) < 2:
        print("\n[E9] Only one workflow available — skipping transfer experiment.")
        return

    wf_a, wf_b = graphs[0], graphs[1]
    print(f"\n[E9] CROSS-WORKFLOW TRANSFER")
    print(f"  Training on: {wf_a.name}")
    print(f"  Testing on:  {wf_b.name}")

    # Train on WF-A only
    train_a = _minority_oversample(
        TemporalTraceGenerator([wf_a], seed=SEED).generate_dataset(N_TRAIN)
    )
    test_b  = TemporalTraceGenerator([wf_b], seed=SEED + 999).generate_dataset(N_TEST // 2)

    # Also train joint model for comparison
    train_ab = _minority_oversample(
        TemporalTraceGenerator(graphs, seed=SEED).generate_dataset(N_TRAIN)
    )

    X_a, Y_a = _tensors(train_a, noise=0.05)
    X_ab, Y_ab = _tensors(train_ab, noise=0.05)
    loader_a  = DataLoader(TensorDataset(X_a,  Y_a),  BATCH_SIZE, shuffle=True)
    loader_ab = DataLoader(TensorDataset(X_ab, Y_ab), BATCH_SIZE, shuffle=True)

    # WF-A-only model
    ors_a = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if CKPT_WFA.exists():
        d = torch.load(CKPT_WFA, weights_only=False)
        ors_a.load_state_dict(d["state_dict"])
        ors_a.eval()
        print(f"  Loaded WF-A model from {CKPT_WFA.name}")
    else:
        print(f"  Training WF-A-only model ({EPOCHS} epochs) ...")
        _train_ors(ors_a, loader_a, EPOCHS, "ORS-WF-A")
        torch.save({"state_dict": ors_a.state_dict()}, CKPT_WFA)

    # Joint model (already trained)
    ors_joint = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    d = torch.load(CKPT_ORS, weights_only=False)
    ors_joint.load_state_dict(d["state_dict"])
    ors_joint.eval()

    def _f1_on(model, test_events):
        from sklearn.metrics import f1_score
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for ev in test_events:
                x    = torch.tensor(ev.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
                pred = model.predict(x)
                y_true.append([ev.label_retry, ev.label_timeout, ev.label_cb, ev.label_fallback])
                y_pred.append([pred.retry, pred.timeout, pred.cb, pred.fallback])
        yt = np.array(y_true); yp = np.array(y_pred)
        per_dim_f1 = [f1_score(yt[:, i], yp[:, i], average="macro", zero_division=0)
                      for i in range(4)]
        exact = np.all(yt == yp, axis=1).mean()
        return np.mean(per_dim_f1), exact

    f1_a,  ex_a  = _f1_on(ors_a,     test_b)
    f1_j,  ex_j  = _f1_on(ors_joint, test_b)

    # Codebook overlap: which codes appear in both WF-A training and WF-B test?
    codes_train_a = set(_collect_codes(ors_a, train_a).tolist())
    codes_test_b  = set(_collect_codes(ors_a, test_b).tolist())
    shared        = codes_train_a & codes_test_b
    transfer_pct  = len(shared) / max(len(codes_train_a), 1)

    W = 80
    print("\n" + "=" * W)
    print("  E9: CROSS-WORKFLOW TRANSFER RESULTS")
    print(f"  Train: {wf_a.name[:50]}  ({len(train_a)} events)")
    print(f"  Test:  {wf_b.name[:50]}  ({len(test_b)} events)")
    print("=" * W)
    print(f"  {'Model':<22}  {'Avg F1':>8}  {'Exact match':>12}")
    print("  " + "-" * (W - 2))
    print(f"  {'ORS-VQ (WF-A only)':<22}  {f1_a:>7.1%}  {ex_a:>11.1%}")
    print(f"  {'ORS-VQ (joint)':<22}  {f1_j:>7.1%}  {ex_j:>11.1%}")
    print("  " + "-" * (W - 2))
    print(f"  ORS codes in WF-A training: {sorted(codes_train_a)}")
    print(f"  ORS codes in WF-B test:     {sorted(codes_test_b)}")
    print(f"  Shared codes: {sorted(shared)}  ({transfer_pct:.0%} of WF-A codes reused on WF-B)")
    print("=" * W)


# ── E10: ORS as retrieval ──────────────────────────────────────────────────────

def _e10_retrieval(model: ORSVQModel, train_events, test_events,
                   n_queries: int = 5, top_k: int = 5) -> None:
    """
    For a sample of query events from the test set:
      1. Encode → find ORS code
      2. Retrieve top-K training events with the same code
      3. Measure: telemetry similarity, scenario match rate, policy match rate
    """
    # Build retrieval index: code → list of training events
    print("\n  Building retrieval index from training events ...")
    train_codes = _collect_codes(model, train_events)
    index: Dict[int, List[TemporalFailureEvent]] = defaultdict(list)
    for ev, c in zip(train_events, train_codes):
        index[c].append(ev)

    # Sample diverse queries (one per scenario)
    queries_by_sc: Dict[int, TemporalFailureEvent] = {}
    for ev in test_events:
        sc = int(ev.scenario)
        if sc not in queries_by_sc:
            queries_by_sc[sc] = ev

    W = 110
    print("\n" + "=" * W)
    print("  E10: ORS AS RETRIEVAL MEMORY")
    print("  Query: new execution context -> ORS code -> retrieve similar historical situations")
    print("  Measures: scenario match rate and policy match rate among retrieved examples")
    print("=" * W)

    feat_names = list(TELEMETRY_FEATURES.keys())
    feat_idxs  = list(TELEMETRY_FEATURES.values())

    model.eval()
    for sc_int in sorted(queries_by_sc.keys()):
        query = queries_by_sc[sc_int]
        with torch.no_grad():
            x_q   = torch.tensor(query.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            code  = model.code_for(x_q)
            pred  = model.predict(x_q)

        retrieved = index.get(code, [])
        if not retrieved:
            continue
        # Sample top_k randomly from the retrieved set (could sort by distance)
        sample = random.sample(retrieved, min(top_k, len(retrieved)))

        sc_match  = sum(1 for ev in sample if ev.scenario == query.scenario) / len(sample)
        pol_match = sum(
            1 for ev in sample
            if (ev.label_retry == query.label_retry and
                ev.label_cb    == query.label_cb    and
                ev.label_fallback == query.label_fallback)
        ) / len(sample)

        # Telemetry stats: query vs retrieved mean
        q_feats = query.to_feature_vector()[feat_idxs]
        r_feats = np.stack([ev.to_feature_vector()[feat_idxs] for ev in sample])
        r_mean  = r_feats.mean(0)

        print(f"\n  Query scenario: {SCENARIO_NAMES[sc_int]:<12}  ORS code: {code}  "
              f"N retrieved: {len(retrieved)}")
        print(f"  Query policy:  retry={RETRY_LABELS[query.label_retry]}, "
              f"cb={CIRCUIT_BREAKER_LABELS[query.label_cb]}, "
              f"fb={FALLBACK_LABELS[query.label_fallback]}")
        print(f"  Predicted:     retry={RETRY_LABELS[pred.retry]}, "
              f"cb={CIRCUIT_BREAKER_LABELS[pred.cb]}, "
              f"fb={FALLBACK_LABELS[pred.fallback]}")
        print(f"  Scenario match in retrieved: {sc_match:.0%}  |  "
              f"Policy match in retrieved: {pol_match:.0%}")
        print(f"  Telemetry comparison (query vs retrieved mean):")
        for fn, qv, rv in zip(feat_names, q_feats, r_mean):
            diff = abs(qv - rv)
            flag = "  <-- close" if diff < 0.05 else ""
            print(f"    {fn:<14} query={qv:.3f}  retrieved_mean={rv:.3f}  diff={diff:.3f}{flag}")

    print("\n" + "=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("\n[Setup] Loading workflows and generating data ...")
    graphs = load_both_workflows()
    for g in graphs:
        print(f"  {g.name[:60]:<60} nodes={len(g.nodes)}")

    # Joint training set (oversampled)
    raw_train = TemporalTraceGenerator(graphs, seed=SEED).generate_dataset(N_TRAIN)
    train_bal = _minority_oversample(raw_train)

    # Balanced test set (equal per workflow)
    n_per_wf  = N_TEST // len(graphs)
    test_wf1  = TemporalTraceGenerator([graphs[0]], seed=SEED + 999).generate_dataset(n_per_wf)
    test_wf2  = TemporalTraceGenerator([graphs[1]], seed=SEED + 999).generate_dataset(n_per_wf)
    test_all  = test_wf1 + test_wf2
    random.shuffle(test_all)

    print(f"  Train: {len(train_bal)}  |  Test: {len(test_all)}")

    # Load ORS-VQ
    print("\n[Setup] Loading ORS-VQ checkpoint ...")
    ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=CODEBOOK_SIZE)
    if not CKPT_ORS.exists():
        raise RuntimeError("ors_vq.pt not found — run ors_poc.py first.")
    d = torch.load(CKPT_ORS, weights_only=False)
    ors.load_state_dict(d["state_dict"])
    ors.eval()

    # ── Run experiments ───────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    _e5_codebook_utilization(ors, test_all)

    print("\n" + "-" * 60)
    _e6_policy_consistency(ors, test_all)

    print("\n" + "-" * 60)
    _e7_telemetry_variance(ors, test_all)

    print("\n" + "-" * 60)
    _e8_robustness(ors, test_all)

    print("\n" + "-" * 60)
    _e9_transfer(graphs)

    print("\n" + "-" * 60)
    print("\n[E10] ORS as retrieval memory ...")
    _e10_retrieval(ors, train_bal, test_all)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
