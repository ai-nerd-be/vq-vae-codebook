"""
evaluator.py

Four-way evaluation: Threshold vs MLP vs VQ-EMA vs Factorized-VQ.

Metrics
───────
  completion_rate      fraction of traces that complete without unrecovered failure
  wasted_retry_rate    wasted retries / total retry attempts across all traces
  avg_latency          mean latency units per trace
  policy_validity_rate fraction of policies with no internal contradictions
  codebook_perplexity  effective entries used (VQ models only; higher = better)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .trace_generator import FailureEvent
from .label_oracle import (
    RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS,
)
from .model.policy_heads import PolicyPrediction
from .model.vqvae_model  import (
    MLPResiliencyModel, VQResiliencyModel, FactorizedVQResiliencyModel,
    TemporalVQResiliencyModel,
)
from .simulator import WorkflowSimulator, ExecutionResult, threshold_policy
from .counterfactual_labeler import _SCENARIO_BASE_SUCCESS, _TIMEOUT_BONUS, _FALLBACK_PROBS


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class PolicyMetrics:
    name:                 str
    completion_rate:      float
    wasted_retry_rate:    float
    avg_latency:          float
    policy_validity_rate: float
    codebook_perplexity:  Optional[str] = None

    def row(self) -> List[str]:
        perp = self.codebook_perplexity if self.codebook_perplexity else "-"
        return [
            self.name,
            f"{self.completion_rate*100:.1f}%",
            f"{self.wasted_retry_rate*100:.1f}%",
            f"{self.avg_latency:.2f}",
            f"{self.policy_validity_rate*100:.1f}%",
            perp,
        ]


@dataclass
class EvaluationReport:
    n_traces: int
    n_events: int
    metrics:  List[PolicyMetrics] = field(default_factory=list)

    def print(self) -> None:
        headers = [
            "Policy", "Completion(+)", "Wasted Retries(-)",
            "Avg Latency(-)", "Valid Policies(+)", "CB Perplexity(+)"
        ]
        widths = [26, 14, 18, 14, 16, 20]

        sep = "+" + "+".join("-" * w for w in widths) + "+"
        print(f"\n{'='*len(sep)}")
        print(f"  Evaluation -- {self.n_traces} traces, {self.n_events} failure events")
        print(f"{'='*len(sep)}")
        print(sep)
        print("|" + "|".join(h.center(w) for h, w in zip(headers, widths)) + "|")
        print(sep)
        for m in self.metrics:
            print("|" + "|".join(v.center(w) for v, w in zip(m.row(), widths)) + "|")
        print(sep)
        print()

        # Delta vs threshold
        if len(self.metrics) >= 2:
            base = self.metrics[0]
            for m in self.metrics[1:]:
                dc = (m.completion_rate - base.completion_rate) * 100
                dw = (base.wasted_retry_rate - m.wasted_retry_rate) * 100
                dv = (m.policy_validity_rate - base.policy_validity_rate) * 100
                sc = "+" if dc >= 0 else ""
                sw = "+" if dw >= 0 else ""
                sv = "+" if dv >= 0 else ""
                print(
                    f"  {m.name} vs {base.name}: "
                    f"completion {sc}{dc:.1f}pp | "
                    f"wasted retries {sw}{dw:.1f}pp | "
                    f"validity {sv}{dv:.1f}pp"
                )
        print()


# ── Model-wrapping policy functions ──────────────────────────────────────────

def _wrap(model) -> Callable[[FailureEvent], PolicyPrediction]:
    model.eval()
    def _fn(event: FailureEvent) -> PolicyPrediction:
        x = torch.tensor(event.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        return model.predict(x)
    return _fn


# ── Core evaluation runner ────────────────────────────────────────────────────

def _evaluate_policy(
    name:       str,
    policy_fn:  Callable[[FailureEvent], PolicyPrediction],
    traces:     List[List[FailureEvent]],
    sim:        WorkflowSimulator,
    perp_str:   Optional[str] = None,
) -> PolicyMetrics:
    results: List[ExecutionResult] = [sim.run_trace(ev, policy_fn) for ev in traces]

    completion_rate = float(np.mean([r.completed for r in results]))

    # Retries = extra attempts beyond the first (0 for CB-open events)
    total_retries = sum(max(0, o.attempts - 1) for r in results for o in r.events)
    total_wasted  = sum(r.wasted_retries for r in results)
    wasted_rate   = float(total_wasted / max(total_retries, 1))

    avg_latency   = float(np.mean([r.total_latency for r in results]))

    all_outcomes  = [o for r in results for o in r.events]
    invalid       = sum(1 for o in all_outcomes if not o.policy.is_valid())
    validity_rate = 1.0 - invalid / max(len(all_outcomes), 1)

    return PolicyMetrics(
        name=name,
        completion_rate=completion_rate,
        wasted_retry_rate=wasted_rate,
        avg_latency=avg_latency,
        policy_validity_rate=float(validity_rate),
        codebook_perplexity=perp_str,
    )


# ── Main evaluate entry point ─────────────────────────────────────────────────

def evaluate(
    test_events:      List[FailureEvent],
    mlp_model:        MLPResiliencyModel,
    vq_model:         VQResiliencyModel,
    fact_model:       Optional[FactorizedVQResiliencyModel] = None,
    traces_per_eval:  int = 200,
    seed:             int = 7,
) -> EvaluationReport:
    sim   = WorkflowSimulator(seed=seed)
    chunk = max(1, len(test_events) // traces_per_eval)
    traces = [
        test_events[i: i + chunk]
        for i in range(0, len(test_events), chunk)
        if test_events[i: i + chunk]
    ][:traces_per_eval]

    failure_events = [e for e in test_events if e.failure_mode != "no_failure"]

    vq_perp_str = f"{vq_model.codebook.perplexity():.1f}/{vq_model.codebook.K}"

    metrics = [
        _evaluate_policy("Threshold (baseline)", threshold_policy, traces, sim),
        _evaluate_policy("MLP (Phase 1)",         _wrap(mlp_model), traces, sim),
        _evaluate_policy("VQ-EMA (Phase 2)",       _wrap(vq_model),  traces, sim,
                         perp_str=vq_perp_str),
    ]

    if fact_model is not None:
        perps = fact_model.codebooks.perplexities()
        dims  = fact_model.codebooks.policy_dims
        perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(perps, dims))
        metrics.append(
            _evaluate_policy("Factorized-VQ (Phase 3)", _wrap(fact_model), traces, sim,
                             perp_str=perp_str)
        )

    return EvaluationReport(n_traces=len(traces), n_events=len(failure_events),
                            metrics=metrics)


# ── Scenario-aware simulator (shared by DSB and synthetic evaluations) ────────

class _ScenarioSimulator(WorkflowSimulator):
    """WorkflowSimulator subclass that uses per-event scenario probabilities."""

    def _run_event(self, event: FailureEvent, policy: PolicyPrediction):
        from .simulator import EventOutcome
        scenario = getattr(event, "_dsb_scenario", None)
        if scenario is None or event.failure_mode == "no_failure":
            return super()._run_event(event, policy)

        base = _SCENARIO_BASE_SUCCESS.get(scenario, 0.5)

        if policy.cb == 2:
            fb_prob = _FALLBACK_PROBS[policy.fallback]
            fb_ok   = self.rng.random() < fb_prob
            return EventOutcome(
                node_name=event.node_name, failure_mode=event.failure_mode,
                policy=policy, attempts=0, succeeded=fb_ok,
                via_fallback=True, was_wasted=False,
                is_partial=(policy.fallback == 3 and fb_ok),
                latency_units=3,
            )

        max_attempts = min(policy.retry + 1, 2 if policy.cb == 1 else policy.retry + 1)
        timeout_bonus = _TIMEOUT_BONUS.get(policy.timeout, 0.0)
        succeeded = False
        attempts  = 0
        for a in range(max_attempts):
            attempts += 1
            prob = min(1.0, base + timeout_bonus + 0.07 * a)
            if self.rng.random() < prob:
                succeeded = True
                break

        was_wasted = (attempts > 1 and not succeeded)
        via_fallback = False
        is_partial   = False
        if not succeeded:
            fb_prob  = _FALLBACK_PROBS[policy.fallback]
            succeeded = self.rng.random() < fb_prob
            is_partial = (policy.fallback == 3 and succeeded)
            via_fallback = True

        return EventOutcome(
            node_name=event.node_name, failure_mode=event.failure_mode,
            policy=policy, attempts=attempts, succeeded=succeeded,
            via_fallback=via_fallback, was_wasted=was_wasted,
            is_partial=is_partial,
            latency_units=attempts + (2 if via_fallback else 0),
        )


# ── Apples-to-apples: all models on the same synthetic temporal test set ──────

def evaluate_all_on_temporal(
    temporal_test_events: list,              # List[TemporalFailureEvent]
    mlp_model:            MLPResiliencyModel,
    vq_model:             VQResiliencyModel,
    fact_model:           FactorizedVQResiliencyModel,
    temp_vq_model:        TemporalVQResiliencyModel,
    seed:                 int = 7,
) -> EvaluationReport:
    """
    Fair simulation comparison: all models run on the same temporal test set
    under identical scenario-aware success probabilities.

    Phases 1-3 receive the 38-dim static features they were trained on.
    Phase 4 receives the full 47-dim temporal features.
    All share the same DSB-calibrated simulator.
    """
    sim = _ScenarioSimulator(seed=seed)

    failure_events = []
    for te in temporal_test_events:
        fe = te.to_failure_event()
        fe._dsb_scenario   = te.scenario
        fe._temporal_feats = te.to_feature_vector()   # 47-dim (Phase 4)
        fe._static_feats   = te.static_features        # 38-dim (Phases 1-3)
        failure_events.append(fe)

    traces = [[fe] for fe in failure_events]

    def _wrap_static(model):
        model.eval()
        def _fn(ev):
            x = torch.tensor(ev._static_feats, dtype=torch.float32).unsqueeze(0)
            return model.predict(x)
        return _fn

    def _wrap_temporal(model):
        model.eval()
        def _fn(ev):
            x = torch.tensor(ev._temporal_feats, dtype=torch.float32).unsqueeze(0)
            return model.predict(x)
        return _fn

    vq_perp_str = f"{vq_model.codebook.perplexity():.1f}/{vq_model.codebook.K}"
    fact_perps  = fact_model.codebooks.perplexities()
    fact_dims   = fact_model.codebooks.policy_dims
    fact_perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(fact_perps, fact_dims))
    temp_perps  = temp_vq_model.codebooks.perplexities()
    temp_dims   = temp_vq_model.codebooks.policy_dims
    temp_perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(temp_perps, temp_dims))

    metrics = [
        _evaluate_policy("Threshold (baseline)",          threshold_policy,              traces, sim),
        _evaluate_policy("MLP (Phase 1, oracle)",         _wrap_static(mlp_model),       traces, sim),
        _evaluate_policy("VQ-EMA (Phase 2, oracle)",      _wrap_static(vq_model),        traces, sim,
                         perp_str=vq_perp_str),
        _evaluate_policy("Fact-VQ (Phase 3, oracle)",     _wrap_static(fact_model),      traces, sim,
                         perp_str=fact_perp_str),
        _evaluate_policy("Temp-VQ (Phase 4, oracle-free)", _wrap_temporal(temp_vq_model), traces, sim,
                         perp_str=temp_perp_str),
    ]

    n_failures = len([fe for fe in failure_events if fe.failure_mode != "no_failure"])
    return EvaluationReport(n_traces=len(traces), n_events=n_failures, metrics=metrics)


# ── DSB evaluation (scenario-aware simulation) ───────────────────────────────

def evaluate_dsb(
    test_events:      list,   # List[TemporalFailureEvent]
    dsb_model:        TemporalVQResiliencyModel,
    seed:             int = 7,
) -> EvaluationReport:
    """
    Simulation evaluation for Phase 4b (DeathStarBench real traces).

    Each TemporalFailureEvent is wrapped as a FailureEvent for the simulator.
    Three policies compared:
      - Threshold (always retry=3, medium timeout, CB closed)
      - DSB-VQ    (trained on real DSB traces, oracle-free)

    The simulator success probabilities are scenario-aware via the DSB
    counterfactual labeler's _SCENARIO_BASE_SUCCESS table.
    """
    from .scenario import ScenarioState

    dsb_sim = _ScenarioSimulator(seed=seed)

    # Build traces: group events into single-event traces (DSB = one request)
    # Attach scenario + pre-computed 44-dim feature vec to each FailureEvent
    failure_events = []
    for te in test_events:
        fe = te.to_failure_event()
        fe._dsb_scenario    = te.scenario
        fe._temporal_feats  = te.to_feature_vector()   # 44-dim — avoids FailureEvent.to_feature_vector
        failure_events.append(fe)

    # Wrap each event as a 1-event trace for run_trace
    traces = [[fe] for fe in failure_events]

    def _wrap_dsb(model):
        model.eval()
        def _fn(event):
            x = torch.tensor(event._temporal_feats, dtype=torch.float32).unsqueeze(0)
            return model.predict(x)
        return _fn

    perps = dsb_model.codebooks.perplexities()
    dims  = dsb_model.codebooks.policy_dims
    perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(perps, dims))

    metrics = [
        _evaluate_policy("Threshold (baseline)",       threshold_policy,      traces, dsb_sim),
        _evaluate_policy("DSB-VQ (real, oracle-free)", _wrap_dsb(dsb_model),  traces, dsb_sim,
                         perp_str=perp_str),
    ]

    # Per-scenario breakdown for DSB-VQ
    _print_per_scenario_metrics(test_events, dsb_model, dsb_sim)

    return EvaluationReport(
        n_traces=len(traces),
        n_events=len([e for e in failure_events if e.failure_mode != "no_failure"]),
        metrics=metrics,
    )


def _print_per_scenario_metrics(test_events: list, model, sim) -> None:
    """Break down DSB-VQ accuracy and completion rate by scenario."""
    from collections import defaultdict
    from .scenario import ScenarioState, SCENARIO_NAMES

    by_scenario = defaultdict(list)
    for te in test_events:
        by_scenario[te.scenario].append(te)

    print("\n  Per-scenario breakdown (DSB-VQ):")
    print(f"  {'Scenario':<20} {'N':>5}  {'CB acc':>7}  {'Retry acc':>10}  {'Completion':>11}")
    print("  " + "-" * 60)

    model.eval()
    for scenario in ScenarioState:
        evs = by_scenario[scenario]
        if not evs:
            continue
        correct_cb    = 0
        correct_retry = 0
        completed     = 0
        for te in evs:
            x    = torch.tensor(te.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
            pred = model.predict(x)
            if pred.cb == te.label_cb:
                correct_cb += 1
            if pred.retry == te.label_retry:
                correct_retry += 1
            fe = te.to_failure_event()
            fe._dsb_scenario   = te.scenario
            fe._temporal_feats = te.to_feature_vector()
            def _dsb_pred(event, _m=model):
                x = torch.tensor(event._temporal_feats, dtype=torch.float32).unsqueeze(0)
                return _m.predict(x)
            result = sim.run_trace([fe], _dsb_pred)
            if result.completed:
                completed += 1

        n = len(evs)
        print(f"  {SCENARIO_NAMES[scenario]:<20} {n:>5}  "
              f"{correct_cb/n*100:>6.1f}%  "
              f"{correct_retry/n*100:>9.1f}%  "
              f"{completed/n*100:>10.1f}%")
    print()


# ── Option B: all models trained and tested on DSB data ──────────────────────

def evaluate_all_on_dsb(
    temporal_test_events: list,              # List[TemporalFailureEvent]
    mlp_model:            MLPResiliencyModel,
    vq_model:             VQResiliencyModel,
    fact_model:           FactorizedVQResiliencyModel,
    temp_vq_model:        TemporalVQResiliencyModel,
    seed:                 int = 7,
) -> EvaluationReport:
    """
    All models use 47-dim temporal features; all tested on the same DSB test set.
    Phases 1-3 were retrained on DSB with oracle labels; Phase 4b with counterfactual.
    """
    sim = _ScenarioSimulator(seed=seed)

    failure_events = []
    for te in temporal_test_events:
        fe = te.to_failure_event()
        fe._dsb_scenario   = te.scenario
        fe._temporal_feats = te.to_feature_vector()   # 47-dim for all models
        failure_events.append(fe)

    traces = [[fe] for fe in failure_events]

    def _wrap_temporal(model):
        model.eval()
        def _fn(ev):
            x = torch.tensor(ev._temporal_feats, dtype=torch.float32).unsqueeze(0)
            return model.predict(x)
        return _fn

    vq_perp_str = f"{vq_model.codebook.perplexity():.1f}/{vq_model.codebook.K}"
    fact_perps  = fact_model.codebooks.perplexities()
    fact_dims   = fact_model.codebooks.policy_dims
    fact_perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(fact_perps, fact_dims))
    temp_perps  = temp_vq_model.codebooks.perplexities()
    temp_dims   = temp_vq_model.codebooks.policy_dims
    temp_perp_str = " / ".join(f"{p:.1f}({k})" for p, k in zip(temp_perps, temp_dims))

    metrics = [
        _evaluate_policy("Threshold (baseline)",           threshold_policy,               traces, sim),
        _evaluate_policy("MLP (Phase 1, oracle, DSB)",     _wrap_temporal(mlp_model),      traces, sim),
        _evaluate_policy("VQ-EMA (Phase 2, oracle, DSB)",  _wrap_temporal(vq_model),       traces, sim,
                         perp_str=vq_perp_str),
        _evaluate_policy("Fact-VQ (Phase 3, oracle, DSB)", _wrap_temporal(fact_model),     traces, sim,
                         perp_str=fact_perp_str),
        _evaluate_policy("DSB-VQ (Phase 4b, cf, DSB)",     _wrap_temporal(temp_vq_model),  traces, sim,
                         perp_str=temp_perp_str),
    ]

    n_failures = len([fe for fe in failure_events if fe.failure_mode != "no_failure"])
    return EvaluationReport(n_traces=len(traces), n_events=n_failures, metrics=metrics)


# ── Shared VQ prototype analysis ──────────────────────────────────────────────

def codebook_profile(
    vq_model:    VQResiliencyModel,
    test_events: List[FailureEvent],
) -> None:
    """Show what each shared codebook entry represents."""
    from collections import defaultdict, Counter
    entry_stats: Dict[int, list] = defaultdict(list)

    vq_model.eval()
    for event in test_events:
        x    = torch.tensor(event.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        code = vq_model.code_for(x)
        pred = vq_model.predict(x)
        entry_stats[code].append((event.failure_mode, pred))

    print("-- VQ-EMA Shared Codebook Prototypes ------------------------------------")
    print(f"  Active entries: {len(entry_stats)} / {vq_model.codebook.K}")
    print(f"  {'Entry':<6} {'Count':>6}  {'Top failure modes':<32} Dominant policy")
    print("  " + "-" * 78)
    for k in sorted(entry_stats.keys()):
        items     = entry_stats[k]
        fm_cnts   = Counter(fm for fm, _ in items)
        top_fm    = ", ".join(f"{fm}({c})" for fm, c in fm_cnts.most_common(2))
        top_p, _  = Counter(p.as_tuple() for _, p in items).most_common(1)[0]
        r, t, c, f = top_p
        ps = (f"retry={RETRY_LABELS[r]} "
              f"to={TIMEOUT_LABELS[t]} "
              f"cb={CIRCUIT_BREAKER_LABELS[c]} "
              f"fb={FALLBACK_LABELS[f]}")
        print(f"  e{k:<5} {len(items):>6}  {top_fm:<32} {ps}")
    print()


# ── Factorized VQ per-dimension analysis ─────────────────────────────────────

def factorized_codebook_profile(
    fact_model:  FactorizedVQResiliencyModel,
    test_events: List[FailureEvent],
) -> None:
    """
    For each policy dimension, show which codebook entries fire and whether
    each entry cleanly maps to a single output class (interpretability check).
    """
    from collections import defaultdict, Counter

    dim_names  = FactorizedVQResiliencyModel.__mro__  # just for import; unused below
    label_maps = [RETRY_LABELS, TIMEOUT_LABELS, CIRCUIT_BREAKER_LABELS, FALLBACK_LABELS]
    dim_labels = ["retry", "timeout", "CB", "fallback"]

    # Collect per-dimension (code_idx, predicted_class) pairs
    dim_stats: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(4)]

    fact_model.eval()
    for event in test_events:
        x     = torch.tensor(event.to_feature_vector(), dtype=torch.float32).unsqueeze(0)
        codes = fact_model.codes_for(x)
        pred  = fact_model.predict(x)
        classes = pred.as_tuple()
        for dim_i, (code, cls) in enumerate(zip(codes, classes)):
            dim_stats[dim_i][code].append(cls)

    print("-- Factorized VQ Codebook Analysis (per dimension) ----------------------")
    perps = fact_model.codebooks.perplexities()
    dims  = fact_model.codebooks.policy_dims
    for dim_i, (dname, lmap, stats, perp, K) in enumerate(
            zip(dim_labels, label_maps, dim_stats, perps, dims)):
        print(f"\n  [{dname}]  K={K}  perplexity={perp:.2f}/{K}")
        print(f"    {'Entry':<6} {'Count':>6}  {'Dominant class':<18} Purity")
        print("    " + "-" * 48)
        for k in sorted(stats.keys()):
            cls_list  = stats[k]
            cls_cnts  = Counter(cls_list)
            top_cls, top_cnt = cls_cnts.most_common(1)[0]
            purity    = top_cnt / len(cls_list)
            print(f"    e{k:<5} {len(cls_list):>6}  "
                  f"{lmap[top_cls]:<18} {purity*100:.1f}%")
    print()
