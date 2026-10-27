"""
diag_abort_reason.py  --  Why does ORS-Hardened have 2.5% ABORTED?

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py -m diag_abort_reason
"""
import torch
from resiliency_poc.dsb_workflow_loader import load_dsb_healthy_and_failure
from resiliency_poc.model.ors_model import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM
from hardening.engine import WorkflowHardeningEngine, _TYPE_TO_SCENARIO
from resiliency_poc.scenario import ScenarioState as _SC

_TYPE_TO_SCENARIO.update({
    'nginx-thrift': _SC.HEALTHY, 'ComposePostService': _SC.TRANSIENT_FAILURE,
    'PostStorageService': _SC.DEGRADED, 'HomeTimelineService': _SC.DEGRADED,
    'UserTimelineService': _SC.DEGRADED, 'WriteHomeTimelineService': _SC.DEGRADED,
    'UserService': _SC.DEGRADED, 'SocialGraphService': _SC.TRANSIENT_FAILURE,
    'UniqueIdService': _SC.HEALTHY, 'TextService': _SC.HEALTHY,
    'MediaService': _SC.TRANSIENT_FAILURE, 'UserMentionService': _SC.HEALTHY,
    'UrlShortenService': _SC.HEALTHY,
})

CB_LBL    = ['closed', 'half_open', 'open']
RETRY_LBL = ['none', '1x', '3x', '5x']
FB_LBL    = ['none', 'alternate', 'cached', 'dead_letter']
FB_PROB   = {0: 0.00, 1: 0.85, 2: 0.95, 3: 1.00}

ors = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
d   = torch.load('resiliency_poc/checkpoints/ors_dsb_vq.pt', weights_only=False)
ors.load_state_dict(d['state_dict'])
ors.eval()
engine = WorkflowHardeningEngine(ors_model=ors, seed=42)
graphs = load_dsb_healthy_and_failure()

print("\n  WHY DOES ORS HAVE 2.5% ABORTED?")
print("  " + "=" * 100)
print("  A node can ABORT the workflow if ALL of: retries fail AND the fallback also fails.")
print("  dead_letter fallback (type 3) has 100% success  ->  never aborts.")
print("  cached fallback      (type 2) has  95% success  ->  5% chance of aborting.")
print("  ORS assigns CIRCUIT_OPEN (cb=open, fallback=dead_letter) to DEGRADED nodes")
print("  -- those never abort. But nodes with HEALTHY/TRANSIENT risk get CAUTIOUS_PROBE")
print("  (cb=half_open, fallback=cached, 95% success) -- those can abort on the 5% miss.\n")

for g in graphs:
    hw = engine.analyze(g)
    print(f'  Graph: {g.name}')
    print(f'  {"Node":<30} {"Risk":<18} {"CB":<11} {"Retry":<6} {"Fallback":<13} {"FbSucc%":<9} {"CanAbort"}')
    print('  ' + '-' * 105)

    visited, queue, order = set(), list(g.entry_nodes), []
    while queue:
        n = queue.pop(0)
        if n in visited:
            continue
        visited.add(n)
        order.append(n)
        queue.extend(g.adj.get(n, []))

    any_abort_risk = False
    for nname in order:
        ann   = hw.nodes[nname].annotation
        fb_p  = FB_PROB[ann.fallback_type]
        # CB=open always jumps to fallback, and dead_letter (type 3) always succeeds
        # So abort is only possible when CB != open AND fb_p < 1.0
        can_abort = ann.cb_policy != 2 and fb_p < 1.0
        if can_abort:
            any_abort_risk = True
        flag = f'  YES -- fallback fails {(1-fb_p)*100:.0f}% of the time' if can_abort else ''
        print(f'  {nname:<30} {ann.risk_scenario:<18} {CB_LBL[ann.cb_policy]:<11} '
              f'{RETRY_LBL[ann.retry]:<6} {FB_LBL[ann.fallback_type]:<13} '
              f'{fb_p*100:<9.0f}%{flag}')

    if not any_abort_risk:
        print('  (no nodes with abort risk in this graph)')
    print()

print("  SUMMARY")
print("  " + "=" * 60)
print("  Nodes with DEGRADED/DOWN risk  ->  CIRCUIT_OPEN policy")
print("    CB=open  -> skip primary entirely, go straight to dead_letter")
print("    dead_letter success = 100%  ->  NEVER aborts")
print()
print("  Nodes with HEALTHY/TRANSIENT risk  ->  CAUTIOUS_PROBE or STORM_RETRY")
print("    CB=half_open or closed, fallback=cached (95% success)")
print("    Under DEGRADED/DOWN simulation scenario:")
print("      - all retries fail (fail_prob = 0.70-0.97)")
print("      - cached fallback then has 5% chance of also failing")
print("      ->  5% * (fraction of runs reaching such a node) = the 2.5% ABORTED")
print()
print("  Fix if desired: assign fallback=dead_letter to CAUTIOUS_PROBE/STORM_RETRY")
print("  nodes too -- but that would hide failures rather than surface them.")
