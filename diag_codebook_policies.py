"""Query every codebook entry and print its policy."""
import torch
from resiliency_poc.model.ors_model import ORSVQModel
from resiliency_poc.temporal_trace_generator import TEMPORAL_FEATURE_DIM

RETRY    = ["none","1x","3x","5x"]
TIMEOUT  = ["5s","30s","120s"]
CB       = ["closed","half_open","open"]
FALLBACK = ["none","alternate","cached","dead_letter"]

model = ORSVQModel(feature_dim=TEMPORAL_FEATURE_DIM, codebook_size=16)
d = torch.load("resiliency_poc/checkpoints/ors_dsb_vq.pt", weights_only=False)
model.load_state_dict(d["state_dict"])
model.eval()

cb_vecs = model.codebook.embeddings.detach()
print(f"\n  {'Code':<8}  {'CB':<12}  {'Retry':<6}  {'Timeout':<8}  {'Fallback':<12}")
print("  " + "-"*54)
with torch.no_grad():
    for i in range(16):
        z = cb_vecs[i].unsqueeze(0)
        p = model.heads.predict(z)
        print(f"  ORS-{i:<4}  {CB[p.cb]:<12}  {RETRY[p.retry]:<6}  {TIMEOUT[p.timeout]:<8}  {FALLBACK[p.fallback]:<12}")
