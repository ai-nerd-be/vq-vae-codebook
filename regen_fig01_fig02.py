"""
regen_fig01_fig02.py  --  Regenerate architecture and pipeline figures
without implementation-specific details (no 47-dim, K=16, 150 sims, etc.)

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py regen_fig01_fig02.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)
SAVEKW = dict(dpi=150, bbox_inches="tight", facecolor="white")

def save(name):
    path = f"{FIGDIR}/{name}.png"
    plt.savefig(path, **SAVEKW)
    plt.close()
    print(f"  saved {path}")


# ==============================================================================
# FIG 1  —  Architecture (generic, no implementation numbers)
# ==============================================================================
print("[1] Architecture diagram ...")

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")

# ── Main pipeline boxes ────────────────────────────────────────────────────────
boxes = [
    # (x, y, w, h, label, colour)
    (0.2,  1.3, 2.2, 1.6,
     "Agent Execution\nEvent\n\n$n$-dim feature vector\n(structural + telemetry)",
     "#E3F2FD"),

    (3.0,  1.3, 2.2, 1.6,
     "Workflow-Context\nEncoder\n\nMLP + LayerNorm\n$\mathbb{R}^n \\to \mathbb{R}^h$",
     "#FFF9C4"),

    (5.8,  1.3, 2.2, 1.6,
     "VQ Codebook\n(EMA + dead-code\nrestart)\n$K$ archetypes",
     "#FCE4EC"),

    (8.6,  1.3, 2.2, 1.6,
     "Policy Heads\n\nOne head per\nresilience dimension",
     "#E8F5E9"),
]

for (x, y, w, h, txt, col) in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                          facecolor=col, edgecolor="#555", linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, txt,
            ha="center", va="center", fontsize=8.5, fontweight="bold",
            linespacing=1.4)

# ── Arrows between main boxes ─────────────────────────────────────────────────
arrow_kw = dict(arrowstyle="-|>", color="#333", lw=2.0)
gap = 0.12
for (x, w) in [(2.2+gap, 2.2), (5.0+gap, 5.8), (7.8+gap, 8.6)]:
    x_end = x + 0.4
    ax.annotate("", xy=(x_end, 2.1), xytext=(x, 2.1),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0))

# Latent labels above arrows
ax.text(2.75, 2.45, r"$z_e$", fontsize=10, ha="center", color="#555", style="italic")
ax.text(5.55, 2.45, r"$z_q$", fontsize=10, ha="center", color="#555", style="italic")

# ── Policy output branches ────────────────────────────────────────────────────
outputs = [
    "Retry policy",
    "Timeout tier",
    "Circuit-breaker\nstate",
    "Fallback\nstrategy",
]
x_head_right = 10.8
y_center = 2.1
y_positions = [3.55, 2.85, 1.85, 1.15]

ax.annotate("", xy=(x_head_right, y_center), xytext=(8.6 + 2.2 + gap, y_center),
            arrowprops=dict(arrowstyle="-", color="#333", lw=1.5))

for yp, lbl in zip(y_positions, outputs):
    ax.annotate("", xy=(x_head_right + 0.55, yp), xytext=(x_head_right, y_center),
                arrowprops=dict(arrowstyle="-|>", color="#2196F3", lw=1.4,
                                connectionstyle="arc3,rad=0.0"))
    ax.text(x_head_right + 0.65, yp, lbl,
            fontsize=8, va="center", color="#1565C0", fontweight="bold")

# ── Footnote ──────────────────────────────────────────────────────────────────
ax.text(5.55, 0.55,
        "Straight-through estimator — gradient bypasses quantisation during training",
        fontsize=8, style="italic", color="#888", ha="center")

ax.set_title("Figure 1: ORS-VQ Architecture", fontsize=11, fontweight="bold", pad=6)
save("fig01_architecture")


# ==============================================================================
# FIG 2  —  Counterfactual labelling pipeline (generic)
# ==============================================================================
print("[2] Counterfactual labelling pipeline ...")

fig, ax = plt.subplots(figsize=(13, 3.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 3.8); ax.axis("off")

steps = [
    # (x, y, w, h, label, colour)
    (0.2,  1.1, 2.0, 1.6,
     "Agent\nExecution\nEvent\n(trace)",
     "#E3F2FD"),

    (2.8,  1.1, 2.0, 1.6,
     "Candidate\nPolicy\nSampling\n(policy space)",
     "#FFF9C4"),

    (5.4,  1.1, 2.0, 1.6,
     "Monte Carlo\nSimulation\n(calibrated\nworkflow sim)",
     "#FCE4EC"),

    (8.0,  1.1, 2.4, 1.6,
     "Composite\nScoring\n$C - \\lambda_w W - \\lambda_l L$",
     "#F3E5F5"),

    (11.0, 1.1, 1.8, 1.6,
     "Best Policy\n→ Training\nLabel\n(oracle)",
     "#E8F5E9"),
]

for (x, y, w, h, txt, col) in steps:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10",
                          facecolor=col, edgecolor="#555", linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, txt,
            ha="center", va="center", fontsize=8.8, fontweight="bold",
            linespacing=1.4)

# Arrows
for (x_tail, x_head) in [(2.2, 2.75), (4.8, 5.35), (7.4, 7.95), (10.4, 10.95)]:
    ax.annotate("", xy=(x_head, 1.9), xytext=(x_tail, 1.9),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0))

# Score variable legend below figure
ax.text(6.5, 0.55,
        r"$C$ = expected completion   $W$ = expected wasted retries   "
        r"$L$ = normalised latency   |   no human annotation required",
        fontsize=8, ha="center", style="italic", color="#555")

ax.set_title("Figure 2: Counterfactual Label Generation Pipeline",
             fontsize=11, fontweight="bold", pad=6)
save("fig02_cf_pipeline")

print("Done.")
