"""
regen_fig_arch_appendix.py
Elaborate top-to-bottom architecture diagram for the appendix.
Includes DeathStarBench call flows and full pipeline detail.
Canvas: 14 x 16 (reduced from 20 so caption fits on same page).

Run:
    cd C:\\Users\\agarw\\OneDrive\\Documents\\research\\codebookvae
    py regen_fig_arch_appendix.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

FIGDIR = "figures"
os.makedirs(FIGDIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_DSB   = "#E3F2FD"
C_FEAT  = "#FFF9C4"
C_ENC   = "#FFF3E0"
C_VQ    = "#FCE4EC"
C_HEAD  = "#E8F5E9"
C_OUT   = "#F3E5F5"
C_CF    = "#E0F7FA"
C_ARROW = "#37474F"
EDGE    = "#455A64"

# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 14)
ax.set_ylim(0, 16)
ax.axis("off")


def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=True,
        edgecolor=EDGE, lw=1.8, linespacing=1.5):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=edgecolor, linewidth=lw)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", linespacing=linespacing)


def arrow(ax, x0, y0, x1, y1, label="", color=C_ARROW, lw=1.8):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        ax.text((x0+x1)/2 + 0.15, (y0+y1)/2, label,
                fontsize=8, style="italic", color="#546E7A", va="center")


def slabel(ax, x, y, text):
    ax.text(x, y, text, fontsize=8.8, fontweight="bold", color="#1A237E",
            va="center",
            bbox=dict(facecolor="#E8EAF6", edgecolor="#3F51B5",
                      boxstyle="round,pad=0.18", linewidth=1.2))


# ==============================================================================
# TIER 0  —  DeathStarBench Input  (y = 13.6 – 15.7)
# ==============================================================================
slabel(ax, 0.2, 15.75, "① DeathStarBench Input")

flows = [
    ("compose_post\nworkflow\n(5 nodes)", 0.5),
    ("read_home_timeline\nworkflow\n(6 nodes)", 4.4),
    ("broken-pipe\nvariants ×2\n(4–7 nodes)", 8.3),
]
for lbl, xf in flows:
    box(ax, xf, 13.7, 3.4, 1.55, lbl, C_DSB, fontsize=8)

box(ax, 0.5, 13.0, 11.2, 0.6,
    "X-Trace spans  ·  9 workload types  ·  4 failure scenarios "
    "(HEALTHY 67% / TRANSIENT 16% / DEGRADED 5% / DOWN 12%)  ·  21,578 events",
    C_DSB, fontsize=8, bold=False)

for xf in [2.2, 6.1, 10.0]:
    arrow(ax, xf, 13.7, xf, 13.6)
arrow(ax, 6.1, 13.0, 6.1, 12.65)

# ==============================================================================
# TIER 1  —  Feature Engineering  (y = 11.0 – 12.6)
# ==============================================================================
slabel(ax, 0.2, 12.78, "② Feature Engineering  (d = 47)")

box(ax, 0.5, 11.05, 5.5, 1.45,
    "Structural features  (38 dim)\n"
    "in-degree · out-degree · node depth\n"
    "critical-path · role encoding (3)\n"
    "historical failure category (32 classes)",
    C_FEAT, fontsize=8)

box(ax, 6.5, 11.05, 5.5, 1.45,
    "Temporal features  (9 dim)\n"
    "latency mean · latency variance\n"
    "invocation rate · error rate · timeout rate\n"
    "retry rate · p95 latency · window age · health trend",
    C_FEAT, fontsize=8)

ax.text(3.25, 10.85, "computed at graph construction time",
        fontsize=7.3, ha="center", style="italic", color="#5D4037")
ax.text(9.25, 10.85, "recomputed every 10 s from rolling telemetry",
        fontsize=7.3, ha="center", style="italic", color="#5D4037")

ax.annotate("", xy=(6.1, 10.52), xytext=(3.25, 11.05),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
ax.annotate("", xy=(6.1, 10.52), xytext=(9.25, 11.05),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
ax.text(6.1, 10.35, "concatenate → x ∈ ℝ⁴⁷",
        fontsize=8.3, ha="center", fontweight="bold", color="#1A237E")

# ==============================================================================
# TIER 2  —  Encoder  (y = 9.1 – 10.2)
# ==============================================================================
slabel(ax, 0.2, 10.42, "③ Workflow-Context Encoder")
arrow(ax, 6.1, 10.3, 6.1, 10.1)

box(ax, 2.0, 9.1, 8.2, 1.0,
    "Two-layer MLP + LayerNorm\n"
    "Linear(47→64) → ReLU → LayerNorm → Linear(64→32)   |   Output: z_e ∈ ℝ³²",
    C_ENC, fontsize=8.3)

arrow(ax, 6.1, 9.1, 6.1, 8.88, label="z_e")

# ==============================================================================
# TIER 3  —  VQ Codebook  (y = 7.55 – 8.85)
# ==============================================================================
slabel(ax, 0.2, 9.02, "④ VQ Codebook  (K = 16, EMA + dead-code restart)")

box(ax, 1.0, 7.6, 12.0, 1.2,
    "k* = argmin_k ‖z_e − e_k‖²  →  z_q = e_{k*}   |   ẑ_q = z_e + sg(z_q − z_e)  (straight-through)\n"
    "EMA: N_k ← γ N_k + (1−γ) n_k,  m_k ← γ m_k + (1−γ) Σ z_e   |   Dead-code restart if N_k < τ\n"
    "Loss:  ℒ = ℒ_task + β‖z_e − sg(z_q)‖²     β=0.25,  γ=0.99,  τ=1.0   |   0% collapse vs 75% gradient-based",
    C_VQ, fontsize=8)

arrow(ax, 6.1, 7.6, 6.1, 7.38, label="z_q  (ORS code k*)")

# ==============================================================================
# TIER 4  —  Policy Heads  (y = 5.9 – 7.35)
# ==============================================================================
slabel(ax, 0.2, 7.52, "⑤ Policy Heads  (4 independent MLPs)")

heads = [
    ("Retry head\n(4 classes)\n0/1/2/3", 0.6),
    ("Timeout head\n(3 classes)\nshort/med/long", 3.8),
    ("Circuit-breaker\n(3 classes)\nclosed/half/open", 7.0),
    ("Fallback head\n(4 classes)\nnone/alt/cached\n/dead-letter", 10.2),
]
for lbl, xh in heads:
    box(ax, xh, 5.95, 3.0, 1.3, lbl, C_HEAD, fontsize=7.8)
    arrow(ax, 6.1, 7.38, xh + 1.5, 7.25)

for xh in [2.1, 5.3, 8.5, 11.7]:
    arrow(ax, xh, 5.95, xh, 5.72)

# ==============================================================================
# TIER 5  —  Policy tuple  (y = 5.05 – 5.65)
# ==============================================================================
box(ax, 2.0, 5.08, 10.0, 0.6,
    "Resilience policy tuple   π = (r, t, c, f)   ← argmax per head",
    C_OUT, fontsize=9, bold=True)

# ==============================================================================
# DUAL-USE  (y = 3.7 – 5.0)
# ==============================================================================
ax.annotate("", xy=(3.2, 4.95), xytext=(4.2, 5.08),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))
ax.annotate("", xy=(10.8, 4.95), xytext=(9.8, 5.08),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.5))

box(ax, 0.4, 3.75, 5.8, 1.1,
    "Generation-time annotation\n"
    "Structural features only (no live telemetry)\n"
    "New node → lookup archetype → attach π to graph",
    "#E8F5E9", fontsize=8)

box(ax, 7.8, 3.75, 5.8, 1.1,
    "Runtime adaptation\n"
    "Full features (structural + telemetry)\n"
    "Re-query codebook every 10 s → update π in-place",
    "#E0F7FA", fontsize=8)

ax.text(7.0, 4.3, "OR", fontsize=11, ha="center", fontweight="bold", color="#455A64")

# Training-only dashed separator
ax.plot([13.2, 13.2], [3.6, 9.0], color="#B0BEC5", lw=1.2, linestyle="--")
ax.text(13.6, 6.3, "Training\nOnly", fontsize=8, ha="center",
        color="#78909C", rotation=90, va="center")

# ==============================================================================
# TIER 6  —  Training Objective  (y = 2.15 – 3.55)
# ==============================================================================
slabel(ax, 0.2, 3.65, "⑥ Training Objective")
arrow(ax, 7.0, 3.75, 7.0, 3.58)

box(ax, 1.5, 2.2, 11.0, 1.25,
    "ℒ_total  =  ℒ_task  +  β · ‖z_e − sg(z_q)‖²\n"
    "ℒ_task  =  Σ  CrossEntropy(head_i logits,  oracle label_i)     (4 heads)\n"
    "Codebook updated via EMA only — no gradient flows into codebook entries",
    C_CF, fontsize=8.2)

# ==============================================================================
# TIER 7  —  Counterfactual Oracle  (y = 0.3 – 2.1)
# ==============================================================================
slabel(ax, 0.2, 2.28, "⑦ Counterfactual Oracle  (label generation, no human annotation)")
arrow(ax, 7.0, 2.2, 7.0, 2.05)

box(ax, 0.5, 0.35, 13.0, 1.6,
    "For each event x_i:  sample 10 candidate policies  →  evaluate each over S=150 Monte Carlo sims\n"
    "Score:  score(π) = C − λ_w W − λ_l L      (λ_w=0.4,  λ_l=0.05)\n"
    "C = completion indicator   W = wasted retry calls   L = normalised latency\n"
    "Best-scoring policy  →  supervised training label   |   oracle agreement: 97.6% on 500 test events",
    C_CF, fontsize=8.2)

# ==============================================================================
# Title
# ==============================================================================
ax.text(7.0, 15.88, "ORS-VQ: Full System Architecture",
        fontsize=12, fontweight="bold", ha="center", va="center", color="#0D47A1")

# ==============================================================================
# Legend
# ==============================================================================
legend_items = [
    mpatches.Patch(facecolor=C_DSB,  edgecolor=EDGE, label="DeathStarBench input"),
    mpatches.Patch(facecolor=C_FEAT, edgecolor=EDGE, label="Feature engineering"),
    mpatches.Patch(facecolor=C_ENC,  edgecolor=EDGE, label="Encoder"),
    mpatches.Patch(facecolor=C_VQ,   edgecolor=EDGE, label="VQ Codebook"),
    mpatches.Patch(facecolor=C_HEAD, edgecolor=EDGE, label="Policy heads"),
    mpatches.Patch(facecolor=C_OUT,  edgecolor=EDGE, label="Policy output / runtime"),
    mpatches.Patch(facecolor=C_CF,   edgecolor=EDGE, label="Training / oracle"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.5,
          framealpha=0.9, edgecolor="#B0BEC5",
          bbox_to_anchor=(0.99, 0.01))

path = f"{FIGDIR}/fig_arch_appendix.png"
plt.tight_layout()
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {path}")
