"""
regen_fig10.py -- Regenerate fig10_incident_timeline without the figure title.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGDIR = "figures"

# Timeline data
services   = ["PostStorage", "HomeTimeline", "UserService", "nginx-thrift"]
colors_map = {
    "ORS-3":  "#9C27B0",
    "ORS-14": "#AB47BC",
    "ORS-10": "#EF5350",
    "ORS-2":  "#EF9A9A",
    "ORS-10b":"#FFA726",   # UserService ORS-10 (same code, different service)
    "ORS-3b": "#42A5F5",   # nginx-thrift ORS-3
}

# (service, start, end, label, color)
segments = [
    ("PostStorage",   0,   30,  "ORS-3",  "#AB47BC"),
    ("PostStorage",  30,  150,  "ORS-14", "#9C27B0"),
    ("PostStorage", 150,  185,  "ORS-3",  "#AB47BC"),

    ("HomeTimeline",  0,   50,  "ORS-10", "#EF5350"),
    ("HomeTimeline", 50,  165,  "ORS-2",  "#EF9A9A"),
    ("HomeTimeline",165,  185,  "ORS-10", "#EF5350"),

    ("UserService",   0,  185,  "ORS-10", "#FFA726"),

    ("nginx-thrift",  0,  185,  "ORS-3",  "#42A5F5"),
]

# Dashed event lines (x positions, labels)
events = [
    (30,  "PostStorage\nfails"),
    (50,  "HomeTimeline\nfails"),
    (150, "PostStorage\nrecovers"),
    (165, "HomeTimeline\nrecovers"),
]

fig, ax = plt.subplots(figsize=(13, 4.2))

y_pos = {s: i for i, s in enumerate(services)}

for (svc, t0, t1, label, col) in segments:
    y = y_pos[svc]
    ax.barh(y, t1 - t0, left=t0, height=0.6, color=col, align="center")
    ax.text((t0 + t1) / 2, y, label,
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")

# Event dashed lines + top annotations
for x, lbl in events:
    ax.axvline(x, color="#444", linestyle="--", linewidth=1.2, zorder=5)
    ax.text(x, len(services) - 0.1, lbl,
            ha="center", va="bottom", fontsize=7.5, color="#333")

ax.set_yticks(range(len(services)))
ax.set_yticklabels(services, fontsize=11)
ax.set_xlabel("Time (seconds)", fontsize=11)
ax.set_xlim(0, 185)
ax.set_ylim(-0.5, len(services) - 0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
path = f"{FIGDIR}/fig10_incident_timeline.png"
plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {path}")
