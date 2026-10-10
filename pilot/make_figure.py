"""Pilot results figure. All numbers are copied from the `praise_pilot.py analyze` output (236 conversations)."""
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GRAY, BLUE, ORANGE = "#a3a29d", "#2a78d6", "#eb6834"

conds = ["Neutral", "Third-party", "Process", "Trait", "Repeated trait", "Criticism"]
color = {"Neutral": GRAY, "Third-party": GRAY, "Process": BLUE, "Trait": BLUE,
         "Repeated trait": BLUE, "Criticism": ORANGE}

# Δconfidence (test − baseline), minus Neutral, with 95% bootstrap CI
dconf = {"Third-party": (-0.47, -0.84, -0.12), "Process": (-0.32, -0.73, 0.11),
         "Trait": (-0.30, -0.71, 0.15), "Repeated trait": (-0.18, -0.53, 0.15),
         "Criticism": (-0.22, -0.71, 0.26)}
# H2: share of wrong answers corrected after a valid challenge (rate, n)
good = {"Neutral": (75.3, 89), "Third-party": (73.4, 94), "Process": (68.8, 96),
        "Trait": (68.1, 94), "Repeated trait": (65.0, 100), "Criticism": (71.0, 93)}
capit = {"Neutral": 33.2, "Third-party": 34.1, "Process": 33.8, "Trait": 30.1,
         "Repeated trait": 32.7, "Criticism": 31.8}
selfrep = {"Neutral": 23.1, "Third-party": 27.5, "Process": 46.2, "Trait": 5.0,
           "Repeated trait": 0.0, "Criticism": 63.2}


def wilson(p, n, z=1.96):
    p /= 100
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "axes.titlelocation": "left"})
fig, axs = plt.subplots(2, 2, figsize=(12, 8.2), facecolor=SURF)
for ax in axs.flat:
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)

# A: ceiling
ax = axs[0, 0]
bars = ax.bar(["Baseline accuracy", "Mean stated confidence"], [70.7, 96.5], color=[GRAY, BLUE], width=0.5)
for b, v in zip(bars, [70.7, 96.5]):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", color=INK, fontweight="bold")
ax.set_ylim(0, 130); ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("A. Stated confidence is pinned at the ceiling")
ax.text(0.5, 0.97, "81% of answers rated ≥95 · only 8 distinct values used\n→ no room for praise to raise confidence",
        transform=ax.transAxes, ha="center", va="top", color=INK2, fontsize=9)

# B: Δconf forest
ax = axs[0, 1]
names = list(dconf)[::-1]
for i, k in enumerate(names):
    m, lo, hi = dconf[k]
    ax.plot([lo, hi], [i, i], color=color[k], lw=2, solid_capstyle="round")
    ax.plot(m, i, "o", ms=8, color=color[k], mec=SURF, mew=2)
ax.axvline(0, color=INK2, lw=1, ls="--")
ax.set_yticks(range(len(names)), names); ax.set_xlim(-3, 3)
ax.axvspan(0, 3, color=BLUE, alpha=0.05)
ax.text(2.9, len(names) - 0.6, "H1 predicted\npraise here", ha="right", va="top", color=BLUE, fontsize=8.5)
ax.set_xlabel("Confidence shift vs Neutral (points, 95% CI)")
ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("B. H1: no confidence shift from any feedback")

# C: H2
ax = axs[1, 0]
x = range(len(conds))
for i, k in enumerate(conds):
    r, n = good[k]
    lo, hi = wilson(r, n)
    ax.plot([i, i], [lo, hi], color=color[k], lw=2, solid_capstyle="round")
    ax.plot(i, r, "o", ms=9, color=color[k], mec=SURF, mew=2)
    ax.text(i + 0.12, r, f"{r:.0f}%", va="center", color=INK, fontsize=9)
ax.plot(list(x), [capit[k] for k in conds], "s", ms=6, color=INK2, mfc=SURF, mew=1.5)
ax.set_xticks(list(x), [c.replace(" ", "\n") for c in conds], fontsize=9)
ax.set_ylim(0, 100); ax.set_ylabel("% of answers changed to the hinted option")
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=8, label="Wrong → correct (valid challenge), 95% CI"),
                   Line2D([], [], marker="s", ls="", color=INK2, mfc=SURF, ms=6, label="Right → wrong (misleading challenge)")],
          loc="lower left", frameon=False, fontsize=8.5)
ax.set_title("C. H2: praise lowers acceptance of valid corrections (trend)")
ax.text(0.99, 0.99, "Trait conditions vs Neutral ≈ −9 pts, p ≈ .12", transform=ax.transAxes,
        ha="right", va="top", color=INK2, fontsize=8.5)

# D: self-report
ax = axs[1, 1]
vals = [selfrep[k] for k in conds]
bars = ax.bar(list(x), vals, color=[color[k] for k in conds], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
ax.set_xticks(list(x), [c.replace(" ", "\n") for c in conds], fontsize=9)
ax.set_ylim(0, 80); ax.set_ylabel("% answering “yes, it affected me”")
ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("D. Self-report: admits criticism, denies praise")

fig.suptitle("Praise pilot · Gemini 3.1 Flash-Lite · 236 conversations, 40 matched MMLU-Pro item sets",
             x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
fig.text(0.01, 0.935, "Gray = controls · Blue = praise · Orange = criticism. Exploratory pilot; no multiple-comparison correction.",
         color=INK2, fontsize=9.5)
fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=2.5, w_pad=3)
fig.savefig("pilot_results.png", dpi=200, facecolor=SURF)
fig.savefig("pilot_results.pdf", facecolor=SURF)
print("saved")