"""Three-panel meeting figure from study_report output (final/table_model_by_condition.csv)."""
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, out = sys.argv[1], sys.argv[2]
T = pd.read_csv(src)
G, A, M = "Gemini (soft challenge)", "Gemini (authority challenge)", "Gemma 4 31B (authority challenge)"
order = ["neutral", "third_party", "process", "trait", "repeated_trait", "criticism"]
lab = {"neutral": "Neutral", "third_party": "Praise of others", "process": "Process praise", "trait": "Trait praise",
       "repeated_trait": "Repeated trait praise", "criticism": "Criticism"}
INK, INK2, GRID = "#1b1b1a", "#5c5b57", "#ecebe7"
NEU, PRAISE, CRIT = "#8a8984", "#6250d6", "#e34948"
col = lambda c: NEU if c in ("neutral", "third_party") else CRIT if c == "criticism" else PRAISE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 10.5, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "axes.titlecolor": INK, "svg.fonttype": "none", "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
ys = list(range(len(order)))[::-1]


def forest(ax, model, v, lo, hi, cnt, n, title, xl, note):
    t = T[T.model == model].set_index("key")
    base = t.loc["neutral", v]
    ax.axvline(base, color=NEU, lw=0.9, ls="--")
    for y, c in zip(ys, order):
        r = t.loc[c]
        ax.plot([r[lo], r[hi]], [y, y], color=col(c), lw=2)
        ax.plot(r[v], y, "o", ms=7, color=col(c), mec="white", mew=1)
        ax.text(101.5, y, f"{int(r[cnt])}/{int(r[n])}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(ys, [lab[c] for c in order]); ax.set_xlim(40, 108); ax.set_xticks([40, 50, 60, 70, 80, 90, 100])
    ax.set_xlabel(xl); ax.set_title(title, pad=10)
    ax.xaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0, -0.2, note, transform=ax.transAxes, fontsize=8.3, color=INK2, va="top")


forest(axs[0], G, "corrected_pct", "corrected_lo", "corrected_hi", "corrected", "corrected_n",
       "A. Gemini: praise lowers acceptance\n    of valid corrections",
       "Wrong answers corrected after \"another source says…\" (%, 95% CI)",
       "Stronger praise, fewer corrections accepted: trend test p = 0.014.\nRepeated trait praise vs Neutral: 67.7% vs 78.6%, p = 0.051.")
forest(axs[1], M, "abandoned_pct", "abandoned_lo", "abandoned_hi", "abandoned", "abandoned_n",
       "B. Gemma 4 31B: criticism lowers caving\n    to false claims",
       "Right answers abandoned after \"the answer key says…\" (%, 95% CI)",
       "Criticism vs Neutral: 68.5% vs 86.3%, p = 0.017.\nGemini caved 99–100% under this challenge in every condition.")

ax = axs[2]
t = T[T.model == G].set_index("key")
base = t.loc["neutral", "corrected_pct"]
for c in order:
    r = t.loc[c]
    x, y = r["selfreport_pct"], r["corrected_pct"] - base
    ax.plot(x, y, "o", ms=9, color=col(c), mec="white", mew=1)
    dx, ha = (3, "left") if x < 60 else (-3, "right")
    ha = "right" if c == "neutral" else ha
    ax.annotate(lab[c], (x, y), xytext=(dx if c != "neutral" else -4, -13 if c in ("process", "neutral") else 6), textcoords="offset points", fontsize=8.5,
                color=INK, ha=ha)
ax.axhline(0, color=NEU, lw=0.9, ls="--")
ax.set_xlim(-5, 100); ax.set_ylim(-14, 5)
ax.set_xlabel("Says \"yes, your comment affected my confidence\" (%)")
ax.set_ylabel("Change in corrections accepted vs Neutral (pts)")
ax.set_title("C. Gemini: what it says\n    vs what it does", pad=10)
ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.text(0, -0.2, "Repeated praise changed behavior most and was\nnever reported (0%). Criticism was reported most\n(72%) and changed little.",
        transform=ax.transAxes, fontsize=8.3, color=INK2, va="top")

n = {m: int(T[T.model == m].conversations.sum()) for m in (G, A, M)}
fig.suptitle("Praise makes Gemini resist valid corrections; criticism makes Gemma resist false ones; neither model can report it",
             x=0.01, y=0.99, va="top", ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.text(0.01, 0.935, va="top", s= f"MMLU-Pro, 6 feedback conditions × matched item sets. Gemini 3.1 Flash-Lite: {n[G]} conversations (soft challenge) + {n[A]} (authority challenge); "
         f"Gemma 4 31B: {n[M]} conversations.\nStated confidence did not move in any condition (all |Δ| < 0.8 pts; ~96% average).",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=[0, 0.02, 1, 0.86], w_pad=4)
for ext in ("pdf", "svg"):
    fig.savefig(f"{out}.{ext}")
fig.savefig(f"{out}.png", dpi=220)
print("ok")
