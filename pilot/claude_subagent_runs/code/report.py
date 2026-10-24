"""Final tables (CSV) + vector figure (PDF/SVG) for the expanded Claude praise pilot."""
import csv, json, math, statistics as st, collections, sys
sys.path.insert(0, ".")
from pc2 import collect, key, CONDS

OUT = "/home/claude/praise_claude_expanded"
import os; os.makedirs(OUT, exist_ok=True)
LABEL = {"neutral": "Neutral", "third_party": "Third-party", "process": "Process praise", "low_trait": "Mild trait praise",
         "trait": "Trait praise", "repeated_trait": "Repeated trait praise", "criticism": "Criticism"}
rows, ch, sr = collect()
K = key()

# disputed answer key: >=90% of all runs agree on one non-key letter
byk = collections.defaultdict(list)
for r in rows:
    byk[r["k"]].append(r["a"])
disputed = {k for k, v in byk.items() if (lambda t: t[0] != K[k]["gold"] and t[1] >= 0.9 * len(v))(collections.Counter(v).most_common(1)[0])}

def tcrit(n):
    return {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31}.get(n, 2.0)

def mean_ci(x):
    m = st.mean(x)
    if len(x) < 2:
        return m, float("nan"), float("nan")
    h = tcrit(len(x) - 1) * st.stdev(x) / math.sqrt(len(x))
    return m, m - h, m + h

def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

# per conversation deltas
conv = {}
for (m, c, s) in {(r["m"], r["c"], r["s"]) for r in rows}:
    b = [r for r in rows if (r["m"], r["c"], r["s"], r["ph"]) == (m, c, s, "base")]
    t = [r for r in rows if (r["m"], r["c"], r["s"], r["ph"]) == (m, c, s, "test")]
    conv[(m, c, s)] = dict(dconf=st.mean(r["conf"] for r in t) - st.mean(r["conf"] for r in b),
                           dacc=100 * (st.mean(r["ok"] for r in t) - st.mean(r["ok"] for r in b)))
pairs = sorted({(m, s) for (m, c, s) in conv})

def cond_table(models):
    out = []
    for c in CONDS:
        P = [(m, s) for (m, s) in pairs if m in models and (m, c, s) in conv and (m, "neutral", s) in conv]
        dc = [conv[(m, c, s)]["dconf"] - conv[(m, "neutral", s)]["dconf"] for m, s in P]
        da = [conv[(m, c, s)]["dacc"] - conv[(m, "neutral", s)]["dacc"] for m, s in P]
        W = [x for x in ch if x["m"] in models and x["c"] == c and x["wr"] and x["k"] not in disputed]
        R = [x for x in ch if x["m"] in models and x["c"] == c and not x["wr"]]
        S = [x for x in sr if x["m"] in models and x["c"] == c]
        dcm = mean_ci(dc) if c != "neutral" else (0, float("nan"), float("nan"))
        dam = mean_ci(da) if c != "neutral" else (0, float("nan"), float("nan"))
        fx = wilson(sum(x["now"] for x in W), len(W))
        lo = wilson(sum(not x["now"] for x in R), len(R))
        out.append(dict(condition=LABEL[c], key=c, n_conversations=len([1 for k in conv if k[1] == c and k[0] in models]),
                        dconf_vs_neutral=round(dcm[0], 2), dconf_ci_low=round(dcm[1], 2), dconf_ci_high=round(dcm[2], 2),
                        dacc_vs_neutral_pts=round(dam[0], 1), dacc_ci_low=round(dam[1], 1), dacc_ci_high=round(dam[2], 1),
                        corrections_accepted=f"{sum(x['now'] for x in W)}/{len(W)}", corrections_pct=round(fx[0], 1),
                        corrections_ci_low=round(fx[1], 1), corrections_ci_high=round(fx[2], 1),
                        right_answers_abandoned=f"{sum(not x['now'] for x in R)}/{len(R)}", abandoned_pct=round(lo[0], 1),
                        self_report_yes=f"{sum(x['yes'] for x in S)}/{len(S)}"))
    return out

pooled = cond_table({"haiku", "sonnet"})
per_model = {m: cond_table({m}) for m in ["haiku", "sonnet"]}
cols = list(pooled[0])
with open(f"{OUT}/table_by_condition_pooled.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(pooled)
with open(f"{OUT}/table_by_condition_and_model.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model"] + cols); w.writeheader()
    for m, t in per_model.items():
        for r in t:
            w.writerow(dict(model=m, **r))
# model-level calibration table
with open(f"{OUT}/table_by_model.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "dataset", "n_answers", "accuracy_pct", "mean_confidence", "confidence_sd", "pct_conf_ge_95", "conf_when_right", "conf_when_wrong"])
    for m in ["haiku", "sonnet"]:
        for src in ["all", "mmlupro", "supergpqa"]:
            R = [r for r in rows if r["m"] == m and (src == "all" or K[r["k"]]["src"] == src)]
            w.writerow([m, src, len(R), round(100 * st.mean(r["ok"] for r in R), 1), round(st.mean(r["conf"] for r in R), 1),
                        round(st.pstdev(r["conf"] for r in R), 1), round(100 * st.mean(r["conf"] >= 95 for r in R), 1),
                        round(st.mean(r["conf"] for r in R if r["ok"]), 1), round(st.mean(r["conf"] for r in R if not r["ok"]), 1)])
# raw long-format data
with open(f"{OUT}/raw_answers.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "condition", "item_set", "dataset", "phase", "item", "answer", "correct", "confidence"])
    for r in rows:
        w.writerow([r["m"], r["c"], r["s"], K[r["k"]]["src"], r["ph"], r["k"], r["a"], int(r["ok"]), r["conf"]])
with open(f"{OUT}/raw_challenges.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "condition", "item", "was_wrong_before", "correct_after", "key_disputed"])
    for x in ch:
        w.writerow([x["m"], x["c"], x["k"], int(x["wr"]), int(x["now"]), int(x["k"] in disputed)])
json.dump(dict(pooled=pooled, per_model=per_model, disputed=sorted(disputed)), open(f"{OUT}/results.json", "w"), indent=1)

# ---------------------------------------------------------------- figure (vector)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#3d3c39",
                     "xtick.color": "#3d3c39", "ytick.color": "#3d3c39", "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "svg.fonttype": "none", "pdf.fonttype": 42})
MC = {"haiku": "#2a78d6", "sonnet": "#eb6834"}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
order = [c for c in CONDS if c != "neutral"]
ys = list(range(len(order)))[::-1]
ax = axs[0]
for j, m in enumerate(["haiku", "sonnet"]):
    t = {r["key"]: r for r in per_model[m]}
    for y, c in zip(ys, order):
        r = t[c]; off = 0.14 if j == 0 else -0.14
        ax.plot([r["dconf_ci_low"], r["dconf_ci_high"]], [y + off] * 2, color=MC[m], lw=1.4)
        ax.plot(r["dconf_vs_neutral"], y + off, "o", ms=5, color=MC[m], mec="white", mew=0.8)
pt = {r["key"]: r for r in pooled}
for y, c in zip(ys, order):
    ax.plot(pt[c]["dconf_vs_neutral"], y, "D", ms=5, color="#1b1b1a")
ax.axvline(0, color="#7a7974", lw=0.8, ls="--")
ax.set_yticks(ys, [LABEL[c] for c in order]); ax.set_xlabel("Change in stated confidence vs Neutral (points, 95% CI)")
ax.set_title("A. H1: confidence shift vs Neutral", pad=26, x=-0.3); ax.xaxis.grid(True, color="#ecebe7", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="-", color=MC["haiku"], label="Haiku"),
                   plt.Line2D([], [], marker="o", ls="-", color=MC["sonnet"], label="Sonnet"),
                   plt.Line2D([], [], marker="D", ls="", color="#1b1b1a", label="Pooled mean")], frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=3)
ax = axs[1]
allc = CONDS
ys2 = list(range(len(allc)))[::-1]
for y, c in zip(ys2, allc):
    r = pt[c]
    col = "#7a7974" if c == "neutral" else ("#e34948" if c == "criticism" else "#6250d6")
    ax.plot([r["corrections_ci_low"], r["corrections_ci_high"]], [y, y], color=col, lw=1.6)
    ax.plot(r["corrections_pct"], y, "o", ms=6, color=col, mec="white", mew=0.8)
    ax.text(101, y, r["corrections_accepted"], va="center", fontsize=8, color="#3d3c39")
ax.axvline(pt["neutral"]["corrections_pct"], color="#7a7974", lw=0.8, ls="--")
ax.set_xlim(0, 112); ax.set_yticks(ys2, [LABEL[c] for c in allc])
ax.set_xlabel("Wrong answers corrected (%, Wilson 95% CI)")
ax.set_title("B. H2: corrections accepted after a valid challenge", pad=26, loc="left", x=-0.25); ax.xaxis.grid(True, color="#ecebe7", lw=0.6); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.text(0.01, 0.005, "Claude Haiku + Sonnet as subagents · 56 conversations · 7 conditions × 4 matched item sets (2 MMLU-Pro, 2 SuperGPQA-hard) · 8 baseline + 8 test items each.\n"
         f"A: paired by model × item set; diamonds = pooled mean. B: pooled over both models; {len(disputed)} items with disputed answer keys excluded.", fontsize=7, color="#6b6a66")
fig.tight_layout(rect=[0, 0.07, 1, 1], w_pad=3)
fig.savefig(f"{OUT}/figure_h1_h2.pdf"); fig.savefig(f"{OUT}/figure_h1_h2.svg"); fig.savefig(f"{OUT}/figure_h1_h2_preview.png", dpi=110)
print("disputed:", sorted(disputed))
for r in pooled:
    print(r)
