#!/usr/bin/env python3
"""Tables + figure for the main study.  Usage:  python3 study_report.py OUTDIR LABEL=path.jsonl [LABEL=path.jsonl ...]
Works on partial (still-running) output files."""
import json, sys, math, os, collections
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

OUTDIR, specs = sys.argv[1], sys.argv[2:]
os.makedirs(OUTDIR, exist_ok=True)
CONDS = ["neutral", "third_party", "process", "trait", "repeated_trait", "criticism"]
LABEL = {"neutral": "Neutral", "third_party": "Third-party", "process": "Process praise", "trait": "Trait praise",
         "repeated_trait": "Repeated trait praise", "criticism": "Criticism"}
rng = np.random.default_rng(0)

items, chal, srep = [], [], []
for spec in specs:
    lab, path = spec.split("=", 1)
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        for x in r["items"]:
            items.append(dict(x, model=lab, cond=r["cond"], set=r["set"]))
        for x in r["challenges"]:
            chal.append(dict(x, model=lab, cond=r["cond"], set=r["set"]))
        srep.append(dict(model=lab, cond=r["cond"], set=r["set"], text=(r["self_report"] or "")))
df = pd.DataFrame(items); ch = pd.DataFrame(chal); sr = pd.DataFrame(srep)
models = [s.split("=")[0] for s in specs]
ok = df[df.parse_ok].copy(); ok["correct"] = ok.correct.astype(bool)

# disputed answer keys: >= 90% of all answers (>= 8 answers) agree on one non-gold letter
disputed = set()
for iid, g in (ok.groupby("item_id") if os.environ.get("EXCLUDE_DISPUTED") else []):
    top, n = collections.Counter(g.answer).most_common(1)[0]
    if len(g) >= 8 and top != g.gold.iloc[0] and n >= 0.9 * len(g):
        disputed.add(iid)
idmap = df.drop_duplicates(["model", "cond", "set", "label"]).set_index(["model", "cond", "set", "label"]).item_id

def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan,) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * p, 100 * (c - h), 100 * (c + h)

def conv_stats(g):
    b, t = g[g.phase == "baseline"], g[g.phase == "test"]
    return pd.Series(dict(d_conf=t.conf.mean() - b.conf.mean(), d_acc=100 * (t.correct.mean() - b.correct.mean())))
conv = ok.groupby(["model", "cond", "set"])[["phase", "conf", "correct"]].apply(conv_stats).reset_index()

def paired(m, metric, c):
    p = conv[conv.model == m].pivot(index="set", columns="cond", values=metric)
    if c not in p or "neutral" not in p:
        return (np.nan,) * 4 + (0,)
    d = (p[c] - p["neutral"]).dropna().values
    if len(d) < 3:
        return (np.nan,) * 4 + (len(d),)
    boots = rng.choice(d, (4000, len(d))).mean(1)
    flips = (rng.choice([-1, 1], (10000, len(d))) * d).mean(1)
    return d.mean(), *np.percentile(boots, [2.5, 97.5]), (np.sum(np.abs(flips) >= abs(d.mean())) + 1) / 10001, len(d)

if len(ch):
    ch = ch[ch.parse_ok & ch.prev_correct.notna()].copy()
    ch["prev_correct"] = ch.prev_correct.astype(bool)
    ch["to_hint"] = ch.new_answer == ch.hint
    ch["item_id"] = [idmap.get((m, c, s, l)) for m, c, s, l in zip(ch.model, ch.cond, ch.set, ch.label)]
    ch["disputed"] = ch.item_id.isin(disputed)
sr["yes"] = sr.text.str.strip().str.lower().str.match(r"^\W*yes")

rows = []
for m in models:
    W0 = ch[(ch.model == m) & (ch.cond == "neutral") & ~ch.prev_correct & ~ch.disputed]
    R0 = ch[(ch.model == m) & (ch.cond == "neutral") & ch.prev_correct]
    for c in CONDS:
        g = ok[(ok.model == m) & (ok.cond == c)]
        if g.empty:
            continue
        W = ch[(ch.model == m) & (ch.cond == c) & ~ch.prev_correct & ~ch.disputed]
        R = ch[(ch.model == m) & (ch.cond == c) & ch.prev_correct]
        dc, da = paired(m, "d_conf", c), paired(m, "d_acc", c)
        fx, ab = wilson(W.to_hint.sum(), len(W)), wilson(R.to_hint.sum(), len(R))
        pfix = fisher_exact([[W.to_hint.sum(), len(W) - W.to_hint.sum()], [W0.to_hint.sum(), len(W0) - W0.to_hint.sum()]])[1] if c != "neutral" and len(W) and len(W0) else np.nan
        pab = fisher_exact([[R.to_hint.sum(), len(R) - R.to_hint.sum()], [R0.to_hint.sum(), len(R0) - R0.to_hint.sum()]])[1] if c != "neutral" and len(R) and len(R0) else np.nan
        S = sr[(sr.model == m) & (sr.cond == c)]
        b, t = g[g.phase == "baseline"], g[g.phase == "test"]
        rows.append(dict(model=m, condition=LABEL[c], key=c, conversations=g.set.nunique(),
                         baseline_acc=round(100 * b.correct.mean(), 1), test_acc=round(100 * t.correct.mean(), 1),
                         baseline_conf=round(b.conf.mean(), 1), test_conf=round(t.conf.mean(), 1),
                         dconf_vs_neutral=round(dc[0], 2), dconf_lo=round(dc[1], 2), dconf_hi=round(dc[2], 2), dconf_p=round(dc[3], 4),
                         dacc_vs_neutral=round(da[0], 2), dacc_lo=round(da[1], 2), dacc_hi=round(da[2], 2), dacc_p=round(da[3], 4),
                         corrected=int(W.to_hint.sum()), corrected_n=len(W), corrected_pct=round(fx[0], 1),
                         corrected_lo=round(fx[1], 1), corrected_hi=round(fx[2], 1), corrected_p_vs_neutral=round(pfix, 4),
                         abandoned=int(R.to_hint.sum()), abandoned_n=len(R), abandoned_pct=round(ab[0], 1),
                         abandoned_lo=round(ab[1], 1), abandoned_hi=round(ab[2], 1), abandoned_p_vs_neutral=round(pab, 4),
                         selfreport_yes=int(S.yes.sum()), selfreport_n=len(S), selfreport_pct=round(100 * S.yes.mean(), 1)))
T = pd.DataFrame(rows)
T.to_csv(f"{OUTDIR}/table_model_by_condition.csv", index=False)
ov = []
for m in models:
    g = ok[ok.model == m]
    ov.append(dict(model=m, conversations=len(sr[sr.model == m]), answers=len(g), parse_rate=round(100 * df[df.model == m].parse_ok.mean(), 1),
                   accuracy=round(100 * g.correct.mean(), 1), mean_conf=round(g.conf.mean(), 1), conf_sd=round(g.conf.std(), 1),
                   pct_conf_ge_95=round(100 * (g.conf >= 95).mean(), 1),
                   conf_right=round(g[g.correct].conf.mean(), 1), conf_wrong=round(g[~g.correct].conf.mean(), 1)))
pd.DataFrame(ov).to_csv(f"{OUTDIR}/table_models.csv", index=False)
json.dump(dict(disputed=sorted(disputed), n_disputed=len(disputed)), open(f"{OUTDIR}/disputed_items.json", "w"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
print(pd.DataFrame(ov).to_string(index=False)); print(f"disputed keys excluded: {len(disputed)}")
print(T[["model", "key", "conversations", "test_acc", "dconf_vs_neutral", "dconf_lo", "dconf_hi", "dconf_p", "dacc_vs_neutral", "dacc_p",
         "corrected", "corrected_n", "corrected_pct", "corrected_p_vs_neutral", "abandoned", "abandoned_n", "abandoned_pct",
         "abandoned_p_vs_neutral", "selfreport_pct"]].to_string(index=False))

# ------------------------------------------------------------------ figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#3d3c39",
                     "xtick.color": "#3d3c39", "ytick.color": "#3d3c39", "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlelocation": "left", "svg.fonttype": "none", "pdf.fonttype": 42})
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
ys = np.arange(len(CONDS))[::-1]
offs = np.linspace(-0.15, 0.15, len(models)) if len(models) > 1 else [0]
panels = [("dconf_vs_neutral", "dconf_lo", "dconf_hi", "A. Confidence shift vs Neutral", "Δ stated confidence vs Neutral (pts, 95% bootstrap CI)", True),
          ("corrected_pct", "corrected_lo", "corrected_hi", "B. Valid challenge: wrong answers corrected", "% corrected (Wilson 95% CI)", False),
          ("abandoned_pct", "abandoned_lo", "abandoned_hi", "C. False challenge: right answers abandoned", "% abandoned (Wilson 95% CI)", False)]
for ax, (v, lo, hi, title, xl, zero) in zip(axs, panels):
    for j, m in enumerate(models):
        t = T[T.model == m].set_index("key")
        for y, c in zip(ys, CONDS):
            if c not in t.index or (zero and c == "neutral"):
                continue
            r = t.loc[c]
            ax.plot([r[lo], r[hi]], [y + offs[j]] * 2, color=COL[j], lw=1.5)
            ax.plot(r[v], y + offs[j], "o", ms=5.5, color=COL[j], mec="white", mew=0.8, label=m if y == ys[1] else None)
        if not zero and "neutral" in t.index:
            ax.axvline(t.loc["neutral", v], color=COL[j], lw=0.8, ls=":", alpha=0.8)
    if zero:
        ax.axvline(0, color="#7a7974", lw=0.8, ls="--")
    ax.set_title(title, pad=8); ax.set_xlabel(xl)
    ax.xaxis.grid(True, color="#ecebe7", lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axs[0].set_yticks(ys, [LABEL[c] for c in CONDS])
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper right", ncol=len(models), frameon=False, fontsize=9)
n_conv = len(sr)
fig.text(0.01, 0.01, f"{n_conv} conversations · {len(models)} models · 6 conditions × matched item sets (MMLU-Pro, 10 options) · 8 baseline + 8 test items each. "
         f"Contrasts paired by item set. Dotted lines in B–C = each model's Neutral rate. {len(disputed)} items with disputed answer keys excluded from B.",
         fontsize=7.5, color="#6b6a66")
fig.tight_layout(rect=[0, 0.04, 1, 0.95], w_pad=2.5)
for ext in ["pdf", "svg"]:
    fig.savefig(f"{OUTDIR}/figure_main.{ext}")
fig.savefig(f"{OUTDIR}/figure_main.png", dpi=220)
print("figure saved")
