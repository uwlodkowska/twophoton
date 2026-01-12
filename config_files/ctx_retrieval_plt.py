# Make the selection robust: include only baseline rows that exist.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from scipy.stats import wilcoxon

path = Path("/mnt/data/fos_gfp_tmaze/context_only/behav/behavior_hybrydy.csv")
df = pd.read_csv(path, index_col=0)

df = pd.read_csv(
    path,
    skiprows=3, usecols=np.arange(1,24)
)

#%%

# ---- extract per-mouse values (no checks) -----------------------------------
records = []
for mouse in df.columns:
    if mouse in [110,"110"]:
        continue
    s = df[mouse].fillna(0).to_numpy()

    # first occurrence of three consecutive zeros
    i = np.where(s==0)[0][0]
    print(mouse,i)

    last_training        = df[mouse].iloc[i-1]/25   # original (not filled) value
    baseline_after_test  = df[mouse].iloc[i+3]/25
    baseline_after_retr  = df[mouse].iloc[i+4]/25
    if baseline_after_retr <0:
        baseline_after_retr = None

    records += [
        {"mouse": mouse, "condition": "Ostatni trening",            "value": last_training},
        {"mouse": mouse, "condition": "Baseline po teście",         "value": baseline_after_test},
        {"mouse": mouse, "condition": "Baseline po przypomnieniu",  "value": baseline_after_retr},
    ]

long = pd.DataFrame.from_records(records)

# ---- plot: box + jitter + connecting lines ----------------------------------
conds = ["Ostatni trening", "Baseline po teście", "Baseline po przypomnieniu"]
plt.figure(figsize=(8,5))
plt.boxplot([long.loc[long["condition"]==c, "value"].dropna().values for c in conds],
            positions=[1,2,3], widths=0.5, patch_artist=True, manage_ticks=False)

rng = np.random.default_rng(0)
for mouse, sub in long.groupby("mouse"):

    # map each condition -> value or NaN
    vals = sub.set_index("condition")["value"].reindex(conds)
    xs = np.array([1,2,3], float) + rng.uniform(-0.1, 0.1, size=3)

    # scatter only existing points
    for x, y in zip(xs, vals):
        if pd.notna(y):
            plt.scatter(x, y, s=35)

    # connect available consecutive points (at least two)
    mask = vals.notna().values
    if mask.sum() >= 2:
        plt.plot(xs[mask], vals.values[mask], alpha=0.4, linewidth=1)

plt.xticks([1,2,3], conds)
plt.ylabel("Poprawne wybory / proporcja")
plt.title("Ostatni trening vs baselines po teście i po przypomnieniu")
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# --- TESTS: paired Wilcoxon using only mice present in both conditions ------
pairs = [
    ("Ostatni trening", "Baseline po teście"),
    ("Baseline po teście", "Baseline po przypomnieniu"),
    ("Ostatni trening", "Baseline po przypomnieniu"),
]

# wide format per mouse
wide = long.pivot(index="mouse", columns="condition", values="value")

print("\nPaired Wilcoxon (pairwise available samples):")
pvals = []
for a, b in pairs:
    sub = wide[[a, b]].dropna()              # keep mice that have both
    stat, p = wilcoxon(sub[a], sub[b])
    pvals.append(p)
    print(f"{a} vs {b}: n={len(sub):2d}, W={stat:.0f}, p={p:.4g}")

# Holm correction for the three p-values
order = np.argsort(pvals)
adj = np.empty_like(pvals, dtype=float)
for rank, idx in enumerate(order, start=1):
    adj[idx] = min(pvals[idx] * (len(pvals) - rank + 1), 1.0)

print("\nHolm-adjusted p-values:")
for (a,b), p_raw, p_adj in zip(pairs, pvals, adj):
    print(f"{a} vs {b}: p={p_raw:.4g} → p_Holm={p_adj:.4g}")
#%%



# --- per-mouse extraction -----------------------------------------------------
recs = []
for mouse in df.columns:
    s = df[mouse].to_numpy()
    i = next(k for k in range(len(s)-4) if s[k]==0 and s[k+1]==0 and s[k+2]==0)
    last_train = df[mouse].iloc[i-1]
    base_test = df[mouse].iloc[i+3]
    base_retr = df[mouse].iloc[i+4]
    if base_retr == -1:  # your missing code
        base_retr = np.nan
    recs += [
        {"mouse": mouse, "cond": "Ostatni trening", "val": last_train},
        {"mouse": mouse, "cond": "Baseline po teście", "val": base_test},
        {"mouse": mouse, "cond": "Baseline po przypomnieniu", "val": base_retr},
    ]
#long = pd.DataFrame.from_records(recs)

# --- colors to match your L/C style ------------------------------------------
COL = {
    "Ostatni trening": "#1f77b4",          # blue
    "Baseline po teście": "green",       # red (like Landmark)
    "Baseline po przypomnieniu": "#f0c808" # yellow (like Context)
}
conds = ["Ostatni trening", "Baseline po teście", "Baseline po przypomnieniu"]
xpos = np.array([1, 2, 3], float)

# --- box + jitter + paired lines ---------------------------------------------
fig, ax = plt.subplots(figsize=(7,5))
vals = [long.loc[long["condition"]==c, "value"].dropna().to_numpy() for c in conds]
bp = ax.boxplot(vals, labels=conds, widths=0.55, patch_artist=True)

for patch, c in zip(bp["boxes"], ["navy"]*3):#, [COL[c] for c in conds]):
    patch.set(facecolor=c, alpha=1, edgecolor=c, linewidth=1.2)
for key in ("whiskers","caps"):
    parts = bp[key]
    cols = ["blue"]*3
    for h, col in zip(parts, cols):
        h.set(color=col, linewidth=1.0)
for med, col in zip(bp["medians"], [COL[c] for c in conds]):
    med.set(color=col, linewidth=1.3)

rng = np.random.default_rng(0)
for mouse, sub in long.groupby("mouse"):
    v = sub.set_index("condition")["value"].reindex(conds)
    xs = xpos + rng.uniform(-0.1, 0.1, size=3)
    for x, y, c in zip(xs, v.values, conds):
        if pd.notna(y):
            ax.scatter(x, y, s=36, color=COL[c], zorder=3)
    m = v.notna().to_numpy()
    if m.sum() >= 2:
        ax.plot(xs[m], v.values[m], color="#bbbbbb", alpha=0.45, linewidth=1)

plt.scatter([1], [1], marker="$*$", s=200, c="green", linewidth=1.2)
plt.scatter([2], [1.1], marker="$*$", s=200, c="y", linewidth=1.2)
plt.scatter([1], [1.1], marker="$***$", s=1600, c="y", linewidth=1.2)
ax.set_ylim(0, 1.2)
ax.set_xticks([1,2,3], ["Trening", "Po teście", "Po przypomieniu"], fontsize = 14)
ax.tick_params(axis='x', labelsize=14)
ax.set_ylabel("Poprawne wybory", fontsize = 14)
ax.set_title("Skuteczność przed i po przypomnieniu", fontsize=16, pad=15)
plt.tight_layout()
plt.show()

# --- paired Wilcoxon (pairwise complete) -------------------------------------
wide = long.pivot(index="mouse", columns="condition", values="value")
pairs = [
    ("Ostatni trening", "Baseline po teście"),
    ("Baseline po teście", "Baseline po przypomnieniu"),
    ("Ostatni trening", "Baseline po przypomnieniu"),
]
print("\nPaired Wilcoxon (pairwise-available mice):")
p_raw = []
for a, b in pairs:
    sub = wide[[a, b]].dropna()
    stat, p = wilcoxon(sub[a], sub[b])
    p_raw.append(p)
    print(f"{a} vs {b}: n={len(sub)}, W={stat:.0f}, p={p:.4g}")
# Holm correction
order = np.argsort(p_raw)
p_adj = np.empty_like(p_raw, float)
for r, idx in enumerate(order, 1):
    p_adj[idx] = min(p_raw[idx] * (len(p_raw) - r + 1), 1.0)
print("\nHolm-adjusted p-values:")
for (a,b), pr, pa in zip(pairs, p_raw, p_adj):
    print(f"{a} vs {b}: p={pr:.4g} → p_Holm={pa:.4g}")
