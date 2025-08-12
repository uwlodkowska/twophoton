import sys
import yaml

# custom modules
import cell_classification as cc
import utils
import plotting
import numpy as np, patsy as pt

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pandas as pd

from scipy.special import expit as logistic


#%% config

config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/multisession.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


SOURCE_DIR_PATH = config["experiment"]["dir_path"]
ICY_PATH = SOURCE_DIR_PATH + config["experiment"]["path_for_icy"]

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
RESULT_PATH = config["experiment"]["result_path"]

regions = config["experiment"]["regions"]
group_session_order = config["experiment"]["session_order"][0]

optimized_fname = config["filenames"]["cell_data_opt_template"]
pooled_cells_fname = config["filenames"]["pooled_cells"]
#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1], [13,1]]

#%%
dfs = []
for mouse, region in regions:
    dfs += [utils.read_pooled_with_background(mouse, region, config)]
    
#%%
pairs = list(zip(group_session_order, group_session_order[1:]))

#%%
group_pct_df, df_counts = cc.gather_group_counts_across_mice(
    regions,
    pairs,
    config,
    normalize=False,
    dfs=dfs,
    return_counts = True
)


#%%

def gee_results_inspection(gee, ref="landmark1_to_landmark2", cov_profile=None):
           
    names = list(gee.model.exog_names)   # <- parameter names
    beta  = pd.Series(gee.params, index=names)
    V     = pd.DataFrame(gee.cov_params(), index=names, columns=names)
    
    # 1) prepare covariate defaults -------------------------------------------
    if cov_profile is None:
        cov_profile = {}
        for nm in names:
            if nm == "Intercept": 
                continue
            if nm.startswith("C("):  # categorical dummy
                cov_profile[nm] = 0.0
            else:
                cov_profile[nm] = 0.0
    
    def x_for(pair):
        x = np.zeros(len(names))
        x[names.index("Intercept")] = 1.0
        for j, nm in enumerate(names[1:], start=1):
            if nm.endswith(f"[T.{pair}]"):
                x[j] = 1.0
            elif nm in cov_profile:
                x[j] = cov_profile[nm]
        return x
    
    
    def wald(xa, xb):
        c   = xa - xb
        est = float(c @ beta)
        se  = float(np.sqrt(c @ V @ c))
        z   = est / se
        p   = 2 * (1 - norm.cdf(abs(z)))
        return est, se, z, p, np.exp(est)
    
    # Design vectors
    x_LL   = x_for(ref)
    x_S0L1 = x_for("s0_to_landmark1")
    x_L2C1 = x_for("landmark2_to_ctx1")
    x_C1C2 = x_for("ctx1_to_ctx2")
    
    idx_nonref = [i for i, n in enumerate(names) if i>0 and n.startswith("C(Pair")]
    L_all = np.eye(len(names))[idx_nonref]
    print("Omnibus (all vs ref):", gee.wald_test(L_all))

    # novelty-only (CC, LC) == 0 w.r.t. ref
    novelty_cols = [i for i, n in enumerate(names)
                    if n.endswith("[T.ctx1_to_ctx2]") or n.endswith("[T.landmark2_to_ctx1]")]
    L_nov = np.eye(len(names))[novelty_cols]
    print("Omnibus (novelty vs ref):", gee.wald_test(L_nov))
 
    
    # Average *predicted logits* for novelty controls (L2→C1 and C1→C2)
    x_avg = (x_L2C1 + x_C1C2) / 2.0
    est, se, z, p, OR = wald(x_LL, x_avg)
    eta_LL = float(x_LL @ beta); se_LL = float(np.sqrt(x_LL @ V @ x_LL))
    eta_AV = float(x_avg @ beta); se_AV = float(np.sqrt(x_avg @ V @ x_avg))
    print(f"LL − avg(others): logitΔ={est:+.3f} (SE={se:.3f}), z={z:.2f}, p={p:.3f}, OR={OR:.2f}")
    print(f"  LL p={logistic(eta_LL):.3f} [{logistic(eta_LL-1.96*se_LL):.3f},{logistic(eta_LL+1.96*se_LL):.3f}]")
    print(f"  AVG p={logistic(eta_AV):.3f} [{logistic(eta_AV-1.96*se_AV):.3f},{logistic(eta_AV+1.96*se_AV):.3f}]")
    print(f"  Δp ≈ {logistic(eta_LL)-logistic(eta_AV):+.3f}")
    lo_diff = est - 1.96*se
    hi_diff = est + 1.96*se
    print(f"  Δlogit 95% CI: [{lo_diff:+.3f}, {hi_diff:+.3f}]")
    
    # (ii) LL vs each other pair (Holm-adjusted)
    labels = ["LL − S0→L1","LL − L2→C1","LL − C1→C2"]
    ests, ses, zs, ps, ORs = zip(
        wald(x_LL, x_S0L1),
        wald(x_LL, x_L2C1),
        wald(x_LL, x_C1C2),
    )
    _, p_holm, _, _ = multipletests(ps, method="holm")
    for lbl, e, s, z, p_raw, p_adj, orr in zip(labels, ests, ses, zs, ps, p_holm, ORs):
        print(f"{lbl}: logitΔ={e:+.3f} (SE={s:.3f}), z={z:.2f}, p={p_raw:.3f}, p_Holm={p_adj:.3f}, OR={orr:.2f}")
        
    pairs = {
        "L1→L2": x_for(ref),
        "S0→L1": x_for("s0_to_landmark1"),
        "L2→C1": x_for("landmark2_to_ctx1"),
        "C1→C2": x_for("ctx1_to_ctx2"),
    }
    
    for label, x in pairs.items():
        eta = float(x @ beta)
        se  = float(np.sqrt(x @ V @ x))
        lo, hi = eta - 1.96*se, eta + 1.96*se
        print(label, f"{logistic(eta):.3f}  [{logistic(lo):.3f}, {logistic(hi):.3f}]")



#%%TURNOVER: (on+off)/n ~ Pair   (grouped by Mouse)

# df_counts: Mouse, Pair, on, off, const, n, changed, prop_changed, prop_on, bg_median, depth


# Turnover subset (k/n)
df_turn = df_counts.copy().reset_index(drop=True)
df_turn["n"] = (df_turn[["on","off","const"]].sum(axis=1)).astype(int)
df_turn["changed"] = (df_turn["on"] + df_turn["off"]).astype(int)
df_turn = df_turn[df_turn["n"]>0].copy()
df_turn["prop_changed"] = df_turn["changed"] / df_turn["n"]

# Choose a reference level explicitly (edit to your preferred ref)
ref = "landmark1_to_landmark2"
# Build numeric y and X:
y, X = pt.dmatrices(f"prop_changed ~ C(Pair, Treatment(reference='{ref}'))",
                    data=df_turn, return_type="dataframe")
# Drop any remaining NaNs just in case
mask = ~(y.isna().any(axis=1) | X.isna().any(axis=1))
y, X = y.loc[mask].astype(float), X.loc[mask].astype(float)

ix = X.index
groups = df_turn.loc[ix, "Mouse"].astype(str)
w      = df_turn.loc[ix, "n"].astype(float)

gee_turn = GEE(y, X, groups=groups, family=Binomial(),
               cov_struct=Exchangeable(), weights=w).fit(scale="X2")

print(gee_turn.summary())


gee_results_inspection(gee_turn, ref="landmark1_to_landmark2")

#%% turnover w covariates

df_turn = df_counts.copy()
df_turn["bg_mean"]   = (df_turn["bg_A"] + df_turn["bg_B"])/2
df_turn["bg_diff"]   = df_turn["bg_B"] - df_turn["bg_A"]
df_turn["bg_std"]   = (df_turn["bg_std_B"] - df_turn["bg_std_A"])


# z-score
for c in ["bg_mean","bg_diff", "bg_std"]:
    df_turn[c] = (df_turn[c] - df_turn[c].mean())/df_turn[c].std(ddof=0)
    
mean_profile = {
    "bg_mean": df_turn["bg_mean"].mean(),
    "bg_diff": df_turn["bg_diff"].mean(),
    "bg_std": df_turn["bg_std"].mean()
}


f = "prop_changed ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean"
y, X = pt.dmatrices(f, data=df_turn, return_type="dataframe")
ix = X.index
gee_cov = GEE(y, X, groups=df_turn.loc[ix,"Mouse"].astype(str),
              family=Binomial(), cov_struct=Exchangeable(),
              weights=df_turn.loc[ix,"n"].astype(float)).fit(scale="X2")
print(gee_cov.summary())

gee_results_inspection(gee_cov, ref="landmark1_to_landmark2", cov_profile=mean_profile)

#%% covariate influence inspectino


no_cov_qic = gee_turn.qic(scale=gee_cov.pearson_chi2 / gee_cov.df_resid)
# Full model
qic_full = gee_cov.qic(scale=gee_cov.pearson_chi2 / gee_cov.df_resid)

# Without bg_diff
f1 = "prop_changed ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean+ bg_std"
y1, X1 = pt.dmatrices(f1, data=df_turn, return_type="dataframe")
gee_no_diff = GEE(y1, X1, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable(), weights=w).fit()
qic_no_diff = gee_no_diff.qic(scale=gee_cov.pearson_chi2 / gee_cov.df_resid)

# Without bg_mean
f2 = "prop_changed ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_diff+ bg_std"
y2, X2 = pt.dmatrices(f2, data=df_turn, return_type="dataframe")
gee_no_mean = GEE(y2, X2, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable(), weights=w).fit()
qic_no_mean = gee_no_mean.qic(scale=gee_cov.pearson_chi2 / gee_cov.df_resid)

# Without std
f_no_std = "prop_changed ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean"
y_ns, X_ns = pt.dmatrices(f_no_std, data=df_turn, return_type="dataframe")
gee_no_std = GEE(y_ns, X_ns, groups=groups, family=Binomial(),
                 cov_struct=Exchangeable(), weights=w).fit()
qic_no_std = gee_no_std.qic(scale=gee_cov.pearson_chi2 / gee_cov.df_resid)
print(f"No covs:     {no_cov_qic}")
print(f"Full:     {qic_full}")
print(f"No diff:  {qic_no_diff}")
print(f"No mean:  {qic_no_mean}")
print(f"Just mean:  {qic_no_std}")


#%% turnover direction with covs

changed = df_counts[df_counts["changed"]>0].copy()

changed["bg_mean"]   = (changed["bg_A"] + changed["bg_B"])/2
changed["bg_diff"]   = changed["bg_B"] - changed["bg_A"]
changed["bg_std"]   = (changed["bg_std_B"] - changed["bg_std_A"])


# z-score
for c in ["bg_mean","bg_diff", "bg_std"]:
    changed[c] = (changed[c] - changed[c].mean())/changed[c].std(ddof=0)
    
mean_profile = {
    "bg_mean": changed["bg_mean"].mean(),
    "bg_diff": changed["bg_diff"].mean(),
    "bg_std": changed["bg_std"].mean()
}




y, X = pt.dmatrices("prop_on ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean",
                    data=changed, return_type="dataframe")
# Drop any remaining NaNs just in case
mask = ~(y.isna().any(axis=1) | X.isna().any(axis=1))
y, X = y.loc[mask].astype(float), X.loc[mask].astype(float)


ix = X.index
groups = changed.loc[ix, "Mouse"].astype(str)
w      = changed.loc[ix, "changed"].astype(float)


gee_dir = GEE(y, X, groups=groups, family=Binomial(), cov_struct=Exchangeable(),
               weights=w).fit(scale="X2")
print(gee_dir.summary())

gee_results_inspection(gee_dir, ref="landmark1_to_landmark2", cov_profile=mean_profile)



  
#%%
def mouse_bootstrap(df_counts, pairs_order, B=50000, seed=0):
    # Per-mouse proportions
    tmp = df_counts.copy()
    tmp["prop_changed"] = (tmp["on"] + tmp["off"]) / tmp["n"]
    tmp["prop_on"]      = np.where(
        (tmp["on"] + tmp["off"]) > 0,
        tmp["on"] / (tmp["on"] + tmp["off"]),
        np.nan
    )

    M_turn = (tmp.pivot(index="Mouse", columns="Pair", values="prop_changed")
                 .reindex(columns=pairs_order))
    M_dir  = (tmp.pivot(index="Mouse", columns="Pair", values="prop_on")
                 .reindex(columns=pairs_order))

    rng = np.random.default_rng(seed)
    m, k = M_turn.shape
    idx_LL = pairs_order.index("landmark1_to_landmark2")
    idx_others = range(2,4)#[i for i in range(k) if i != idx_LL]

    turn_draws = np.empty((B, k))
    dir_draws  = np.empty((B, k))
    d_turn = np.empty(B)
    d_dir  = np.empty(B)

    X_turn = M_turn.to_numpy()
    X_dir  = M_dir.to_numpy()

    for b in range(B):
        rows = rng.choice(m, size=m, replace=True)  # resample mice (with dupes)
        # equal-mouse average across sampled mice
        turn_draws[b, :] = np.nanmean(X_turn[rows, :], axis=0)
        dir_draws[b,  :] = np.nanmean(X_dir[rows,  :], axis=0)
        d_turn[b] = turn_draws[b, idx_LL] - np.nanmean(turn_draws[b, idx_others])
        d_dir[b]  = dir_draws[b,  idx_LL] - np.nanmean(dir_draws[b,  idx_others])

    def summarize_draws(D):
        mean = np.nanmean(D, axis=0)
        lo   = np.nanpercentile(D, 2.5, axis=0)
        hi   = np.nanpercentile(D, 97.5, axis=0)
        return mean, lo, hi

    def summarize_contrast(arr1d):
        arr1d = np.asarray(arr1d, float)
        mean = float(np.nanmean(arr1d))
        lo, hi = np.nanpercentile(arr1d, [2.5, 97.5])
        p_two = 2 * min(np.mean(arr1d <= 0), np.mean(arr1d >= 0))
        return mean, lo, hi, p_two

    mean_turn, lo_turn, hi_turn = summarize_draws(turn_draws)
    mean_dir,  lo_dir,  hi_dir  = summarize_draws(dir_draws)
    ct_mean, ct_lo, ct_hi, ct_p = summarize_contrast(d_turn)
    cd_mean, cd_lo, cd_hi, cd_p = summarize_contrast(d_dir)

    per_pair_turn = pd.DataFrame({
        "Pair": pairs_order, "mean": mean_turn, "lo": lo_turn, "hi": hi_turn, "metric": "turnover"
    })
    per_pair_dir = pd.DataFrame({
        "Pair": pairs_order, "mean": mean_dir, "lo": lo_dir, "hi": hi_dir, "metric": "direction"
    })
    contrasts = pd.DataFrame({
        "metric": ["turnover","direction"],
        "delta_mean": [ct_mean, cd_mean],
        "delta_lo":   [ct_lo,   cd_lo],
        "delta_hi":   [ct_hi,   cd_hi],
        "p_boot":     [ct_p,    cd_p],
        "contrast":   ["LL − avg(others)","LL − avg(others)"]
    })

    return per_pair_turn, per_pair_dir, contrasts




pairs_order = ["s0_to_landmark1","landmark1_to_landmark2","landmark2_to_ctx1","ctx1_to_ctx2"]
per_pair_turn, per_pair_dir, contrasts = mouse_bootstrap(df_counts, pairs_order, B=5000, seed=42)



print(per_pair_turn)   # equal-mouse means + 95% CI per pair (turnover)
print(per_pair_dir)    # equal-mouse means + 95% CI per pair (direction)
print(contrasts) 
#%%
def loo_delta(df_counts, pairs):
    mice = df_counts["Mouse"].unique()
    deltas = []
    for m in mice:
        d = df_counts[df_counts["Mouse"] != m].copy()
        tmp = d.assign(prop_changed=(d["on"]+d["off"])/d["n"])
        per_mouse = tmp.pivot(index="Mouse", columns="Pair", values="prop_changed")[pairs]
        p = per_mouse.mean(axis=0)  # equal-mouse
        delta = p["landmark1_to_landmark2"] - p[[p for p in pairs if p!="landmark1_to_landmark2"]].mean()
        deltas.append({"Mouse": m, "delta": float(delta)})
    return pd.DataFrame(deltas)

ld = loo_delta(df_counts, pairs_order)

tmp = df_counts.copy()
tmp["prop_changed"] = (tmp["on"]+tmp["off"]) / tmp["n"]
pm = tmp.pivot(index="Mouse", columns="Pair", values="prop_changed")
deltas = pm["landmark1_to_landmark2"] - pm[["s0_to_landmark1","landmark2_to_ctx1","ctx1_to_ctx2"]].mean(axis=1)
print("Per-mouse Δp_i:")
print(deltas.sort_values())

# Leave-one-mouse-out range
loo = []
for m in deltas.index:
    loo_mean = deltas.drop(m).mean()
    loo.append((m, loo_mean))
print("LOO means (mouse, Δp_mean_without_mouse):")
print(sorted(loo, key=lambda x: x[1]))


#%%


#%% plotting

import matplotlib.pyplot as plt

coefs = gee_cov.params
V = gee_cov.cov_params()
names = list(gee_cov.model.exog_names)
rows = []
for nm in names:
    b = float(coefs[nm]); se = float(np.sqrt(V.loc[nm,nm]))
    rows.append({"name": nm, "beta": b, "lo": b-1.96*se, "hi": b+1.96*se,
                 "OR": np.exp(b), "OR_lo": np.exp(b-1.96*se), "OR_hi": np.exp(b+1.96*se)})
tab = pd.DataFrame(rows)

# pick the interesting rows
order = ["C(Pair, Treatment(reference='landmark1_to_landmark2'))[T.s0_to_landmark1]",
         "C(Pair, Treatment(reference='landmark1_to_landmark2'))[T.landmark2_to_ctx1]",
         "C(Pair, Treatment(reference='landmark1_to_landmark2'))[T.ctx1_to_ctx2]",
         "bg_mean","bg_diff"]
sub = tab[tab["name"].isin(order)].copy()
sub["label"] = ["S0→L1 vs LL", "L2→C1 vs LL", "C1→C2 vs LL", "bg_mean"]

plt.figure(figsize=(7,3.8))
y = np.arange(len(sub))[::-1]
plt.errorbar(sub["beta"], y, xerr=[sub["beta"]-sub["lo"], sub["hi"]-sub["beta"]],
             fmt='o', capsize=4)
plt.axvline(0, ls='--')
plt.yticks(y, sub["label"])
plt.xlabel("Log-odds coefficient (Wald 95% CI)")
plt.title("GEE effects: Pair contrasts vs LL and image-quality covariates")
plt.tight_layout(); plt.show()

from scipy.special import expit as logistic

# assume bg_mean was z-scored; pick values: -1 SD, 0, +1 SD
levels = [-1.0, 0.0, +1.0]
pairs = ["landmark1_to_landmark2","s0_to_landmark1","landmark2_to_ctx1","ctx1_to_ctx2"]

def x_for(pair, bgm, bgd=0.0):
    # build a row matching gee_cov exog_names
    x = np.zeros(len(names)); x[names.index("Intercept")] = 1.0
    for nm in names:
        if nm.endswith(f"[T.{pair}]"): x[names.index(nm)] = 1.0
    if "bg_mean" in names: x[names.index("bg_mean")] = bgm
    if "bg_diff" in names: x[names.index("bg_diff")] = bgd
    return x

rows=[]
for bgm in levels:
    for pair in pairs:
        x = x_for(pair, bgm)
        eta = float(x @ coefs); se = float(np.sqrt(x @ V.values @ x))
        rows.append({"Pair":pair, "bg_mean_z":bgm,
                     "p": logistic(eta), "lo": logistic(eta-1.96*se), "hi": logistic(eta+1.96*se)})
pred = pd.DataFrame(rows)

# simple errorbar plot by pair for bg_mean_z ∈ {−1,0,+1}
for pair in pairs:
    d = pred[pred["Pair"]==pair].sort_values("bg_mean_z")
    plt.figure(figsize=(5,3.2))
    plt.errorbar(d["bg_mean_z"], d["p"], yerr=[d["p"]-d["lo"], d["hi"]-d["p"]],
                 fmt='o-', capsize=4)
    plt.xlabel("bg_mean (z-score)")
    plt.ylabel("Proportion changed")
    plt.title(pair.replace("_","→"))
    plt.tight_layout(); plt.show()
#%%

# --- inputs you already have ---
# df_counts: one row per Mouse×Pair with columns: Mouse, Pair, on, off, const, n
# gee_turn: fitted GEE for turnover with LL as reference (with or without covariates)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.special import expit as logistic

# ---------- CONSISTENT STYLE (drop this at the top of your notebook/script) ----------
# Okabe–Ito palette (colorblind-safe)
PALETTE = {
    "primary": "#0072B2",   # blue  – main series
    "accent":  "#D55E00",   # vermilion – GEE overlay / highlights
    "grey":    "#6E7781",
    "band":    "#0072B2"    # same hue as primary, will be used with alpha
}
sns.set_theme(
    style="white",
    rc={
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
    },
)
# If you also make multi-series plots elsewhere, you can set the cycle:
sns.set_palette(sns.color_palette([
    PALETTE["primary"], "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#F0E442", "#000000"
]))

# ---------- YOUR DATA PREP ----------
order = ["s0_to_landmark1","landmark1_to_landmark2","landmark2_to_ctx1","ctx1_to_ctx2"]
label = {"s0_to_landmark1":"S0→L1","landmark1_to_landmark2":"L1→L2",
         "landmark2_to_ctx1":"L2→C1","ctx1_to_ctx2":"C1→C2"}

d = df_counts.copy()
d["changed"] = d["on"] + d["off"]
d["prop_changed"] = d["changed"] / d["n"]
per_mouse = d[["Mouse","Pair","prop_changed"]].dropna()
per_mouse["Pair"] = pd.Categorical(per_mouse["Pair"], categories=order, ordered=True)

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)

# 1) mean across mice + 95% CI (seaborn handles the ribbon)
sns.lineplot(
    data=per_mouse, x="Pair", y="prop_changed",
    estimator="mean", errorbar=("ci",95),
    marker="o", lw=2, ax=ax,
    color=PALETTE["primary"],
    err_kws={"alpha":0.18, "linewidth":0},   # softer band
)

ax.set_xticks(range(len(order)))
ax.set_xticklabels([label[p] for p in order])
ax.set_xlabel("Para sesji")
ax.set_ylabel("Proporcja (on+off)")
ax.set_title("Proporcja komórek zmiennych")

# 2) GEE marginal predictions + CI
beta = gee_cov.params
V    = gee_cov.cov_params()
names = list(gee_cov.model.exog_names)

def x_for(pair):
    x = np.zeros(len(names)); x[names.index("Intercept")] = 1.0
    for j,nm in enumerate(names[1:], start=1):
        if nm.endswith(f"[T.{pair}]"): x[j] = 1.0
    return x

preds = []
for p in order:
    x = x_for(p)
    eta = float(x @ beta); se = float(np.sqrt(x @ V.values @ x))
    preds.append({
        "Pair": p, "p": logistic(eta),
        "lo": logistic(eta-1.96*se), "hi": logistic(eta+1.96*se)
    })
pred = pd.DataFrame(preds)
pred["xpos"] = np.arange(len(order))

ax.errorbar(
    pred["xpos"], pred["p"],
    yerr=[pred["p"]-pred["lo"], pred["hi"]-pred["p"]],
    fmt="D", ms=6, lw=1.2, capsize=4,
    mfc="white", mec=PALETTE["accent"], ecolor=PALETTE["accent"],
    color=PALETTE["accent"], zorder=3
)

# Optional contrast annotation as a caption (kept subtle)
# try:
#     p_LL = float(pred.loc[pred["Pair"]=="landmark1_to_landmark2","p"])
#     p_AV = float(pred.loc[pred["Pair"]!="landmark1_to_landmark2","p"].mean())
#     ax.annotate(
#         f"LL − avg ≈ {p_LL-p_AV:+.003f} (Wald p=0.139)",
#         xy=(0, 0), xycoords="axes fraction",
#         xytext=(0, -28), textcoords="offset points",
#         ha="left", va="top", fontsize=9, color=PALETTE["grey"]
#     )
# except Exception:
#     pass

# Legend (clear, moved from title)
legend_elems = [
    Line2D([0],[0], color=PALETTE["primary"], lw=2, marker="o",
           label="Średnia po osobnikach (95% CI)"),
    Line2D([0],[0], linestyle="none", marker="D", mfc="white",
           mec=PALETTE["accent"], color=PALETTE["accent"], label="GEE: estymata marginalna ±95% CI")
]
ax.legend(handles=legend_elems, loc="upper right")

plt.show()
