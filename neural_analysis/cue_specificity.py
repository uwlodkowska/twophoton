#%% imports

import sys
import yaml

# custom modules
import intersession as its
import cell_preprocessing as cp
import cell_classification as cc
import utils
import plotting

import numpy as np, pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, mannwhitneyu, rankdata
import matplotlib.pyplot as plt, seaborn as sns
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
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1]]

#%%
dfs = []
for mouse, region in regions:
    df = utils.read_pooled_with_background(mouse, region, config)
    df["Mouse"] = mouse 
    df = cp.intensity_depth_detrend(df, group_session_order)
    dfs += [df]
#%%

def wilcoxon_with_effect(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    x_nz = x[x != 0]
    n_eff = x_nz.size
    if n_eff == 0:
        return {"n_eff": 0, "W": np.nan, "p": np.nan,
                "HL": np.nan, "median": np.nan}
    W, p = wilcoxon(x_nz, zero_method="wilcox", method="auto")
    HL = float(np.median(x))          # Hodges–Lehmann for paired = median(diff)
    return {"n_eff": int(n_eff), "W": float(W), "p": float(p),
            "HL": HL, "median": float(np.median(x))}

def per_mouse_lspec_cspec_ctrl(dfs, tst_only=False):
    """
    Returns per-mouse dataframe with:
      - proportions: Lspec, Cspec, Ctrl, and diffs (diff_spec = Lspec - Cspec, diff_ctrl = Lspec - Ctrl)
      - intensity contrasts (per session and per-mouse averages):
          lspec_other_L1_diff, lspec_other_L2_diff, lspec_other_mean_diff
          cspec_other_C1_diff, cspec_other_C2_diff, cspec_other_mean_diff

    Assumes each df has columns:
      - 'Mouse'
      - 'detected_in_sessions' (set/list/tuple of str: 's0','landmark1','landmark2','ctx1','ctx2')
      - intensity columns after detrending/standardization:
          'int_optimized_landmark1_rstd', 'int_optimized_landmark2_rstd',
          'int_optimized_ctx1_rstd',      'int_optimized_ctx2_rstd'
    """
    rows = []
    for i, df in enumerate(dfs):
        m, r = regions[i]
        if m == 13:
            continue

        sub = df.copy()

        # Optionally drop S0-only rows (if tst_only)
        S0 = (sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "s0")) )#& ~sub["is_dim_by_bg_s0"])
        if tst_only:
            sub = sub[~S0]

        # Depth/whatever detrending for intensity; keep index aligned to 'sub'
        #sub_tmp = cp.intensity_depth_detrend(sub, group_session_order)
        sub_tmp = sub.copy()
        # Session presence masks (boolean Series aligned to 'sub'/'sub_tmp')
        L1 = (sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "landmark1")))# & ~sub["is_dim_by_bg_landmark1"])
        L2 = (sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "landmark2")))# & ~sub["is_dim_by_bg_landmark2"])
        C1 = (sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "ctx1")) )#& ~sub["is_dim_by_bg_ctx1"])
        C2 = (sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "ctx2")))#& ~sub["is_dim_by_bg_ctx2"])

        # Cell classes
        Lspec = (L1 & L2) & (~C1 & ~C2)           # landmark-specific
        Cspec = (C1 & C2) & (~L1 & ~L2)           # context-specific
        Ctrl  = (L2 & C1) & (~L1 & ~C2)           # control (mixed)

        print("landmark specific ", Lspec.sum(), "ctx specific ", Cspec.sum(), "ctrl ", Ctrl.sum(), "out of ", len(sub))

        # --- Proportions per mouse ---
        p_Lspec = Lspec.mean()
        p_Cspec = Cspec.mean()
        p_Ctrl  = Ctrl.mean()

        # --- Intensity contrasts: L-specific vs other L-active ---
        L_active = (L1 | L2)
        L_other  = L_active #& (~Lspec)

        mean_L1_Lspec = sub_tmp.loc[L1 & Lspec,  "int_optimized_landmark1_rstd"].mean()
        mean_L1_Other = sub_tmp.loc[L1 & L_other, "int_optimized_landmark1_rstd"].mean()
        lspec_other_L1_diff = mean_L1_Lspec - mean_L1_Other

        mean_L2_Lspec = sub_tmp.loc[L2 & Lspec,  "int_optimized_landmark2_rstd"].mean()
        mean_L2_Other = sub_tmp.loc[L2 & L_other, "int_optimized_landmark2_rstd"].mean()
        lspec_other_L2_diff = mean_L2_Lspec - mean_L2_Other

        lspec_other_mean_diff = np.nanmean([lspec_other_L1_diff, lspec_other_L2_diff])

        # --- Intensity contrasts: C-specific vs other C-active ---
        C_active = (C1 | C2)
        C_other  = C_active #& (~Cspec)

        mean_C1_Cspec = sub_tmp.loc[C1 & Cspec,  "int_optimized_ctx1_rstd"].mean()
        mean_C1_Other = sub_tmp.loc[C1 & C_other, "int_optimized_ctx1_rstd"].mean()
        cspec_other_C1_diff = mean_C1_Cspec - mean_C1_Other

        mean_C2_Cspec = sub_tmp.loc[C2 & Cspec,  "int_optimized_ctx2_rstd"].mean()
        mean_C2_Other = sub_tmp.loc[C2 & C_other, "int_optimized_ctx2_rstd"].mean()
        cspec_other_C2_diff = mean_C2_Cspec - mean_C2_Other

        cspec_other_mean_diff = np.nanmean([cspec_other_C1_diff, cspec_other_C2_diff])
        
        # --- Intensity contrasts: control spec vs other control-active ---
        Ctrl_active = (L2 | C1)
        Ctrl_other  = Ctrl_active #& (~Ctrl)
        
        # L2 session (use LANDMARK2 intensity)
        mean_L2_Ctrl  = sub_tmp.loc[L2 & Ctrl,      "int_optimized_landmark2_rstd"].mean()
        mean_L2_Other = sub_tmp.loc[L2 & Ctrl_other,"int_optimized_landmark2_rstd"].mean()
        ctrl_other_L2_diff = mean_L2_Ctrl - mean_L2_Other
        
        # C1 session (use CTX1 intensity)
        mean_C1_Ctrl  = sub_tmp.loc[C1 & Ctrl,      "int_optimized_ctx1_rstd"].mean()
        mean_C1_Other = sub_tmp.loc[C1 & Ctrl_other,"int_optimized_ctx1_rstd"].mean()
        ctrl_other_C1_diff = mean_C1_Ctrl - mean_C1_Other
        
        # Per-mouse average across the two control sessions
        ctrl_other_mean_diff = np.nanmean([ctrl_other_L2_diff, ctrl_other_C1_diff])

        rows.append({
            "Mouse": m,
            # proportions
            "Lspec": p_Lspec, "Cspec": p_Cspec, "Ctrl": p_Ctrl,
            "diff_spec": p_Lspec - p_Cspec,       # Lspec minus Cspec
            "diff_ctrl": p_Lspec - p_Ctrl,        # Lspec minus Ctrl
            # intensity diffs (per session)
            "lspec_other_L1_diff": lspec_other_L1_diff,
            "lspec_other_L2_diff": lspec_other_L2_diff,
            "cspec_other_C1_diff": cspec_other_C1_diff,
            "cspec_other_C2_diff": cspec_other_C2_diff,
            # intensity diffs (per-mouse average over the two sessions)
            "lspec_other_mean_diff": lspec_other_mean_diff,
            "cspec_other_mean_diff": cspec_other_mean_diff,
            "ctrl_other_mean_diff": ctrl_other_mean_diff,
        })

    return pd.DataFrame(rows)


# --- 2) cluster bootstrap over mice ------------------------------------------
def cluster_bootstrap_lspec_cspec_ctrl(dfs, B=10000, rng_seed=0, stat="mean", tst_only=False):
    rng = np.random.default_rng(rng_seed)
    print("full population? ", str((not tst_only)))

    pm = per_mouse_lspec_cspec_ctrl(dfs, tst_only=tst_only)
    if pm.empty:
        raise ValueError("No eligible data.")

    reducer = np.mean if stat == "mean" else np.median
    
    pm_idx = pm.set_index("Mouse")
    mice = pm["Mouse"].to_numpy()
    M = len(mice)

    # --- Observed point estimates (proportions) ---
    point_diff_spec = reducer(pm["diff_spec"].to_numpy())
    point_diff_ctrl = reducer(pm["diff_ctrl"].to_numpy())

    # --- Observed point estimates (intensities; per-mouse averages) ---
    point_lspec_other = reducer(pm["lspec_other_mean_diff"].dropna().to_numpy()) \
                        if pm["lspec_other_mean_diff"].notna().any() else np.nan
    point_cspec_other = reducer(pm["cspec_other_mean_diff"].dropna().to_numpy()) \
                        if pm["cspec_other_mean_diff"].notna().any() else np.nan
    point_ctrl_other = reducer(pm["ctrl_other_mean_diff"].dropna().to_numpy()) \
                        if pm["ctrl_other_mean_diff"].notna().any() else np.nan

    # --- Bootstrap with cluster multiplicity preserved via .loc[sample] ---
    boots_spec = np.empty(B, dtype=float)
    boots_ctrl = np.empty(B, dtype=float)
    boots_lspec_other = np.empty(B, dtype=float)
    boots_cspec_other = np.empty(B, dtype=float)

    for b in range(B):
        sample = rng.choice(mice, size=M, replace=True)  # resample mice with replacement

        # proportions
        boots_spec[b] = reducer(pm_idx.loc[sample, "diff_spec"].to_numpy())
        boots_ctrl[b] = reducer(pm_idx.loc[sample, "diff_ctrl"].to_numpy())

        # intensities (drop NaNs within the resample for the reducer)
        vL = pm_idx.loc[sample, "lspec_other_mean_diff"].to_numpy()
        vC = pm_idx.loc[sample, "cspec_other_mean_diff"].to_numpy()
        vL = vL[~np.isnan(vL)]
        vC = vC[~np.isnan(vC)]
        boots_lspec_other[b] = reducer(vL) if vL.size else np.nan
        boots_cspec_other[b] = reducer(vC) if vC.size else np.nan

    def ci_percentile(a, alpha=0.05):
        a = a[~np.isnan(a)]
        if a.size == 0:
            return (np.nan, np.nan)
        lo, hi = np.percentile(a, [100*alpha/2, 100*(1-alpha/2)])
        return (float(lo), float(hi))

    # --- Wilcoxon per-mouse tests vs 0 (as before) ---
    wilc = {
        "spec_vs_zero": wilcoxon_with_effect(pm["diff_spec"]),
        "ctrl_vs_zero": wilcoxon_with_effect(pm["diff_ctrl"])
    }
    # Intensity contrasts vs 0 (drop NaNs)
    if pm["lspec_other_mean_diff"].notna().sum() >= 1:
        wilc["lspec_other_vs_zero"] = wilcoxon_with_effect(pm["lspec_other_mean_diff"].dropna())
    if pm["cspec_other_mean_diff"].notna().sum() >= 1:
        wilc["cspec_other_vs_zero"] = wilcoxon_with_effect(pm["cspec_other_mean_diff"].dropna())
    if pm["ctrl_other_mean_diff"].notna().sum() >= 1:
        wilc["ctrl_other_mean_diff"] = wilcoxon_with_effect(pm["ctrl_other_mean_diff"].dropna())

    return {
        "per_mouse": pm,
        # proportions
        "point_diff_spec": float(point_diff_spec),
        "ci_diff_spec": ci_percentile(boots_spec),
        "point_diff_ctrl": float(point_diff_ctrl),
        "ci_diff_ctrl": ci_percentile(boots_ctrl),
        # intensities (per-mouse average across the two sessions)
        "point_lspec_other": float(point_lspec_other) if not np.isnan(point_lspec_other) else np.nan,
        "ci_lspec_other": ci_percentile(boots_lspec_other),
        "point_cspec_other": float(point_cspec_other) if not np.isnan(point_cspec_other) else np.nan,
        "ci_cspec_other": ci_percentile(boots_cspec_other),
        "point_ctrl_other": float(point_ctrl_other),
        # tests
        "wilcoxon": wilc,
        # bootstrap info
        "B": int(B), "stat": stat
    }



def _top_quantile_share(x, q=0.75):
    if x.size == 0:
        return np.nan
    thr = np.quantile(x, q)
    return float((x >= thr).mean())

def per_mouse_top_enrichment(dfs, tst_only=False):
    """
    Per-mouse top-25% enrichment diffs for:
      - L-specific vs other L-active (average of L1/L2)
      - C-specific vs other C-active (average of C1/C2)
      - Ctrl (L2&C1) vs other (L2|C1) (average of L2/C1)
    Returns a DataFrame with one row per mouse.
    """
    rows = []
    for i, df in enumerate(dfs):
        m, r = regions[i]
        if m == 13:
            continue
        sub = df.copy()
        S0 = sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "s0"))
        if tst_only:
            sub = sub[~S0]
        sub_tmp = cp.intensity_depth_detrend(sub, group_session_order)

        L1 = sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "landmark1"))
        L2 = sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "landmark2"))
        C1 = sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "ctx1"))
        C2 = sub["detected_in_sessions"].apply(lambda s: utils.in_s(s, "ctx2"))

        Lspec = (L1 & L2) & ~(C1 | C2)
        Cspec = (C1 & C2) & ~(L1 | L2)
        Ctrl  = (L2 & C1) & ~(L1 | C2)

        # L sessions
        L_other = (L1 | L2) & ~Lspec
        shares = []
        for sess_mask, col in [(L1, "int_optimized_landmark1_rstd"),
                               (L2, "int_optimized_landmark2_rstd")]:
            pool = sub_tmp.loc[sess_mask, col].to_numpy()
            a = sub_tmp.loc[sess_mask & Lspec,  col].to_numpy()
            b = sub_tmp.loc[sess_mask & L_other, col].to_numpy()
            if a.size and b.size:
                share_a = _top_quantile_share(a, q=0.75)
                share_b = _top_quantile_share(b, q=0.75)
                shares.append(share_a - share_b)
        lspec_top25_diff = float(np.nanmean(shares)) if shares else np.nan

        # C sessions
        C_other = (C1 | C2) & ~Cspec
        shares = []
        for sess_mask, col in [(C1, "int_optimized_ctx1_rstd"),
                               (C2, "int_optimized_ctx2_rstd")]:
            a = sub_tmp.loc[sess_mask & Cspec,  col].to_numpy()
            b = sub_tmp.loc[sess_mask & C_other, col].to_numpy()
            if a.size and b.size:
                shares.append(_top_quantile_share(a) - _top_quantile_share(b))
        cspec_top25_diff = float(np.nanmean(shares)) if shares else np.nan

        # Control: L2 and C1 sessions
        Ctrl_other = (L2 | C1) & ~Ctrl
        shares = []
        for sess_mask, col in [(L2, "int_optimized_landmark2_rstd"),
                               (C1, "int_optimized_ctx1_rstd")]:
            a = sub_tmp.loc[sess_mask & Ctrl,      col].to_numpy()
            b = sub_tmp.loc[sess_mask & Ctrl_other,col].to_numpy()
            if a.size and b.size:
                shares.append(_top_quantile_share(a) - _top_quantile_share(b))
        ctrl_top25_diff = float(np.nanmean(shares)) if shares else np.nan

        rows.append({"Mouse": m,
                     "lspec_top25_diff": lspec_top25_diff,
                     "cspec_top25_diff": cspec_top25_diff,
                     "ctrl_top25_diff":  ctrl_top25_diff})
    return pd.DataFrame(rows)

def cluster_bootstrap_top_enrichment(dfs, B=10000, rng_seed=0, stat="mean", tst_only=False):
    rng = np.random.default_rng(rng_seed)
    pm = per_mouse_top_enrichment(dfs, tst_only=tst_only).dropna()
    if pm.empty:
        raise ValueError("No eligible data.")
    reducer = np.mean if stat == "mean" else np.median
    pm_idx = pm.set_index("Mouse")
    mice = pm["Mouse"].to_numpy(); M = len(mice)

    obs = {k: reducer(pm[k].to_numpy()) for k in ["lspec_top25_diff","cspec_top25_diff","ctrl_top25_diff"]}
    boots = {k: np.empty(B, float) for k in obs}
    for b in range(B):
        sample = rng.choice(mice, size=M, replace=True)
        for k in obs:
            v = pm_idx.loc[sample, k].to_numpy()
            v = v[~np.isnan(v)]
            boots[k][b] = reducer(v) if v.size else np.nan

    def ci(a): 
        a = a[~np.isnan(a)]
        return (np.nan, np.nan) if a.size==0 else tuple(np.percentile(a, [2.5, 97.5]))

    tests = {}
    for k in obs:
        x = pm[k].dropna().to_numpy()
        tests[k] = {"wilcoxon": (wilcoxon_with_effect(x)) if x.size>=1 else (np.nan, np.nan), "n": int(x.size)}

    return {"per_mouse": pm, "point": obs, "ci": {k: ci(boots[k]) for k in obs}, "tests": tests}


#%%
res_full = cluster_bootstrap_lspec_cspec_ctrl(dfs, B=10000, stat='median', tst_only = False)
res_no_tst = cluster_bootstrap_lspec_cspec_ctrl(dfs, B=10000, stat='median', tst_only = True)
df_full = res_full["per_mouse"].copy()
df_no_tst = res_no_tst["per_mouse"].copy()

#%%
enrichment_res = cluster_bootstrap_top_enrichment(dfs, stat = "median", tst_only=False)
enrichment_res_tst = cluster_bootstrap_top_enrichment(dfs, stat = "median", tst_only=True)
#%%
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
for k, v in res_full.items():
    if k != "per_mouse":
        print(k, v , "\n")
print("===========================")
for k, v in res_no_tst.items():
    if k != "per_mouse":
        print(k, v , "\n")
print("===========================")
for k, v in enrichment_res.items():
    if k != "per_mouse":
        print(k, v , "\n")
print("===========================")
for k, v in enrichment_res_tst.items():
    if k != "per_mouse":
        print(k, v , "\n")
print("===========================")
#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(res_full["per_mouse"])
    print("===========================")
    print(res_no_tst["per_mouse"])
    print("===========================")
    print(enrichment_res["per_mouse"])
    print("===========================")
    print(enrichment_res_tst["per_mouse"])
    print("===========================")

#%%
fig, axes = plt.subplots(1, 2, figsize=(9, 4))#, gridspec_kw={'width_ratios':[2, 1]})

# Left panel: paired points
for _, row in df_full.iterrows():
    axes[0].plot(["Lspec", "Cspec"], [row["Lspec"], row["Cspec"]], color="gray", alpha=0.5)
    axes[0].plot(["Lspec", "Ctrl"], [row["Lspec"], row["Ctrl"]], color="lightblue", alpha=0.5)

sns.pointplot(data=df_full.melt(id_vars="Mouse", value_vars=["Lspec", "Cspec", "Ctrl"]),
              x="variable", y="value", join=False, ax=axes[0], color="black")
axes[0].set_xticklabels(["L-spec", "C-spec", "Kontrola"])
axes[0].set_ylabel("Proporcja")
axes[0].set_xlabel("")
axes[0].set_title("Cała populacja")
for _, row in df_no_tst.iterrows():
    axes[1].plot(["Lspec", "Cspec"], [row["Lspec"], row["Cspec"]], color="gray", alpha=0.5)
    axes[1].plot(["Lspec", "Ctrl"], [row["Lspec"], row["Ctrl"]], color="lightblue", alpha=0.5)
sns.pointplot(data=df_no_tst.melt(id_vars="Mouse", value_vars=["Lspec", "Cspec", "Ctrl"]),
              x="variable", y="value", join=False, ax=axes[1], color="black")
axes[1].set_xticklabels(["L-spec", "C-spec", "Kontrola"])
axes[1].set_ylabel("")
axes[1].set_xlabel("")
axes[1].set_title("Komórki selektywne dla testu")

fig.suptitle("Proporcje komórek selektywnych względem typu sesji", fontsize=14, fontweight='bold')
plt.tight_layout()
fig.text(0.5, 0.04, "Rodzaj sesji", ha='center', fontsize=12)
plt.subplots_adjust(top=0.8, bottom=0.2) 
ylims = axes[1].get_ylim()  # current limits
y_ast = 0.07 + 0.005  # a bit above the highest L-spec point

axes[1].annotate("*",
                 xy=(0.5, y_ast),
                 xycoords='data',
                 ha='center', va='bottom',
                 fontsize=16, fontweight='bold')

# Force same y-limits for both panels
axes[0].set_ylim(0.01, 0.095)  # adjust min/max if needed
axes[1].set_ylim(0.01, 0.095)
plt.show()