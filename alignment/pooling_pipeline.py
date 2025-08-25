#%% imports

import sys
import yaml

# custom modules
import intersession as its
import cell_preprocessing as cp
import cell_classification as cc
import utils
import plotting

import pandas as pd

#%% config

config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


SOURCE_DIR_PATH = config["experiment"]["dir_path"]
ICY_PATH = SOURCE_DIR_PATH + config["experiment"]["path_for_icy"]

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
RESULT_PATH = config["experiment"]["result_path"]

regions = config["experiment"]["regions"]
group_session_order = config["experiment"]["session_order"][1]

optimized_fname = config["filenames"]["cell_data_opt_template"]
pooled_cells_fname = config["filenames"]["pooled_cells"]

#%% ready regions

regions = [[13,2],[14,1], [16, 1], [8,1], [20,2], [2,2], [5,1], [10,1], [11,1]]
regions = [[1,2], [3,2],[4,1], [7,2], [9,1],[12,1], [17,1], [18,2]]

#%% reading  and prepping detection results from icy

for mouse, region in regions:
    sessions = utils.read_single_session_cell_data(
        mouse, 
        region, 
        group_session_order, 
        config, 
        test=False, 
        optimized=False
        )
    
    imgs = utils.read_images(mouse, region, group_session_order, config)
    
    for i, s in enumerate(sessions):
        s = s.loc[s["Interior (px)"]>150].copy()
        # tolerance 2, because I don't expect icy to place the centroid outside 
        # of the cell, which is about 7 px in xy
        s = cp.optimize_centroids(s, imgs[i], suff="", tolerance = 2)
        s.to_csv(ICY_PATH + optimized_fname.format(
            mouse, 
            region, 
            group_session_order[i]))
        sessions[i] = s
#%% pooling

#!warning! here tolerance is not in pixels, but um
for mouse, region in regions:        
    df_reseg = its.pool_cells_globally(mouse, region, group_session_order, config, 7)
    for sid in group_session_order:
        img = utils.read_image(mouse, region, sid, config)
        df_reseg = cp.optimize_centroids(df_reseg, img, suff=f"_{sid}", tolerance=3, update_coords=False)
    df_reseg.to_csv(RESULT_PATH+pooled_cells_fname.format(mouse, region))
   
#%% here i'm looking at how cells detected in session differ from the ones undetected in terms of fluorescence in that session
df = utils.read_pooled_with_background(1, 1, config)
plotting.session_detection_vs_background(df, group_session_order, sub_bgr = True)

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1]]

#%%
dfs = []
for mouse, region in regions:
    dfs += [utils.read_pooled_with_background(mouse, region, config)]

#%%
pairs = list(zip(group_session_order, group_session_order[1:]))
plotting.plot_cohort_tendencies(regions, pairs, config, groups=["on", "off", "const"], dfs=dfs)
#plotting.plot_cohort_tendencies(regions, pairs, config, groups=["const"], dfs=dfs)
#plotting.plot_cohort_tendencies(regions, pairs, config, groups=["up", "down", "stable"], ttype="", dfs=dfs)

#%%
classes = ["landmark_specific", "ctx_specific", "is_mixed", "test_specific"]
plotting.plot_class_distribution(regions, config, classes, dfs=dfs)

#%%
classes = ["is_transient", "is_intermediate", "is_persistent"]
plotting.plot_class_distribution(regions, config, classes)

#%%
dfs_trans = []  # collect all mouse-region DataFrames

for i, df in enumerate(dfs):
    m, r = regions[i]
    dfi = cc.mark_cells_specificity_class(df).copy()
    dfi = dfi.loc[dfi["is_transient"]]

    dfi["sid"] = dfi["detected_in_sessions"].apply(lambda s: list(s)[0])
    dfi["Mouse"] = m  # add mouse ID

    total_cells = dfi.shape[0]  # number of transient cells for this mouse

    # Count cells per session
    tst = dfi.groupby("sid").size().reset_index(name="count")
    tst["Mouse"] = m
    tst["fraction"] = tst["count"] / total_cells

    dfs_trans.append(tst)
plot_df = pd.concat(dfs_trans, ignore_index=True)

#%%
dfs_trans = []

for i, df in enumerate(dfs):
    m, r = regions[i]
    df = cc.mark_cells_specificity_class(df)
    df["Mouse"] = m
    
    all_transient = df[df["is_transient"]].shape[0]
    for sid in group_session_order:
        # Count all cells detected in this session
        all_count = df["detected_in_sessions"].apply(lambda s: sid in s).sum()

        # Count transient cells detected in this session
        transient_count = df[df["is_transient"]]["detected_in_sessions"].apply(lambda s: sid in s).sum()

        dfs_trans.append({
            "Mouse": m,
            "sid": sid,
            "transient_count": transient_count,
            "all_count": all_count,
            "fraction": transient_count / all_transient/all_count if all_count > 0 else 0
        })
plot_df = pd.DataFrame(dfs_trans)

#%%
plotting.generic_class_plot(plot_df, "sid", "fraction", "Transient cells per session", hue="Mouse")
#%%
dfs_trans = []
pairs = list(zip(group_session_order, group_session_order[1:]))+[["s0", "landmark2"], ["s0", "ctx1"], ["s0", "ctx2"]]
for i, df in enumerate(dfs):
    m, r = regions[i]
    if m ==6:
        continue

    for idpair in pairs:
        ex_pair = cc.cell_count_for_sessions(df, list(idpair), exclusive=True)

        ex_pair /= df.shape[0]

        dfs_trans.append({
            "Mouse": m,
            "pair_id": f"{idpair[0]}_and_{idpair[1]}",
            "fraction": ex_pair
        })
plot_df = pd.DataFrame(dfs_trans)

plotting.generic_class_plot(plot_df, "pair_id", "fraction", "Cells specific for pairs", hue="Mouse")

#%%

#%%

from upsetplot import UpSet, from_indicators
import pandas as pd

all_cells = pd.concat(dfs)
all_cells['detected_in_sessions'] = all_cells['detected_in_sessions'].apply(frozenset)
for session in group_session_order:
    all_cells[session] = all_cells['detected_in_sessions'].apply(lambda x: session in x).astype(bool)
# grouped = all_cells.groupby(group_session_order).size()
# upsetplot.plot(grouped) 




# 1) pick your exact session order (rows order in the matrix)
session_order = ["s0", "landmark1", "landmark2", "ctx1", "ctx2"]   # <- your order

# df_cells must have boolean indicator columns for each session in session_order
# e.g., df_cells["landmark1"] = df_cells["detected_in_sessions"].apply(lambda s: "landmark1" in s)

# Build the Series for upsetplot, in the column order you want
s = from_indicators(session_order, all_cells[session_order])

# 2) custom intersection order:
#    sort by degree (number of sessions in intersection), and inside *each degree*
#    sort by decreasing size
tmp = s.index.to_frame(index=False)   # columns: same as session_order


tmp["size"] = s.value_counts()
tmp["degree"] = tmp[session_order].sum(axis=1)

# Sort intersections: first by degree (e.g., high→low), then by size (high→low)
order_idx = (tmp.sort_values(["degree", "size"], ascending=[False, False])
               .set_index(session_order)
               .index)

# Reindex the Series to this custom order
s_sorted = s.reindex(order_idx)

# 3) plot: disable built-in sorting so our order is respected
up = UpSet(
    s_sorted,
    sort_by=None,                 # don't re-sort intersections
    sort_categories_by=None,      # don't re-sort row categories
    show_counts=True
)
up.plot()

#%%
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, t

# --- 1) per-mouse proportions -----------------------------------------------
def _contains(sess, key):
    if isinstance(sess, (set, list, tuple)):
        return key in sess
    # if it's a string like "['l1','l2']" try a safe fallback
    try:
        return key in set(sess)
    except Exception:
        return False

def per_mouse_lspec_cspec_ctrl(dfs, tst_only = False):
    """
    Returns per-mouse dataframe with p_Lspec, p_Cspec, n_eligible.
    Assumes df has columns: Mouse, detected_in_sessions (set/list/tuple of str).
    """
    rows = []
    for i, df in enumerate(dfs):
        m,r = regions[i]
        if m==13:
            continue
        sub =df.copy()
       
        S0 = sub["detected_in_sessions"].apply(lambda s: _contains(s, "s0"))
        if tst_only:
            sub = sub[~S0]
        sub_tmp = cp.intensity_depth_detrend(sub, group_session_order)

        L1 = sub["detected_in_sessions"].apply(lambda s: _contains(s, "landmark1"))
        L2 = sub["detected_in_sessions"].apply(lambda s: _contains(s, "landmark2"))
        C1 = sub["detected_in_sessions"].apply(lambda s: _contains(s, "ctx1"))
        C2 = sub["detected_in_sessions"].apply(lambda s: _contains(s, "ctx2"))
        # Definitions
        Lspec = (L1 & L2) & (~C1 & ~C2)
        Cspec = (C1 & C2) & (~L1 & ~L2)
        Ctrl  = (L2 & C1) & (~L1 & ~C2)

        # Proportions
        
        print("landmark ",
              sub_tmp.loc[Lspec]["int_optimized_landmark1_rstd"].mean(),
              sub_tmp.loc[Lspec]["int_optimized_landmark2_rstd"].mean()
              )
        
        print("ctx ",
              sub_tmp.loc[Cspec]["int_optimized_ctx1_rstd"].mean(),
              sub_tmp.loc[Cspec]["int_optimized_ctx2_rstd"].mean()
              )
        
        print("mixed ",
              sub_tmp.loc[Ctrl]["int_optimized_landmark2_rstd"].mean(),
              sub_tmp.loc[Ctrl]["int_optimized_ctx1_rstd"].mean()
              )
        
        p_Lspec = Lspec.mean()
        p_Cspec = Cspec.mean()
        p_Ctrl  = Ctrl.mean()

        rows.append({
            "Mouse": m,
            "Lspec": p_Lspec,
            "Cspec": p_Cspec,
            "Ctrl":  p_Ctrl,
            "diff_spec": p_Lspec - p_Cspec,
            "diff_ctrl": p_Lspec - p_Ctrl
        })

    return pd.DataFrame(rows)

# --- 2) cluster bootstrap over mice ------------------------------------------
def cluster_bootstrap_lspec_cspec_ctrl(df_cells, B=10000, rng_seed=0, stat="mean", tst_only = False):
    rng = np.random.default_rng(rng_seed)
    print("full population? ", str((not tst_only)))
    pm = per_mouse_lspec_cspec_ctrl(df_cells, tst_only = tst_only)
    if pm.empty:
        raise ValueError("No eligible data.")

    reducer = np.mean if stat == "mean" else np.median
    mice = pm["Mouse"].tolist()

    # Observed
    obs = {col: reducer(pm[col]) for col in ["diff_spec", "diff_ctrl"]}

    boots_spec = np.empty(B)
    boots_ctrl = np.empty(B)
    for b in range(B):
        sample = rng.choice(mice, size=len(mice), replace=True)
        boots_spec[b] = reducer(pm.loc[pm["Mouse"].isin(sample), "diff_spec"])
        boots_ctrl[b] = reducer(pm.loc[pm["Mouse"].isin(sample), "diff_ctrl"])

    def ci_percentile(samples):
        return tuple(np.percentile(samples, [2.5, 97.5]))

    return {
        "per_mouse": pm,
        "point_diff_spec": obs["diff_spec"], "ci_diff_spec": ci_percentile(boots_spec),
        "point_diff_ctrl": obs["diff_ctrl"], "ci_diff_ctrl": ci_percentile(boots_ctrl),
        "wilcoxon": {
            "spec_vs_zero": wilcoxon(pm["diff_spec"]),
            "ctrl_vs_zero": wilcoxon(pm["diff_ctrl"])
        }
    }
#%%
# Example usage:
res_full = cluster_bootstrap_lspec_cspec_ctrl(dfs, B=10000, stat='median', tst_only = False)
res_no_tst = cluster_bootstrap_lspec_cspec_ctrl(dfs, B=10000, stat='median', tst_only = True)
# print(res["per_mouse"])
# print(res["point_diff_spec"], res["ci_diff_spec"])
# print(res["point_diff_ctrl"], res["ci_diff_ctrl"])
# print(res["wilcoxon"])
#%%
import matplotlib.pyplot as plt, seaborn as sns
df_full = res_full["per_mouse"].copy()
df_no_tst = res_no_tst["per_mouse"].copy()
def bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(data, size=len(data), replace=True).mean()
                  for _ in range(n_boot)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.median(data), lower, upper

# Get bootstrap results
boot_results = {
    "Lspec − Cspec": bootstrap_ci(df_full["diff_spec"]),
    "Lspec − Ctrl":  bootstrap_ci(df_full["diff_ctrl"])
}

boot_df_full = pd.DataFrame(boot_results, index=["median", "ci_low", "ci_high"]).T

boot_results_no_tst = {
    "Lspec − Cspec": bootstrap_ci(df_no_tst["diff_spec"]),
    "Lspec − Ctrl":  bootstrap_ci(df_no_tst["diff_ctrl"])
}

boot_df_np_tst = pd.DataFrame(boot_results_no_tst, index=["median", "ci_low", "ci_high"]).T
#%%
# ==== PLOT ====
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
axes[0].set_ylim(0.01, 0.09)  # adjust min/max if needed
axes[1].set_ylim(0.01, 0.09)
plt.show()
