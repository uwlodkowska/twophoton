

import sys
import yaml

# custom modules
import cell_classification as cc
import cell_preprocessing as cp
import utils
import numpy as np, patsy as pt
import matplotlib.pyplot as plt

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pandas as pd
if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items
from scipy.special import expit as logistic
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats

from constants import ICY_COLNAMES
import statsmodels.api as sm
import gee_plotting as gp
from scipy.stats import f as fdist


#%% config
idx=0
suff=""
with open("config_files/ctx_landmark.yaml", "r") as file:
    config = yaml.safe_load(file)

regions = config["experiment"]["regions"][idx]
SESSIONS = config["experiment"]["session_order"][idx]

dfs = []
for mouse, region in regions:
    df_mouse = utils.read_pooled_with_background(mouse, region, config, idx=idx)
    df_mouse = cp.intensity_depth_detrend(df_mouse, SESSIONS)
    df_mouse["mouse"] = f'{mouse}{suff}'
    df_mouse[ICY_COLNAMES['zcol']] = df_mouse[ICY_COLNAMES['zcol']]/df_mouse[ICY_COLNAMES['zcol']].max()
    df_mouse["cell_id"] = df_mouse.index.map(lambda i: f"{mouse}{suff}_{i}")
    dfs += [df_mouse]
all_mice = pd.concat(dfs)

#%% df building helpers

pairs = list(zip(SESSIONS, SESSIONS[1:]))


#%%
# s = "landmark"
# for df in dfs:
#     print(df[f"is_dim_by_bg_{s}"].mean())
#     df_tmp = df.loc[df[f"is_dim_by_bg_{s}"]]
#     ssets = df_tmp["detected_in_sessions"].map(_parse_setlike)
#     print((ssets.apply(lambda x: s in x)).sum())
#     plt.hist(df_tmp[ICY_COLNAMES['zcol']], bins=30)
#     plt.show()

#%%

from scipy.stats import ttest_1samp, wilcoxon, t

def per_mouse_counts(df):
    #df= df.loc[df["n_sessions"]<3]
    # expects a column 'detected_in_sessions' with sets/lists like {'s0','landmark1',...}
    labs = SESSIONS
    counts = {k: 0 for k in SESSIONS}
    df = df.loc[df["n_sessions"]!=3]
    for sid in SESSIONS:
        tmp_df = df.copy()
        tmp_df = tmp_df.loc[~tmp_df[f'is_dim_by_bg_{sid}']]
        for s in tmp_df["detected_in_sessions"]:
            if not isinstance(s, (set,list,tuple)): continue
            counts[sid] += (sid in s)
    print(counts)
    return counts

def build_ratios_by_mouse(dfs, reference=SESSIONS[0]):
    rows = []
    for d in dfs:
        mouse = d["mouse"].iloc[0]
        c = per_mouse_counts(d)


        r_diff = c[SESSIONS[0]] / c[SESSIONS[1]]
        r_same  = c[SESSIONS[1]] / c[SESSIONS[2]]
        rows.append({"mouse": mouse, "ratio_diff": r_diff, "ratio_same": r_same})

    return pd.DataFrame(rows)

def summarize_logratio(vec):
    vec = np.asarray(vec, float)
    vec = vec[~np.isnan(vec)]
    G = vec.size
    out = {"n": int(G)}
    if G < 2:
        out.update({"gmean_ratio": np.nan, "ci95": (np.nan, np.nan), "t_df": None, "t": np.nan, "p_t": np.nan,
                    "wilcoxon_Wplus": np.nan, "p_wilcoxon": np.nan, "HL_median_ratio": np.nan})
        return out
    y = np.log(vec)
    tstat, p_t = ttest_1samp(y, 0.0)
    ybar = y.mean()
    se = y.std(ddof=1)/np.sqrt(G)
    crit = t.ppf(0.975, G-1)
    ci = (np.exp(ybar - crit*se), np.exp(ybar + crit*se))
    # Wilcoxon on raw ratios vs 1 (median estimand)
    w = wilcoxon(vec - 1.0, zero_method="wilcox", method="auto")
    HL = float(np.median(vec))
    out.update({"gmean_ratio": float(np.exp(ybar)), "ci95": (float(ci[0]), float(ci[1])),
                "t_df": int(G-1), "t": float(tstat), "p_t": float(p_t),
                "wilcoxon_Wplus": float(w.statistic), "p_wilcoxon": float(w.pvalue),
                "HL_median_ratio": HL})
    return out

#%%
pm = build_ratios_by_mouse(dfs, reference="Lmean")  # or "S0"
print(pm)
res_diff = summarize_logratio(pm["ratio_diff"])    # S0 vs L reference
res_same  = summarize_logratio(pm["ratio_same"])     # C vs L reference
print(res_diff)
print("======================")
print(res_same)

