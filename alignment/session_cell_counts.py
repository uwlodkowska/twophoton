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

config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/multisession.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


SOURCE_DIR_PATH = config["experiment"]["dir_path"]
ICY_PATH = SOURCE_DIR_PATH + config["experiment"]["path_for_icy"]

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
RESULT_PATH = config["experiment"]["result_path"]

regions = config["experiment"]["regions"]
SESSIONS = config["experiment"]["session_order"][0]

optimized_fname = config["filenames"]["cell_data_opt_template"]
pooled_cells_fname = config["filenames"]["pooled_cells"]
#%% df building helpers


pairs = list(zip(SESSIONS, SESSIONS[1:]))

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1], [13,1]]
#%%

mouse_ids = [r[0] for r in regions]
print(mouse_ids)
#%%
dfs = []
for mouse, region in regions:
    df_mouse = utils.read_pooled_with_background(mouse, region, config)
    df_mouse["Mouse"] = mouse
    dfs += [df_mouse]
#%%

from scipy.stats import ttest_1samp, wilcoxon, t

def per_mouse_counts(df):
    # expects a column 'detected_in_sessions' with sets/lists like {'s0','landmark1',...}
    labs = ["s0","landmark1","landmark2","ctx1","ctx2"]
    counts = {k: 0 for k in ["s0","L1","L2","C1","C2"]}
    for s in df["detected_in_sessions"]:
        if not isinstance(s, (set,list,tuple)): continue
        counts["s0"] += ("s0" in s)
        counts["L1"] += ("landmark1" in s)
        counts["L2"] += ("landmark2" in s)
        counts["C1"] += ("ctx1" in s)
        counts["C2"] += ("ctx2" in s)
    return counts

def build_ratios_by_mouse(dfs, reference="Lmean"):
    rows = []
    for d in dfs:
        mouse = d["Mouse"].iloc[0]
        c = per_mouse_counts(d)
        nS0 = c["s0"]; nL1, nL2 = c["L1"], c["L2"]; nC1, nC2 = c["C1"], c["C2"]
        L_vals = [v for v in [nL1, nL2] if np.isfinite(v)]
        C_vals = [v for v in [nC1, nC2] if np.isfinite(v)]
        if mouse == 13:
            C_vals = [v for v in [nC1] if np.isfinite(v)]
        nL = np.mean(L_vals) if L_vals else np.nan
        nC = np.mean(C_vals) if C_vals else np.nan
        if reference.lower() == "lmean":
            r_S0 = nS0 / nL if (nL and nL>0) else np.nan
            r_C  = nC  / nL if (nL and nL>0) else np.nan
            rows.append({"mouse": mouse, "ratio_S0_vs_L": r_S0, "ratio_C_vs_L": r_C})
        else:  # reference = S0
            r_L = nL / nS0 if (nS0 and nS0>0) else np.nan
            r_C = nC / nS0 if (nS0 and nS0>0) else np.nan
            rows.append({"mouse": mouse, "ratio_L_vs_S0": r_L, "ratio_C_vs_S0": r_C})
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
res_S0 = summarize_logratio(pm["ratio_S0_vs_L"])    # S0 vs L reference
res_C  = summarize_logratio(pm["ratio_C_vs_L"])     # C vs L reference
print(res_S0)
print("======================")
print(res_C)
#%%
pm = build_ratios_by_mouse(dfs, reference="S0")  # or "S0"
res_S0 = summarize_logratio(pm["ratio_L_vs_S0"])    # S0 vs L reference
res_C  = summarize_logratio(pm["ratio_C_vs_S0"])     # C vs L reference
print(res_S0)
print("======================")
print(res_C)
