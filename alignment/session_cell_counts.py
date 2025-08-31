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
with open("config_files/multisession.yaml", "r") as file:
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

#%%
SESSION_MAP = {
    "s0": "S0",
    "landmark1": "L1",
    "landmark2": "L2",
    "ctx1": "C1",
    "ctx2": "C2",  # keep out of LLCC unless you explicitly want 5 levels
}
from scipy.stats import ttest_1samp, wilcoxon, t

def per_mouse_counts(df):
    # expects a column 'detected_in_sessions' with sets/lists like {'s0','landmark1',...}
    labs = ["s0","landmark1","landmark2","ctx1","ctx2"]
    counts = {k: 0 for k in ["S0","L1","L2","C1","C2"]}
    for sid in labs:
        if f'is_dim_by_bg_{sid}' not in df.columns:
            continue
        tmp_df = df.copy()
        tmp_df = tmp_df.loc[~tmp_df[f'is_dim_by_bg_{sid}']]
        for s in tmp_df["detected_in_sessions"]:
            counts[SESSION_MAP[sid]] += (sid in s)
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


# ===== ANOVA on L1-normalized counts + planned log(L1/S0) test =====
try:
    import pingouin as pg
except ImportError as e:
    raise SystemExit("Please `pip install pingouin` to enable rm_anova + sphericity.") from e

# Map raw session keys -> short labels used in the ANOVA figure/story

def side_filter_baseline_engaged_dynamic(d, min_tests=1, drop_always=True):
    Tcount = d["detected_in_sessions"].apply(
        lambda s: sum(utils.in_s(s, t) for t in ["landmark1","landmark2","ctx1","ctx2"])
    )
    mask = ((Tcount >= min_tests) & (Tcount<4))
    if drop_always:
        mask = mask & (~d["n_sessions"].eq(5))
    return d.loc[mask].copy()

def counts_long_per_mouse_from_dfs(dfs, session_map=SESSION_MAP):
    """
    Reuse your per_mouse_counts() to build a tidy long table with per-mouse counts.
    Returns columns: mouse, session, count
    """
    rows = []
    for d in dfs:
        mouse_id = d["mouse"].iloc[0]
        
        d = side_filter_baseline_engaged_dynamic(d)

        c = per_mouse_counts(d)  # uses your existing logic
        for raw, short in session_map.items():
            if raw == "s0":
                cnt = c["S0"]
            elif raw == "landmark1":
                cnt = c["L1"]
            elif raw == "landmark2":
                cnt = c["L2"]
            elif raw == "ctx1":
                cnt = c["C1"]
            elif raw == "ctx2":
                cnt = c["C2"]
            else:
                continue
            rows.append({"mouse": mouse_id, "session": short, "count": int(cnt)})
    return pd.DataFrame(rows)

def l1_normalize_long(long_counts):
    """
    For each mouse, divide every session's count by that mouse's L1 count.
    Drops mice with missing/nonpositive L1.
    Returns columns: mouse, session, count, norm
    """
    wide = long_counts.pivot(index="mouse", columns="session", values="count")
    if "S0" not in wide.columns:
        raise ValueError("L1 column missing; cannot normalize by L1.")
    keep = wide.index[(wide["S0"].notna()) & (wide["S0"] > 0)]
    wide = wide.loc[keep]
    norm = wide.divide(wide["S0"], axis=0)
    long_norm = norm.reset_index().melt(id_vars="mouse", var_name="session", value_name="norm")
    # merge back counts (handy to keep)
    out = long_counts[long_counts["mouse"].isin(keep)].merge(long_norm, on=["mouse","session"], how="inner")
    return out

def rm_anova_l1_normalized(long_norm):
    """
    Repeated-measures ANOVA across sessions (S0, L1, L2, C1) on L1-normalized counts.
    Includes sphericity test and GG/HF epsilons (Pingouin).
    """
    wanted = ["L1","L2","C1", "C2"]
    d = long_norm[long_norm["session"].isin(wanted)].copy()
    d=d.loc[~(d["mouse"]=="13")]
    print(d)
    # drop incomplete mice (for ANOVA only)
    wide = d.pivot(index="mouse", columns="session", values="norm")
    complete = wide.dropna(axis=0, how="any").index
    d_comp = d[d["mouse"].isin(complete)].copy()
    print(d_comp)
    if len(complete) < 2 or d_comp["session"].nunique() < 2:
        return {"message": "Not enough complete mice or session levels for RM-ANOVA.",
                "dropped_mice": list(set(d["mouse"]) - set(complete))}
    # sphericity + eps

    #W, chi2, df_sph, p_sph = pg.sphericity(d_comp, dv="norm", subject="mouse", within="session")
    eps = pg.epsilon(d_comp, dv="norm", subject="mouse", within="session")  # {'gg':..., 'hf':...}
    # RM-ANOVA (GG applied if needed)
    anova = pg.rm_anova(data=d_comp, dv="norm", within="session",
                        subject="mouse", detailed=True, correction=True, effsize="np2")
    return {
        "anova_table": anova,
        "epsilon": eps,
        "wide_complete": wide.loc[complete],
        "dropped_mice": list(set(d["mouse"]) - set(complete))
    }

def planned_log_L1_over_S0(long_counts):
    """
    Planned novelty test exactly per your pipeline:
      - compute per-mouse ratio R = L1/S0
      - test mean(log R) vs 0 (one-sample t-test)
      - also return your summarize_logratio() on R for geom. mean & Wilcoxon
    """
    wide = long_counts.pivot(index="mouse", columns="session", values="count")
    if not {"L1","S0"}.issubset(wide.columns):
        return {"message": "Need both L1 and S0 to compute logratio."}
    sub = wide.dropna(subset=["L1","S0"]).copy()
    sub = sub[(sub["L1"] > 0) & (sub["S0"] > 0)]
    if len(sub) < 2:
        return {"message": "Not enough mice with positive L1 and S0."}

    ratios = (sub["L1"] / sub["S0"]).values
    # your helper returns t/Wilcoxon + CI on the ratio scale
    summary = summarize_logratio(ratios)

    # also expose the exact one-sample t-test on log-ratios (redundant with summarize_logratio but explicit)
    logR = np.log(ratios)
    t_stat, p_t = stats.ttest_1samp(logR, popmean=0.0)
    summary.update({
        "t_stat_logratio": float(t_stat),
        "p_t_logratio": float(p_t),
        "n_used": int(len(logR)),
    })
    return summary

from itertools import combinations
from scipy.stats import shapiro

def shapiro_on_pairwise_diffs(long_df, dv="norm", subject="mouse", within="session"):
    """
    Shapiro–Wilk normality tests on within-mouse paired differences
    for all session contrasts used in the RM-ANOVA.
    """
    wide = long_df.pivot(index=subject, columns=within, values=dv).dropna()
    levels = list(wide.columns)
    rows = []
    for a, b in combinations(levels, 2):
        diffs = (wide[a] - wide[b]).dropna()
        if len(diffs) >= 3:
            W, p = shapiro(diffs)
        else:
            W, p = float("nan"), float("nan")
        rows.append({"contrast": f"{a} - {b}", "n": int(len(diffs)), "W": W, "p": p})
    return pd.DataFrame(rows)

def shapiro_logratio(long_counts, num="L1", den="S0"):
    wide = long_counts.pivot(index="mouse", columns="session", values="count")
    sub = wide.dropna(subset=[num, den])
    sub = sub[(sub[num] > 0) & (sub[den] > 0)]
    if len(sub) < 3:
        return {"n": len(sub), "W": float("nan"), "p": float("nan")}
    lr = np.log(sub[num] / sub[den])
    W, p = shapiro(lr)
    return {"n": int(len(lr)), "W": float(W), "p": float(p)}

SESSIONS_RAW = {"s0":"S0","landmark1":"L1","landmark2":"L2","ctx1":"C1","ctx2":"C2"}

def counts_long_from_filtered(dfs):
    rows_all, rows_side = [], []
    for d in dfs:
        mouse = d["mouse"].iloc[0]

        # -- ALL cells for S0 denominator
        c_all = per_mouse_counts(d)
        for raw, short in SESSIONS_RAW.items():
            rows_all.append({"mouse": mouse, "session": short, "count": int(c_all[short])})

        # -- FILTERED cells for test numerators
        df = side_filter_baseline_engaged_dynamic(d)
        c_side = per_mouse_counts(df)
        for raw, short in SESSIONS_RAW.items():
            rows_side.append({"mouse": mouse, "session": short, "count": int(c_side[short])})

    long_all  = pd.DataFrame(rows_all)
    long_side = pd.DataFrame(rows_side)
    return long_all, long_side

def s0_normalize_external(long_all, long_side):
    # S0 from ALL cells
    s0 = (long_all.query("session=='S0'")[["mouse","count"]]
                 .rename(columns={"count":"S0"}))

    # Use FILTERED counts for tests; keep only tests
    d = long_side[long_side["session"].isin(["L1","L2","C1","C2"])].merge(s0, on="mouse", how="left")
    d = d[d["S0"] > 0].copy()
    d["norm"] = d["count"] / d["S0"]
    return d

def rm_anova_L_only(long_norm):
    wide = long_norm.pivot(index="mouse", columns="session", values="norm")
    complete = wide.dropna(axis=0, how="any").index
    d_comp = long_norm[long_norm["mouse"].isin(complete)].copy()
    eps = pg.epsilon(d_comp, dv="norm", subject="mouse", within="session")
    anova = pg.rm_anova(d_comp, dv="norm", within="session",
                        subject="mouse", detailed=True, correction=True, effsize="np2")
    return anova, eps, wide.loc[complete]

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    long_all, long_side = counts_long_from_filtered(dfs)  # or _2plus
    long_norm = s0_normalize_external(long_all, long_side)
    
    print(long_all.head(10))
    print(long_side.head(10))
    print(long_norm.head(10))
    
    
    anova, eps, wide = rm_anova_L_only(long_norm)
    print(anova); print(eps)
    
    
    # Paired post-hoc across L1,L2,C1,C2 (Holm-adjusted), with effect sizes
    post = pg.pairwise_ttests(data=long_norm,
                              dv="norm", within="session", subject="mouse",
                              padjust="holm", effsize="cohen")  # Cohen's dz for paired
    print(post)
    post = post.loc[post["A"].isin(["L1","L2","C1","C2"]) & post["B"].isin(["L1","L2","C1","C2"])]
    print(post[["A","B","T","dof","p-unc","p-corr","cohen"]])




#%%
# ---- Run the steps (fits right after your dfs are built) ----
long_counts = counts_long_per_mouse_from_dfs(dfs, SESSION_MAP)
#long_counts = long_counts.loc[~(long_counts["n_sessions"]==5)]
# ANOVA on L1-normalized counts
long_norm = l1_normalize_long(long_counts)
long_norm_a=long_norm.loc[~(long_norm["mouse"]=="13")]
anova_out = rm_anova_l1_normalized(long_norm_a)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print("=== RM-ANOVA on L1-normalized counts (S0,L1,L2,C1) ===")
    if "anova_table" in anova_out:
        print(anova_out["anova_table"])
        print("Epsilons  :", anova_out["epsilon"])
        print("Dropped mice:", anova_out["dropped_mice"])
    else:
        print(anova_out["message"])

# Planned test: log(L1/S0) vs 0
planned_out = planned_log_L1_over_S0(long_counts)
print("\n=== Planned test: log(L1/S0) vs 0 ===")
print(planned_out)


levels = ["L1","L2","C1", "C2"]                 # e.g., ["L1","L2","C1"] (S0 excluded)
d_for_normcheck = long_norm_a[long_norm_a["session"].isin(levels)]

normality_diffs = shapiro_on_pairwise_diffs(d_for_normcheck, dv="norm", subject="mouse", within="session")
print("\n=== Shapiro on paired differences ===")
print(normality_diffs)


print("\n=== Shapiro: log(L1/S0) ===")
print(shapiro_logratio(long_counts, "L1", "S0"))
#%%
# pm = build_ratios_by_mouse(dfs, reference="Lmean")  # or "S0"
# print(pm)
# res_S0 = summarize_logratio(pm["ratio_S0_vs_L"])    # S0 vs L reference
# res_C  = summarize_logratio(pm["ratio_C_vs_L"])     # C vs L reference
# print(res_S0)
# print("======================")
# print(res_C)
# #%%
# pm = build_ratios_by_mouse(dfs, reference="S0")  # or "S0"
# res_S0 = summarize_logratio(pm["ratio_L_vs_S0"])    # S0 vs L reference
# res_C  = summarize_logratio(pm["ratio_C_vs_S0"])     # C vs L reference
# print(res_S0)
# print("======================")
# print(res_C)
