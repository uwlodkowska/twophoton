import sys
import yaml

# custom modules
import cell_classification as cc
import utils
import numpy as np, patsy as pt

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pandas as pd

from scipy.special import expit as logistic
import statsmodels.api as sm


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
cols = ["bg_mean", "bg_std", "bg_diff"]  # adjust as needed
desc = df_turn[cols].agg(["mean","std","min","max"]).T
print(desc)

#%%
from itertools import combinations
from scipy.stats import ttest_rel, shapiro
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

# -------------------------------------------------------------------
# 1) Build DVs: prop_const and prop_on_given_change
#    Expected columns: mouse_id, Pair (transition), n, on, off, const
#    If proportions already exist (prop_on, prop_changed), they’re ignored here.
# -------------------------------------------------------------------
def build_stability_direction_table(df: pd.DataFrame,
                                    mouse_col="mouse_id",
                                    transition_col="Pair",
                                    n_col="n",
                                    on_col="on",
                                    off_col="off",
                                    const_col="const") -> pd.DataFrame:
    d = df.copy()

    # Basic checks
    for c in [mouse_col, transition_col, n_col, on_col, off_col, const_col]:
        if c not in d.columns:
            raise ValueError(f"Missing required column: {c}")

    # Stability
    d["prop_const"] = d[const_col] / d[n_col]

    # Direction among changers
    change_den = d[on_col] + d[off_col]
    d["prop_on_given_change"] = np.where(change_den > 0, d[on_col] / change_den, np.nan)

    # Keep tidy columns
    keep = [mouse_col, transition_col, "prop_const", "prop_on_given_change"]
    return d[keep]


# -------------------------------------------------------------------
# 2) RM-ANOVA across transitions for ONE DV
#    - Drops mice missing any transition (for this DV only)
#    - Reports ANOVA main effect of transition
#    - Post-hoc paired t-tests across transitions + Holm
#    - Normality checks on all paired difference vectors
#    - If pingouin is available: GG/HF sphericity correction
# -------------------------------------------------------------------


def rm_anova_one_factor_wide(df_long: pd.DataFrame,
                             dv: str,
                             mouse_col="mouse",
                             within_col="transition"):
    d = df_long[[mouse_col, within_col, dv]].dropna(subset=[dv]).copy()

    # Pivot to Wide: mouse × transition
    wide = d.pivot_table(index=mouse_col, columns=within_col, values=dv, aggfunc="mean")

    # Drop incomplete mice (this analysis only)
    complete = wide.dropna()
    dropped_mice = list(set(wide.index) - set(complete.index))

    # Back to long for AnovaRM
    long_c = complete.reset_index().melt(id_vars=mouse_col, var_name=within_col, value_name=dv)

    # Need at least 2 mice and at least 2 levels of transition
    if long_c[mouse_col].nunique() < 2 or long_c[within_col].nunique() < 2:
        return {"message": "Not enough complete subjects or transition levels.",
                "dropped_mice": dropped_mice}

    # --- Uncorrected RM-ANOVA (statsmodels) ---
    anova = AnovaRM(long_c, depvar=dv, subject=mouse_col, within=[within_col]).fit()
    anova_table = anova.anova_table.copy()

    # Extract F and dfs for the within effect
    try:
        F_value = float(anova_table.loc[within_col, "F Value"])
        df1 = float(anova_table.loc[within_col, "Num DF"])
        df2 = float(anova_table.loc[within_col, "Den DF"])
    except KeyError:
        # Some statsmodels versions label the row differently (e.g., 'C(within_col)')
        row_key = [idx for idx in anova_table.index if within_col in str(idx)]
        if not row_key:
            raise
        rk = row_key[0]
        F_value = float(anova_table.loc[rk, "F Value"])
        df1 = float(anova_table.loc[rk, "Num DF"])
        df2 = float(anova_table.loc[rk, "Den DF"])

    # --- Sphericity + epsilon (manual, works on Py3.9) ---
    long_c_ren = long_c.rename(columns={mouse_col: "mouse",
                                        within_col: "transition",
                                        dv: "value"})
    # Mauchly test
    try:
        spher = mauchly_sphericity(long_c_ren, dv="value", subject="mouse", within="transition")
    except Exception:
        spher = {"W": np.nan, "chi2": np.nan, "df": np.nan, "p": np.nan}

    # GG/HF epsilons
    try:
        eps = gg_hf_epsilon(long_c_ren, dv="value", subject="mouse", within="transition")
        eps_gg, eps_hf = eps["eps_gg"], eps["eps_hf"]
    except Exception:
        # If only 2 levels, epsilon is 1 and sphericity is moot
        eps_gg = eps_hf = 1.0

    # Apply corrections to df and p-value
    gg_corr = apply_epsilon_correction(F_value=F_value, df1=df1, df2=df2, eps=eps_gg)
    hf_corr = apply_epsilon_correction(F_value=F_value, df1=df1, df2=df2, eps=eps_hf)

    gg_correction = {"epsilon": eps_gg,
                     "df1_corr": gg_corr["df1_corr"],
                     "df2_corr": gg_corr["df2_corr"],
                     "p_corr": gg_corr["p_corr"]}

    hf_correction = {"epsilon": eps_hf,
                     "df1_corr": hf_corr["df1_corr"],
                     "df2_corr": hf_corr["df2_corr"],
                     "p_corr": hf_corr["p_corr"]}

    # (Optional) If you also want Pingouin’s table for reference when available:
    try:
        import pingouin as pg
        pg_table = pg.rm_anova(data=long_c_ren, dv="value", within="transition",
                               subject="mouse", detailed=True, correction=True)
    except Exception:
        pg_table = None

    # --- Post-hoc paired t-tests across transitions (+ Holm) ---
    pairs = list(combinations(complete.columns, 2))
    post = []
    for a, b in pairs:
        diffs = (complete[a] - complete[b]).dropna()
        if len(diffs) >= 2:
            t, p = ttest_rel(complete[a], complete[b], nan_policy="omit")
            dz = diffs.mean() / (diffs.std(ddof=1) + 1e-12)  # Cohen's dz
        else:
            t, p, dz = np.nan, np.nan, np.nan
        post.append({"contrast": f"{a} - {b}", "t": t, "p_uncorrected": p, "dz": dz, "n": len(diffs)})
    post = pd.DataFrame(post)
    if not post.empty:
        post["p_holm"] = multipletests(post["p_uncorrected"], method="holm")[1]

    # --- Normality on paired differences (assumption check) ---
    norm_rows = []
    for a, b in pairs:
        diffs = (complete[a] - complete[b]).dropna()
        if len(diffs) >= 3:
            W, pW = shapiro(diffs)
        else:
            W, pW = np.nan, np.nan
        norm_rows.append({"contrast": f"{a} - {b}", "shapiro_W": W, "shapiro_p": pW, "n": len(diffs)})
    normality = pd.DataFrame(norm_rows)

    return {
        "anova_table_uncorrected": anova_table,
        "sphericity": spher,                 # Mauchly W, chi2, df, p
        "gg_correction": gg_correction,      # epsilon, corrected dfs, corrected p
        "hf_correction": hf_correction,      # epsilon, corrected dfs, corrected p
        "pingouin_table": pg_table,          # optional; None if not available
        "posthoc": post,
        "normality": normality,
        "dropped_mice": dropped_mice,
        "wide_complete": complete
    }



# -------------------------------------------------------------------
# 3) Convenience runner: execute both ANOVAs on your dataframe
# -------------------------------------------------------------------
def run_two_anovas(df_counts: pd.DataFrame):
    tidy = build_stability_direction_table(df_counts)
    tidy = tidy.rename(columns={"mouse_id": "mouse", "Pair": "transition"})  # align names

    # ANOVA #1: Stability across transitions
    res_stability = rm_anova_one_factor_wide(tidy, dv="prop_const",
                                             mouse_col="mouse", within_col="transition")

    # ANOVA #2: Direction across transitions
    res_direction = rm_anova_one_factor_wide(tidy, dv="prop_on_given_change",
                                             mouse_col="mouse", within_col="transition")





    return res_stability, res_direction

from scipy.stats import chi2

def mauchly_sphericity(df, dv, subject, within):
    """
    Mauchly's test of sphericity.
    
    Parameters
    ----------
    df : pd.DataFrame (long format)
    dv : str, dependent variable column (e.g., "value")
    subject : str, subject column (e.g., "mouse")
    within : str, within-subject factor column (e.g., "transition")
    
    Returns
    -------
    dict with W, chi2, df, p
    """
    # pivot to wide: subjects × conditions
    wide = df.pivot(index=subject, columns=within, values=dv).dropna()
    X = wide.values
    n, k = X.shape  # subjects, levels
    
    # covariance matrix across conditions
    S = np.cov(X, rowvar=False)
    
    # compute Mauchly’s W
    det_S = np.linalg.det(S)
    tr_S = np.trace(S) / k
    W = det_S / (tr_S ** k)
    
    # correction factor
    c = (2*k**2 + k + 2) / (6*(k-1)*(n-1))
    df_chi = (k*(k-1))//2 - 1
    chi2_stat = -(n-1) * (1-c) * np.log(W)
    pval = 1 - chi2.cdf(chi2_stat, df_chi)
    
    return {"W": W, "chi2": chi2_stat, "df": df_chi, "p": pval}

def to_wide(df: pd.DataFrame, dv: str, subject: str, within: str) -> pd.DataFrame:
    """Pivot long -> wide (subjects × levels), dropping subjects with any missing cells."""
    wide = df.pivot(index=subject, columns=within, values=dv)
    wide = wide.dropna(axis=0, how="any")
    return wide

def gg_hf_epsilon(df_long: pd.DataFrame, dv: str, subject: str, within: str):
    """
    Compute Greenhouse–Geisser (GG) and Huynh–Feldt (HF) epsilons
    for a one-way within-subject factor.
    Returns dict: eps_gg, eps_hf, n, k
    """
    wide = to_wide(df_long, dv, subject, within)
    X = wide.values
    n, k = X.shape
    if n < 2 or k < 3:
        raise ValueError("Need at least 2 subjects and 3 levels for epsilon estimation.")

    # Sample covariance across levels
    S = np.cov(X, rowvar=False, ddof=1)

    # Centering matrix C = I - (1/k) * 11^T
    I = np.eye(k)
    J = np.ones((k, k)) / k
    C = I - J

    # S_tilde = C S C (project onto deviations from the mean level)
    S_tilde = C @ S @ C

    # GG epsilon estimator (Maxwell & Delaney, 2004)
    num = (np.trace(S_tilde)) ** 2
    den = (k - 1) * np.trace(S_tilde @ S_tilde)
    eps_gg = num / den if den > 0 else 1.0

    # Bound to [1/(k-1), 1]
    lower_bound = 1.0 / (k - 1)
    eps_gg = float(np.clip(eps_gg, lower_bound, 1.0))

    # HF epsilon (Van den Brink / Huynh-Feldt)
    # Common sample-size adjusted formula:
    eps_hf = ((n * k - 2) * eps_gg + 2) / ((k - 1) * (n - 1))
    eps_hf = float(np.clip(eps_hf, lower_bound, 1.0))

    return {"eps_gg": eps_gg, "eps_hf": eps_hf, "n": n, "k": k}
from scipy.stats import f as fd
def apply_epsilon_correction(F_value: float, df1: float, df2: float, eps: float):
    """
    Apply epsilon (GG or HF) to numerator & denominator df for a one-way RM effect.
    Returns dict: df1_corr, df2_corr, p_corr
    """
    df1_corr = eps * df1
    df2_corr = eps * df2
    p_corr = fd.sf(F_value, df1_corr, df2_corr)  # upper-tail (survival function)
    return {"df1_corr": df1_corr, "df2_corr": df2_corr, "p_corr": p_corr}

#%%

res_stability, res_direction = run_two_anovas(df_counts)

# Inspect
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for res in [res_stability, res_direction]:
        print(res["anova_table_uncorrected"])
        print("Dropped mice:", res["dropped_mice"])
        print("Anova table")
        print(res["anova_table_uncorrected"])
        print("Sphericity")
        print(res["sphericity"])
        
        print("gg_correction")
        print(res["gg_correction"])
        
        print("hf_correction")
        print(res["hf_correction"])
        
        print("pingouin_table")
        print(res["pingouin_table"])
        
        print("\nPost-hoc (Holm):")
        print(res["posthoc"])
        
        print("\nNormality checks:")
        print(res["normality"])


#%%


from scipy.stats import ttest_rel, shapiro, wilcoxon, norm, t as t_dist


# --- Helper: pick AABB mice and compute outcomes ---
def make_aabb_outcomes(df,
                       mouse_col="mouse_id", pair_col="Pair",
                       on="on", off="off", const="const", n_col="n"):
    # Keep mice that have AABB transitions (with S0->landmark1)
    pairs_needed = {"s0_to_landmark1", "landmark1_to_landmark2", "landmark2_to_ctx1", "ctx1_to_ctx2"}
    has_all = df.groupby(mouse_col)[pair_col].transform(lambda s: pairs_needed.issubset(set(s)))
    d = df[has_all].copy()

    # Outcomes
    d["prop_const"] = d[const] / d[n_col]
    denom = d[on] + d[off]
    d["prop_on_given_change"] = np.where(denom > 0, d[on] / denom, np.nan)

    return d

# --- Helper: wide mouse×transition table for a DV ---
def wide_by_pair(d, dv, mouse_col="mouse_id", pair_col="Pair"):
    keep = [mouse_col, pair_col, dv]
    ww = d[keep].pivot_table(index=mouse_col, columns=pair_col, values=dv, aggfunc="mean")
    return ww

# Planned contrasts (AABB mapping)
# Baseline vs First test: s0_to_landmark1 (Baseline) vs landmark1_to_landmark2 (LL)
# The other three among LL, LC, CC
CONTRASTS = [
    ("s0_to_landmark1",       "landmark1_to_landmark2", "ttest"),  # Baseline vs First test (non-normal)
    ("landmark1_to_landmark2","landmark2_to_ctx1",      "ttest"),     # LL vs LC (primary)
    ("landmark1_to_landmark2","ctx1_to_ctx2",           "ttest"),     # LL vs CC
#    ("landmark2_to_ctx1",     "ctx1_to_ctx2",           "ttest"),     # LC vs CC
]

def paired_t_with_ci(x, y):
    # x,y are aligned arrays (no NaNs), len>=2
    diffs = x - y
    n = len(diffs)
    m = diffs.mean()
    sd = diffs.std(ddof=1) if n>1 else np.nan
    t_stat, p = ttest_rel(x, y, nan_policy="omit")
    dz = m / (sd + 1e-12)  # Cohen's dz for paired
    # 95% CI for mean difference
    se = sd / np.sqrt(n) if n>1 else np.nan
    tcrit = t_dist.ppf(0.975, df=n-1) if n>1 else np.nan
    ci_lo = m - tcrit*se if n>1 else np.nan
    ci_hi = m + tcrit*se if n>1 else np.nan
    return dict(n=n, mean_diff=m, t=t_stat, p=p, dz=dz, ci_lo=ci_lo, ci_hi=ci_hi)

def wilcoxon_with_effect(x, y):
    # Wilcoxon signed-rank; effect size as rank-biserial r
    diffs = x - y
    diffs_nonzero = diffs[diffs != 0]
    n = len(diffs_nonzero)
    if n < 1:
        return dict(n=len(diffs), W=np.nan, p=np.nan, r_rb=np.nan,
                    hl=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    W_stat, p = wilcoxon(diffs_nonzero)  # two-sided
    # Approximate z from p (two-sided), keep sign from median diff
    z_abs = norm.isf(p/2) if p>0 else np.inf
    sign = np.sign(np.median(diffs_nonzero)) if np.isfinite(z_abs) else 0.0
    z = sign * z_abs
    r_rb = z / np.sqrt(n)  # rank-biserial approximation
    # Hodges–Lehmann estimator (median of pairwise differences) ~ median(diffs)
    hl = np.median(diffs)
    # Simple bootstrap CI for HL (percentile)
    rng = np.random.default_rng(12345)
    B = 5000
    if n >= 3:
        boots = []
        idx = np.arange(len(diffs))
        for _ in range(B):
            bidx = rng.choice(idx, size=len(idx), replace=True)
            boots.append(np.median(diffs[bidx]))
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    else:
        ci_lo = ci_hi = np.nan
    return dict(n=len(diffs), W=W_stat, p=p, r_rb=r_rb, hl=hl, ci_lo=ci_lo, ci_hi=ci_hi)

def run_planned_contrasts_S0AABB(df, dv):
    d = make_aabb_outcomes(df)
    W = wide_by_pair(d, dv)

    rows = []
    for a, b, method in CONTRASTS:
        # Keep mice that have both
        sub = W[[a, b]].dropna()
        if sub.empty or len(sub) < 2:
            rows.append({"contrast": f"{a} - {b}", "n": len(sub), "method": method,
                         "p_raw": np.nan})
            continue
        x, y = sub[a].to_numpy(), sub[b].to_numpy()

        # Normality check (useful to log)
        try:
            Wsh, Psh = shapiro(x - y) if len(sub) >= 3 else (np.nan, np.nan)
        except Exception:
            Wsh, Psh = (np.nan, np.nan)

        if method == "ttest":
            res = paired_t_with_ci(x, y)
            rows.append({
                "contrast": f"{a} - {b}", "n": res["n"], "method": "paired_t",
                "mean_diff": res["mean_diff"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                "t": res["t"], "dz": res["dz"], "p_raw": res["p"], "shapiro_W": Wsh, "shapiro_p": Psh
            })
        else:  # wilcoxon
            res = wilcoxon_with_effect(x, y)
            rows.append({
                "contrast": f"{a} - {b}", "n": res["n"], "method": "wilcoxon",
                "median_diff": res["hl"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
                "W": res["W"], "r_rb": res["r_rb"], "p_raw": res["p"], "shapiro_W": Wsh, "shapiro_p": Psh
            })

    out = pd.DataFrame(rows)

    # Holm correction across the FOUR planned contrasts
    if out["p_raw"].notna().any():
        out["p_holm"] = np.nan
        mask = out["p_raw"].notna()
        out.loc[mask, "p_holm"] = multipletests(out.loc[mask, "p_raw"], method="holm")[1]

    return out.sort_values("contrast").reset_index(drop=True)
#%%
# ---- Example usage ----

# Stability:
tab_stab = run_planned_contrasts_S0AABB(df_counts, dv="prop_const")                # const/n already computed inside
# # Direction:
tab_dir  = run_planned_contrasts_S0AABB(df_counts, dv="prop_on_given_change")      # on/(on+off)
# print(tab_stab); print(tab_dir)

#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(tab_dir)



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
        
    print("QIC ", gee.qic(scale=1))


#%%TURNOVER: (on+off)/n ~ Pair   (grouped by Mouse)

# df_counts: Mouse, Pair, on, off, const, n, changed, prop_changed, prop_on, bg_median, depth


# Turnover subset (k/n)
df_turn = df_counts.copy().reset_index(drop=True)
df_turn["n"] = (df_turn[["on","off","const"]].sum(axis=1)).astype(int)
df_turn["changed"] = (df_turn["on"] + df_turn["off"]).astype(int)
df_turn = df_turn[df_turn["n"]>0].copy()
df_turn["prop_changed"] = df_turn["changed"] / df_turn["n"]
df_turn["prop_const"] = df_turn["const"] / df_turn["n"]

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
               cov_struct=Exchangeable()).fit(scale="X2")

print(gee_turn.summary())

w
gee_results_inspection(gee_turn, ref="landmark1_to_landmark2")
#%%
ORDER_AABB = {
    "s0_to_landmark1": 0,
    "landmark1_to_landmark2": 1,
    "landmark2_to_ctx1": 2,
    "ctx1_to_ctx2": 3,
}

# BBAA sequence (if you have them later): S0→C1=0, C1→C2=1, C2→L1=2, L1→L2=3
ORDER_BBAA = {
    "s0_to_ctx1": 0,
    "ctx1_to_ctx2": 1,
    "ctx2_to_landmark1": 2,
    "landmark1_to_landmark2": 3,
}

def add_time_ix(df, mouse_col="Mouse", pair_col="Pair"):
    d = df.copy()
    # Detect sequence per mouse from which S0 transition is present
    seq = d.groupby(mouse_col)[pair_col].transform(
        lambda s: "AABB" if "s0_to_landmark1" in set(s) else ("BBAA" if "s0_to_ctx1" in set(s) else "UNKNOWN")
    )
    d["sequence"] = seq

    def _map_time(row):
        if row["sequence"] == "AABB":
            return ORDER_AABB.get(row[pair_col], np.nan)
        elif row["sequence"] == "BBAA":
            return ORDER_BBAA.get(row[pair_col], np.nan)
        else:
            return np.nan

    d["time_ix"] = d.apply(_map_time, axis=1).astype("float")
    # Keep rows with a valid time
    d = d.dropna(subset=["time_ix"]).copy()
    d["time_ix"] = d["time_ix"].astype(int)

    # Sort and drop accidental duplicates (mouse, time)
    d = d.sort_values([mouse_col, "time_ix"]).drop_duplicates([mouse_col, "time_ix"])

    # Sanity checks
    bad_dupes = d.duplicated([mouse_col, "time_ix"]).any()
    bad_order = d.groupby(mouse_col)["time_ix"].apply(lambda s: not s.is_monotonic_increasing).any()
    too_few = (d.groupby(mouse_col)["time_ix"].nunique() < 2).any()
    if bad_dupes or bad_order or too_few:
        print("⚠️ Time index issues:",
              {"duplicates": bool(bad_dupes), "non_monotonic": bool(bad_order), "mouse_with_<2_timepoints": bool(too_few)})
    return d


def make_prop_df(df, dv="prop_const"):  # or "prop_on_given_change"
    d = df.copy()
    if dv == "prop_const":
        d["prop"] = d["const"] / d["n"]
    elif dv == "prop_on_given_change":
        den = d["on"] + d["off"]
        d["prop"] = np.where(den > 0, d["on"] / den, np.nan)
    else:
        raise ValueError("dv must be 'prop_const' or 'prop_on_given_change'")
    return d



#%% turnover w covariates
from statsmodels.genmod.cov_struct import Independence, Exchangeable, Autoregressive, Stationary
#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(ix)

#%%
df_turn = df_counts.copy()
df_turn["bg_mean"]   = (df_turn["bg_A"] + df_turn["bg_B"])/2
df_turn["bg_diff"]   = df_turn["bg_B"] - df_turn["bg_A"]
df_turn["bg_std"]   = (df_turn["bg_std_B"] - df_turn["bg_std_A"])
df_turn["prop_const"] = df_turn["const"] / df_turn["n"]

# z-score
for c in ["bg_mean","bg_diff", "bg_std"]:
    df_turn[c] = (df_turn[c] - df_turn[c].mean())/df_turn[c].std(ddof=1)
df_turn["bg_std_mean_mouse"] = df_turn.groupby("Mouse")["bg_std"].transform("mean")
df_turn["bg_std_within"]     = df_turn["bg_std"] - df_turn["bg_std_mean_mouse"]    
mean_profile = {
    # "bg_mean": df_turn["bg_mean"].mean(),
    # "bg_diff": df_turn["bg_diff"].mean(),
    # "bg_std": df_turn["bg_std"].mean()
}

covs = [Independence(), Exchangeable(), Stationary(), Autoregressive()]
covs = [Exchangeable()]

d0 = add_time_ix(df_turn, mouse_col="Mouse", pair_col="Pair")
d1 = make_prop_df(d0, dv="prop_const").dropna(subset=["prop", "time_ix"]).copy()


rhs = "C(Pair, Treatment(reference='landmark1_to_landmark2'))"
X_df = pt.dmatrix("1 + " + rhs, data=d1, return_type="dataframe")

# 3) Convert EVERYTHING passed to GEE into NumPy arrays (1D for y/groups/time)
y_1d      = d1["prop"]                       # Series → fine
groups_1d = d1["Mouse"].astype("category").cat.codes.to_numpy()
time_1d   = d1["time_ix"].to_numpy(dtype=int)  

# f = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_diff+ bg_std"
# y, X = pt.dmatrices(f, data=df_turn, return_type="dataframe")
# ix = X.index

for cov_struct in covs:
    res = GEE(endog=y_1d, exog=X_df, groups=groups_1d, time=time_1d,
          family=Binomial(), cov_struct=cov_struct).fit(
              cov_type="bias_reduced", scale=1.0
          )
    print(res.summary())
    gee_results_inspection(res, ref="landmark1_to_landmark2", cov_profile=mean_profile)
    
m_std = res.get_margeff(at='overall', method='dydx')
print(m_std.summary()) 
#%%

import numpy as np
import pandas as pd
import patsy as pt
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable  # or your chosen structure




# ---- EMMs: adjusted probabilities per transition (marginal over covariates) ----
def gee_emm(res, data, formula_exog, factor, levels, weights=None):
    """Return adjusted (marginal) probability for each level, with 95% CI."""
    if weights is None:
        w = np.ones(len(data))
    else:
        w = np.asarray(weights)
    w = w / w.sum()  # normalize

    emms = []
    cov = res.cov_params()
    linkinv = res.model.family.link.inverse

    for lvl in levels:
        # Build exog where the factor is set to the target level for every row
        data_l = data.copy()
       
        
        data_l[factor] = pd.Categorical([lvl]*len(data), categories=levels)
        
               
        X_l = pt.dmatrix("1 + " + rhs, data_l, return_type="dataframe")
        X_l = X_l.reindex(columns=res.params.index, fill_value=0.0)
        eta = np.dot(X_l, res.params)
        mu  = linkinv(eta)                      # predicted prob per row
        p_hat = np.sum(w * mu)                  # weighted average (EMM)

        # Delta-method gradient for the mean
        grad_rows = (w * (mu * (1 - mu)))[:, None] * X_l.values
        g = grad_rows.sum(axis=0)
        se = float(np.sqrt(g @ cov @ g))
        ci_lo, ci_hi = p_hat - 1.96*se, p_hat + 1.96*se
        emms.append(dict(level=lvl, p=p_hat, se=se, ci_lo=ci_lo, ci_hi=ci_hi))
    return pd.DataFrame(emms)



# ---- Planned contrasts on EMMs (model-based Wald tests) ----
def gee_emm_contrast(res, data, formula_exog, factor, level_a, level_b, weights=None):
    # Grab RHS irrespective of the LHS name
    rhs = formula_exog.split("~", 1)[1].strip() if "~" in formula_exog else formula_exog

    def _emm_and_grad(level):
        data_l = data.copy()
        # keep your existing 'levels' list if you have it defined outside
        try:
            data_l[factor] = pd.Categorical([level] * len(data), categories=levels)
        except NameError:
            # fallback: use categories from the data if 'levels' not defined
            if pd.api.types.is_categorical_dtype(data_l[factor]):
                cats = data_l[factor].cat.categories
            else:
                cats = sorted(pd.Series(data_l[factor].unique()).tolist())
            data_l[factor] = pd.Categorical([level] * len(data), categories=cats)

        # build exog with intercept from RHS and align columns to res.params
        X_l = pt.dmatrix("1 + " + rhs, data_l, return_type="dataframe")
        X_l = X_l.reindex(columns=res.params.index, fill_value=0.0)

        eta = np.dot(X_l.to_numpy(dtype=float), np.asarray(res.params))
        mu  = np.asarray(res.model.family.link.inverse(eta), dtype=float)

        if weights is None:
            w = np.ones(len(mu), dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
        w /= w.sum()

        p_hat = float(np.dot(w, mu))
        grad_rows = (w * (mu * (1.0 - mu)))[:, None] * X_l.to_numpy(dtype=float)
        g = grad_rows.sum(axis=0)
        return p_hat, g

    pa, ga = _emm_and_grad(level_a)
    pb, gb = _emm_and_grad(level_b)
    delta = pa - pb
    g = ga - gb
    cov = np.asarray(res.cov_params())
    var = float(g @ cov @ g)
    se = np.sqrt(var)
    z = delta / se if se > 0 else np.nan
    p = 2 * norm.sf(abs(z))
    ci_lo, ci_hi = delta - 1.96 * se, delta + 1.96 * se
    return dict(contrast=f"{level_a} - {level_b}",
                delta=delta, se=se, z=z, p=p, ci_lo=ci_lo, ci_hi=ci_hi)

# Pick the level order you want to display
levels = ["s0_to_landmark1", "landmark1_to_landmark2", "landmark2_to_ctx1", "ctx1_to_ctx2"]
#%%

emm_tab = gee_emm(res, df_turn, formula_exog=f, factor="Pair", levels=levels)
print(emm_tab)
planned = [
    ("landmark1_to_landmark2","landmark2_to_ctx1"),  # LL - LC (primary)
    ("landmark1_to_landmark2","ctx1_to_ctx2"),       # LL - CC
    ("landmark2_to_ctx1","ctx1_to_ctx2"),            # LC - CC
    ("landmark1_to_landmark2","s0_to_landmark1"),    # LL - S0→L1 (baseline)
]

contr_rows = [gee_emm_contrast(res, df_turn, f, "Pair", a, b) for a,b in planned]
contr = pd.DataFrame(contr_rows)
# Holm across the planned family
contr["p_holm"] = multipletests(contr["p"], method="holm")[1]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(contr)


#%%
gee_cov.summary()







#%% covariate influence inspectino
f1 = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2'))"
y1, X1 = pt.dmatrices(f1, data=df_turn, return_type="dataframe")
gee_turn = GEE(y1, X1, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable()).fit(scale=1, cov_type="bias_reduced")

no_cov_qic = gee_turn.qic(scale=1)


f1 = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean+ bg_std+ bg_diff"
y1, X1 = pt.dmatrices(f1, data=df_turn, return_type="dataframe")
gee_cov = GEE(y1, X1, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable()).fit(scale=1, cov_type="bias_reduced")
qic_full = gee_cov.qic(scale=1)

print(no_cov_qic, qic_full)
#%%
# Without bg_diff
w=None
f1 = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_mean+ bg_std"
y1, X1 = pt.dmatrices(f1, data=df_turn, return_type="dataframe")
gee_no_diff = GEE(y1, X1, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable(), weights=w).fit(scale=1, cov_type="bias_reduced")
qic_no_diff = gee_no_diff.qic(scale=1)

# Without bg_mean
f2 = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_diff+ bg_std"
y2, X2 = pt.dmatrices(f2, data=df_turn, return_type="dataframe")
gee_no_mean = GEE(y2, X2, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable(), weights=w).fit(scale=1, cov_type="bias_reduced")
qic_no_mean = gee_no_mean.qic(scale=1)

# Without std
f_no_std = "prop_const ~ C(Pair, Treatment(reference='landmark1_to_landmark2')) + bg_diff+ bg_mean"
y_ns, X_ns = pt.dmatrices(f_no_std, data=df_turn, return_type="dataframe")
gee_no_std = GEE(y_ns, X_ns, groups=groups, family=Binomial(),
                 cov_struct=Exchangeable(), weights=w).fit(scale=1, cov_type="bias_reduced")
qic_no_std = gee_no_std.qic(scale=1)
print(f"No covs:     {no_cov_qic}")
print(f"Full:     {qic_full}")
print(f"No diff:  {qic_no_diff}")
print(f"No mean:  {qic_no_mean}")
print(f"Just mean:  {qic_no_std}")

import numpy as np

def gee_block_wald(res, terms, verbose=True):
    """
    Block Wald test H0: all coefficients for `terms` == 0
    res   : GEEResults (already fit; e.g., gee_cov)
    terms : list of coefficient names to test (must match res.params.index)
    """
    names = list(res.params.index)
    idx = [i for i, n in enumerate(names) if n in terms]
    if len(idx) == 0:
        raise ValueError(f"None of {terms} found in params: {names}")

    # Contrast matrix L: one row per tested coefficient
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0

    wt = res.wald_test(L, scalar=True)  # chi^2 test with df = len(idx)

    # Be robust to different return shapes
    stat = float(np.asarray(wt.statistic))
    pval = float(np.asarray(wt.pvalue))
    df   = L.shape[0]

    if verbose:
        pretty = ", ".join(f"{names[j]}=0" for j in idx)
        print(f"H0: {pretty}")
        print(f"Block Wald: chi2({df}) = {stat:.3f}, p = {pval:.3f}")

    return {"chi2": stat, "df": df, "p": pval, "tested": [names[j] for j in idx]}


wald_all = gee_block_wald(gee_cov, ["bg_mean", "bg_std", "bg_diff"])

# 2) Optional: single-covariate Wald tests inside the full model
wald_std  = gee_block_wald(gee_cov, ["bg_std"])
wald_mean = gee_block_wald(gee_cov, ["bg_mean"])
wald_diff = gee_block_wald(gee_cov, ["bg_diff"])

print(wald_all,wald_std, wald_mean, wald_diff )
#%%
gee_results_inspection(gee_no_mean, ref="landmark1_to_landmark2", cov_profile=mean_profile)
#%%
gee_no_mean.summary()
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
