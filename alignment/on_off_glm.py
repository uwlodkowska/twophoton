# --- imports ---
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ========== 1) aggregate on/off/const per Mouse × Pair ==========
def build_df_counts(df, mouse_col="Mouse", region_col=None):
    # find your A_to_B status columns
    pair_cols = [c for c in df.columns if "_to_" in c]
    if not pair_cols:
        raise ValueError("No columns with '_to_' found. Run on_off_cells(...) first.")

    id_vars = [mouse_col] + ([region_col] if region_col else [])
    tall = df.melt(id_vars=id_vars, value_vars=pair_cols,
                   var_name="Pair", value_name="Status")
    tall["Status"] = tall["Status"].astype(str).str.lower().str.strip()

    valid = {"on","off","const"}
    tall = tall[tall["Status"].isin(valid)].copy()

    group_keys = id_vars + ["Pair","Status"]
    counts = (tall.groupby(group_keys).size()
                   .rename("count").reset_index())

    pivot_index = id_vars + ["Pair"]
    df_counts = (counts.pivot_table(index=pivot_index,
                                    columns="Status", values="count",
                                    fill_value=0)
                        .reset_index())

    # ensure cols exist
    for col in ["on","off","const"]:
        if col not in df_counts.columns:
            df_counts[col] = 0

    df_counts["n"] = df_counts[["on","off","const"]].sum(axis=1)
    df_counts["changed"] = df_counts["on"] + df_counts["off"]
    df_counts["prop_changed"] = np.where(df_counts["n"]>0,
                                         df_counts["changed"]/df_counts["n"], np.nan)
    df_counts["prop_on"] = np.where(df_counts["changed"]>0,
                                    df_counts["on"]/df_counts["changed"], np.nan)
    return df_counts

# ========== 2) fit GLMs (turnover; direction), cluster-robust by Mouse ==========
def fit_models(df_counts, mouse_col="Mouse"):
    # Turnover: (on+off)/n ~ Pair
    mod_turn = smf.glm("prop_changed ~ C(Pair)",
                       data=df_counts,
                       family=sm.families.Binomial(),
                       freq_weights=df_counts["n"]
                       ).fit(cov_type="cluster",
                             cov_kwds={"groups": df_counts[mouse_col]})

    # Direction among changed: on/(on+off) ~ Pair
    df_changed = df_counts[df_counts["changed"] > 0].copy()
    mod_dir = smf.glm("prop_on ~ C(Pair)",
                      data=df_changed,
                      family=sm.families.Binomial(),
                      freq_weights=df_changed["changed"]
                      ).fit(cov_type="cluster",
                            cov_kwds={"groups": df_changed[mouse_col]})
    return mod_turn, mod_dir

# ========== 3) overdispersion (Pearson χ² / df) ==========
def pearson_dispersion(glm_res):
    return float((glm_res.resid_pearson**2).sum() / glm_res.df_resid)

# ========== 4) plots: model bars + CI, raw dots (size ~ sqrt(denom)) ==========
def plot_turnover(df_counts, mod_turn):
    new = pd.DataFrame({"Pair": sorted(df_counts["Pair"].unique())})
    pred = mod_turn.get_prediction(new, transform=True).summary_frame()
    pred["Pair"] = new["Pair"]

    plt.figure(figsize=(9,4))
    x = np.arange(len(pred))
    plt.bar(x, pred["mean"])
    yerr = np.vstack([pred["mean"]-pred["mean_ci_lower"],
                      pred["mean_ci_upper"]-pred["mean"]])
    plt.errorbar(x, pred["mean"], yerr=yerr, fmt='none', capsize=4)

    # raw per-mouse points
    for i, p in enumerate(pred["Pair"]):
        yy = df_counts.loc[df_counts["Pair"]==p, "prop_changed"]
        nn = df_counts.loc[df_counts["Pair"]==p, "n"]
        xj = i + (np.random.rand(len(yy)) - 0.5) * 0.22
        sizes = 20 * np.sqrt(nn / (nn.max() if nn.max()>0 else 1))
        plt.scatter(xj, yy, s=sizes, alpha=0.7)

    plt.xticks(x, pred["Pair"], rotation=45)
    plt.ylabel("Turnover (on+off)/union")
    plt.title("Turnover across consecutive pairs\n(bars = GLM estimate ±95% CI; dots = mice, size ~ √n)")
    plt.tight_layout()
    plt.show()

def plot_direction(df_counts, mod_dir):
    df_changed = df_counts[df_counts["changed"] > 0].copy()
    new = pd.DataFrame({"Pair": sorted(df_changed["Pair"].unique())})
    pred = mod_dir.get_prediction(new, transform=True).summary_frame()
    pred["Pair"] = new["Pair"]

    plt.figure(figsize=(9,4))
    x = np.arange(len(pred))
    plt.bar(x, pred["mean"])
    yerr = np.vstack([pred["mean"]-pred["mean_ci_lower"],
                      pred["mean_ci_upper"]-pred["mean"]])
    plt.errorbar(x, pred["mean"], yerr=yerr, fmt='none', capsize=4)

    for i, p in enumerate(pred["Pair"]):
        yy = df_changed.loc[df_changed["Pair"]==p, "prop_on"]
        mm = df_changed.loc[df_changed["Pair"]==p, "changed"]
        xj = i + (np.random.rand(len(yy)) - 0.5) * 0.22
        sizes = 20 * np.sqrt(mm / (mm.max() if mm.max()>0 else 1))
        plt.scatter(xj, yy, s=sizes, alpha=0.7)

    plt.xticks(x, pred["Pair"], rotation=45)
    plt.ylabel("Direction among changed (on / (on+off))")
    plt.title("Direction among changed\n(bars = GLM estimate ±95% CI; dots = mice, size ~ √m)")
    plt.tight_layout()
    plt.show()

# ========== 5) optional: pairwise contrasts between pairs (Wald tests) ==========
def wald_contrast_on_pairs(glm_res, level1, level2):
    """
    Wald test for difference in log-odds between Pair=level1 and Pair=level2.
    Works with GLM using C(Pair) coding (reference is the first level in the data).
    """
    coef = glm_res.params
    cov = glm_res.cov_params()
    # parameter names look like: 'Intercept', 'C(Pair)[T.<level>]', ...
    def name_for(level):
        return f"C(Pair)[T.{level}]"
    # If a level is the reference, its coefficient is 0 and not in params/cov.
    n1 = name_for(level1)
    n2 = name_for(level2)
    b1 = coef[n1] if n1 in coef.index else 0.0
    b2 = coef[n2] if n2 in coef.index else 0.0
    # Var(a - b) = Var(a)+Var(b)-2Cov(a,b), treating missing (reference) as 0 variance and 0 cov
    v1 = cov.loc[n1, n1] if (n1 in cov.index and n1 in cov.columns) else 0.0
    v2 = cov.loc[n2, n2] if (n2 in cov.index and n2 in cov.columns) else 0.0
    c12 = cov.loc[n1, n2] if (n1 in cov.index and n2 in cov.columns) else 0.0
    diff = b1 - b2
    se = np.sqrt(v1 + v2 - 2*c12)
    z = diff / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"diff_logit": diff, "se": se, "z": z, "p": p}

# -------------------- USAGE --------------------
# df = ...  # your dataframe with Mouse and *_to_* columns
# df_counts = build_df_counts(df, mouse_col="Mouse")  # region_col=... if you want stratification
# mod_turn, mod_dir = fit_models(df_counts, mouse_col="Mouse")

# print summaries
# print(mod_turn.summary()); print("Dispersion(turnover):", pearson_dispersion(mod_turn))
# print(mod_dir.summary());  print("Dispersion(direction):", pearson_dispersion(mod_dir))

# plots
# plot_turnover(df_counts, mod_turn)
# plot_direction(df_counts, mod_dir)

# example contrast: compare 's0_to_landmark1' vs 'landmark1_to_landmark2' for TURNOVER
# res = wald_contrast_on_pairs(mod_turn, "s0_to_landmark1", "landmark1_to_landmark2")
# print(res)
# (apply Holm/FDR across many contrasts with multipletests)
