import pandas as pd
import numpy as np
import patsy as pt
import statsmodels.api as sm
import utils

from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from scipy.special import expit
#%%


dfs = [utils.get_concatenated_df_from_config("config_files/multisession.yaml", 0, "_multi"),
utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", 0, "_cll"),
utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", 1, "_lcc")]



#%%
import ast

# ----------------------------
# 1) Canonical session helpers
# ----------------------------
# We drop baseline (B) from modeling; days are the exposure index among L/C only.
SCHEDULE = {
    "exp3a": [("ctx1","C",1), ("landmark1","L",2), ("landmark2","L",3)],   # CLL
    "exp3b": [("landmark1","L",1), ("ctx1","C",2), ("ctx2","C",3)],        # LCC
    "exp5" : [("landmark1","L",1), ("landmark2","L",2), ("ctx1","C",3), ("ctx2","C",4)],  # LLCC (baseline s0 excluded)
}
CANON_MAP = {"ctx":"ctx1", "landmark":"landmark1"}  # unify names

def _parse_sessions(x):
    """Return a canonicalized set of session names for one row."""
    if isinstance(x, set):
        s = x
    elif isinstance(x, (list, tuple)):
        s = set(x)
    elif isinstance(x, str):
        try:
            s = ast.literal_eval(x)  # "{'ctx1','landmark1'}" etc.
            if not isinstance(s, (set, list, tuple)):
                s = set()
            else:
                s = set(s)
        except Exception:
            # fall back: split on commas/spaces
            s = set([t.strip() for t in x.replace("{","").replace("}","").replace("[","").replace("]","").split(",") if t.strip()])
    else:
        s = set()
    # canonicalize
    return {CANON_MAP.get(k, k) for k in s}

def _make_XL_XC(X, term='C(sess_ident)[T.L]'):
    """Two design matrices where the L indicator is 1 (XL) vs 0 (XC)."""
    XL = X.copy()
    XC = X.copy()
    if term not in X.columns:
        raise KeyError(f"Column {term!r} not in design matrix.")
    XL.loc[:, term] = 1.0
    XC.loc[:, term] = 0.0
    return XL, XC

def ame_L_minus_C(res, X, agg, by_mouse_equal=True, term='C(sess_ident)[T.L]'):
    """
    Average marginal effect (Δp = p_L - p_C) with delta-method SE using the
    model's (cluster-robust) covariance.
    by_mouse_equal=True -> each mouse contributes equally.
    Otherwise weight rows by n (pooled population weighting).
    """
    beta = res.params.values
    V    = res.cov_params().values
    XL, XC = _make_XL_XC(X, term=term)

    eta_L = XL.values @ beta
    eta_C = XC.values @ beta
    pL = expit(eta_L)
    pC = expit(eta_C)
    dp = pL - pC  # row-wise Δp

    if by_mouse_equal:
        # weight rows within each mouse by their n, then average equally across mice
        w = agg["n"].to_numpy()
        mice = agg["mouse"].to_numpy()
        # per-mouse weighted mean of dp
        df_tmp = pd.DataFrame({"mouse": mice, "dp": dp, "w": w})
        per_mouse = (df_tmp
                     .groupby("mouse", as_index=False)
                     .apply(lambda g: np.average(g["dp"], weights=g["w"]))
                     .rename(columns={None: "dp_mouse"}))
        ame = per_mouse["dp_mouse"].mean()
        # gradient via delta method: average of row-wise gradients aggregated the same way
        sL = pL*(1-pL); sC = pC*(1-pC)
        g_rows = (sL[:,None]*XL.values - sC[:,None]*XC.values)  # shape (R, K)
        # per-mouse weighted mean of gradients
        Gm = (pd.DataFrame(g_rows, columns=X.columns)
                .assign(mouse=mice, w=w)
                .groupby("mouse")
                .apply(lambda g: (g.drop(columns=["w","mouse"]).mul(g["w"], axis=0).sum(axis=0)
                                  / g["w"].sum()))
                .to_numpy())
        g = Gm.mean(axis=0)  # K-length
    else:
        # pooled-n weighting across all rows
        w = agg["n"].to_numpy()
        ame = np.average(dp, weights=w)
        sL = pL*(1-pL); sC = pC*(1-pC)
        g_rows = (sL[:,None]*XL.values - sC[:,None]*XC.values) * w[:,None] / w.sum()
        g = g_rows.sum(axis=0)

    se = float(np.sqrt(g @ V @ g))
    lo, hi = ame - 1.96*se, ame + 1.96*se
    return {"AME": ame, "SE": se, "lo": lo, "hi": hi}

# -----------------------------------
# 2) Build cell × day (exposures) long
# -----------------------------------
def make_long_cells(df, exp):
    """
    From pooled wide df (one row per cell), produce one row per cell × exposure day (L/C only).
    Columns returned: mouse, cell_id, experiment, day, sess_key, sess_ident, active,
                      prev_active, same_as_prev, has_prev
    """
    recs = []
    # pre-extract to speed up attribute access
    mouse_col   = df["mouse"].to_numpy()
    cell_col    = df["cell_id"].to_numpy()
    det_sessraw = df["detected_in_sessions"].to_numpy()

    for mouse, cell, sess_raw in zip(mouse_col, cell_col, det_sessraw):
        sset = _parse_sessions(sess_raw)
        schedule = SCHEDULE[exp]  # list of (sess_key, ident, day_index)

        # Build rows for exposure days (ignore baseline)
        rows = []
        for sess_key, ident, day in schedule:
            rows.append({
                "mouse": mouse,
                "cell_id": cell,
                "experiment": exp,
                "day": day,                  # exposure index within experiment
                "sess_key": sess_key,
                "sess_ident": ident,         # "C" or "L"
                "active": 1 if sess_key in sset else 0,
            })
        # lag-based features per cell within its experiment (by day order)
        rows = sorted(rows, key=lambda r: r["day"])
        for i, r in enumerate(rows):
            if i == 0:
                r["prev_active"] = 0
                r["same_as_prev"] = 0
                r["has_prev"] = 0
            else:
                prev = rows[i-1]
                r["prev_active"] = prev["active"]
                r["same_as_prev"] = 1 if (r["sess_ident"] == prev["sess_ident"]) else 0
                r["has_prev"] = 1
            recs.append(r)

    long = pd.DataFrame.from_records(recs)
    # enforce dtypes
    long["active"] = long["active"].astype(np.int8)
    long["prev_active"] = long["prev_active"].astype(np.int8)
    long["same_as_prev"] = long["same_as_prev"].astype(np.int8)
    long["has_prev"] = long["has_prev"].astype(np.int8)
    long["sess_ident"] = long["sess_ident"].astype("category")
    long["experiment"] = long["experiment"].astype("category")
    return long

# --------------------------------------------------------
# 3) Aggregate to mouse × experiment × day with fixed n
# --------------------------------------------------------
def aggregate_mouse_day(long):
    """
    For each mouse×experiment, define denominator n as the count of unique cells
    that were active in at least one exposure day (C or L). Then per day compute y.
    Also create prev_prop and same_as_prev at the day level for that mouse×experiment.
    """
    # eligible cells per mouse×experiment (any exposure-day activity)
    g1 = (long.groupby(["mouse","experiment","cell_id"], observed=True)
              .agg(any_active=("active","max"))
              .reset_index())

    elig = (g1.query("any_active==1")
              .groupby(["mouse","experiment"], observed=True)
              .agg(n=("cell_id","nunique"))
              .reset_index())

    # y per day
    yday = (long.groupby(["mouse","experiment","day","sess_ident"], observed=True)
                 .agg(y=("active","sum"))
                 .reset_index())

    agg = yday.merge(elig, on=["mouse","experiment"], how="left", validate="m:1")
    agg = agg.dropna(subset=["n"]).copy()
    agg["n"] = agg["n"].astype(int)
    agg["prop"] = agg["y"] / agg["n"]

    # same_as_prev / prev_prop
    agg = agg.sort_values(["mouse","experiment","day"]).reset_index(drop=True)
    agg["prev_prop"] = agg.groupby(["mouse","experiment"], observed=True)["prop"].shift(1).fillna(0.0)

    prev_ident = agg.groupby(["mouse","experiment"], observed=True)["sess_ident"].shift(1)
    agg["same_as_prev"] = (agg["sess_ident"].values == prev_ident.values).astype(int)
    agg["same_as_prev"] = agg["same_as_prev"].fillna(0).astype(int)

    agg["has_prev"] = (agg.groupby(["mouse","experiment"], observed=True)["day"]
                         .rank(method="first") > 1).astype(int)

    # keep tidy dtypes
    agg["sess_ident"] = agg["sess_ident"].astype("category")
    agg["experiment"] = agg["experiment"].astype("category")
    return agg


# --------------------------------------------------------
# 4) Models
# --------------------------------------------------------
def fit_mouse_level_glm_clean(agg):
    # enforce reference level for sess_ident so we always get [T.L]
    agg = agg.copy()
    agg["sess_ident"] = pd.Categorical(agg["sess_ident"], categories=["C","L"])

    X = pt.dmatrix('1 + C(day)  + C(sess_ident)' ,
                   agg, return_type='dataframe')
    endog = np.column_stack([agg["y"].to_numpy(), (agg["n"] - agg["y"]).to_numpy()])
    model = sm.GLM(endog, X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster",
                    cov_kwds={"groups": agg["mouse"].to_numpy(), "use_correction": True})
    return res, X

def fit_cell_level_gee(long):
    """
    Optional: cell-level marginal model.
    GEE (Binomial logit), exchangeable within mouse, bias-reduced covariance.
    (Large N but degrees of freedom ~ #mice; this keeps p-values from exploding.)
    """
    X = pt.dmatrix('1 + day + prev_active + C(sess_ident) + same_as_prev + C(experiment)',
                   long, return_type='dataframe')
    y = long["active"].to_numpy()
    groups = long["mouse"].to_numpy()  # cluster by mouse (you can switch to mouse×cell if you prefer)
    gee = sm.GEE(endog=y, exog=X, groups=groups,
                 family=sm.families.Binomial(),
                 cov_struct=sm.cov_struct.Independence())
    res = gee.fit(cov_type="bias_reduced")
    return res, X.columns


#%%
longs = [
    make_long_cells(dfs[0][1], "exp5"),
    make_long_cells(dfs[1][1], "exp3a"),
    make_long_cells(dfs[2][1], "exp3b")
    ]
#%%
long_tst = pd.concat(longs)
#%%
long_tst["experiment"] = (long_tst["experiment"]
                          .replace({"exp3a":"exp3", "exp3b":"exp3"})
                          .astype("category"))
agg  = aggregate_mouse_day(long_tst)


res_glm, X_glm = fit_mouse_level_glm_clean(agg)
print(res_glm.summary())
print("Odds ratios:\n", np.exp(res_glm.params))
#%%
ame_eq = ame_L_minus_C(res_glm, X_glm, agg, by_mouse_equal=True)
print("Equal over mice: Δp = %.2f pp [%.2f, %.2f]" % 
      (100*ame_eq["AME"], 100*ame_eq["lo"], 100*ame_eq["hi"]))
#%%split
def fit_and_report(agg, exp):
    sub = agg[agg["experiment"] == exp].copy()
    
    print(pd.crosstab(sub["day"], sub["same_as_prev"]))
    print(pd.crosstab([sub["experiment"], sub["day"]], sub["same_as_prev"]))
    
    res, X = fit_mouse_level_glm_clean(sub)  # same helper you already have
    print(f"\n=== {exp} ===")
    print(res.summary())
    ame = ame_L_minus_C(res, X, sub, by_mouse_equal=True)
    print("AME (L−C, mouse-equal): %.2f pp [%.2f, %.2f]" %
          (100*ame["AME"], 100*ame["lo"], 100*ame["hi"]))
    return res, X

res3, X3 = fit_and_report(agg, "exp3")
#res5, X5 = fit_and_report(agg, "exp5")

#%%


#%% truncated

agg_trunc = agg[~((agg["experiment"]=="exp5") & (agg["day"]==4))].copy()
res_pool, X_pool = fit_mouse_level_glm_clean(agg_trunc)
print(res_pool.summary())
ame_pool = ame_L_minus_C(res_pool, X_pool, agg_trunc, by_mouse_equal=True)
print("Pooled exp3+exp5 (days 1–3): Δp = %.2f pp [%.2f, %.2f]" %
      (100*ame_pool["AME"], 100*ame_pool["lo"], 100*ame_pool["hi"]))