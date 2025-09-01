# --- Third-party
import numpy as np
import pandas as pd
import patsy as pt
from scipy import stats
from scipy.stats import norm, f as fdist
from scipy.special import expit as logistic

# ------------------------------
# Core helpers
# ------------------------------

def names_beta_V(res):
    """
    Return (names, beta, V) from a statsmodels GEEResults-like object.
    """
    names = list(res.model.exog_names)
    beta  = pd.Series(res.params, index=names, dtype=float)
    V     = pd.DataFrame(res.cov_params(), index=names, columns=names, dtype=float)
    return names, beta, V


def build_design(df, formula, set_fixed=None):
    """
    Build a design matrix with optional 'fixed covariates' overrides.

    set_fixed: dict like {"z": 0.0, "bg_mean": df["bg_mean"].median(), ...}
               Values can be scalars or 1D arrays aligned to df.
    """
    d = df.copy()
    if set_fixed:
        for k, v in set_fixed.items():
            if k not in d.columns:
                raise KeyError(f"set_fixed key {k!r} not found in df")
            d.loc[:, k] = v
    # Build design and DROP NA rows used by the formula
    X = pt.dmatrix(formula, d, return_type="dataframe", NA_action="drop")
    # Align data rows to the design matrix and reset to positional indices
    d = d.loc[X.index].copy()
    d.reset_index(drop=True, inplace=True)
    X = X.reset_index(drop=True)
    return d, X


def _x_for_level(names, level_label, factor_prefix="C(pair"):
    """
    Construct a row vector for a given categorical level in a Treatment-coded factor.
    Matches columns ending with '[T.<level_label>]'.
    """
    x = np.zeros(len(names), dtype=float)
    # Intercept always present for Treatment coding.
    if "Intercept" in names:
        x[names.index("Intercept")] = 1.0
    # Add the indicator for the requested level
    tag = f"[T.{level_label}]"
    for j, nm in enumerate(names):
        if nm.startswith(factor_prefix) and nm.endswith(tag):
            x[j] = 1.0
    return x


# ------------------------------
# EMMs on probability scale
# ------------------------------

def emm_pair(
    res,
    df,
    pair_label,
    ref_label,
    *,
    formula,
    averaging="cell",              # "cell" | "mouse" | "weights"
    mouse_col="mouse",
    weights=None,
    set_fixed=None
):
    """
    Compute an EMM (predicted probability) for a given 'pair' level and its
    delta-method gradient matching the averaging scheme.

    averaging = "cell":   simple row mean
                "mouse":  mean within mouse, then mean across mice
                "weights": custom row weights (array aligned to df), normalized to sum=1
    set_fixed: dict of fixed-covariate overrides applied before building X.
    """
    # Force the pair level for all rows (avg of predictions approach)
    d = df.copy()
    all_levels = pd.Categorical(df["pair"]).categories
    d["pair"] = pd.Categorical([pair_label] * len(d), categories=all_levels)

    d, X = build_design(d, formula, set_fixed=set_fixed)
    eta = np.asarray(X @ res.params.values, dtype=float).reshape(-1)
    p   = logistic(eta)  # predictions on probability scale

    X_arr = np.asarray(getattr(X, "values", X), dtype=float)

    if averaging == "weights":
        if weights is None:
            raise ValueError("averaging='weights' requires weights array")
        w = np.asarray(weights, dtype=float).reshape(-1)
        if len(w) != len(p):
            raise ValueError("weights must have same length as df")
        w = w / w.sum()
        p_bar = float((w * p).sum())
        grad  = (w[:, None] * (p * (1 - p))[:, None] * X_arr).sum(axis=0)
        return p_bar, grad

    if averaging == "cell":
        p_bar = float(p.mean())
        grad  = ((p * (1 - p))[:, None] * X_arr).mean(axis=0)
        return p_bar, grad

    if averaging == "mouse":
        if mouse_col not in d:
            raise KeyError(f"mouse_col {mouse_col!r} not in df")
        p_bar = float(d.assign(_p=p).groupby(mouse_col)["_p"].mean().mean())
        mats = []
        for _, sub in d.groupby(mouse_col, sort=False):
            idx = sub.index.to_numpy()
            gm  = ((p[idx] * (1 - p[idx]))[:, None] * X_arr[idx, :]).mean(axis=0)
            mats.append(gm)
        grad = np.vstack(mats).mean(axis=0)
        return p_bar, grad

    raise ValueError("averaging must be 'cell', 'mouse', or 'weights'")


def emm_prob_ci(
    res,
    df,
    pair_label,
    ref_label,
    *,
    formula,
    alpha=0.05,
    averaging="cell",
    mouse_col="mouse",
    weights=None,
    set_fixed=None
):
    """
    EMM and (delta-method) CI on the probability scale for a single pair level.
    """
    p_bar, grad = emm_pair(
        res, df, pair_label, ref_label,
        formula=formula,
        averaging=averaging, mouse_col=mouse_col,
        weights=weights, set_fixed=set_fixed
    )
    V   = np.asarray(res.cov_params(), dtype=float)
    var = float(grad @ V @ grad)
    se  = max(var, 0.0) ** 0.5
    z   = norm.ppf(1 - alpha/2)
    lo, hi = p_bar - z*se, p_bar + z*se
    return {"prob": p_bar, "lo": max(0.0, lo), "hi": min(1.0, hi), "se_prob": se}


# ------------------------------
# Contrasts vs reference
# ------------------------------

def contrast_vs_ref_logit(res, pair_label, ref_label, factor_prefix="C(pair"):
    """
    Logit-scale (linear predictor) contrast: pair_label - ref_label with
    large-sample z (or t if you do that externally).
    """
    names, beta, V = names_beta_V(res)
    xa = _x_for_level(names, pair_label, factor_prefix=factor_prefix)
    xb = _x_for_level(names, ref_label,  factor_prefix=factor_prefix)
    c  = xa - xb
    est = float(c @ beta)
    var = float(c @ V @ c)
    se  = max(var, 0.0) ** 0.5
    z   = est / se if se > 0 else np.nan
    p   = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p


def delta_prob_test(
    res,
    df,
    pair_label,
    ref_label,
    *,
    formula,
    df_t=None,                 # supply (#clusters - 1) if you want t correction
    averaging="cell",
    mouse_col="mouse",
    weights=None,
    set_fixed=None
):
    """
    Δp = P(pair_label) - P(ref_label) on probability scale with delta-method SE and p-value.
    """
    p_pair, g_pair = emm_pair(
        res, df, pair_label, ref_label,
        formula=formula,
        averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
    )
    p_ref,  g_ref  = emm_pair(
        res, df, ref_label, ref_label,
        formula=formula,
        averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
    )
    V = np.asarray(res.cov_params(), dtype=float)
    delta   = p_pair - p_ref
    g_delta = g_pair - g_ref
    se      = float((g_delta @ V @ g_delta) ** 0.5)

    if se <= 1e-12:
        p_val = 1.0
        stat  = np.nan
    else:
        stat = delta / se
        if df_t is None:
            p_val = 2 * norm.sf(abs(stat))
        else:
            p_val = 2 * stats.t.sf(abs(stat), df_t)

    return {"delta": delta, "se": se, "stat": stat, "p": p_val}


# ------------------------------
# Omnibus (factor-level) Wald test
# ------------------------------

def wald_omnibus_for_factor(res, factor_prefix="C(pair", n_clusters=None):
    """
    Omnibus Wald test over all columns whose name starts with factor_prefix,
    e.g., "C(pair, Treatment(...)" columns. Returns (p_value, statistic).
    If res.cov_type != 'bias_reduced', provide n_clusters to compute an F with df1 = k, df2 = n_clusters-1.
    """
    names = res.model.exog_names
    idx   = [i for i, n in enumerate(names) if n.startswith(factor_prefix)]
    if not idx:
        return np.nan, np.nan

    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0

    w = res.wald_test(L, scalar=True)
    df1 = len(idx)
    if res.cov_type == "bias_reduced":
        # statsmodels returns a chi-square under large-sample;
        # for bias_reduced we typically report the chi-square p-value
        return float(w.pvalue), float(np.asarray(w.statistic))

    if n_clusters is None:
        raise ValueError("n_clusters required for F correction when cov_type != 'bias_reduced'")

    df2   = max(int(n_clusters) - 1, 1)
    chi2  = float(np.asarray(w.statistic))
    Fstat = chi2 / df1
    p     = 1.0 - fdist.cdf(Fstat, df1, df2)
    return float(p), float(Fstat)


def prepare_covariates(
    df: pd.DataFrame,
    formula_for_fit: str,
    *,
    mouse_col: str = "mouse",
    mundlak_var: str = "bg_mean",           # background column to split into within/between
    center_vars: tuple = ("snr_pre_std","snr_post_std","z"),  # recentre these to mean 0 on fitting rows
    copy: bool = True
):
    """
    Align to the rows actually used by the model (via `formula_for_fit` with NA_action='drop'),
    then create Mundlak terms for `mundlak_var` and mean-center the chosen `center_vars`.

    Returns
    -------
    d          : DataFrame aligned and augmented with:
                 - '{mundlak_var}_mouse'  (between-mouse, mean-centered)
                 - '{mundlak_var}_within' (within-mouse, mean 0 by construction)
                 - centered versions in-place for any vars listed in `center_vars` that exist
    fixed_zero : dict mapping covariates to 0.0 for EMMs under "typical conditions"
    stats      : dict with means used for centering and the index used for fitting
    """
    # 1) Build design once to determine the fitting subset (handles NaNs consistently with your model)
    X = pt.dmatrix(formula_for_fit, df, return_type="dataframe", NA_action="drop")
    idx = X.index
    d = df.loc[idx].copy() if copy else df.loc[idx]

    # 2) Mundlak: split background into between-mouse and within-mouse components
    made_mundlak = False
    if mundlak_var in d.columns and mouse_col in d.columns:
        # between-mouse mean
        d[f"{mundlak_var}_mouse"] = d.groupby(mouse_col)[mundlak_var].transform("mean")
        # within-mouse deviation
        d[f"{mundlak_var}_within"] = d[mundlak_var] - d[f"{mundlak_var}_mouse"]
        # center the between-mouse term so 0 == "typical mouse"
        mu_between = float(d[f"{mundlak_var}_mouse"].mean())
        d[f"{mundlak_var}_mouse"] = d[f"{mundlak_var}_mouse"] - mu_between
        made_mundlak = True
    else:
        mu_between = np.nan  # recorded for stats

    # 3) Mean-center chosen covariates (if present) on the fitting subset
    means_used = {}
    centered_cols = []
    for c in center_vars:
        if c in d.columns:
            m = float(d[c].mean())
            d[c] = d[c] - m
            means_used[c] = m
            centered_cols.append(c)

    # 4) Build a fixed dict where 0.0 = typical conditions
    fixed_zero = {}
    # centered vars → 0
    for c in centered_cols:
        fixed_zero[c] = 0.0
    # Mundlak terms (if created)
    if made_mundlak:
        fixed_zero[f"{mundlak_var}_within"] = 0.0   # at mouse's typical background
        fixed_zero[f"{mundlak_var}_mouse"]  = 0.0   # typical mouse

    stats = {
        "index": idx,
        "means_centered": means_used,
        f"{mundlak_var}_mouse_mean": mu_between if made_mundlak else None,
        "centered_vars": tuple(centered_cols),
        "mundlak_made": made_mundlak,
    }
    return d, fixed_zero, stats


import numpy as np
import pandas as pd
from scipy.stats import norm, t
from statsmodels.stats.multitest import multipletests

def summarize_pairs_delta_and_or(
    res,
    df,
    pair_labels,
    ref_label,
    *,
    formula,
    df_t=None,                  # e.g., clusters-1 for small-sample t
    averaging="mouse",          # "cell" | "mouse" | "weights"
    mouse_col="mouse",
    weights=None,
    set_fixed=None,
    alpha=0.05
):
    """
    For each pair in `pair_labels`, compute:
      - Δp = P(pair) - P(ref) on the probability scale (EMMs)
      - SE(Δp), (1-alpha) CI, raw p, Holm-adjusted p (across the pairs provided)
      - Logit-contrast odds ratio (OR) with CI
      - The two EMM probabilities (p_pair, p_ref) for context

    Returns
    -------
    pandas.DataFrame with one row per comparison.
    """
    rows = []
    zcrit = t.ppf(1 - alpha/2, df_t) if df_t is not None else norm.ppf(1 - alpha/2)
    stat_name = f"t (df={int(df_t)})" if df_t is not None else "z"

    for lab in pair_labels:
        # Δp + SE + p (marginal, per your averaging/fixed scheme)
        d = delta_prob_test(
            res, df, lab, ref_label,
            formula=formula, df_t=df_t,
            averaging=averaging, mouse_col=mouse_col,
            weights=weights, set_fixed=set_fixed
        )
        delta, se, stat, p_delta = float(d["delta"]), float(d["se"]), float(d["stat"]), float(d["p"])
        ci_lo, ci_hi = delta - zcrit*se, delta + zcrit*se

        # EMM probabilities for each level (useful to report)
        p_pair, _ = emm_pair(
            res, df, lab, ref_label, formula=formula,
            averaging=averaging, mouse_col=mouse_col,
            weights=weights, set_fixed=set_fixed
        )
        p_ref, _  = emm_pair(
            res, df, ref_label, ref_label, formula=formula,
            averaging=averaging, mouse_col=mouse_col,
            weights=weights, set_fixed=set_fixed
        )

        # Logit contrast → OR (+ CI on logit scale, exponentiated)
        est_logit, se_logit, z_logit, p_logit = contrast_vs_ref_logit(res, lab, ref_label)
        OR    = float(np.exp(est_logit))
        OR_lo = float(np.exp(est_logit - zcrit*se_logit)) if np.isfinite(se_logit) else np.nan
        OR_hi = float(np.exp(est_logit + zcrit*se_logit)) if np.isfinite(se_logit) else np.nan

        rows.append({
            "pair": lab,
            "ref": ref_label,
            # probability-scale estimand
            "p_ref": float(p_ref),
            "p_pair": float(p_pair),
            "delta": delta,
            "se_delta": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "stat": stat,
            "stat_name": stat_name,
            "p_raw": p_delta,
            # logit-scale (conditional) estimand
            "OR": OR,
            "OR_lo": OR_lo,
            "OR_hi": OR_hi,
            "p_logit": float(p_logit),
        })


def add_holm(df, p_col="p_delta", alpha=0.05):
    if len(df) <= 1:
        df = df.copy()
        df["p_holm"] = df[p_col]
        df["reject_holm"] = False
        return df
    reject, p_holm, _, _ = multipletests(df[p_col], method="holm", alpha=alpha)
    df = df.copy()
    df["p_holm"] = p_holm
    df["reject_holm"] = reject
    return df