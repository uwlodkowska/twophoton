# --- deps
import numpy as np, pandas as pd, patsy as pt
from scipy.special import expit as logistic
from scipy.stats import norm, t as tdist, f as fdist
from statsmodels.stats.multitest import multipletests

# ========= basic helpers (names, beta, V, & design) =========
def _gee_names_beta_V(res):
    names = list(res.model.exog_names)
    beta  = pd.Series(res.params, index=names)
    V     = pd.DataFrame(res.cov_params(), index=names, columns=names)
    return names, beta, V

def _design(df, formula, names_like=None):
    """Design matrix as DataFrame; optionally reindex columns to 'names_like'."""
    X = pt.dmatrix(formula, df, return_type='dataframe')
    if names_like is not None:
        # ensure same column order as used in fit
        X = X.reindex(columns=names_like)
    return X

def _raw_cols_from_formula(formula: str):
    import re
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)
    skip = {"C","bs","Treatment","reference","pair"}
    return [t for t in toks if t not in skip]

def _per_mouse_template(df, formula, pair_level, set_fixed=None, mouse_col="mouse"):
    cols = _raw_cols_from_formula(formula)
    mice = pd.Index(df[mouse_col].unique(), name=mouse_col)
    # start with per-mouse means (independent of pair) for any needed covariates
    base = (df.groupby(mouse_col)[[c for c in cols if c in df.columns]]
              .mean().reindex(mice).reset_index())
    base["pair"] = pair_level
    # override with fixed values, if provided
    if set_fixed:
        for k, v in set_fixed.items():
            if k in base.columns:
                base[k] = v
    # fill any missing needed columns with global means or 0.0
    for c in cols:
        if c not in base.columns:
            base[c] = df[c].mean() if c in df.columns else 0.0
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0)
    return base  # exactly 1 row per mouse

# ========= EMM on probability scale + delta-method + t-correction =========
def pair_emm_and_grad(res, df, formula, pair_label, ref_label,
                      averaging="mouse", mouse_col="mouse", weights=None, set_fixed=None):
    """
    EMM on prob. scale for a given 'pair' level, with gradient matching the averaging.
    averaging: "mouse" = mean within mouse -> mean across mice; "row" = simple row mean.
    If weights is provided (len==len(df)), compute a weighted row-mean instead.
    """
    d = df.copy()
    
    if set_fixed:
        for k, v in set_fixed.items():
            if k not in d.columns:
                raise KeyError(f"set_fixed key {k!r} not found in df")
            d.loc[:, k] = v     
    # Keep all categories; do NOT drop unused
    if "pair" not in d.columns:
        raise KeyError("'pair' column missing")
    if not pd.api.types.is_categorical_dtype(d["pair"]):
        d["pair"] = pd.Categorical(d["pair"])
    all_levels = list(d["pair"].cat.categories)
    if pair_label not in all_levels:
        raise ValueError(f"Requested pair {pair_label!r} not in categories {all_levels}")

    have_pair = set(df.loc[df["pair"] == pair_label, mouse_col].unique())
    have_ref  = set(df.loc[df["pair"] == ref_label,  mouse_col].unique())
    keep_mice = have_pair & have_ref
    if len(keep_mice) == 0:
        raise ValueError(f"No mice observed for both {pair_label!r} and {ref_label!r}")
    d = d[d[mouse_col].isin(list(keep_mice))].copy()
    d["pair"] = pd.Categorical([pair_label]*len(d), categories=all_levels)

    # Build the same design used at fit-time; keep same column order as model
    names, beta, _ = _gee_names_beta_V(res)
    X = _design(d, formula, names_like=names)

    eta = (X.values @ beta.values)
    p   = logistic(eta) 
    p   = np.asarray(p, dtype=float).reshape(-1)
    Xv  = X.values

    # Averaging & gradient
    if weights is not None:
        
        w = np.asarray(weights, dtype=float).reshape(-1)
        # if len(w) != len(df):
        #     raise ValueError("weights must match df length before filtering")
        pos = df.index.get_indexer(d.index)
        w = w[pos]
        if w.shape[0] != len(d):
            raise ValueError("weights length must equal len(df)")

        w = w / w.sum()
            

        p_bar = float((w * p).sum())
        grad  = (w[:,None] * (p*(1-p))[:,None] * Xv).sum(axis=0)
        return p_bar, grad



    if averaging == "row":
        p_bar = float(p.mean())
        grad  = ((p*(1-p))[:,None] * Xv).mean(axis=0)
        return p_bar, grad

    # mouse-balanced
    if mouse_col not in d.columns:
        raise KeyError(f"{mouse_col!r} column missing for mouse-balanced averaging")
    p_bar = float(pd.DataFrame({"mouse": d[mouse_col].values, "p": p}).groupby("mouse")["p"].mean().mean())
    mats = []
    for _, sub in d.groupby(mouse_col, sort=False):
        idx = d.index.get_indexer(sub.index)
        gm  = ((p[idx]*(1-p[idx]))[:,None] * Xv[idx,:]).mean(axis=0)
        mats.append(gm)
    grad = np.vstack(mats).mean(axis=0)
    return p_bar, grad

def pair_emm(res, df, formula, pair_label, ref_label,
             averaging="mouse", mouse_col="mouse", weights=None,
             p0=0.5, use_t=True, dof=None, set_fixed=None):
    """
    Wraps pair_emm_and_grad to return prob, CI, se, and test vs null (p0=0.5 for logit).
    Small-cluster t: df = (#unique mice - 1) unless overridden.
    """
    names, beta, V = _gee_names_beta_V(res)
    p_bar, grad = pair_emm_and_grad(res, df, formula, pair_label, ref_label,
                                    averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed  )
    var = float(grad @ V.values @ grad)
    se  = float(np.sqrt(max(var, 0.0)))

    # degrees of freedom for t
    if use_t:
        if dof is not None:
            df_eff = int(dof)
        else:
            if averaging == "mouse" and mouse_col in df.columns:
                G = df[mouse_col].nunique()
            else:
                G = df[mouse_col].nunique() if mouse_col in df.columns else None
            df_eff = (G - 1) if (G is not None and G > 1) else 1
        crit = tdist.ppf(0.975, df_eff)
    else:
        df_eff = None
        crit = norm.ppf(0.975)

    lo, hi = p_bar - crit*se, p_bar + crit*se

    lo, hi = max(0.0, lo), min(1.0, hi)
    stat = (p_bar - float(p0)) / se if se > 1e-15 else np.nan
    

    if use_t and df_eff is not None:
        pval = 2 * (1 - tdist.cdf(abs(stat), df_eff))
        dist = "t"
    else:
        pval = 2 * (1 - norm.cdf(abs(stat)))
        dist = "z"
    
    return dict(pair=pair_label, prob=p_bar, lo=lo, hi=hi, se_prob=se,
                stat=stat, pval=pval, dof=df_eff, dist=dist)

# ========= table of EMMs for many pairs + Holm correction =========
def emms_by_pair(res, df, formula, pair_levels, ref_label,
                 averaging="mouse", mouse_col="mouse", weights=None,
                 p0=0.5, use_t=True, dof=None, adjust="holm",
                 set_fixed=None):
    rows = []
    for lvl in pair_levels:
        out = pair_emm(res, df, formula, lvl, ref_label,
                       averaging=averaging, mouse_col=mouse_col, weights=weights,
                       p0=p0, use_t=use_t, dof=dof, set_fixed=set_fixed)
        rows.append(out)
    tab = pd.DataFrame(rows, columns=["pair","prob","lo","hi","se_prob","stat","pval","dof","dist"])
    rej, p_adj, _, _ = multipletests(tab["pval"].values, method=adjust)
    tab["pval_adj"] = p_adj
    tab["reject"]   = rej
    tab["adjust_method"] = adjust
    return tab

# ========= t-based contrasts vs reference with Holm (Welch df) =========
def contrast_prob_diff_cluster(emm_ref, emm_alt, n_clusters, rho=0.0):
    """
    Contrast two EMMs on the probability scale (alt - ref) using
    small-cluster t with df = n_clusters - 1. No Welch.

    Parameters
    ----------
    emm_ref, emm_alt : dict-like with keys {'prob','se_prob'}
    n_clusters : int
        Number of clusters (e.g., mice) used in the model/EMM averaging.
    rho : float in [-1, 1]
        Assumed correlation between the two EMM estimators (default 0).

    Returns
    -------
    dict: diff, se, lo, hi, stat, dof, dist, pval
    """
    m1, s1 = float(emm_ref["prob"]), float(emm_ref["se_prob"])
    m2, s2 = float(emm_alt["prob"]), float(emm_alt["se_prob"])
    diff = m2 - m1

    # variance of difference with optional correlation
    var = s1**2 + s2**2 - 2.0 * rho * s1 * s2
    var = max(var, 0.0)
    se  = float(np.sqrt(var))

    dof = max(int(n_clusters) - 1, 1)
    stat = diff / se if se > 0 else np.nan
    crit = tdist.ppf(0.975, dof)
    pval = 2 * (1 - tdist.cdf(abs(stat), dof))
    lo, hi = diff - crit * se, diff + crit * se

    dz = (stat / np.sqrt(n_clusters)) if (n_clusters and np.isfinite(stat)) else np.nan

    return dict(diff=diff, se=se, lo=lo, hi=hi, stat=stat,
                dof=dof, dist="t", pval=pval, dz = dz)


def contrasts_vs_reference_cluster(emms_df, ref_level, n_clusters,
                                   adjust="holm", rho=0.0):
    """
    Run contrasts (each level vs reference) with small-cluster t
    using df = n_clusters - 1. No Welch. Holm/FDR/etc. correction supported.

    Parameters
    ----------
    emms_df : DataFrame with columns ['pair','prob','se_prob']
    ref_level : str
    n_clusters : int
    adjust : str, default 'holm'
    rho : float, assumed correlation between EMMs (default 0)

    Returns
    -------
    DataFrame with: pair, ref, diff, lo, hi, se, stat, dof, dist, pval, pval_adj, reject, adjust_method
    """
    need = {"pair", "prob", "se_prob"}
    miss = need - set(emms_df.columns)
    if miss:
        raise ValueError(f"emms_df missing columns: {sorted(miss)}")
    if ref_level not in set(emms_df["pair"]):
        raise ValueError(f"ref_level {ref_level!r} not found in emms_df['pair']")

    ref = emms_df.loc[emms_df["pair"] == ref_level].iloc[0].to_dict()

    rows = []
    for _, r in emms_df.iterrows():
        if r["pair"] == ref_level:
            continue
        alt = r.to_dict()
        res = contrast_prob_diff_cluster(ref, alt, n_clusters=n_clusters, rho=rho)
        res["pair"] = alt["pair"]
        res["ref"]  = ref_level
        rows.append(res)

    out = pd.DataFrame(rows, columns=["pair","ref","diff","lo","hi","se","stat","dof","dist","pval", "dz"])
    if not out.empty:
        rej, p_adj, _, _ = multipletests(out["pval"].values, method=adjust)
        out["pval_adj"] = p_adj
        out["reject"] = rej
        out["adjust_method"] = adjust
    return out

def contrast_prob_diff_delta(res, df, formula, pair_alt, pair_ref,
                             averaging="mouse", mouse_col="mouse", set_fixed=None):
    # reuse your EMM gradient machinery
    _, _, V = _gee_names_beta_V(res)
    p_alt, g_alt = pair_emm_and_grad(res, df, formula, pair_alt, pair_ref,
                                     averaging=averaging, mouse_col=mouse_col, set_fixed=set_fixed)
    p_ref, g_ref = pair_emm_and_grad(res, df, formula, pair_ref, pair_ref,
                                     averaging=averaging, mouse_col=mouse_col, set_fixed=set_fixed)
    diff = p_alt - p_ref
    g    = g_alt - g_ref
    var  = float(g @ V.values @ g)
    se   = float(np.sqrt(max(var, 0.0)))
    G    = df[mouse_col].nunique()
    dof  = max(G-1, 1)
    stat = diff / se if se > 0 else np.nan
    pval = 2 * (1 - tdist.cdf(abs(stat), dof))
    crit = tdist.ppf(0.975, dof)
    lo, hi = diff - crit*se, diff + crit*se
    return dict(diff=diff, se=se, lo=lo, hi=hi, stat=stat, dof=dof, dist="t", pval=pval)


# ========= omnibus Wald for C(pair, ...) with small-sample F =========
def wald_omnibus_pair(res, n_clusters, reference="ctx1_to_ctx2", by="z", term="pair"):
    names = res.model.exog_names
    base = f'C({term}, Treatment(reference="{reference}"))'
    idx = [i for i, n in enumerate(names)
           if n.startswith(base + "[T")]
    if not idx:
        return dict(p=np.nan, stat=np.nan, df1=np.nan, df2=np.nan, note="no pair:z terms")
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx): L[r, j] = 1.0

    w = res.wald_test(L, scalar=False)
    chi2 = float(np.asarray(w.statistic))
    df1  = len(idx)

    if res.cov_type == "bias_reduced":
        return dict(p=float(w.pvalue), stat=chi2, df1=df1, df2=None, note="chi2 (bias_reduced)")

    df2 = max(int(n_clusters) - 1, 1)
    F = chi2 / df1
    p = 1.0 - fdist.cdf(F, df1, df2)
    return dict(p=float(p), stat=float(F), df1=df1, df2=df2, note="F small-sample")

# ========= convenience: build EMM table, then contrasts =========
def build_emms_and_contrasts(res, df, formula, pair_levels, ref_label,
                             averaging="mouse", mouse_col="mouse",
                             weights=None, p0=0.5, adjust="holm"):
    emms = emms_by_pair(res, df, formula, pair_levels, ref_label,
                        averaging=averaging, mouse_col=mouse_col, weights=weights,
                        p0=p0, use_t=True, dof=None, adjust=adjust)
    n_clusters = df["mouse"].nunique()
    contr = contrasts_vs_reference_cluster(emms_df=emms, ref_level=ref_label,
                                       n_clusters=n_clusters,
                                       adjust="holm", rho=0.0)
    return emms, contr



import numpy as np
import pandas as pd
import patsy as pt
from scipy.special import expit as logistic
from scipy.stats import f as fdist

def wald_omnibus_pair_prob(
    res,
    data_like: pd.DataFrame,
    formula: str,
    pair_levels,
    reference="ctx1_to_ctx2",
    set_fixed: dict = None,
    n_clusters: int = None,
    note="prob-scale omnibus (pair)"
):
    """
    Omnibus test that all pair probabilities equal the reference probability
    at given covariate values (nonlinear Wald on probability scale).

    Parameters
    ----------
    res : fitted statsmodels results (GEE/GLM) with .params and .cov_params()
    data_like : DataFrame used to build design matrices (column names/codings)
    formula : Patsy RHS formula used for fitting (e.g., '1 + C(pair, Treatment(...)) + z + snr')
    pair_levels : list of all levels of 'pair' in the fitted model (including reference)
    reference : reference level of pair
    set_fixed : dict of {col: value} to fix covariates (e.g., {'z': 0.5, 'snr': 0})
    n_clusters : for small-sample F df2 correction; if None, tries to infer from res
    note : string stored in output
    """
    # pick a template row and set covariates
    d0 = data_like.iloc[[0]].copy()
    if set_fixed:
        for k, v in set_fixed.items():
            if k in d0: d0[k] = v

    # make 'pair' categorical with full levels
    d0["pair"] = pd.Categorical([reference], categories=list(pair_levels))

    # helper: design row for a given pair
    def design_for(pair):
        d = d0.copy()
        d["pair"] = pd.Categorical([pair], categories=list(pair_levels))
        X = pt.dmatrix(formula, d, return_type="dataframe")
        # align to model params
        X = X.reindex(columns=res.params.index, fill_value=0.0)
        return X.iloc[0].to_numpy()[None, :]  # shape (1, p)

    # compute probs p_j and gradients g_j = dp/dβ = p(1-p) * x_j
    beta = res.params.to_numpy()
    V = res.cov_params().to_numpy() if hasattr(res.cov_params(), "to_numpy") else np.asarray(res.cov_params())
    X_ref = design_for(reference)
    eta_ref = float(X_ref @ beta)
    p_ref = logistic(eta_ref)
    g_ref = p_ref * (1.0 - p_ref) * X_ref  # shape (1,p)

    pairs_alt = [lvl for lvl in pair_levels if lvl != reference]
    k = len(pairs_alt)
    if k == 0:
        return dict(p=np.nan, stat=np.nan, df1=np.nan, df2=np.nan, note="only reference level")

    h = np.zeros((k, 1))
    G = np.zeros((k, beta.size))
    for i, lvl in enumerate(pairs_alt):
        Xj = design_for(lvl)
        eta = float(Xj @ beta)
        pj = logistic(eta)
        gj = pj * (1.0 - pj) * Xj
        h[i, 0] = pj - p_ref
        G[i, :] = (gj - g_ref)

    # Wald statistic on prob scale: W = h' (G V G')^-1 h
    GVGt = G @ V @ G.T
    # pseudo-inverse for safety
    try:
        inv = np.linalg.pinv(GVGt, rcond=1e-12)
    except Exception:
        return dict(p=np.nan, stat=np.nan, df1=k, df2=np.nan, note=note + " (singular GVGt)")
    W = float(h.T @ inv @ h)  # chi^2_k under large-sample

    # small-sample F or chi2 depending on cov_type
    if getattr(res, "cov_type", "") == "bias_reduced":
        return dict(p=float(1.0 - fdist.cdf(W / k, k, 1e9)), stat=W, df1=k, df2=None,
                    note=note + " (chi2 approx; bias_reduced)")
    # df2
    if n_clusters is None:
        try:
            n_clusters = int(getattr(res.model, "groups").n_groups)
        except Exception:
            n_clusters = None
    if n_clusters is None:
        # fall back to chi2 if we can't infer clusters
        from scipy.stats import chi2
        pval = float(1.0 - chi2.cdf(W, df=k))
        return dict(p=pval, stat=W, df1=k, df2=None, note=note + " (chi2)")
    df1 = k
    df2 = max(int(n_clusters) - 1, 1)
    F = W / df1
    pval = float(1.0 - fdist.cdf(F, df1, df2))
    return dict(p=pval, stat=F, df1=df1, df2=df2, note=note + " (F small-sample)")

# omni_prob = wald_omnibus_pair_prob(
#     res=res,
#     data_like=long_nobgr,                                 # the df used at fit
#     formula='1 + C(pair, Treatment(reference="ctx1_to_ctx2")) + z_std + snr_dest_std',
#     pair_levels=PAIR_LEVELS,
#     reference="ctx1_to_ctx2",
#     set_fixed={"z_std": 0.0, "snr_dest_std": 0.0},        # fix covariates here
#     n_clusters=long_nobgr["mouse"].nunique()
# )
# print(omni_prob)


from scipy.special import expit as logistic
from scipy.stats import t as tdist, norm

def marginal_rd_pair(res, df, formula, pair_alt, pair_ref,
                     averaging="mouse", mouse_col="mouse", use_t=True):
    """
    Average risk difference (alt - ref) on the probability scale at the *empirical*
    covariate distribution in df, with a delta-method SE (small-cluster t).
    """
    names, beta, V = _gee_names_beta_V(res)

    def p_and_grad(pair):
        d = df.copy()
        # ensure 'pair' categorical keeps all levels used at fit
        if "pair" not in d.columns:
            raise KeyError("'pair' column missing")
        if not pd.api.types.is_categorical_dtype(d["pair"]):
            d["pair"] = pd.Categorical(d["pair"])
        levels = list(d["pair"].cat.categories)
        d["pair"] = pd.Categorical([pair]*len(d), categories=levels)

        X = _design(d, formula, names_like=names).values
        eta = X @ beta.values
        p   = logistic(eta).reshape(-1)
        gx  = (p*(1-p))[:, None] * X  # rowwise gradient

        if averaging == "mouse":
            # mean-by-mouse, then mean; gradient averages the same way
            p_bar = float(pd.DataFrame({"m": d[mouse_col].values, "p": p}).groupby("m")["p"].mean().mean())
            mats = []
            for _, sub in d.groupby(mouse_col, sort=False):
                idx = sub.index
                
                idx = d.index.get_indexer(sub.index)
                # gm  = ((p[idx]*(1-p[idx]))[:,None] * Xv[idx,:]).mean(axis=0)
                # mats.append(gm)
                
                mats.append(gx[idx, :].mean(axis=0))
            grad = np.vstack(mats).mean(axis=0)
        else:
            p_bar = float(p.mean())
            grad  = gx.mean(axis=0)
        return p_bar, grad

    p_alt, g_alt = p_and_grad(pair_alt)
    p_ref, g_ref = p_and_grad(pair_ref)

    diff = p_alt - p_ref
    g = g_alt - g_ref
    var = float(g @ V.values @ g)
    se  = float(np.sqrt(max(var, 0.0)))

    # small-cluster t
    G = df[mouse_col].nunique() if mouse_col in df.columns else None
    if use_t and G and G > 1:
        df_eff = G - 1
        crit = tdist.ppf(0.975, df_eff)
        pval = 2*(1 - tdist.cdf(abs(diff/se), df_eff)) if se > 0 else float("nan")
        dist = "t"
    else:
        df_eff = None
        crit = norm.ppf(0.975)
        pval = 2*(1 - norm.cdf(abs(diff/se))) if se > 0 else float("nan")
        dist = "z"

    lo, hi = diff - crit*se, diff + crit*se
    return {"diff": diff, "se": se, "lo": lo, "hi": hi,
            "stat": (diff/se if se > 0 else float("nan")),
            "dof": df_eff, "dist": dist, "pval": pval}



# --- NEW: delta-method omnibus on probability scale with averaging/weights ---

def wald_omnibus_pair_delta(
    res,
    df,
    formula,
    pair_levels,
    ref_label,
    averaging="mouse",
    mouse_col="mouse",
    weights=None,
    set_fixed=None,
    n_clusters=None,
):
    """
    Nonlinear Wald test that all pair probabilities equal the reference
    at the chosen covariate setting (delta method on prob scale).

    averaging: "mouse" (default) balances mice; "row" averages rows.
    set_fixed: dict to fix covariates (e.g., {'snr': 0}); if None, averages over df.
    Small-sample uses F(df1=k, df2=n_clusters-1). For bias_reduced cov, uses chi2.
    """
    names, beta, V = _gee_names_beta_V(res)

    # reference p and gradient
    p_ref, g_ref = pair_emm_and_grad(
        res, df, formula, ref_label, ref_label,
        averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
    )

    alts = [lvl for lvl in pair_levels if lvl != ref_label]
    k = len(alts)
    if k == 0:
        return dict(p=float("nan"), stat=float("nan"), df1=float("nan"), df2=float("nan"),
                    note="only reference level")

    # assemble h (differences) and G (gradients of differences)
    import numpy as np
    h = np.zeros((k, 1))
    G = np.zeros((k, len(beta)))
    for i, lvl in enumerate(alts):
        p_alt, g_alt = pair_emm_and_grad(
            res, df, formula, lvl, ref_label,
            averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
        )
        h[i, 0] = p_alt - p_ref
        G[i, :] = (g_alt - g_ref)

    GVGt = G @ V.values @ G.T
    inv = np.linalg.pinv(GVGt, rcond=1e-12)
    W = float(h.T @ inv @ h)           # ~ chi2_k large-sample
    df1 = k

    # small-sample correction
    if n_clusters is None and (mouse_col in df.columns):
        n_clusters = int(df[mouse_col].nunique())

    if getattr(res, "cov_type", "") == "bias_reduced" or not n_clusters:
        # fall back to chi2 if bias-reduced or unknown clusters
        from scipy.stats import chi2
        p = float(1.0 - chi2.cdf(W, df=df1))
        return dict(p=p, stat=W, df1=df1, df2=None, note="prob-scale omnibus (chi2)")

    df2 = max(int(n_clusters) - 1, 1)
    F = W / df1
    p = float(1.0 - fdist.cdf(F, df1, df2))
    return dict(p=p, stat=F, df1=df1, df2=df2, note="prob-scale omnibus (F small-sample)")


# --- NEW: single contrast (alt - ref) on prob scale via delta method + small-sample t ---
def contrast_prob_delta(
    res,
    df,
    formula,
    pair_alt,
    pair_ref,
    averaging="mouse",
    mouse_col="mouse",
    weights=None,
    set_fixed=None,
    n_clusters=None,
    use_t=True,
):
    """
    Risk difference (alt - ref) on prob scale using delta method at the chosen
    covariate setting (or empirical averaging), with small-sample t.
    Returns: dict with diff, se, lo, hi, stat, dof, dist, pval, dz, pair, ref
    """
    import numpy as np
    names, beta, V = _gee_names_beta_V(res)

    p_alt, g_alt = pair_emm_and_grad(
        res, df, formula, pair_alt, pair_ref,
        averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
    )
    p_ref, g_ref = pair_emm_and_grad(
        res, df, formula, pair_ref, pair_ref,
        averaging=averaging, mouse_col=mouse_col, weights=weights, set_fixed=set_fixed
    )

    diff = p_alt - p_ref
    g = g_alt - g_ref
    var = float(g @ V.values @ g)
    se  = float(np.sqrt(max(var, 0.0)))

    if n_clusters is None and (mouse_col in df.columns):
        n_clusters = int(df[mouse_col].nunique())

    if use_t and n_clusters and n_clusters > 1:
        dof = n_clusters - 1
        crit = tdist.ppf(0.975, dof)
        pval = 2 * (1 - tdist.cdf(abs(diff / se), dof)) if se > 0 else float("nan")
        dist = "t"
    else:
        dof = None
        crit = norm.ppf(0.975)
        pval = 2 * (1 - norm.cdf(abs(diff / se))) if se > 0 else float("nan")
        dist = "z"

    lo, hi = diff - crit * se, diff + crit * se
    dz = (diff / se) / np.sqrt(n_clusters) if (se > 0 and n_clusters) else float("nan")

    return dict(pair=pair_alt, ref=pair_ref, diff=diff, se=se, lo=lo, hi=hi,
                stat=(diff / se if se > 0 else float("nan")),
                dof=dof, dist=dist, pval=pval, dz=dz)


# --- NEW: many contrasts vs reference (Holm correction) ---
from statsmodels.stats.multitest import multipletests

def contrasts_vs_reference(
    res,
    df,
    formula,
    pair_levels,
    ref_label,
    averaging="mouse",
    mouse_col="mouse",
    weights=None,
    set_fixed=None,
    n_clusters=None,
    adjust="holm",
):
    """
    Evaluate selected contrasts against a reference on the prob scale.
    Returns a DataFrame with: pair, ref, diff, lo, hi, se, stat, dof, dist, pval, pval_adj, reject, adjust_method, dz
    """
    if n_clusters is None and (mouse_col in df.columns):
        n_clusters = int(df[mouse_col].nunique())

    alts = [lvl for lvl in pair_levels if lvl != ref_label]
    rows = [
        contrast_prob_delta(
            res, df, formula, pair_alt=alt, pair_ref=ref_label,
            averaging=averaging, mouse_col=mouse_col,
            weights=weights, set_fixed=set_fixed, n_clusters=n_clusters
        )
        for alt in alts
    ]
    out = pd.DataFrame(rows, columns=["pair","ref","diff","lo","hi","se","stat","dof","dist","pval","dz"])
    if not out.empty:
        rej, p_adj, _, _ = multipletests(out["pval"].values, method=adjust)
        out["pval_adj"] = p_adj
        out["reject"] = rej
        out["adjust_method"] = adjust
    return out


# --- NEW: friendly alias for EMMs per pair (prob + CI); this just forwards to your EMM table ---
def pair_estimands(
    res, df, formula, pair_levels, ref_label,
    averaging="mouse", mouse_col="mouse", weights=None,
    use_t=True, dof=None, adjust="holm", set_fixed=None
):
    return emms_by_pair(
        res, df, formula, pair_levels, ref_label,
        averaging=averaging, mouse_col=mouse_col, weights=weights,
        p0=0.5, use_t=use_t, dof=dof, adjust=adjust, set_fixed=set_fixed
    )


# --- UPDATED convenience: build EMM table, then delta-method contrasts (keeps same name) ---
def build_emms_and_contrasts(
    res, df, formula, pair_levels, ref_label,
    averaging="mouse", mouse_col="mouse",
    weights=None, adjust="holm", set_fixed=None
):
    emms = pair_estimands(
        res, df, formula, pair_levels, ref_label,
        averaging=averaging, mouse_col=mouse_col, weights=weights,
        adjust=adjust, set_fixed=set_fixed
    )
    n_clusters = df[mouse_col].nunique() if mouse_col in df.columns else None
    contr = contrasts_vs_reference(
        res, df, formula, pair_levels, ref_label,
        averaging=averaging, mouse_col=mouse_col, weights=weights,
        set_fixed=set_fixed, n_clusters=n_clusters, adjust=adjust
    )
    return emms, contr


# --- BACKWARD-COMPAT alias: the older name now points to the delta-method omnibus ---
wald_omnibus_pair_prob = wald_omnibus_pair_delta

# --- Cluster influence: leave-one-mouse-out diagnostics for a single contrast ---
import numpy as np, pandas as pd
from scipy.stats import t as tdist

def loso_contrast_table(
    fit_callable,
    df,
    formula,
    pair_alt,
    pair_ref,
    averaging="mouse",
    mouse_col="mouse",
    set_fixed=None,
    weights=None,
):
    """
    fit_callable: function(df_subset) -> fitted GEE results (same formula/design)
                  e.g., lambda d: fit_gees(d, cov_struct=sm.cov_struct.Independence())
    Returns: (table, summary)
      table: per-mouse LOSO effects with flags
      summary: dict with full fit, jackknife SE/CI, and thresholds
    """
    # full fit
    _, res_full = fit_callable(df)
    base = contrast_prob_delta(
        res_full, df, formula, pair_alt, pair_ref,
        averaging=averaging, mouse_col=mouse_col,
        set_fixed=set_fixed, weights=weights
    )
    m = int(df[mouse_col].nunique())
    mice = list(df[mouse_col].unique())

    rows = []
    for mouse in mice:
        d_sub = df.loc[df[mouse_col] != mouse].copy()
        if d_sub.empty:
            continue
        _, res_i = fit_callable(d_sub)
        r = contrast_prob_delta(
            res_i, d_sub, formula, pair_alt, pair_ref,
            averaging=averaging, mouse_col=mouse_col,
            set_fixed=set_fixed, weights=weights
        )
        change = r["diff"] - base["diff"]
        dfbeta = (base["diff"] - r["diff"]) / base["se"] if base["se"] > 0 else np.nan
        rows.append(dict(
            mouse=str(mouse),
            diff_full=base["diff"], se_full=base["se"], dof_full=base["dof"],
            diff_loso=r["diff"], se_loso=r["se"], dof_loso=r["dof"],
            change=change, abs_change=abs(change),
            dfbeta=dfbeta, pval_loso=r["pval"]
        ))

    tab = pd.DataFrame(rows).sort_values("abs_change", ascending=False).reset_index(drop=True)

    # Robust outlier flags on change (MAD z) and DFBETA threshold
    if not tab.empty:
        med = np.median(tab["change"])
        mad = 1.4826 * np.median(np.abs(tab["change"] - med))
        tab["z_mad"] = (tab["change"] - med) / mad if mad > 0 else np.nan
        tab["flag_mad"] = tab["z_mad"].abs() > 3.0
        thr = 2.0 / np.sqrt(max(m, 1))
        tab["flag_dfbeta"] = tab["dfbeta"].abs() > thr
    else:
        thr = np.nan

    # Jackknife SE/CI for the contrast (delete-1 clusters)
    diffs_loso = tab["diff_loso"].to_numpy()
    theta_dot = np.nanmean(diffs_loso) if diffs_loso.size else np.nan
    se_jack = np.sqrt(((m - 1) / m) * np.nansum((diffs_loso - theta_dot) ** 2)) if m > 1 else np.nan
    dof = m - 1 if m > 1 else np.nan
    crit = tdist.ppf(0.975, dof) if (isinstance(dof, int) and dof > 0) else np.nan
    ci_jack = (base["diff"] - crit * se_jack, base["diff"] + crit * se_jack) if np.isfinite(crit) else (np.nan, np.nan)

    summary = dict(
        diff_full=base["diff"], se_full=base["se"], dof_full=base["dof"], pval_full=base["pval"],
        m_clusters=m, dfbeta_threshold=thr, se_jack=se_jack, dof_jack=dof, ci95_jack=ci_jack
    )
    return tab, summary


# --- Optional: per-mouse raw (unmodeled) contrast to eyeball outliers quickly ---
def per_mouse_raw_contrast(
    df, pair_alt, pair_ref, resp_col="y_const", pair_col="pair", mouse_col="mouse"
):
    """
    Returns per-mouse proportions and the raw difference: p_alt - p_ref,
    with simple Wald SE for quick screening (model-free).
    """
    out = []
    for mouse, d in df.groupby(mouse_col):
        a = d.loc[d[pair_col] == pair_alt, resp_col]
        r = d.loc[d[pair_col] == pair_ref, resp_col]
        n_a, n_r = int(a.notna().sum()), int(r.notna().sum())
        if n_a == 0 or n_r == 0:
            continue
        p_a, p_r = float(a.mean()), float(r.mean())
        diff = p_a - p_r
        se = np.sqrt((p_a * (1 - p_a) / max(n_a, 1)) + (p_r * (1 - p_r) / max(n_r, 1)))
        out.append(dict(mouse=str(mouse), n_alt=n_a, n_ref=n_r, p_alt=p_a, p_ref=p_r, diff=diff, se=se))
    tbl = pd.DataFrame(out).sort_values("diff").reset_index(drop=True)
    if not tbl.empty:
        med = tbl["diff"].median()
        mad = 1.4826 * np.median(np.abs(tbl["diff"] - med))
        tbl["z_mad"] = (tbl["diff"] - med) / mad if mad > 0 else np.nan
        tbl["flag_mad"] = tbl["z_mad"].abs() > 3.0
    return tbl


def emm_prob_observed(res, df_sub, formula, pair_label, mouse_col="mouse", set_fixed=None):
    """
    Compute mouse-balanced EMM (probability scale) for a given pair,
    keeping observed covariates for each row/mouse.

    res        : fitted statsmodels GEE result
    df_sub     : subset dataframe (only mice to include)
    formula    : Patsy formula string used at fit
    pair_label : which pair to predict
    mouse_col  : column with mouse IDs
    set_fixed  : optional dict of covariates to fix to a value (overrides df_sub)
    """
    d = df_sub.copy()

    # make sure 'pair' is categorical with full category set
    if not pd.api.types.is_categorical_dtype(d["pair"]):
        d["pair"] = pd.Categorical(d["pair"])
    if pair_label not in d["pair"].cat.categories:
        d["pair"] = d["pair"].cat.add_categories([pair_label])

    # override covariates if requested
    if set_fixed:
        for k, v in set_fixed.items():
            d[k] = v

    # set target pair, keep other covariates as observed
    d.loc[:, "pair"] = pair_label

    # build design aligned to fitted model
    names = list(res.model.exog_names)
    X = pt.dmatrix(formula, d, return_type="dataframe", NA_action="raise") \
          .reindex(columns=names, fill_value=0.0)
    eta = X.values @ res.params.to_numpy()
    p   = 1.0 / (1.0 + np.exp(-eta))

    # mouse-balanced mean probability
    per_mouse = (pd.DataFrame({"m": d[mouse_col].to_numpy(), "p": p})
                   .groupby("m")["p"].mean())
    return float(per_mouse.mean()), per_mouse.sort_index()
#%%


import numpy as np, pandas as pd, patsy as pt

def _mice_for_contrast(df, pair_a, pair_b, mouse_col="mouse"):
    A = set(df.loc[df["pair"]==pair_a, mouse_col].unique())
    B = set(df.loc[df["pair"]==pair_b, mouse_col].unique())
    return sorted(A & B)

def _emm_fixed_obs_and_grad(res, d, formula, pair_label, set_fixed, mouse_col="mouse"):
    # keep observed rows, but overwrite specified covariates and pair
    d = d.copy()
    if not pd.api.types.is_categorical_dtype(d["pair"]):
        d["pair"] = pd.Categorical(d["pair"])
    if pair_label not in d["pair"].cat.categories:
        d["pair"] = d["pair"].cat.add_categories([pair_label])
    d.loc[:, "pair"] = pair_label
    if set_fixed:
        for k,v in set_fixed.items():
            d[k] = v

    names = list(res.model.exog_names)
    X = pt.dmatrix(formula, d, return_type="dataframe", NA_action="raise") \
           .reindex(columns=names, fill_value=0.0).values
    beta = res.params.to_numpy()
    eta  = X @ beta
    p    = 1.0/(1.0+np.exp(-eta))

    # mouse-balanced mean + matching gradient
    mice, inv = np.unique(d[mouse_col].to_numpy(), return_inverse=True)
    n_m = len(mice)
    p_m = np.zeros(n_m)
    G_m = np.zeros((n_m, X.shape[1]))
    for m_idx in range(n_m):
        idx = (inv == m_idx)
        pm  = p[idx]
        Xm  = X[idx,:]
        p_m[m_idx]   = pm.mean()
        G_m[m_idx,:] = ((pm*(1-pm))[:,None] * Xm).mean(axis=0)

    p_bar = float(p_m.mean())
    grad  = G_m.mean(axis=0)
    return p_bar, grad

def contrast_same_mice_refit_fixed(fit_callable, df, formula, pair_alt, pair_ref,
                                   set_fixed=None, mouse_col="mouse"):
    # 1) same-mice subset for this contrast
    keep = _mice_for_contrast(df, pair_alt, pair_ref, mouse_col)
    if not keep:
        raise ValueError("No mice have both pairs for this contrast.")
    dsub = df[df[mouse_col].isin(keep)].copy()

    # 2) REFIT on that subset (captures β-sensitivity without m13, etc.)
    _,res = fit_callable(dsub)

    # 3) EMMs at fixed covariates
    p_alt, g_alt = _emm_fixed_obs_and_grad(res, dsub, formula, pair_alt, set_fixed, mouse_col)
    p_ref, g_ref = _emm_fixed_obs_and_grad(res, dsub, formula, pair_ref, set_fixed, mouse_col)


    # Delta-method on prob. scale using NumPy
    g = (g_alt - g_ref).reshape(-1)                 # shape (p,)
    V = res.cov_params()
    V = V.to_numpy() if hasattr(V, "to_numpy") else np.asarray(V)
    if V.shape[0] != g.shape[0]:
        raise ValueError(f"grad len {g.shape[0]} != cov shape {V.shape}")
    var = float(g @ V @ g)     
    se  = np.sqrt(max(var, 0.0))
    diff = p_alt - p_ref
    return {
        "kept_mice": keep,
        "n_mice": len(keep),
        "diff": diff,
        "se": se,
        "ci95": (diff - 1.96*se, diff + 1.96*se),
        "res": res,
    }
from math import isfinite
import numpy as np
from scipy.stats import norm, t as student_t

def add_significance_to_contrast(result_dict, alpha=0.05, small_sample=True):
    """
    result_dict must contain: {'diff': float, 'se': float, 'kept_mice': [...]}.
    Returns the same dict with Wald stats, p-values, and CIs added.
    """
    diff = float(result_dict["diff"])
    se   = float(result_dict["se"])
    m    = int(len(result_dict["kept_mice"]))
    if se <= 0 or not isfinite(se):
        # degenerate case
        result_dict.update({
            "stat": np.inf if diff != 0 else 0.0,
            "df": m - 1,
            "p_norm": 0.0 if diff != 0 else 1.0,
            "p_t": 0.0 if (diff != 0 and m > 1) else 1.0,
            "ci95_norm": (diff, diff),
            "ci95_t": (diff, diff),
        })
        return result_dict

    stat = diff / se
    df   = max(m - 1, 1)  # conservative small-sample df

    # Two-sided p-values
    p_norm = 2 * norm.sf(abs(stat))
    p_t    = 2 * student_t.sf(abs(stat), df=df)

    # 95% CIs
    zcrit  = norm.ppf(0.975)
    tcrit  = student_t.ppf(0.975, df=df)
    ci95_norm = (diff - zcrit * se, diff + zcrit * se)
    ci95_t    = (diff - tcrit * se, diff + tcrit * se)

    result_dict.update({
        "stat": stat,
        "df": df,
        "p_norm": p_norm,
        "p_t": p_t if small_sample else p_norm,   # pick which you’ll report
        "ci95_norm": ci95_norm,
        "ci95_t": ci95_t,
    })
    return result_dict


def _signed_log(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))

def _winsor_per_mouse(df, col, mouse_col="mouse", lo=0.5, hi=99.5):
    qlo, qhi = lo/100.0, hi/100.0

    def _clip(s: pd.Series) -> pd.Series:
        v = s.dropna()
        if v.empty:                
            return s               
        lo_v, hi_v = v.quantile([qlo, qhi])
        return s.clip(lower=lo_v, upper=hi_v)
    return df.groupby(mouse_col)[col].transform(_clip)