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
from statsmodels.genmod.cov_struct import Exchangeable, Independence
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

SESSIONS, cll = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", suff= "_cll")
SESSIONS, lcc = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", idx = 1, suff= "_lcc")
#%%
all_mice = lcc#pd.concat([cll,lcc])
#%% df building helpers


pairs = list(zip(SESSIONS, SESSIONS[1:]))




#%%
def classify_transitions(
    df, id_pairs, method='raw', threshold=0.2, sd_threshold=1,
    session_ids=None, canonical=False
):
    """
    method: 'raw' | 'rz' | 'depthdc' | 'bgratio'
    threshold: multiplicative window for raw/depthdc/bgratio, e.g. 0.2 => ±20%
    sd_threshold: additive window for 'rz' (in SD units)
    include_bgr: if True, label 'bgr' when bgr_rule is met
    bgr_rule: 'any' (either session bg) or 'both'
    """
    res = df.copy()


    def colname(sid):
        base = f'int_optimized_{sid}'
        return {
            'raw': base,
            'robust': f'{base}_rstd',
        }[method]

    for id1, id2 in id_pairs:
        c1, c2 = colname(id1), colname(id2)

        if c1 not in res.columns or c2 not in res.columns:
            continue
        # coerce numeric
        res[[c1, c2]] = res[[c1, c2]].apply(pd.to_numeric, errors='coerce')
        outcol = f'{id1}_to_{id2}_{method}'
        if(canonical):
            outcol = utils.canonical_pair(id1, id2)
        res[outcol] = 'stable'
        #TODO może dodać bgr
        if method == 'robust':
            diff = res[c2] - res[c1]
            res.loc[diff >  sd_threshold, outcol] = 'up'
            res.loc[diff < -sd_threshold, outcol] = 'down'
        else:
            lower = res[c1] * (1 - threshold)
            upper = res[c1] * (1 + threshold)
            res.loc[res[c2] > upper, outcol] = 'up'
            res.loc[res[c2] < lower, outcol] = 'down'

        
    return res
#%%
#classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = 0.75)
classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = 0.75, method = 'robust', canonical = True)

#%%
classified_df.columns
    
#%%

import patsy as pt
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def build_long(df, 
               id_pairs, 
               depth_col=ICY_COLNAMES['zcol'], 
               group_cols=('mouse','cell_id'), 
               method = "robust", canonical=False):
    recs = []
    
    suff = "_rstd"
    if method == 'raw':
        suff = ""
    
    for (id1, id2) in id_pairs:
        col = f'{id1}_to_{id2}_{method}'
        if canonical:
            col = utils.canonical_pair(id1, id2)
        tmp = df[[*group_cols, depth_col, col,
                  f'int_optimized_{id1}{suff}',
                  f"background_{id1}_rstd",
                  f"background_{id2}_rstd",
                  f"is_dim_by_bg_{id1}",
                  f"is_dim_by_bg_{id2}",
                  'detected_in_sessions',
                  'n_sessions']].copy()
        
        tmp['pair'] = f'{id1}_to_{id2}'
        tmp["bg_mean"]   = tmp[f"background_{id2}_rstd"]
        #tmp["bg_mean"] = (tmp["bg_mean"] - tmp["bg_mean"].mean())/tmp["bg_mean"].std(ddof=0)
        tmp["bg_mean"] = tmp["bg_mean"].fillna(0)

        tmp.rename(columns={f'int_optimized_{id1}{suff}':'baseline_int', depth_col:'z'}, inplace=True)
        tmp['y_up']     = (tmp[col] == 'up').astype(int)
        tmp['y_down']   = (tmp[col] == 'down').astype(int)
        tmp['is_bgr']   = tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) and utils.in_s(x, id2)))
        tmp['is_bgr'] = (~(tmp['n_sessions'] ==3))
        #tmp['is_bgr'] = False
        
        recs.append(tmp)
    long = pd.concat(recs, ignore_index=True)

    # Switch between all and active in both sessions only - maybe one session too?
    long_nobgr = long[long['is_bgr'] == 0].copy()
    #long_nobgr = long.copy()

    # Construct cluster id robustly
    long_nobgr['cluster'] = long_nobgr[group_cols[0]].astype(str)

    return long_nobgr, long

def cluster_weights(groups):
    sizes = pd.Series(groups).groupby(groups).size()
    w = sizes.mean() / sizes  
    labels = np.unique(np.asarray(groups))
    return w.reindex(labels).to_numpy(), labels

def fit_gees(long_nobgr, add_pair_effect=True, method='robust', cov_struct=sm.cov_struct.Independence(), cov_type="robust", 
             weighted=False, ref="ctx1_to_ctx2", canonical=False):
    # Design matrix (tune as you like)
    # Include baseline_int (start-session intensity), depth (and optional quadratic), and pair fixed effects.
    need = ["baseline_int","bg_mean","z","pair"]
    long_nobgr = long_nobgr.dropna(subset=need).copy()

    if canonical:
        ref = utils.canonical_from_pair_label(ref)
    formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="{ref}"))'

    X = pt.dmatrix(formula, long_nobgr, return_type='dataframe')
    

        
    groups = long_nobgr['mouse'].to_numpy()
    assert groups.shape[0] == len(long_nobgr)

    # --- per-row weights constant within mouse: w_i ∝ 1 / n_mouse ---
    n_per_mouse = long_nobgr.groupby('mouse').size()
    w_per_mouse = (n_per_mouse.mean() / n_per_mouse).to_dict()
    if weighted:
        w = long_nobgr['mouse'].map(w_per_mouse).to_numpy()
        cov_type="robust"
    else:
        w=None
        cov_type="bias_reduced"

    cov_up  = type(cov_struct)()  
    cov_down= type(cov_struct)()

    # Up vs not-up
    gee_up = GEE(endog=long_nobgr['y_up'], exog=X, groups=groups,
                 cov_struct=cov_up, family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1.0)

    # Down vs not-down
    gee_down = GEE(endog=long_nobgr['y_down'], exog=X, groups=groups,
                   cov_struct=cov_down, family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1.0)
    
    qic_up = gee_up.qic(scale=1.0)
    qic_down = gee_down.qic(scale=1.0)
    print("QIC up: ",qic_up,"QIC down: ",qic_down)
    #print(gee_up.summary())
    #print(gee_down.summary())
    return gee_up, gee_down, w




#%%
#TODO  marked to reuse for inspector problems

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(classified_df[["detected_in_sessions"]+[f"is_dim_by_bg_{sid}" for sid in SESSIONS]].head(1))
for sid in SESSIONS:    
    print(classified_df[f"is_dim_by_bg_{sid}"].mean())



#%% sweeper definition
import patsy as pt
from scipy.stats import norm
from scipy.special import expit as logistic

# --- small utils --------------------------------------------------------------


def _gee_names_beta_V(res):
    names = list(res.model.exog_names)
    beta  = pd.Series(res.params, index=names)
    V     = pd.DataFrame(res.cov_params(), index=names, columns=names)
    return names, beta, V

def _x_for_pair(names, pair_label):
    x = np.zeros(len(names))
    x[names.index("Intercept")] = 1.0
    if pair_label is not None:
        for j, nm in enumerate(names):
            if nm.endswith(f"[T.{pair_label}]"):
                x[j] = 1.0
    return x

def _wald_omnibus_pair(res, prefix='C(pair', n_clusters=None):
    names = res.model.exog_names
    idx   = [i for i,n in enumerate(names) if n.startswith(prefix)]
    if not idx:
        return np.nan
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0
    df1 = len(idx)
    w = res.wald_test(L, scalar=True)
    print("degrees of freedom sanity check ", df1)
    if res.cov_type == "bias_reduced":
        return float(w.pvalue), float(np.asarray(w.statistic))
    else:
        if n_clusters is None:
            raise ValueError("n_clusters required for F correction")
        df2 = max(int(n_clusters) - 1, 1)
        chi2 = float(np.asarray(w.statistic))
        Fstat = chi2 / df1
        p = 1.0 - fdist.cdf(Fstat, df1, df2)
        return float(p), float(Fstat)

def _pair_prob_ci(names, beta, V, pair_label):
    x = _x_for_pair(names, pair_label)
    eta = float(x @ beta)
    var = float(x @ V @ x)
    se  = np.sqrt(max(var, 0.0))
    p   = logistic(eta)
    lo, hi = logistic(eta - 1.96*se), logistic(eta + 1.96*se)
    return dict(eta=eta, se=se, prob=p, lo=lo, hi=hi)

def _contrast_vs_ref_p(names, beta, V, pair_label, ref_label):
    xa = _x_for_pair(names, pair_label)
    xb = _x_for_pair(names, ref_label)
    c  = xa - xb
    est = float(c @ beta)
    var = float(c @ V @ c)
    se  = np.sqrt(max(var, 0.0))
    z   = est / se if se > 0 else np.nan
    p   = 2*(1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return est, se, z, p

def pair_prob_ci_avgpred(res, df, pair_label,ref_label, alpha=0.05, weights=None):
    p_bar, grad = pair_emm_and_grad(res, df, pair_label,ref_label, weights=weights)
    V = np.asarray(res.cov_params(), dtype=float)
    var = float(grad @ V @ grad)
    se = var**0.5

    z = norm.ppf(1 - alpha/2)
    lo, hi = p_bar - z*se, p_bar + z*se
    return {"prob": p_bar, "lo": max(0.0, lo), "hi": min(1.0, hi), "se_prob": se}

def _pair_avgpred_and_grad(res, df, pair_label, ref_label):
    d = df.copy()
    ALL_LEVELS = df["pair"].unique().tolist()


    d["pair"] = pd.Categorical([pair_label]*len(d), categories=ALL_LEVELS)
    formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference={ref_label}))'
    X = pt.dmatrix(formula, d, return_type='dataframe')
    eta = X @ res.params.values
    p = 1.0 / (1.0 + np.exp(-eta))
    p_bar = float(p.mean())
    p_arr = np.asarray(p)
    X_arr = np.asarray(getattr(X, "values", X))
    grad = (p_arr * (1 - p_arr))[:, None] * X_arr
    grad = grad.mean(axis=0)
    return p_bar, grad

def pair_emm_and_grad(res, df, pair_label, ref_label, mouse_col="mouse", weights=None, estimand="cell"):
    """
    EMM on the probability scale for a given pair level, with a delta-method gradient
    that matches the averaging scheme.

    If row_weights is None, we compute a mouse-balanced EMM:
      - predict for each row with pair fixed to `pair_label`
      - average within mouse (simple mean of that mouse's rows)
      - average equally across mice
    If row_weights is provided (1-D array aligned to df), we do a weighted average
    using those same weights (normalized to sum to 1).
    """
    d = df.copy()
    all_levels = pd.Categorical(df["pair"]).categories
    d["pair"] = pd.Categorical([pair_label]*len(d), categories=all_levels)


    formula = f"1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference='{ref_label}'))"
    X = pt.dmatrix(formula, d, return_type='dataframe')

    eta = X @ res.params.values
    p   = 1.0 / (1.0 + np.exp(-eta))
    p = np.asarray(p, dtype=float).reshape(-1)
    print(weights)
    X_arr = np.asarray(getattr(X, "values", X))
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        w = w / w.sum()
        p_bar = float((w * p).sum())
        # gradient: d mean / d beta = sum_i w_i * p_i*(1-p_i) * x_i
        X_arr = np.asarray(getattr(X, "values", X))
        grad  = (w[:,None] * (p*(1-p))[:,None] * X_arr).sum(axis=0)
        return p_bar, grad



    if estimand == "cell":
        # simple row-average (cell-level estimand)
        p_bar = float(p.mean())
        grad  = (((p*(1-p))[:, None]) * X_arr).mean(axis=0)
        return p_bar, grad

    if estimand == "mouse":
        # mouse-balanced: within-mouse mean, then mean across mice
        p_bar = float(d.assign(p=p).groupby(mouse_col)["p"].mean().mean())
        mats = []
        for _, sub in d.groupby(mouse_col, sort=False):   # <-- 3)
            idx = sub.index.to_numpy()
            gm  = ((p[idx]*(1-p[idx]))[:, None] * X_arr[idx, :]).mean(axis=0)
            mats.append(gm)
        grad = np.vstack(mats).mean(axis=0)
        return p_bar, grad


def _LL_vs_avg_stats(res, df, V, ref_label, LC_label, CC_label, alpha=0.05, df_t=None, weights = None):
    # LC and CC: marginal means + gradients
    # p_LC, g_LC = _pair_avgpred_and_grad(res, df, LC_label)
    # p_CC, g_CC = _pair_avgpred_and_grad(res, df, CC_label)

    p_LC, g_LC = pair_emm_and_grad(res, df, LC_label,ref_label, weights = weights)
    p_CC, g_CC = pair_emm_and_grad(res, df, CC_label,ref_label, weights = weights)

    # Average of LC and CC
    p_avg  = 0.5 * (p_LC + p_CC)
    g_avg  = 0.5 * (g_LC + g_CC)
    var_avg = float(g_avg @ V @ g_avg)
    se_avg  = np.sqrt(max(var_avg, 0.0))

    
    p_REF, g_REF = pair_emm_and_grad(res, df, ref_label,ref_label, weights = weights)

    # Contrast (REF − AVG) on probability scale
    delta   = p_REF - p_avg
    g_delta = g_REF - g_avg
    var_delta = float(g_delta @ V @ g_delta)
    se_delta  = np.sqrt(max(var_delta, 0.0))
    z_delta   = delta / se_delta if se_delta > 0 else np.nan

    # p-value: t-correct if needed
    if df_t is None:
        p_val = 2 * norm.sf(abs(z_delta))
    else:
        p_val = 2 * stats.t.sf(abs(z_delta), df_t)

    z = norm.ppf(1 - alpha/2)
    avg_lo, avg_hi = p_avg - z*se_avg, p_avg + z*se_avg

    return {
        "p_LL_vs_avg": p_val,
        "p_LL_vs_avg_Holm": p_val,     # single contrast → Holm = p
        "avg_coeff": p_avg,
        "avg_CI_low": max(0.0, avg_lo),
        "avg_CI_hi":  min(1.0, avg_hi),
        "avg_SE_prob": se_avg,
        "delta_p": delta,
        # (optionally return REF too if you want)
    }

# --- main collector -----------------------------------------------------------


def sweep_gee_to_csv(
    thresholds,
    classify_fn,
    build_long_fn,
    fit_fn,                 # function(long_df, ...) -> (gee_up, gee_down, weights)
    pairs,                  # list of (before, after[, label])
    sessions,               # SESSIONS
    cells_label="no_bgr",
    ref_label="landmark1_to_landmark2",
    weighted=False,
    canonical=False
):
    """
    Writes a wide CSV with one row per (threshold, direction).
    - Auto-discovers available transition labels from the data (canonical or original).
    - No hardcoded pair names; columns are created per discovered label.
    - Reference level picked automatically if not present in the data.
    - Works unchanged with your fit_fn that uses C(pair, ...).

    The following per-label fields are produced:
        <slug>_p_prob, <slug>_Holm_prob, <slug>_p_logit, <slug>_Holm_logit,
        <slug>_coeff, <slug>_CI_low, <slug>_CI_hi, <slug>_SE_prob
    where slug = "pair_<label>", e.g. pair_landmark_to_ctx.
    """


    rows = []

    # Build the *string* labels from the 'pairs' argument

    if canonical:
        ref_label = utils.canonical_from_pair_label(ref_label)
    #orig_labels = [ _label_from_tuple(t) for t in pairs ]
    #cand_labels = [ utils.canonical_from_pair_label(l) for l in orig_labels ] if canonical else orig_labels

    # Choose ref sensibly (can be overridden by ref_label arg if present in labels)
    ref_fallback = ref_label#_pick_ref_label(cand_labels, canonical, ref_label)

    for t_val in thresholds:
        # 1) classify & build long
        classified = classify_fn(all_mice, pairs, session_ids=sessions, sd_threshold=t_val, method='robust')
        long_df, _ = build_long_fn(classified, pairs, group_cols=['mouse'], method='robust')

        # 2) optionally canonicalize 'pair' *before* fitting (so fit_fn uses C(pair, ...))
        long_df = utils.ensure_pair_canonical(long_df, canonical=canonical)

        # discover which labels actually appear in this data cut
        data_labels = sorted(str(x) for x in pd.Series(long_df.get("pair", [])).dropna().unique())
        if not data_labels:
            continue

        ref_use = ref_label#_pick_ref_label(data_labels, canonical, ref_fallback)
        assert ref_use in data_labels, f"reference level '{ref_use}' not present in data labels {data_labels}"



        need_cols = ["y_up","y_down","baseline_int","bg_mean","z","pair","mouse"]

        formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="{ref_label}"))'





        # 3) fit
        gee_up_ex,  gee_down_ex,  weights = fit_fn(
             long_df, cov_type="robust", cov_struct=Exchangeable(), weighted=weighted, canonical=canonical
         )
        gee_up_ind, gee_down_ind, weights = fit_fn(
             long_df, cov_type="robust", cov_struct=Independence(),  weighted=weighted, canonical=canonical
         )

        
        ref_use = ref_label

        
        try:
             rho_hat   = float(getattr(gee_up_ex.cov_struct,  "dep_params", np.nan))
             rho_hat_d = float(getattr(gee_down_ex.cov_struct,"dep_params", np.nan))
        except Exception:
             rho_hat = rho_hat_d = np.nan
             
        
         # QICs for both structures
        ex_qic_up    = gee_up_ex.qic(scale=1.0)
        ex_qic_down  = gee_down_ex.qic(scale=1.0)
        ind_qic_up   = gee_up_ind.qic(scale=1.0)
        ind_qic_down = gee_down_ind.qic(scale=1.0)

        # 4) iterate for UP/DOWN models
        for outcome_label, res in (("UP", gee_up_ind), ("DOWN", gee_down_ind)):
            N = int(getattr(res.model.endog, "shape", [len(long_df)])[0])
            n_clusters = long_df["mouse"].nunique()
            df_t = n_clusters - 1 if (res.cov_type != "bias_reduced") else None
            p_src = "t_corrected" if df_t is not None else "bias_reduced"

            sizes = long_df.groupby('mouse').size()
            cluster_mean = float(sizes.mean()); cluster_min = int(sizes.min()); cluster_max = int(sizes.max())

            group_p, group_chi = _wald_omnibus_pair(res, n_clusters=n_clusters)
            names, beta, V = _gee_names_beta_V(res)

            # choose which labels to report: all except the reference
            comp_labels = [lab for lab in data_labels if lab != ref_use]

            # accumulators
            per = {}
            pvals_prob, pvals_logit = [], []
            tmp = {}

            # compute ref (prob-scale EMM) once

            ref_emm = pair_prob_ci_avgpred(res, long_df, ref_use, ref_use, weights=weights)
            se_prob_ref = ref_emm["se_prob"]
            
            for lab in comp_labels:
                key = utils.slug_for_cols(lab)

                # LOGIT-scale contrast vs ref
                est, se, z, p_logit = _contrast_vs_ref_p(names, beta, V, lab, ref_use)
                if df_t is not None and np.isfinite(z):
                    p_logit = 2 * stats.t.sf(abs(z), df_t)

                # PROB-scale Δp via delta method (avg-of-preds gradient)
                p_pair, g_pair = pair_emm_and_grad(res, long_df, lab,ref_use, weights=weights)
                p_ref,  g_ref  = pair_emm_and_grad(res, long_df, ref_use,ref_use, weights=weights)
                Vnp = np.asarray(res.cov_params(), dtype=float)
                delta   = p_pair - p_ref
                g_delta = g_pair - g_ref
                se_delta = float((g_delta @ Vnp @ g_delta) ** 0.5)
                if se_delta <= 1e-12:
                    p_prob = 1.0   
                else:                                
                    if df_t is not None and se_delta > 0:
                        tstat = delta / se_delta
                        p_prob = 2 * stats.t.sf(abs(tstat), df_t)
                    else:
                        zstat = delta / se_delta if se_delta > 0 else np.nan
                        p_prob = 2 * norm.sf(abs(zstat)) if np.isfinite(zstat) else np.nan

                # Probabilities & CI shown (avg-of-predictions)
                pr = pair_prob_ci_avgpred(res, long_df, lab,ref_use, weights=weights)

                tmp[lab] = dict(
                    p_prob=p_prob, p_logit=p_logit,
                    est_logit=est, se_logit=se, z_logit=z,
                    prob=pr["prob"], lo=pr["lo"], hi=pr["hi"], se_prob=pr["se_prob"],
                    key=key
                )
                pvals_prob.append(p_prob); pvals_logit.append(p_logit)
                print(f"[{outcome_label}] {lab} vs {ref_use} → types:",
                      type(p_prob).__name__, type(p_logit).__name__)

            # Holm corrections (over all pair-vs-ref tests)
            from statsmodels.stats.multitest import multipletests
            _, p_holm_prob,  _, _ = multipletests([p for p in pvals_prob if pd.notnull(p)],  method="holm") if any(pd.notnull(p) for p in pvals_prob) else (None, [], None, None)
            _, p_holm_logit, _, _ = multipletests([p for p in pvals_logit if pd.notnull(p)], method="holm") if any(pd.notnull(p) for p in pvals_logit) else (None, [], None, None)

            # Map Holm back to labels in the same order
            idx_prob = 0; idx_logit = 0
            for lab in comp_labels:
                key = tmp[lab]["key"]
                # raw p
                per[f"{key}_p_prob"]  = tmp[lab]["p_prob"]
                per[f"{key}_p_logit"] = tmp[lab]["p_logit"]
                # Holm (if available)
                if np.isfinite(tmp[lab]["p_prob"]) and idx_prob < len(p_holm_prob):
                    per[f"{key}_Holm_prob"]  = float(p_holm_prob[idx_prob]);  idx_prob  += 1
                else:
                    per[f"{key}_Holm_prob"]  = np.nan
                if np.isfinite(tmp[lab]["p_logit"]) and idx_logit < len(p_holm_logit):
                    per[f"{key}_Holm_logit"] = float(p_holm_logit[idx_logit]); idx_logit += 1
                else:
                    per[f"{key}_Holm_logit"] = np.nan
                # coefficients you display (probability scale)
                per[f"{key}_coeff"]   = tmp[lab]["prob"]
                per[f"{key}_CI_low"]  = tmp[lab]["lo"]
                per[f"{key}_CI_hi"]   = tmp[lab]["hi"]
                per[f"{key}_SE_prob"] = tmp[lab]["se_prob"]

            # residual diagnostics
            from scipy.stats import skew, kurtosis
            yhat = res.fittedvalues
            var  = res.family.variance(yhat)
            pearson = (res.model.endog - yhat) / np.sqrt(var)
            resid_skew = skew(pearson, bias=False)
            resid_kurt = kurtosis(pearson, bias=False) + 3.0


            if outcome_label == "UP":
                qic_exch = ex_qic_up
                qic_ind  = ind_qic_up
            else:
                qic_exch = ex_qic_down
                qic_ind  = ind_qic_down


            row = {
                "size": N,
                "cluster_mean": cluster_mean,
                "cluster_min": cluster_min,
                "cluster_max": cluster_max,
                "what cells": cells_label,
                "direction": outcome_label,
                "qic": res.qic(scale=1.0),
                "qic_exch": qic_exch,                   
                "qic_ind":  qic_ind,
                "sd cutoff": t_val,
                "group wald p": group_p,
                "group wald chi": group_chi,
                "ref_label": ref_use,
                "ref_coeff": ref_emm["prob"],
                "ref_CI_low": ref_emm["lo"],
                "ref_CI_hi":  ref_emm["hi"],
                "ref_SE_prob": ref_emm["se_prob"],
                "p_source": p_src,
                "rho_hat_up": rho_hat,
                "rho_hat_down": rho_hat_d,
                "ex_qic_up": ex_qic_up,
                "ex_qic_down": ex_qic_down,
                "skew": resid_skew,
                "kurtosis": resid_kurt,
            }
            row.update(per)
            rows.append(row)

    out = pd.DataFrame(rows)

    # Order columns: meta first, then per-label blocks in alphabetical order
    meta_cols = [
        "size","cluster_mean","cluster_min","cluster_max",
        "what cells","direction","qic","qic_exch","sd cutoff","group wald p","group wald chi",
        "ref_label","ref_coeff","ref_CI_low","ref_CI_hi","ref_SE_prob",
        "rho_hat_up","rho_hat_down",#"ex_qic_up","ex_qic_down",
        "skew","kurtosis","p_source"
    ]
    per_cols = sorted([c for c in out.columns if c.startswith("pair_")])
    cols = [c for c in meta_cols if c in out.columns] + per_cols + [c for c in out.columns if c not in meta_cols + per_cols]
    out = out[cols]

    if weighted:
        cells_label = f"{cells_label}_weighted"
    if canonical:
        cells_label = f"{cells_label}_canonical"

    csv_path = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/{cells_label}.csv"
    out.to_csv(csv_path, index=False)
    return gee_up_ind, gee_down_ind




#%%

#sweep_thresholds([0.5, 0.75, 1, 1.25], classify_transitions, build_long, fit_gees, contrast_fn=None)

#%% main sweeper call
filteredby = "lcc_active3"

thresholds = [0.75, 1]
pairs = [['landmark','ctx1'], ['ctx1','ctx2']]#['landmark1','landmark2'], ['ctx','landmark1']]#, ['landmark','ctx1'], ['ctx1','ctx2']]
all_mice = lcc
gee_up_ind, gee_down_ind = sweep_gee_to_csv(
    thresholds,
    classify_transitions,
    build_long,
    fit_gees,                 # function(long_df, cov_type) -> (gee_up, gee_down)
    pairs,                  # list of (before, after, label)
    SESSIONS,               # SESSIONS
    cells_label=filteredby,
    ref_label="ctx_to_ctx",#landmark1_to_landmark2",
    weighted = True,
    canonical = True
)
#%%
gee_down_ind.summary()


#%%
#%%

cutoffs = [0.75,1]
directions = ["DOWN", "UP"]

lcc_pairs = [["landmark", "ctx1"], ["ctx1", "ctx2"]]
cll_pairs = [["ctx", "landmark1"], ["landmark1", "landmark2"]]


filteredby = "lcc_active3"
pairs = lcc_pairs
all_mice = lcc

for cutoff in cutoffs:
    classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = cutoff, method = 'robust')
    long_df, _ = build_long(classified_df, pairs ,group_cols=(['mouse']), method="robust")
    overlay_up = gp.build_per_mouse_overlay(long_df, "mouse", "pair", "y_up",
                                          canonical=True,
                                          pair_order=["landmark_to_landmark",
                                                      "landmark_to_ctx",
                                                      "ctx_to_landmark",
                                                      "ctx_to_ctx"])
    overlay_down = gp.build_per_mouse_overlay(long_df, "mouse", "pair", "y_down",
                                          canonical=True,
                                          pair_order=["landmark_to_landmark",
                                                      "landmark_to_ctx",
                                                      "ctx_to_landmark",
                                                      "ctx_to_ctx"])
    CSV_PATH = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/{filteredby}_weighted_canonical.csv"
    
    gp.fig_pair_effect_at_cutoff(
        csv_path=CSV_PATH,
        direction="UP",
        cutoff=cutoff,
        per_mouse_overlay=overlay_up,
        ref_label="ctx_to_ctx",
        canonical=True,
        label_order=["landmark_to_landmark", "landmark_to_ctx", "ctx_to_landmark", "ctx_to_ctx"],
        pretty_map={
            "landmark_to_landmark":"L→L",
            "landmark_to_ctx":"L→C",
            "ctx_to_landmark":"C→L",
            "ctx_to_ctx":"C→C"
        }
    )
    
    gp.fig_pair_effect_at_cutoff(
        csv_path=CSV_PATH,
        direction="DOWN",
        cutoff=cutoff,
        per_mouse_overlay=overlay_down,
        ref_label="ctx_to_ctx",
        canonical=True,
        label_order=["landmark_to_landmark", "landmark_to_ctx", "ctx_to_landmark", "ctx_to_ctx"],
        pretty_map={
            "landmark_to_landmark":"L→L",
            "landmark_to_ctx":"L→C",
            "ctx_to_landmark":"C→L",
            "ctx_to_ctx":"C→C"
        }
    )



#%%
import matplotlib.pyplot as plt

def plot_gee_sensitivity(df):
    """
    df: DataFrame with columns ['threshold', 'outcome', 'est', 'se']
         outcome in {'UP','DOWN'}
    """
    fig, ax = plt.subplots(figsize=(5,4))
    for outcome, style in [('UP','-'), ('DOWN','--')]:
        sub = df[df['outcome']==outcome]
        ax.errorbar(sub['threshold'], sub['logitΔ'],
                    yerr=1.96*sub['se'], label=outcome,
                    fmt=style+'o', capsize=3)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel("SD threshold for binarization")
    ax.set_ylabel("LL − avg(LC,CC) Δp")
    ax.legend()
    ax.set_title("GEE pair effect sensitivity to cutoff")
    plt.tight_layout()
    return fig
def plot_class_balance(class_counts):
    """
    class_counts: DataFrame with columns ['threshold', 'pair', 'y_mean']
                  y_mean = mean(y) = proportion "up" or "down"
    """
    fig, ax = plt.subplots(figsize=(6,4))
    for pair in class_counts['pair'].unique():
        sub = class_counts[class_counts['pair']==pair]
        ax.plot(sub['threshold'], sub['y_mean'], marker='o', label=pair)
    ax.set_xlabel("SD threshold")
    ax.set_ylabel("Proportion positive")
    ax.set_title("Class balance by threshold and pair")
    ax.legend()
    plt.tight_layout()
    return fig
#%%



#%% summary plots
pair_key_map = {
    "s0_to_landmark1": "s0_l1",
    "landmark1_to_landmark2": "ref",
    "landmark2_to_ctx1": "l2_c1",
    "ctx1_to_ctx2": "c1_c2",
}



for v in ["", "_weighted"]:
    CSV_PATH = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/{filteredby}{v}.csv"
    summary_df = pd.read_csv(CSV_PATH)

    for cutoff in [0.75,1]:    
        classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = cutoff, method = 'robust')
        long_nobgr, _ = build_long(classified_df, pairs ,group_cols=(['mouse']), method="robust")
        overlay_up = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_up", pair_key_map)
        overlay_down = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_down", pair_key_map)
        gp.fig_pair_effect_at_cutoff(CSV_PATH, "UP", cutoff, per_mouse_overlay=overlay_up, xlim=(0.02, 0.2))
        gp.fig_pair_effect_at_cutoff(CSV_PATH, "DOWN", cutoff, per_mouse_overlay=overlay_down, xlim=(0.085, 0.2))
    
   
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)  # sharex; y ranges differ
    gp.fig_fragility_p_vs_cutoff(CSV_PATH, "DOWN", y_max=1, title="DOWN", ax=axs[0])
    gp.fig_fragility_p_vs_cutoff(CSV_PATH, "UP",   y_max=1, title="UP",   ax=axs[1])
    
    fig.suptitle("Wrażliwość modelu na próg odcięcia", fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    
#%%
filteredby="active_in_both"
CSV_PATH = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/{filteredby}{v}.csv"
gp.plot_emm_vs_cutoff(CSV_PATH, "UP", what_cells=f'{filteredby}',
                       ref_vlines=(0.75, 1.0), title=None)
gp.plot_emm_vs_cutoff(CSV_PATH, "DOWN", what_cells=f'{filteredby}',
                       ref_vlines=(0.75, 1.0), title=None)
#%%
gp.plot_p_vs_cutoff_with_counts(
    CSV_PATH,
    direction="DOWN",
    count_cols=("cluster_mean", "cluster_min", "cluster_max")
)


#%%

