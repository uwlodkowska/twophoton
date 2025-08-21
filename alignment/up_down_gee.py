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
    dfs += [df_mouse]
    
#%%
preprocessed_dfs = []
for i, df in enumerate(dfs):
    df = df.copy()
    mouse, region = regions[i]
    df["mouse"] = mouse
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]/df[ICY_COLNAMES['zcol']].max()
    df["cell_id"] = df.index.values
    df = cp.intensity_depth_detrend(df, SESSIONS)
    preprocessed_dfs += [df]
all_mice = pd.concat(preprocessed_dfs)
#%%
all_mice.shape
#%%
all_mice[ICY_COLNAMES['zcol']].max()
#%%
def classify_transitions(
    df, id_pairs, method='raw', threshold=0.2, sd_threshold=1,
    session_ids=None
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
classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = 0.75, method = 'robust')

#%%
plt.scatter(classified_df[ICY_COLNAMES['zcol']],classified_df['int_optimized_landmark2_rstd'], s=3)
plt.show()
#%%
classified_df.columns
#%%
print(classified_df.shape)
a = "landmark1_to_landmark2_raw"
b = "landmark1_to_landmark2_robust"
for tag in["up", "down", "stable"]:
    print(tag, classified_df[classified_df[a] == tag].shape)
    print(tag, classified_df[classified_df[b] == tag].shape)
    
#%%

import patsy as pt
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def build_long(df, 
               id_pairs, 
               depth_col=ICY_COLNAMES['zcol'], 
               group_cols=('mouse','cell_id'), 
               method = "robust"):
    recs = []
    
    suff = "_rstd"
    if method == 'raw':
        suff = ""
    
    for (id1, id2) in id_pairs:
        col = f'{id1}_to_{id2}_{method}'
        tmp = df[[*group_cols, depth_col, col,
                  f'int_optimized_{id1}{suff}',
                  f"background_{id1}_rstd",
                  f"background_{id2}_rstd",
                  f"is_dim_by_bg_{id1}",
                  f"is_dim_by_bg_{id2}",
                  'detected_in_sessions',
                  'n_sessions']].copy()
        
        tmp['pair'] = f'{id1}_{id2}'
        tmp["bg_mean"]   = tmp[f"background_{id2}_rstd"]
        #tmp["bg_mean"] = (tmp["bg_mean"] - tmp["bg_mean"].mean())/tmp["bg_mean"].std(ddof=0)
        tmp["bg_mean"] = tmp["bg_mean"].fillna(0)
        tmp.rename(columns={f'int_optimized_{id1}{suff}':'baseline_int', depth_col:'z'}, inplace=True)
        tmp['y_up']     = (tmp[col] == 'up').astype(int)
        tmp['y_down']   = (tmp[col] == 'down').astype(int)
        tmp['is_bgr']   = df['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) and utils.in_s(x, id2)))
        tmp['is_bgr'] = ~(tmp['n_sessions'] ==4)
        
        recs.append(tmp)
    long = pd.concat(recs, ignore_index=True)

    # Switch between all and active in both sessions only - maybe one session too?
    long_nobgr = long[long['is_bgr'] == 0].copy()
    #long_nobgr = long.copy()

    # Construct cluster id robustly
    long_nobgr['cluster'] = long_nobgr[group_cols[0]].astype(str)
    # (long_nobgr[group_cols[0]].astype(str) + '_' +
    #                          long_nobgr[group_cols[1]].astype(str))
    return long_nobgr, long

def cluster_weights(groups):
    sizes = pd.Series(groups).groupby(groups).size()
    w = sizes.mean() / sizes  
    labels = np.unique(np.asarray(groups))
    return w.reindex(labels).to_numpy(), labels

def fit_gees(long_nobgr, add_pair_effect=True, method='robust', cov_struct=sm.cov_struct.Independence(), cov_type="robust", weighted=False):
    # Design matrix (tune as you like)
    # Include baseline_int (start-session intensity), depth (and optional quadratic), and pair fixed effects.
    formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="landmark1_landmark2"))'
    #formula = f'1  + C(pair, Treatment(reference="landmark1_landmark2"))'
    # if add_pair_effect:
    #     formula += ' + C(pair)'

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
                 cov_struct=cov_up, family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1)

    # Down vs not-down
    gee_down = GEE(endog=long_nobgr['y_down'], exog=X, groups=groups,
                   cov_struct=cov_down, family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1)
    
    qic_up = gee_up.qic(scale=1.0)
    qic_down = gee_down.qic(scale=1.0)
    print("QIC up: ",qic_up,"QIC down: ",qic_down)
    #print(gee_up.summary())
    #print(gee_down.summary())
    return gee_up, gee_down

#%%
long_nobgr, _ = build_long(classified_df, pairs ,group_cols=(['mouse']), method="robust")
#%%
gee_up, gee_down = fit_gees(long_nobgr, cov_struct=sm.cov_struct.Independence(), cov_type="robust")
#%%
gee_up.summary()
#%%


#%%
print(long_nobgr.isna().sum().sort_values(ascending=False).head(20)      )
long_nobgr['baseline_int'].describe()
long_nobgr['bg_mean'].describe()

#%%
g = long_nobgr.groupby('mouse')['bg_mean']
print("within-mouse SD median:", g.transform('std').median())
print("between-mouse SD:", g.mean().std())

# 2) Collinearity with your other covariates
long_nobgr[['bg_mean','baseline_int','z']].corr()

#%%
long_nobgr[['detected_in_sessions', 'pair', 'is_bgr']]

#%%
print(gee_up.summary())
print(gee_down.summary())




#%%
formula = '1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="landmark1_landmark2"))'

#%%
#TODO  marked to reuse for inspector problems

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(classified_df[["detected_in_sessions"]+[f"is_dim_by_bg_{sid}" for sid in SESSIONS]].head(1))
for sid in SESSIONS:    
    print(classified_df[f"is_dim_by_bg_{sid}"].mean())
#%%
plt.hist(classified_df["int_optimized_landmark1"], bins=100)
plt.hist(classified_df["background_landmark1"], alpha = 0.5, bins=100)
plt.axvline(classified_df["background_landmark1"].mean()+6*classified_df["background_landmark1"].std(), color='red')
#%%

def sweep_thresholds(thresholds, classify_fn, build_long_fn, fit_fn, contrast_fn=None):
    rows = []
    for t in thresholds:
        classified = classify_fn(all_mice, pairs, session_ids = SESSIONS, sd_threshold=t, method="robust")
        long_df, _ = build_long_fn(classified, pairs, group_cols=['mouse'], method="robust")
        gee_up, gee_down = fit_fn(long_df, cov_struct=sm.cov_struct.Independence(), cov_type="bias_reduced")
        print("qic up ", gee_up.qic(scale=1.0), "qic down ", gee_down.qic(scale=1.0))
        # for label, res in [("UP", gee_up), ("DOWN", gee_down)]:
        #     # pull LL vs avg(LC,CC) contrast
        #     est, se, z, p = contrast_fn(res)   # return logitΔ, se, z, p
        #     rows.append({"threshold": t, "outcome": label, "logitΔ": est, "se": se, "p": p})
    return pd.DataFrame(rows)



#%%
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

def _wald_omnibus_pair(res, prefix='C(pair'):
    names = res.model.exog_names
    idx   = [i for i,n in enumerate(names) if n.startswith(prefix)]
    if not idx:
        return np.nan
    L = np.zeros((len(idx), len(names)))
    for r, j in enumerate(idx):
        L[r, j] = 1.0
    w = res.wald_test(L, scalar=True)
    return float(w.pvalue), float(np.asarray(w.statistic))

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

def pair_prob_ci_avgpred(res, df, pair_label, alpha=0.05):
    d = df.copy()
    ALL_LEVELS = df["pair"].unique().tolist()

    d["pair"] = pd.Categorical([pair_label]*len(d), categories=ALL_LEVELS)
    formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="landmark1_landmark2"))'
    X = pt.dmatrix(formula, d, return_type='dataframe')

    eta = X @ res.params.values
    p_i = 1.0 / (1.0 + np.exp(-eta))
    p_bar = float(p_i.mean())

    p_arr = np.asarray(p_i)
    X_arr = np.asarray(X)
    grad = (p_arr * (1 - p_arr))[:, None] * X_arr
    grad = grad.mean(axis=0)
    
        
    V = res.cov_params().values
    var = float(grad @ V @ grad)
    se = var**0.5

    z = norm.ppf(1 - alpha/2)
    lo, hi = p_bar - z*se, p_bar + z*se
    return {"prob": p_bar, "lo": max(0.0, lo), "hi": min(1.0, hi), "se_prob": se}

def _pair_avgpred_and_grad(res, df, pair_label):
    d = df.copy()
    ALL_LEVELS = df["pair"].unique().tolist()

    d["pair"] = pd.Categorical([pair_label]*len(d), categories=ALL_LEVELS)
    formula = f'1 + baseline_int + bg_mean + bs(z, df=4) + C(pair, Treatment(reference="landmark1_landmark2"))'
    X = pt.dmatrix(formula, d, return_type='dataframe')
    eta = X @ res.params.values
    p = 1.0 / (1.0 + np.exp(-eta))
    p_bar = float(p.mean())
    p_arr = np.asarray(p)
    X_arr = np.asarray(getattr(X, "values", X))
    grad = (p_arr * (1 - p_arr))[:, None] * X_arr
    grad = grad.mean(axis=0)
    return p_bar, grad

def _LL_vs_avg_stats(res, df, V, ref_label, LC_label, CC_label, alpha=0.05, df_t=None):
    # LC and CC: marginal means + gradients
    p_LC, g_LC = _pair_avgpred_and_grad(res, df, LC_label)
    p_CC, g_CC = _pair_avgpred_and_grad(res, df, CC_label)

    # Average of LC and CC
    p_avg  = 0.5 * (p_LC + p_CC)
    g_avg  = 0.5 * (g_LC + g_CC)
    var_avg = float(g_avg @ V @ g_avg)
    se_avg  = np.sqrt(max(var_avg, 0.0))

    # REF on the same scale
    p_REF, g_REF = _pair_avgpred_and_grad(res, df, "landmark1_landmark2")

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
    fit_fn,                 # function(long_df, cov_type) -> (gee_up, gee_down)
    pairs,                  # list of (before, after, label)
    sessions,               # SESSIONS
    cells_label="no_bgr",
    ref_label="landmark1_to_landmark2",
    map_cols=("s0_landmark1","landmark2_ctx1","ctx1_ctx2"),  # order: s0_l1, l2_c1, c1_c2
    weighted=False
):
    """
    Writes a wide CSV with one row per (threshold, direction), columns:
      size, what cells, direction, sd cutoff, group wald p,
      then for s0_l1 / l2_c1 / c1_c2: p, Holm, coeff, CI low, CI hi,
      then for avg(LC,CC): p, Holm, coeff, CI low, CI hi, delta p.

    'coeff' and CIs are on the **probability** scale.
    """
    rows = []
    labels = [f'{a}_to_{b}' for a,b in pairs]
    assert ref_label in labels, f"ref '{ref_label}' not found in pairs"

    LC_label, CC_label = "landmark2_ctx1", "ctx1_ctx2"  # used for 'avg' block

    for t in thresholds:
        # classify & long
        classified = classify_fn(all_mice, pairs, session_ids=sessions, sd_threshold=t, method='robust')
        long_df, _ = build_long_fn(classified, pairs, group_cols=['mouse'], method='robust')

        # fit
        
        gee_up, gee_down = fit_fn(long_df, cov_type="robust", cov_struct=Exchangeable(), weighted=weighted)
        rho_hat = float(gee_up.cov_struct.dep_params) 
        rho_hat_d = float(gee_down.cov_struct.dep_params)      # or res.cov_struct.dep_params[0]
        
        print(id(gee_up.cov_struct), id(gee_down.cov_struct))
        
        ex_qic_up = gee_up.qic(scale=1.0)
        ex_qic_down = gee_down.qic(scale=1.0)
    
        gee_up, gee_down = fit_fn(long_df, weighted=weighted)
        
        print(gee_up.summary())
        print(gee_down.summary())
        
        for outcome_label, res in (("UP", gee_up), ("DOWN", gee_down)):
            # meta
            N = int(res.model.endog.shape[0]) if hasattr(res.model, "endog") else len(long_df)
            n_clusters = long_df["mouse"].nunique()
            df_t = n_clusters - 1 if (res.cov_type != "bias_reduced") else None
            p_source_res = "t_corrected" if df_t is not None else "bias_reduced"
                        
            sizes = long_df.groupby('mouse').size()
            cluster_mean = float(sizes.mean())
            cluster_min  = int(sizes.min())
            cluster_max  = int(sizes.max())
                        
            group_p, group_chi = _wald_omnibus_pair(res)

            names, beta, V = _gee_names_beta_V(res)

            # per-pair blocks (prob scale) + p-values vs ref
            per = {}
            # Holm adjust across the three pair-vs-ref p-values
            pvals_prob = []   # for Holm on probability-scale tests
            pvals_logit = []  # for Holm on logit-scale tests
            tmp = {}
            
            for key, pair_lab in zip(("s0_l1","l2_c1","c1_c2"), map_cols):
                # --- LOGIT-SCALE Wald test (same as before)
                est, se, z, p_logit = _contrast_vs_ref_p(names, beta, V, pair_lab, ref_label)
                if df_t is not None:
                    # if you want t-correction on logits too:
                    p_logit = 2 * stats.t.sf(abs(z), df_t)
            
                # --- PROBABILITY-SCALE Wald test (Δp via delta method, averaged over covariates)
                p_pair, g_pair = _pair_avgpred_and_grad(res, long_df, pair_lab)
                p_ref,  g_ref  = _pair_avgpred_and_grad(res, long_df, "landmark1_landmark2")
                Vnp = res.cov_params().values
                delta   = p_pair - p_ref
                g_delta = g_pair - g_ref
                se_delta = float((g_delta @ Vnp @ g_delta) ** 0.5)
                z_delta  = delta / se_delta if se_delta > 0 else np.nan
                p_prob   = (2 * stats.t.sf(abs(z_delta), df_t)) if df_t is not None else (2 * norm.sf(abs(z_delta)))
            
                # --- Probabilities & CI you display (avg-of-predictions)
                pr = pair_prob_ci_avgpred(res, long_df, pair_lab)
            
                tmp[key] = dict(
                    p_prob=p_prob, p_logit=p_logit,
                    est_logit=est, se_logit=se, z_logit=z,      # (kept for reference)
                    prob=pr["prob"], lo=pr["lo"], hi=pr["hi"], se_prob=pr["se_prob"]
                )
                pvals_prob.append(p_prob)
                pvals_logit.append(p_logit)
                
            ref = pair_prob_ci_avgpred(res, long_df, "landmark1_landmark2")
            se_prob_ref = ref["se_prob"]

            
            # Holm correction
            from statsmodels.stats.multitest import multipletests
            _, p_holm_prob,  _, _ = multipletests(pvals_prob,  method="holm")
            _, p_holm_logit, _, _ = multipletests(pvals_logit, method="holm")
            
            for (key, php, phl) in zip(("s0_l1","l2_c1","c1_c2"), p_holm_prob, p_holm_logit):
                per[f"{key}_p_prob"]     = tmp[key]["p_prob"]
                per[f"{key}_Holm_prob"]  = php
                per[f"{key}_p_logit"]    = tmp[key]["p_logit"]
                per[f"{key}_Holm_logit"] = phl
            
                # coefficients you display (probability scale)
                per[f"{key}_coeff"]   = tmp[key]["prob"]
                per[f"{key}_CI_low"]  = tmp[key]["lo"]
                per[f"{key}_CI_hi"]   = tmp[key]["hi"]
                per[f"{key}_SE_prob"] = tmp[key]["se_prob"]

            # avg(LC,CC) block + delta p and LL vs avg p
            V = res.cov_params().values  # ensure V matches this res
            avgd = _LL_vs_avg_stats(res, long_df, V, ref_label, LC_label, CC_label, alpha=0.05, df_t=df_t)
            x_ref = _x_for_pair(names, ref_label)
            x_LC  = _x_for_pair(names, LC_label)
            x_CC  = _x_for_pair(names, CC_label)
            c     = x_ref - 0.5*(x_LC + x_CC)
            est_LA = float(c @ beta)
            var_LA = float(c @ V @ c)
            se_LA  = np.sqrt(max(var_LA, 0.0))
            z_LA   = est_LA / se_LA if se_LA > 0 else np.nan
            p_LA_logit = (2*stats.t.sf(abs(z_LA), df_t)) if df_t is not None else (2*norm.sf(abs(z_LA)))
            
            per.update({
                "avg_p_prob":  avgd["p_LL_vs_avg"],        # prob-scale p (you already compute)
                "avg_coeff":   avgd["avg_coeff"],
                "avg_CI_low":  avgd["avg_CI_low"],
                "avg_CI_hi":   avgd["avg_CI_hi"],
                "avg_SE_prob": avgd["avg_SE_prob"],
                "delta_p":     avgd["delta_p"],
            
                "avg_p_logit": p_LA_logit                  # logit-scale p for supplement
            })

            
            
            from scipy.stats import skew, kurtosis

            y = res.model.endog
            mu = res.fittedvalues
            var = res.family.variance(mu)
            pearson_resid = (y - mu) / np.sqrt(var)

            resid_skew = skew(pearson_resid, bias=False)
            resid_kurtosis = kurtosis(pearson_resid, bias=False) + 3.0
            
            row = {
                "size": N,
                "cluster_mean": cluster_mean,
                "cluster_min": cluster_min,
                "cluster_max": cluster_max,
                "what cells": cells_label,
                "direction": outcome_label,
                "qic": res.qic(scale=1.0),
                "sd cutoff": t,
                "group wald p": group_p,
                "group wald chi": group_chi,
                "ref_coeff": ref["prob"],          
                "ref_CI_low": ref["lo"],
                "ref_CI_hi": ref["hi"],
                "ref_SE_prob": se_prob_ref,
                "p_source": p_source_res,
                "rho_hat_up": rho_hat,
                "rho_hat_down": rho_hat_d,
                "ex_qic_up"  :ex_qic_up,
                "ex_qic_down": ex_qic_down,
                "skew": resid_skew,                
                "kurtosis": resid_kurtosis
            }
            row.update(per)
            rows.append(row)

    out = pd.DataFrame(rows)

    # final column order to match your sketch
    ordered_cols = [
        "size", "cluster_mean","cluster_min","cluster_max",
        "what cells","direction","qic", "sd cutoff","group wald p", "group wald chi",
        "s0_l1_p_prob","s0_l1_Holm_prob","s0_l1_p_logit","s0_l1_Holm_logit",
        "l2_c1_p_prob","l2_c1_Holm_prob","l2_c1_p_logit","l2_c1_Holm_logit",
        "c1_c2_p_prob","c1_c2_Holm_prob","c1_c2_p_logit","c1_c2_Holm_logit",

        "s0_l1_coeff","s0_l1_CI_low","s0_l1_CI_hi","s0_l1_SE_prob",
        "l2_c1_coeff","l2_c1_CI_low","l2_c1_CI_hi","l2_c1_SE_prob",
        "c1_c2_coeff","c1_c2_CI_low","c1_c2_CI_hi","c1_c2_SE_prob",
        "avg_p_prob","avg_p_logit","avg_coeff","avg_CI_low","avg_CI_hi","avg_SE_prob","delta_p",
        "ref_coeff","ref_CI_low","ref_CI_hi","ref_SE_prob",
        "rho_hat_up", "rho_hat_down", "ex_qic_up", "ex_qic_down",
        "skew","kurtosis", "p_source"
    ]


    # keep any extras too, but order the main ones
    cols = [c for c in ordered_cols if c in out.columns] + [c for c in out.columns if c not in ordered_cols]
    out = out[cols]
    csv_path = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/{cells_label}.csv"
    out.to_csv(csv_path, index=False)
    return out




#%%

sweep_thresholds([0.5, 0.75, 1, 1.25], classify_transitions, build_long, fit_gees, contrast_fn=None)

#%%

thresholds = [0.5, 0.75, 1, 1.25]
sweep_gee_to_csv(
    thresholds,
    classify_transitions,
    build_long,
    fit_gees,                 # function(long_df, cov_type) -> (gee_up, gee_down)
    pairs,                  # list of (before, after, label)
    SESSIONS,               # SESSIONS
    cells_label="test_spec",
    ref_label="landmark1_to_landmark2",
    map_cols=("s0_landmark1","landmark2_ctx1","ctx1_ctx2")  # order: s0_l1, l2_c1, c1_c2
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
long_nobgr.columns

#%%
pair_key_map = {
    "s0_landmark1": "s0_l1",
    "landmark1_landmark2": "ref",
    "landmark2_ctx1": "l2_c1",
    "ctx1_ctx2": "c1_c2",
}
CSV_PATH = "/mnt/data/fos_gfp_tmaze/results/gee_transition/always_active.csv"
summary_df = pd.read_csv(CSV_PATH)

overlay_up = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_up", pair_key_map)
overlay_down = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_down", pair_key_map)
#%%
for cutoff in [0.5,0.75,1,1.25]:    
    classified_df = classify_transitions(all_mice, pairs, session_ids = SESSIONS, sd_threshold = cutoff, method = 'robust')
    long_nobgr, _ = build_long(classified_df, pairs ,group_cols=(['mouse']), method="robust")
    overlay_up = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_up", pair_key_map)
    overlay_down = gp.build_per_mouse_overlay(long_nobgr, "mouse", "pair", "y_down", pair_key_map)
    gp.fig_pair_effect_at_cutoff(CSV_PATH, "UP", cutoff, per_mouse_overlay=overlay_up)
    gp.fig_pair_effect_at_cutoff(CSV_PATH, "DOWN", cutoff, per_mouse_overlay=overlay_down)

#%%
gp.plot_forest_at_cutoff(summary_df, "UP", 0.75, save_path=None, title_suffix="")
gp.plot_forest_at_cutoff(summary_df, "DOWN", 0.75, save_path=None, title_suffix="")
#%%
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)  # sharex; y ranges differ
gp.fig_fragility_p_vs_cutoff(CSV_PATH, "DOWN", y_max=1.2, title="DOWN — p vs cutoff", ax=axs[0])
gp.fig_fragility_p_vs_cutoff(CSV_PATH, "UP",   y_max=1.2, title="UP — p vs cutoff",   ax=axs[1])

fig.tight_layout()
