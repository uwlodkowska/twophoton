# custom modules

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

import gee_utils as gu
import up_down_utils as gemm

#%% config

SESSIONS_cll, cll = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", suff= "_cll")
SESSIONS_lcc, lcc = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", idx = 1, suff= "_lcc")
#%%
all_mice = lcc#pd.concat([cll,lcc])
#%% pairs
SESSIONS = SESSIONS_lcc
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
    
#%%long df builder

import patsy as pt
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable

def build_long(df, 
               id_pairs, 
               depth_col='zcol', 
               group_cols=('mouse','cell_id'), 
               method = "robust", canonical=False, exclude=None):
    recs = []
    
    suff = "_rstd"
    if method == 'raw':
        suff = ""

    session_ids = sorted({sid for (id1,id2) in id_pairs for sid in (id1,id2)})
    for sid in session_ids:
        raw = f"snr_robust_{sid}"
        log = f"snr_robust_{sid}_log"
        w   = f"snr_robust_{sid}_log_w"
        df.loc[:, log] = gu._signed_log(df[raw].to_numpy(dtype=float))
        df.loc[:, w]   = gu._winsor_per_mouse(df, log, mouse_col="mouse", lo=0.5, hi=99.5)
    
    
    for (id1, id2) in id_pairs:
        col = f'{id1}_to_{id2}_{method}'
        if canonical:
            col = utils.canonical_pair(id1, id2)

        tmp = df.copy()
        
        tmp['pair'] = f'{id1}_to_{id2}'
        tmp["bg_mean"]   = tmp[f"background_{id2}_rstd"]
        tmp["bg_mean"] = (tmp["bg_mean"] - tmp["bg_mean"].mean())/tmp["bg_mean"].std(ddof=0)
        tmp["bg_mean"] = tmp["bg_mean"].fillna(0)
        tmp["snr_pre_log_w"] = tmp[f"snr_robust_{id1}_log_w"]#np.minimum(tmp[f"snr_robust_{id1}_log_w"],tmp[f"snr_robust_{id2}_log_w"])
        tmp["snr_post_log_w"] = tmp[f"snr_robust_{id2}_log_w"]
        tmp["snr_overall_log_w"] = 0.5 * (
            tmp[f"snr_robust_{id1}_log_w"] + tmp[f"snr_robust_{id2}_log_w"]
        )
        tmp["snr_asym_log_w"] = (
            tmp[f"snr_robust_{id2}_log_w"] - tmp[f"snr_robust_{id1}_log_w"]
        )
        tmp.rename(columns={f'int_optimized_{id1}{suff}':'baseline_int', depth_col:'z'}, inplace=True)
        tmp['y_up']     = (tmp[col] == 'up').astype(int)
        tmp['y_down']   = (tmp[col] == 'down').astype(int)
        
        if exclude == "not_pair":
            tmp['is_bgr']   = tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) and utils.in_s(x, id2)))
        elif exclude == "changing":
            tmp['is_bgr'] = (~(tmp['n_sessions'] ==3))
        else:
            tmp['is_bgr'] = False
        
        recs.append(tmp)
    long = pd.concat(recs, ignore_index=True)

    # Switch between all and active in both sessions only - maybe one session too?
    long_nobgr = long[long['is_bgr'] == 0].copy()
    #long_nobgr = long.copy()

    for c in ["snr_pre_log_w", "snr_post_log_w", "snr_overall_log_w", "snr_asym_log_w"]:
        m, s = long_nobgr[c].mean(), long_nobgr[c].std(ddof=0) or 1.0
        long_nobgr[c.replace("_log_w","_std")] = (long_nobgr[c] - m) / s
        
    #long_nobgr["snr"] = long_nobgr["snr_min_std"]

    # Construct cluster id robustly
    long_nobgr['cluster'] = long_nobgr[group_cols[0]].astype(str)

    return long_nobgr, long

def cluster_weights(groups):
    sizes = pd.Series(groups).groupby(groups).size()
    w = sizes.mean() / sizes  
    labels = np.unique(np.asarray(groups))
    return w.reindex(labels).to_numpy(), labels

def fit_gees(long_nobgr, formula = None, add_pair_effect=True, method='robust', cov_struct=sm.cov_struct.Independence(), cov_type="robust", 
             weighted=False, ref="ctx1_to_ctx2", canonical=False):
    # Design matrix (tune as you like)
    # Include baseline_int (start-session intensity), depth (and optional quadratic), and pair fixed effects.
    need = ["baseline_int","bg_mean","z","pair"]
    long_nobgr = long_nobgr.dropna(subset=need).copy()

    if canonical:
        ref = utils.canonical_from_pair_label(ref)
    
    print(formula)
    X = pt.dmatrix(formula, long_nobgr, return_type='dataframe')
    long_nobgr = long_nobgr.loc[X.index].copy()

        
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
    #print("QIC up: ",qic_up,"QIC down: ",qic_down)

    return gee_up, gee_down, w




#%%
#TODO  marked to reuse for inspector problems

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):


#%% sweeper definition
import patsy as pt
from scipy.stats import norm
from scipy.special import expit as logistic



# --- main collector -----------------------------------------------------------


def sweep_gee_to_csv(
    thresholds,
    classify_fn,
    build_long_fn,
    fit_fn,                 # function(long_df, ...) -> (gee_up, gee_down, weights)
    pairs,                  # list of (before, after[, label])
    sessions,  
    formula=None,             
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

        pre_formula = (
            f'1 + snr_pre_std + snr_post_std  + z '
            f'+ C(pair, Treatment(reference="{ref}"))'
        )
        # long_df, fixed0, prep_stats = gemm.prepare_covariates(
        #     long_df,
        #     formula_for_fit=pre_formula,         # NA handling aligned to your model
        #     mouse_col="mouse",
        #     mundlak_var="bg_mean",
        #     center_vars=("snr_pre_std","snr_post_std","z")
        # )

        formula=pre_formula

        need_cols = ["y_up","y_down","baseline_int","bg_mean","z","pair","mouse"]


        # 3) fit
        gee_up_ex,  gee_down_ex,  weights = fit_fn(
             long_df, formula = formula, cov_type="robust", cov_struct=Exchangeable(), weighted=weighted, canonical=canonical, ref=ref_label
         )
        gee_up_ind, gee_down_ind, weights = fit_fn(
             long_df, formula = formula, cov_type="robust", cov_struct=Independence(),  weighted=weighted, canonical=canonical, ref = ref_label
         )

        print(gee_up_ind.summary())
        print(gee_down_ind.summary())
        ref_use = ref_label
        pair_label = "landmark_to_ctx"  # example
        dp_base = gemm.delta_prob_test(gee_up_ex, long_df, pair_label, ref_label, formula=pre_formula, df_t=8-1, averaging="mouse", mouse_col="mouse")["delta"]
        print(dp_base)
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
        print("QIC : ", ind_qic_up, ind_qic_down)
        # 4) iterate for UP/DOWN models
        for outcome_label, res in (("UP", gee_up_ind), ("DOWN", gee_down_ind)):
            N = int(getattr(res.model.endog, "shape", [len(long_df)])[0])
            n_clusters = long_df["mouse"].nunique()
            df_t = n_clusters - 1 if (res.cov_type != "bias_reduced") else None
            p_src = "t_corrected" if df_t is not None else "bias_reduced"

            sizes = long_df.groupby('mouse').size()
            cluster_mean = float(sizes.mean()); cluster_min = int(sizes.min()); cluster_max = int(sizes.max())

            group_p, group_chi = gemm.wald_omnibus_for_factor(res, n_clusters=n_clusters)
            names, beta, V = gemm.names_beta_V(res)

            # choose which labels to report: all except the reference
            comp_labels = [lab for lab in data_labels if lab != ref_use]

            # accumulators
            per = {}
            pvals_prob, pvals_logit = [], []
            tmp = {}

            # compute ref (prob-scale EMM) once

            ref_emm = ref_emm = gemm.emm_prob_ci(
                res, long_df, ref_use, ref_use,
                formula=formula,
                averaging=("weights" if weights is not None else "cell"),
                weights=weights
            )
            se_prob_ref = ref_emm["se_prob"]
            
            for lab in comp_labels:
                key = utils.slug_for_cols(lab)

                # LOGIT-scale contrast vs ref
                est, se, z, p_logit = gemm.contrast_vs_ref_logit(res, lab, ref_use)
                if df_t is not None and np.isfinite(z):
                    p_logit = 2 * stats.t.sf(abs(z), df_t)

                df_t = n_clusters - 1 if (res.cov_type != "bias_reduced") else None
                delta_info = gemm.delta_prob_test(
                    res, long_df, lab, ref_use,
                    formula=formula,
                    df_t=df_t,
                    averaging=("weights" if weights is not None else "cell"),
                    weights=weights
                )
                p_prob = delta_info["p"]

                # Probabilities & CI shown (avg-of-predictions)
                pr = gemm.emm_prob_ci(
                    res, long_df, lab, ref_use,
                    formula=formula,
                    averaging=("weights" if weights is not None else "cell"),
                    weights=weights
                )

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



#%% main sweeper call
filteredby = "lcc_active3"
SESSIONS = SESSIONS_lcc
thresholds = [1]
pairs = [['landmark','ctx1'], ['ctx1','ctx2']]#['landmark1','landmark2'], ['ctx','landmark1']]#, ['landmark','ctx1'], ['ctx1','ctx2']]
#pairs = [["ctx", "landmark1"], ["landmark1", "landmark2"]]
all_mice = lcc

#" + bg_mean_within + bg_mean_mouse  "
ref = "ctx_to_ctx"#"landmark_to_landmark"#
formula = f'1 + snr_post_std + snr_pre_std + bg_mean_mouse + bs(z, df=4) + C(pair, Treatment(reference="{ref}"))'





gee_up_ind, gee_down_ind = sweep_gee_to_csv(
    thresholds,
    classify_transitions,
    build_long,
    fit_gees,                 # function(long_df, cov_type) -> (gee_up, gee_down)
    pairs,                  # list of (before, after, label)
    SESSIONS,  
    formula=formula,
    cells_label=filteredby,
    ref_label=ref,
    weighted = True,
    canonical = True
)

#%% covariate model evaluation
classified_df = classify_transitions(lcc, pairs, session_ids = SESSIONS, sd_threshold = 0.75, method = 'robust', canonical = False)
long, _ = build_long(classified_df, pairs, group_cols=['mouse'], method='robust', canonical = False, exclude="changing")

ref = "ctx_to_ctx"
pair = "landmark_to_ctx"

long = utils.ensure_pair_canonical(long, canonical=True)

pre_formula = (
    f'1 + snr_pre_std + snr_post_std +  z'
    f'+ C(pair, Treatment(reference="{ref}"))'
)
long_prep, fixed0, prep_stats = gemm.prepare_covariates(
    long,
    formula_for_fit=pre_formula,         # NA handling aligned to your model
    mouse_col="mouse",
    mundlak_var="bg_mean",
    center_vars=("snr_pre_std","snr_post_std","z")
)


form_mund = f'1 + snr_pre_std + snr_post_std + bg_mean_within + bg_mean_mouse + C(pair, Treatment(reference="{ref}"))'

up_m, dn_m, wm = fit_gees(long_prep, formula=form_mund, weighted=True, ref=ref, canonical=True, cov_struct = Exchangeable())

def contrast_stats(res, df, form):
    out = gemm.delta_prob_test(res, df, pair, ref, formula=form, df_t=df["mouse"].nunique()-1,
                               averaging="mouse", mouse_col="mouse")#, set_fixed = fixed0)
    return out["delta"], out["se"], out["p"]

dp_m, se_m, p_m = contrast_stats(up_m, long_prep, form_mund)

# print(f"Δp base={dp_b:.4f}, Δp +bg_mouse={dp_m:.4f}, shift={dp_m-dp_b:.4f} ({100*(dp_m-dp_b):.1f} pp)")
# print(f"SE base={se_b:.4f}, SE +bg_mouse={se_m:.4f}, standardized shift S={S:.2f}")
# print(f"p base={p_b:.4g}, p +bg_mouse={p_m:.4g}")
print("QIC UP  =", up_m.qic(scale=1.0)[0], dn_m.qic(scale=1.0)[0])


print("For up ",dp_m,"p_val ", p_m)
dp_m, se_m, p_m = contrast_stats(dn_m, long_prep, form_mund)
print("For down: delta ",dp_m,"p_val ", p_m)
    


#%%
print(up_m.summary())

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
long[["bg_mean"]].describe()
