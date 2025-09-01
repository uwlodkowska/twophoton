from statsmodels.genmod.cov_struct import Exchangeable
import numpy as np
import pandas as pd
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

    long_nobgr = long[long['is_bgr'] == 0].copy()


    for c in ["snr_pre_log_w", "snr_post_log_w", "snr_overall_log_w", "snr_asym_log_w"]:
        m, s = long_nobgr[c].mean(), long_nobgr[c].std(ddof=0) or 1.0
        long_nobgr[c.replace("_log_w","_std")] = (long_nobgr[c] - m) / s

    long_nobgr['cluster'] = long_nobgr[group_cols[0]].astype(str)

    return long_nobgr, long




















def sweep_cutoffs_and_summarize(
    lcc,
    pairs,
    SESSIONS,
    thresholds,
    ref="ctx_to_ctx",
    pair_labels=("landmark_to_ctx",),   # compare these to ref at each cutoff
    mouse_col="mouse",
    use_model="wm",                      # choose which result from fit_gees: "wm", "up", "dn"
    alpha=0.05
):
    """
    Returns
    -------
    results_df : tidy DataFrame with two rows per cutoff:
                 mode ∈ {"set_fixed","mouse_avg"} plus Δp, CI, OR, Holm, etc.
    omnibus_df : one row per cutoff with omnibus Wald test for the pair factor
    """
    all_rows = []
    omni_rows = []

    # --- formulas (ref injected into both) ---
    pre_formula = (
        f'1 + snr_pre_std + snr_post_std + z'
        f' + C(pair, Treatment(reference="{ref}"))'
    )
    form_mund = (
        f'1 + snr_pre_std + snr_post_std'
        f' + bg_mean_within + bg_mean_mouse'
        f' + C(pair, Treatment(reference="{ref}"))'
    )

    for cutoff in thresholds:
        # 1) classify + long format
        classified_df = classify_transitions(
            lcc, pairs, session_ids=SESSIONS,
            sd_threshold=cutoff, method="robust", canonical=False
        )
        long, _ = build_long(
            classified_df, pairs,
            group_cols=[mouse_col], method="robust",
            canonical=False, exclude="changing"
        )
        # canonicalize pair labels for modeling
        long = utils.ensure_pair_canonical(long, canonical=True)

        # 2) align NA handling + Mundlak prep
        long_prep, fixed0, prep_stats = gemm.prepare_covariates(
            long,
            formula_for_fit=pre_formula,
            mouse_col=mouse_col,
            mundlak_var="bg_mean",
            center_vars=("snr_pre_std", "snr_post_std", "z")
        )

        # 3) fit GEE(s)
        up_m, dn_m, wm = fit_gees(
            long_prep, formula=form_mund, weighted=True,
            ref=ref, canonical=True, cov_struct=Exchangeable()
        )
        res = {"up": up_m, "dn": dn_m, "wm": wm}[use_model]

        # degrees of freedom for small-sample t
        n_clusters = long_prep[mouse_col].nunique()
        df_t = max(n_clusters - 1, 1)

        # 4) per-cutoff comparisons: (A) set_fixed, (B) mouse-averaged
        for mode, set_fixed, averaging in [
            ("set_fixed", fixed0, "mouse"),     # fix covariates at prepared values
            ("mouse_avg", None,    "mouse"),    # average within mouse -> across mice
        ]:
            tmp = summarize_pairs_delta_and_or(
                res=res,
                df=long_prep,
                pair_labels=pair_labels,
                ref_label=ref,
                formula=form_mund,
                df_t=df_t,
                averaging=averaging,
                mouse_col=mouse_col,
                weights=None,          # set if you want weighted averaging
                set_fixed=set_fixed,
                alpha=alpha
            )
            # add cutoff/mode context
            tmp["cutoff"] = cutoff
            tmp["mode"]   = mode

            # Holm across the provided pair_labels within this cutoff+mode block
            # NOTE: summarize_pairs_delta_and_or stores p in column "p_raw"
            tmp = add_holm(tmp, p_col="p_raw", alpha=alpha)

            all_rows.append(tmp)

        # 5) omnibus Wald over the pair factor
        p_omni, stat_omni = wald_omnibus_for_factor(
            res, factor_prefix="C(pair", n_clusters=n_clusters
        )
        omni_rows.append({
            "cutoff": cutoff,
            "n_clusters": n_clusters,
            "wald_stat": stat_omni,
            "wald_p": p_omni,
            "cov_type": getattr(res, "cov_type", None)
        })

    results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    omnibus_df = pd.DataFrame(omni_rows)

    # nice ordering
    sort_cols = ["cutoff", "mode", "pair"]
    keep_cols = [
        "cutoff","mode","pair","ref",
        "p_ref","p_pair","delta","se_delta","ci_lo","ci_hi",
        "stat","stat_name","p_raw","p_holm","reject_holm",
        "OR","OR_lo","OR_hi","p_logit"
    ]
    results_df = (results_df
                  .sort_values(sort_cols, kind="mergesort")
                  .reindex(columns=[c for c in keep_cols if c in results_df.columns]))

    return results_df, omnibus_df
