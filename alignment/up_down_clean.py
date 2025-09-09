from statsmodels.genmod.cov_struct import Exchangeable
import pandas as pd
import utils
import patsy as pt


from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence

if not hasattr(pd.DataFrame, "iteritems"):
    pd.DataFrame.iteritems = pd.DataFrame.items



import statsmodels.api as sm


import gee_utils as gu
import up_down_utils as gemm

import up_down_plots as uplt

#%%
SESSIONS_cll, cll = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", suff= "_cll")
SESSIONS_lcc, lcc = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", idx = 1, suff= "_lcc")

SESSIONS_ALL,all_mice = utils.get_concatenated_df_from_config("config_files/multisession.yaml")

SESSIONS_ctx, ctx_mice = utils.get_concatenated_df_from_config("config_files/context_only.yaml", suff= "_cll")
#%%
SESSIONS_ret, ret_mice = utils.get_concatenated_df_from_config("config_files/context_only.yaml", suff= "_ret", idx = 1)
#%% pairs
SESSIONS = SESSIONS_ctx
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
        outcol = f'{id1}_to_{id2}'
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
        col = f'{id1}_to_{id2}'
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
        
        
        Tcount = tmp["detected_in_sessions"].apply(
            lambda s: sum(utils.in_s(s, t) for t in ["landmark1","landmark2","ctx1", "ctx2", "ctx", "landmark", "s1", "s2", "s3"])
        )
        
        test_sessions = len(session_ids)
        # if (test_sessions>3):
        #     test_sessions -= 1
        
        Bcount = tmp["detected_in_sessions"].apply(
            lambda s: sum(utils.in_s(s, t) for t in ["s0"])
        )
        print(test_sessions)
        mask = (Tcount == test_sessions)
        
        # print(tmp["detected_in_sessions"].head(25))
        # print(mask.head(25))
        
        if exclude == "not_pair":
            tmp['is_bgr']   = tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) and utils.in_s(x, id2)))
        elif exclude == "strict_not_pair":
            tmp['is_bgr']   = ((tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) 
                                                                                and utils.in_s(x, id2)))) | (tmp["n_sessions"]==3))
        elif exclude == "changing":
            
            tmp['is_bgr'] = (tmp["n_sessions"]<3)
        else:
            tmp['is_bgr'] = (Tcount == 0)
        
        
        recs.append(tmp)
    long = pd.concat(recs, ignore_index=True)

    long_nobgr = long[long['is_bgr'] == 0].copy()


    for c in ["snr_pre_log_w", "snr_post_log_w", "snr_overall_log_w", "snr_asym_log_w"]:
        m, s = long_nobgr[c].mean(), long_nobgr[c].std(ddof=0) or 1.0
        long_nobgr[c.replace("_log_w","_std")] = (long_nobgr[c] - m) / s

    long_nobgr['cluster'] = long_nobgr[group_cols[0]].astype(str)

    return long_nobgr, long
#%%

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
def sweep_cutoffs_and_summarize(
    df,
    pairs,
    SESSIONS,
    thresholds,
    ref="ctx_to_ctx",
    pair_labels=("landmark_to_ctx",),   # compare these to ref at each cutoff
    mouse_col="mouse",
    cells_to_exclude = "",                     # choose which result from fit_gees: "wm", "up", "dn"
    alpha=0.05,
    canonical=False
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
            df, pairs, session_ids=SESSIONS,
            sd_threshold=cutoff, method="robust", canonical=canonical
        )
        long, _ = build_long(
            classified_df, pairs,
            group_cols=[mouse_col], method="robust",
            canonical=canonical, exclude=cells_to_exclude
        )
        # canonicalize pair labels for modeling
        if canonical:
            long = utils.ensure_pair_canonical(long, canonical=True)

        # 2) align NA handling + Mundlak prep
        print(cutoff)
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
            ref=ref, canonical=canonical, cov_struct=Independence()
        )
        for direct, res in [["UP",up_m], ["DOWN", dn_m]]:

            # degrees of freedom for small-sample t
            n_clusters = long_prep[mouse_col].nunique()
            df_t = max(n_clusters - 1, 1)
            omni_tmp = []
            # 4) per-cutoff comparisons: (A) set_fixed, (B) mouse-averaged
            for mode, set_fixed, averaging in [
                ("set_fixed", fixed0, "mouse"),     # fix covariates at prepared values
                ("mouse_avg", None,    "mouse"),    # average within mouse -> across mice
            ]:
                tmp = gemm.summarize_pairs_delta_and_or(
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
    
                tmp["cutoff"] = cutoff
                tmp["mode"]   = mode
                tmp["direction"] = direct
    
                tmp = gemm.add_holm(tmp, p_col="p_raw", alpha=alpha)
    
                all_rows.append(tmp)
  
                omni_tmp.append(gemm.prob_wald_multi(
                    res,
                    df = long_prep,
                    formula = form_mund,
                    pair_labels = pair_labels,           
                    ref_label = ref,
                    averaging="mouse",
                    mouse_col="mouse",
                    set_fixed = set_fixed,
                ))
            omni_tmp = pd.concat(omni_tmp, ignore_index = True)
            omni_tmp['direction'] = direct
            omni_tmp['cutoff'] = cutoff
            omni_rows.append(omni_tmp)
            print(res.summary())

    results_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    stats_omni = pd.concat(omni_rows)

    sort_cols = ["cutoff", "mode", "pair"]
    keep_cols = [
        "cutoff","mode", "direction", "pair","ref",
        "p_ref","p_pair","delta","se_delta","ci_lo","ci_hi",
        "stat","stat_name","p_raw","p_holm","reject_holm",
        "OR","OR_lo","OR_hi","p_logit"
    ]
    results_df = (results_df
                  .sort_values(sort_cols, kind="mergesort")
                  .reindex(columns=[c for c in keep_cols if c in results_df.columns]))

    return results_df, stats_omni

#%%
thresholds = [0.5, 0.75, 1, 1.25]
pairs_all =list(zip(SESSIONS_ALL[1:], SESSIONS_ALL[2:]))
pair_labels = [("landmark1_to_landmark2", "ctx1_to_ctx2"), ("landmark2_to_ctx1","ctx1_to_ctx2"), ("landmark1_to_landmark2","landmark2_to_ctx1")]

results = sweep_cutoffs_and_summarize(all_mice, 
                                      pairs_all, 
                                      SESSIONS_ALL, 
                                      thresholds, 
                                      ref="landmark1_to_landmark2",#"ctx_to_ctx", 
                                      pair_labels=pair_labels,#"s0_to_landmark1",),#("ctx_to_landmark",),#
                                      mouse_col="mouse",
                                      cells_to_exclude = "strict_not_pair",
                                      canonical=False
                                      )

#%%
csv_path = f"/mnt/data/fos_gfp_tmaze/results/gee_transition/multi_always_tx.csv"
results[0].to_csv(csv_path)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results[0])
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results[1])
    
    #%%
    print(len([("landmark1_to_landmark2", "ctx1_to_ctx2"), ("landmark2_to_ctx1","ctx1_to_ctx2"), ("landmark1_to_landmark2","landmark2_to_ctx1")]))
#%%
to_compare = {
    'LCC':[("landmark_to_ctx", "ctx_to_ctx")],
    'CLL':[("ctx_to_landmark", "landmark_to_landmark")],
    'LLC':[("landmark_to_landmark",'landmark_to_ctx')],
    'rozszerzona':[("ctx1_to_ctx2","s0_to_landmark1"),("landmark1_to_landmark2", "s0_to_landmark1"), 
                   ("landmark2_to_ctx1","s0_to_landmark1"), ("landmark1_to_landmark2", "ctx1_to_ctx2")],
    'kontekst':[("s1_to_s2", "s2_to_s3")],
    'kontekst_ret':[("ret1_to_ret2", "ret2_to_ret3")]
    }
#%%

less_mice = all_mice.loc[~(all_mice["mouse"]=="13")].copy()

#%%plotting

for exp, df, sessions in [
        #('LCC', lcc, SESSIONS_lcc),
        #('CLL', cll, SESSIONS_cll),
        #('LLC', all_mice, SESSIONS_ALL[1:4]),
        #('rozszerzona', less_mice, SESSIONS_ALL),
        #('kontekst', ctx_mice, SESSIONS_ctx[:3]),
        ('kontekst_ret', ret_mice, SESSIONS_ret)
        ]:
    pairs = list(zip(sessions, sessions[1:]))
    for suff, to_exclude in [
        ("w parze", "strict_not_pair"),  
        ("zawsze", "changing")
    ]:
        uplt.run_all_groups(df, pairs, sessions, to_compare[exp][0][1], to_compare[exp],sweep_cutoffs_and_summarize, exp, suff, to_exclude)
