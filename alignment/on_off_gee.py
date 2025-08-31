import sys
import yaml

# custom modules
import cell_classification as cc
import utils
import numpy as np, patsy as pt

from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import pandas as pd

from scipy.special import expit as logistic
import statsmodels.api as sm
from scipy import stats

import presence_plots as pp
import gee_utils as gu
#%% config

#SESSIONS, all_mice = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", suff= "_cll", idx=0)
SESSIONS, all_mice = utils.get_concatenated_df_from_config("config_files/multisession.yaml")

#%%
id_pairs = list(zip(SESSIONS, SESSIONS[1:]))#[1:3]
#%%
all_mice = cc.on_off_cells(all_mice, id_pairs)

#%%

less_mice = all_mice.loc[all_mice["mouse"]!="13"]
#%%
import numpy as np
import pandas as pd


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

def build_long(df, 
               id_pairs, 
               depth_col='zcol', 
               method = "robust", canonical=False):
    recs = []
    
    suff = "_rstd"
    df = df.copy()
    before = len(df)

    print("Nans dropped " , before - df.shape[0])

    session_ids = sorted({sid for (id1,id2) in id_pairs for sid in (id1,id2)})
    for sid in session_ids:
        raw = f"snr_robust_{sid}"
        log = f"snr_robust_{sid}_log"
        w   = f"snr_robust_{sid}_log_w"
        df.loc[:, log] = _signed_log(df[raw].to_numpy(dtype=float))
        df.loc[:, w]   = _winsor_per_mouse(df, log, mouse_col="mouse", lo=0.5, hi=99.5)

    for (id1, id2) in id_pairs:
        col = f'{id1}_to_{id2}'
        
        if canonical:
            col = utils.canonical_pair(id1, id2)
        tmp = df.copy()
        
        tmp = tmp.dropna(subset=[f"snr_robust_{id1}", f"snr_robust_{id2}"])
        if tmp.empty:
            continue
        # df[["mouse", "cell_id", depth_col, col,
        #           f'int_optimized_{id1}{suff}',
        #           f"background_{id1}_rstd",
        #           f"background_{id2}_rstd",
        #           f"is_dim_by_bg_{id1}",
        #           f"is_dim_by_bg_{id2}",
        #           f"snr_robust_{id1}",
        #           f"snr_robust_{id2}",
        #           'detected_in_sessions',
        #           'n_sessions']].copy()
        
        tmp['pair'] = f'{id1}_to_{id2}'
        tmp["bg_mean"]   = tmp[f"background_{id2}_rstd"]
        #tmp["bg_mean"] = (tmp["bg_mean"] - tmp["bg_mean"].mean())/tmp["bg_mean"].std(ddof=0)
        tmp["bg_mean"] = tmp["bg_mean"].fillna(0)

        tmp.rename(columns={f'int_optimized_{id1}{suff}':'baseline_int', depth_col:'z'}, inplace=True)
        tmp["depth"] = tmp['z']<0.5
        tmp["snr_min_log_w"] = np.minimum(tmp[f"snr_robust_{id1}_log_w"],tmp[f"snr_robust_{id2}_log_w"])
        tmp['y_const']  = (tmp[col] == 'const').astype(int)
        tmp['y_on']     = (tmp[col] == 'on').astype(int)
        tmp['y_off']    = (tmp[col] == 'off').astype(int)

        tmp["snr_overall_log_w"] = 0.5 * (
            tmp[f"snr_robust_{id1}_log_w"] + tmp[f"snr_robust_{id2}_log_w"]
        )
        tmp["snr_asym_log_w"] = (
            tmp[f"snr_robust_{id2}_log_w"] - tmp[f"snr_robust_{id1}_log_w"]
        )

        Tcount = tmp["detected_in_sessions"].apply(
            lambda s: sum(utils.in_s(s, t) for t in ["landmark1","landmark2","ctx1"])#,"ctx2"])
        )
        mask = ((Tcount<2))
        
        Bcount = tmp["detected_in_sessions"].apply(
            lambda s: sum(utils.in_s(s, t) for t in ["s0"])
        )
        

        m13_mask = ((tmp["mouse"]=='13') | (Tcount == 3))
        #tmp['is_bgr'] = mask | m13_mask
        tmp['is_bgr'] =  (tmp["n_sessions"] == 5)
        tmp['is_bgr'] = False
        recs.append(tmp)

    long = pd.concat(recs, ignore_index=True)

    long_nobgr = long.loc[long['is_bgr'] == 0].copy()
    #long_nobgr = long.copy()

    long_nobgr['cluster'] = long_nobgr["mouse"].astype(str)

    
    for c in ["snr_overall_log_w", "snr_asym_log_w", "snr_min_log_w"]:
        m, s = long_nobgr[c].mean(), long_nobgr[c].std(ddof=0) or 1.0
        long_nobgr[c.replace("_log_w","_std")] = (long_nobgr[c] - m) / s
        
    long_nobgr["snr"] = long_nobgr["snr_min_std"]
    
    needed = ["y_on","y_off","baseline_int","z","snr_min_std","bg_mean","pair","mouse","cluster"]
    long_nobgr = long_nobgr.dropna(subset=[c for c in needed if c in long_nobgr.columns]).reset_index(drop=True)
    #long_nobgr = long_nobgr[needed].copy()
    
    return long_nobgr, long

#%%
PAIR_LEVELS = [
    "ctx1_to_ctx2",
    "landmark2_to_ctx1",
    "landmark1_to_landmark2",
    "s0_to_landmark1"
    ]
#PAIR_LEVELS = ["landmark1_to_landmark2", "ctx_to_landmark1"]
#PAIR_LEVELS = ["ctx1_to_ctx2", "landmark_to_ctx1"]

long_mice, _ = build_long(less_mice, id_pairs)

#long_mice["pair"] = pd.Categorical(long_mice["pair"], categories=PAIR_LEVELS, ordered=True)

long_mice["z_std"] = (long_mice["z"] - long_mice["z"].mean()) / long_mice["z"].std(ddof=0)

#%%
long_mice_changed = (long_mice.loc[long_mice["y_const"]==0]).copy()
#%%

def make_mouse_weights(df, mouse_col="mouse"):
    n = df.groupby(mouse_col).size()
    w = (n.mean() / n).astype(float)  
    return w.to_dict()




def attach_weights(d, mouse_col="mouse", name="w_full"):
    w_map = make_mouse_weights(d, mouse_col="mouse")
    w = d[mouse_col].map(w_map)
    if w.isna().any():
        w = w.fillna(1.0)
    d[name] = w.astype(float)
    return d

long_mice = attach_weights(long_mice, name="w_full")
long_mice_changed = attach_weights(long_mice_changed, name="w_full")



def fit_gees(long_nobgr, cov_struct=sm.cov_struct.Independence(), cov_type="robust", 
             weighted=False, 
             ref="ctx1_to_ctx2", 
             canonical=False, 
             formula = None):
    if canonical: 
        ref = utils.canonical_from_pair_label(ref) 
     
    X = pt.dmatrix(formula, long_nobgr, return_type='dataframe').astype(float)     
    y = pd.to_numeric(long_nobgr['y_const'], errors='coerce').to_numpy()        # <- ensure float
    groups = long_nobgr['mouse'].astype('category').cat.codes.to_numpy()


    assert groups.shape[0] == len(long_nobgr) 
    
    # --- per-row weights constant within mouse: w_i ∝ 1 / n_mouse --- 
    n_per_mouse = long_nobgr.groupby('mouse').size()
    w_per_mouse = (n_per_mouse.mean() / n_per_mouse).to_dict()
    if weighted:
        #w = long_nobgr['mouse'].map(w_per_mouse)
        w = long_nobgr['w_full']
        w = pd.to_numeric(w, errors='coerce').fillna(0.0).to_numpy()
        cov_type = "robust"
    else: 
        w=None 
        cov_type="bias_reduced" 
        
        
    cov_up = type(cov_struct)() 

    

    gee_const = GEE(endog=long_nobgr['y_const'], exog=X, 
                    groups=groups, cov_struct=cov_up, 
                    family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1.0) 
    
    qic_const = gee_const.qic(scale=1.0) 
    print("QIC up: ",qic_const) 
    return w, gee_const

#%%
ref = "landmark1_to_landmark2"
ref = "ctx1_to_ctx2"

formula = f'1 + C(pair, Treatment(reference="{ref}")) + snr'
weights, res = fit_gees(long_mice, weighted = True, ref = ref, cov_struct=Independence(), formula=formula)

print(res.summary())


#%%

emms = gu.pair_estimands(
    res, long_mice, formula, PAIR_LEVELS, ref_label=ref,
    averaging="mouse", set_fixed={"snr": 0}  
)

# 2) Contrasts vs reference (Holm-adjusted), with Cohen’s dz
contr = gu.contrasts_vs_reference(
    res, long_mice, formula, PAIR_LEVELS, ref_label=ref,
    averaging="mouse", set_fixed={"snr": 0}
)

# 3) Omnibus Wald on probability scale (delta method, small-sample F)
omni = gu.wald_omnibus_pair_delta(
    res, long_mice, formula, PAIR_LEVELS, ref_label=ref,
    averaging="mouse", set_fixed={"snr": 0},
    n_clusters=long_mice["mouse"].nunique()
)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(emms)
    print(contr)
    print(omni)

#%%
fit_callable = lambda d: fit_gees(
    d, cov_struct=sm.cov_struct.Independence(),formula=formula, cov_type="robust", weighted=True, ref=ref
)

# 1) LOSO influence on the biological contrast (ctx→l1 vs l1→l2)
tab, summ = gu.loso_contrast_table(
    fit_callable, long_mice, formula,
    pair_alt="landmark1_to_landmark2", pair_ref=ref,
    averaging="mouse", set_fixed={"snr": 0}   
)
#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(tab.head(10))   # sorted by absolute change
    print(summ)           # jackknife SE/CI and DFBETA threshold (2/sqrt(m))
    
    # 2) Per-mouse raw (unmodeled) contrast, to sanity-check the same outlier
    raw = gu.per_mouse_raw_contrast(long_mice, "ctx_to_landmark1", "landmark1_to_landmark2",
                                    resp_col="y_const", pair_col="pair", mouse_col="mouse")
    print(raw.sort_values("diff"))