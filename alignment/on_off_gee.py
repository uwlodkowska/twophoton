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

#SESSIONS, all_mice = utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", suff= "_lcc", idx=1)
SESSIONS, all_mice = utils.get_concatenated_df_from_config("config_files/multisession.yaml")
#SESSIONS, all_mice = utils.get_concatenated_df_from_config("config_files/context_only.yaml", suff= "_cll")
#%%
#SESSIONS = SESSIONS[:3]
id_pairs = list(zip(SESSIONS, SESSIONS[1:]))#[1:3]
#%%
all_mice = cc.on_off_cells(all_mice, id_pairs)
#%%
all_mice.columns

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
            lambda s: sum(utils.in_s(s, t) for t in ["landmark1","landmark2","ctx1","ctx2"])
        )
        mask = ((Tcount==4))
        
        Bcount = tmp["detected_in_sessions"].apply(
            lambda s: sum(utils.in_s(s, t) for t in ["s0"])
        )
        

        
        #tmp['is_bgr'] = mask | m13_mask
        tmp['is_bgr'] =  (tmp["n_sessions"] == 3)
        tmp['is_bgr']   = ((((tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) 
                                                                               or utils.in_s(x, id2)))) | (tmp[col]=="_")))|(mask))
        
        
        tmp['is_bgr']   = ((((tmp['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, id1) 
                                                                               or utils.in_s(x, id2)))) | (tmp[col]=="_")))|(tmp["n_sessions"] == 5))
        #tmp["is_bgr"] = False
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
#PAIR_LEVELS= ["s1_to_s2", "s2_to_s3"]

long_mice, _ = build_long(less_mice, id_pairs)

#long_mice["pair"] = pd.Categorical(long_mice["pair"], categories=PAIR_LEVELS, ordered=True)

long_mice["z_std"] = (long_mice["z"] - long_mice["z"].mean()) / long_mice["z"].std(ddof=0)

#%%

import numpy as np, pandas as pd
from collections import defaultdict

def _sess_set(val, SESSIONS, utils_in_s):
    """Return a set of sessions from val (set/list/tuple or string)."""
    if isinstance(val, (set, list, tuple)):
        return set(val)
    # string-like
    return {sid for sid in SESSIONS if utils_in_s(val, sid)}

def per_mouse_cell_sessions(all_mice, SESSIONS,
                            mouse_col="mouse", cell_col="cell_id", det_col="detected_in_sessions",
                            utils_in_s=None):
    """Dict: mouse -> {cell_id -> set(sessions)} (collapses duplicates by union)."""
    d = defaultdict(dict)
    for _, r in all_mice[[mouse_col, cell_col, det_col]].iterrows():
        m, cid, raw = r[mouse_col], r[cell_col], r[det_col]
        sset = _sess_set(raw, SESSIONS, utils_in_s)
        if cid in d[m]:
            d[m][cid] |= sset
        else:
            d[m][cid] = set(sset)
    return d

def k2_consecutive_fraction(by_mouse, SESSIONS):
    pos = {sid:i for i,sid in enumerate(SESSIONS)}
    rows=[]
    for m, cells in by_mouse.items():
        pairs=[]
        for sset in cells.values():
            if len(sset)==2:
                a,b = sorted(sset, key=lambda s: pos.get(s, -10))
                if a in pos and b in pos:
                    pairs.append(abs(pos[a]-pos[b])==1)
        n=len(pairs)
        rows.append({"mouse":m, "n_k2":n, "frac_consecutive": (np.mean(pairs) if n>0 else np.nan)})
    return pd.DataFrame(rows)


def retention_by_lag(by_mouse, SESSIONS):
    pos = {sid:i for i,sid in enumerate(SESSIONS)}
    out=[]
    for m,cells in by_mouse.items():
        T=len(SESSIONS)
        daysets=[set() for _ in range(T)]
        for cid, sset in cells.items():
            for s in sset:
                if s in pos: daysets[pos[s]].add(cid)
        for lag in range(1,T):
            vals=[]
            for t in range(T-lag):
                A, B = daysets[t], daysets[t+lag]
                if len(A)>0:
                    vals.append(len(A & B)/len(A))
            if vals:
                out.append({"mouse":m,"lag":lag,"retention":float(np.mean(vals)),"n_pairs":len(vals)})
    return pd.DataFrame(out)

def overlap_lift(by_mouse, SESSIONS):
    pos = {sid:i for i,sid in enumerate(SESSIONS)}
    out=[]
    for m,cells in by_mouse.items():
        T=len(SESSIONS); N=len(cells)
        daysets=[set() for _ in range(T)]
        for cid, sset in cells.items():
            for s in sset:
                if s in pos: daysets[pos[s]].add(cid)
        for i in range(T):
            for j in range(i+1,T):
                A,B = daysets[i], daysets[j]
                obs = len(A & B)
                exp = (len(A)*len(B)/N) if N>0 else np.nan
                lift = (obs/exp) if (exp and exp>0) else np.nan
                out.append({"mouse":m,"i":i,"j":j,"lag":j-i,
                            "overlap":obs,"expected":exp,"lift":lift})
    return pd.DataFrame(out)


# Build per-mouse cell session sets
by_mouse = per_mouse_cell_sessions(less_mice, SESSIONS, utils_in_s=utils.in_s)

# (A) K=2 fraction consecutive
k2 = k2_consecutive_fraction(by_mouse, SESSIONS)
print(k2.sort_values("n_k2", ascending=False).head(10))
print("Across mice: mean frac_consecutive (K=2) =", np.nanmean(k2["frac_consecutive"]))

# (B) Retention vs lag
ret = retention_by_lag(by_mouse, SESSIONS)
print(ret.pivot_table(index="mouse", columns="lag", values="retention"))
print("Mouse-mean retention by lag:\n", ret.groupby("lag")["retention"].mean())

# (C) Lift vs lag
lift = overlap_lift(by_mouse, SESSIONS)
print("Mouse-mean lift by lag:\n", lift.groupby("lag")["lift"].mean())
# Optional: compare lag=1 vs all lag>1 at mouse level
lag1 = lift[lift["lag"]==1].groupby("mouse")["lift"].mean()
lagN = lift[lift["lag"]>1].groupby("mouse")["lift"].mean()
common = lag1.index.intersection(lagN.index)
diff = (lag1.loc[common] - lagN.loc[common]).dropna()
print(f"Mean(lift lag1 − lag>1) across mice = {diff.mean():.3f}  (n={len(diff)})")










#%%
long_mice_changed = (long_mice.loc[long_mice["y_const"]==0]).copy()


#%%
# from statsmodels.stats.proportion import proportion_confint

# d = all_mice.copy()
# d = d[d["n_sessions"] == 2]
# d["all_active"] = (d["n_sessions"]==2)

# # 1) Pooled fraction + Wilson 95% CI (descriptive)
# k = int(d["all_active"].sum())
# N = int(len(d))
# p_pooled = k / N
# ci_lo, ci_hi = proportion_confint(k, N, method="wilson")

# # 2) Per-mouse mean ± SD (primary summary)
# per_mouse = d.groupby(["mouse"])["all_active"].mean()   # one proportion per mouse
# mean_pm = per_mouse.mean()
# sd_pm   = per_mouse.std(ddof=1)
# sem_pm  = sd_pm / np.sqrt(per_mouse.size)

# print(f"Pooled across cells: {p_pooled:.3f} (Wilson 95% CI {ci_lo:.3f}–{ci_hi:.3f}), Ncells={N}")
# print(f"Per-mouse: {mean_pm:.3f} ± {sd_pm:.3f} SD (SEM {sem_pm:.3f}), Nmice={per_mouse.size}")



#%%



def attach_weights_ttest_equivalent(d, mouse_col="mouse", pair_col="pair", name="w_paired"):
    dd = d.copy()
    n_mp = dd.groupby([mouse_col, pair_col])["pair"].transform("size")
    dd[name] = 1.0 / n_mp

    # optional sanity check: sums to 1 within each mouse×pair
    assert np.allclose(dd.groupby([mouse_col, pair_col])[name].sum().values, 1.0)

    return dd


long_mice = attach_weights_ttest_equivalent(long_mice, name="w_full")
long_mice_changed = attach_weights_ttest_equivalent(long_mice_changed, name="w_full")



def fit_gees(long_nobgr, cov_struct=sm.cov_struct.Independence(), cov_type="robust", 
             weighted=False, 
             ref="ctx1_to_ctx2", 
             canonical=False, 
             formula = None,
             fit_col = 'y_const'):
    if canonical: 
        ref = utils.canonical_from_pair_label(ref) 
    
        
    
    X = pt.dmatrix(formula, long_nobgr, return_type='dataframe').astype(float)     
    y = pd.to_numeric(long_nobgr[fit_col], errors='coerce').to_numpy()        # <- ensure float
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

    

    gee_const = GEE(endog=long_nobgr[fit_col], exog=X, 
                    groups=groups, cov_struct=cov_up, 
                    family=Binomial(), weights=w).fit(cov_type=cov_type, scale=1.0) 
    
    qic_const = gee_const.qic(scale=1.0) 
    print("QIC up: ",qic_const) 
    return w, gee_const

#%%
ref = "landmark1_to_landmark2"
ref = "ctx1_to_ctx2"
#ref = "s0_to_landmark1"
#ref= "s2_to_s3"

formula = f'1 + C(pair, Treatment(reference="{ref}")) + snr'
weights, res = fit_gees(long_mice, weighted = True, ref = ref, cov_struct=Independence(), formula=formula)#, fit_col="y_on")

print(res.summary())



#%%

set_fixed = None#{"snr": 0}
d = long_mice_changed.copy()
d = long_mice.copy()

emms = gu.pair_estimands(
    res, d, formula, PAIR_LEVELS,weights=None, ref_label=ref,
    averaging="mouse", set_fixed=set_fixed  
)

# 2) Contrasts vs reference (Holm-adjusted), with Cohen’s dz
contr = gu.contrasts_vs_reference(
    res, d, formula, PAIR_LEVELS,weights=None, ref_label=ref,
    averaging="mouse", set_fixed=set_fixed
)

#3) Omnibus Wald on probability scale (delta method, small-sample F)
omni = gu.wald_omnibus_pair_delta(
    res, d, formula, PAIR_LEVELS,weights=None, ref_label=ref,
    averaging="mouse", set_fixed=set_fixed,
    n_clusters=d["mouse"].nunique()
)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(emms)
    print(contr)
    print(omni)

#%%
fit_callable = lambda d: fit_gees(
    d, cov_struct=sm.cov_struct.Independence(),formula=formula, cov_type="robust", weighted=True, ref=ref
)
tst_ref = "ctx1_to_ctx2"
# 1) LOSO influence on the biological contrast (ctx→l1 vs l1→l2)
tab, summ = gu.loso_contrast_table(
    fit_callable, long_mice, formula,
    pair_alt="landmark1_to_landmark2", pair_ref=ref,
    averaging="mouse",weights=weights, set_fixed={"snr": 0}   
)
#%%
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(tab.head(10))   # sorted by absolute change
    print(summ)           # jackknife SE/CI and DFBETA threshold (2/sqrt(m))
    
    # 2) Per-mouse raw (unmodeled) contrast, to sanity-check the same outlier
    raw = gu.per_mouse_raw_contrast(long_mice, tst_ref, "landmark1_to_landmark2",
                                    resp_col="y_const", pair_col="pair", mouse_col="mouse")
    print(raw.sort_values("diff"))
#%%
fit_callable = lambda d: fit_gees(
    d,
    cov_struct=sm.cov_struct.Independence(),
    formula=formula,
    cov_type="bias_reduced",   # or "robust"
    weighted=True,
    ref="landmark1_to_landmark2"
)

out = gu.contrast_same_mice_refit_fixed(
    fit_callable,
    long_mice,
    formula,
    pair_alt="landmark1_to_landmark2",
    pair_ref="landmark2_to_ctx1",
    set_fixed={"snr": 0},      # <- fixed covariates here
    mouse_col="mouse",
)

out = gu.add_significance_to_contrast(out, alpha=0.05, small_sample=True)
print({
    "n_mice": out["n_mice"],
    "diff": out["diff"],
    "se": out["se"],
    "t(df)": (out["stat"], out["df"]),
    "p_t": out["p_t"],
    "ci95_t": out["ci95_t"],
})
