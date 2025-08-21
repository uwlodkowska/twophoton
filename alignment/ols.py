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
    df_mouse['mouse'] = mouse
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

import numpy as np, pandas as pd, patsy as pt
from patsy import bs

def build_long_continuous(df, pairs, z_col='Center Z (px)'):
    rows = []
    for before, after, label in pairs:
        aI = f'int_optimized_{before}_rstd'
        bI = f'int_optimized_{after}_rstd'
        aB = f'background_{before}_rstd'
        bB = f'background_{after}_rstd'
        keep = df[[aI, bI, aB, bB]].notna().all(axis=1)
        is_bgr = df['detected_in_sessions'].apply(lambda x: not (utils.in_s(x, before) or utils.in_s(x, after)))
        is_persistent = df['n_sessions'] == 5
        keep = keep & ~(is_bgr) & ~(is_persistent)

        d = pd.DataFrame({
            'mouse': df.loc[keep, 'mouse'].values,
            'cell_id': df.loc[keep, 'cell_id'].values,
            'pair': label,
            'z': df.loc[keep, z_col].astype(float).values,
            # outcome: robust standardized change
            'y_cont': df.loc[keep, bI].values - df.loc[keep, aI].values,
            # covariates (same spirit as your GEE)
            'baseline_int': df.loc[keep, aI].values,
            'bg_mean': 0.5*(df.loc[keep, aB].values + df.loc[keep, bB].values),
            # (optional) background change if you want it instead of bg_mean:
            # 'bg_delta': df.loc[keep, bB].values - df.loc[keep, aB].values,
        })
        rows.append(d)

    long = pd.concat(rows, ignore_index=True)
    # Ensure pair is a clean categorical with your preferred reference level later in the formula
    long['pair'] = long['pair'].astype('category')
    return long
#%%
import statsmodels.formula.api as smf

def fit_ols_cluster(long_df, ref_level="landmark1_to_landmark2"):
    # Relevel pair
    long_df = long_df.copy()
    long_df['pair'] = long_df['pair'].cat.reorder_categories(
        [ref_level] + [c for c in long_df['pair'].cat.categories if c != ref_level],
        ordered=True
    )
    formula = 'y_cont ~ C(pair, Treatment(reference="@ref_level")) + baseline_int + bg_mean + bs(z, df=4)'
    # patsy can't see Python var in Treatment(...), so we inject ref level directly:
    formula = formula.replace('@ref_level', ref_level)

    ols = smf.ols(formula, data=long_df).fit(
        cov_type='cluster',
        cov_kwds={'groups': long_df['mouse'], 'use_correction': True}  # bias-corrected cluster SEs
    )
    return ols
#%%
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests

def lm_pair_contrasts(ols_res, ref="landmark1_to_landmark2"):
    names = list(ols_res.model.exog_names)
    beta  = pd.Series(ols_res.params, index=names)
    V     = pd.DataFrame(ols_res.cov_params(), index=names, columns=names)

    def x_for(pair):
        x = np.zeros(len(names)); 
        x[names.index('Intercept')] = 1.0
        for j, nm in enumerate(names):
            if nm.endswith(f"[T.{pair}]"): x[j] = 1.0
        # other covariates at 0 by default; you can plug in means if preferred
        return x

    pair_levels = [ref] + [n.split("[T.")[1][:-1] for n in names if n.startswith("C(pair")]
    x_ref = x_for(ref)

    rows = []
    for p in pair_levels:
        x = x_for(p)
        est = float(x @ beta)
        se  = float(np.sqrt(x @ V @ x))
        if p == ref:
            z = pval = np.nan
        else:
            diff = x - x_ref
            est = float(diff @ beta)
            se  = float(np.sqrt(diff @ V @ diff))
            z   = est / se
            from scipy.stats import norm
            pval = 2*(1 - norm.cdf(abs(z)))
        rows.append({"pair": p, "est_vs_ref": est, "SE": se, "z": z, "p": pval})
    out = pd.DataFrame(rows)
    mask = out['pair'] != ref
    out.loc[mask, 'p_holm'] = multipletests(out.loc[mask, 'p'].values, method='holm')[1]
    return out
#%%

from scipy.stats import norm

def lm_LL_vs_avg_LC_CC(ols_res, ref="landmark1_to_landmark2", LC="landmark2_ctx1", CC="ctx1_ctx2"):
    names = list(ols_res.model.exog_names)
    beta  = pd.Series(ols_res.params, index=names)
    V     = pd.DataFrame(ols_res.cov_params(), index=names, columns=names)

    def x_for(pair):
        x = np.zeros(len(names)); x[names.index('Intercept')] = 1.0
        for j, nm in enumerate(names):
            if nm.endswith(f"[T.{pair}]"): x[j] = 1.0
        return x

    x_ref = x_for(ref); x_LC = x_for(LC); x_CC = x_for(CC)
    x_avg = 0.5*(x_LC + x_CC)
    c = x_ref - x_avg
    est = float(c @ beta)
    se  = float(np.sqrt(c @ V @ c))
    z   = est / se
    p   = 2*(1 - norm.cdf(abs(z)))
    return {"contrast": "LL - avg(LC,CC)", "est": est, "SE": se, "z": z, "p": p}
#%%

pairs = [('s0','landmark1','s0_landmark1'),
          ('landmark1','landmark2','landmark1_to_landmark2'),
          ('landmark2','ctx1','landmark2_ctx1'),
          ('ctx1','ctx2','ctx1_ctx2')]
#%%

long_cont = build_long_continuous(all_mice, pairs, z_col='Center Z (px)')
ols = fit_ols_cluster(long_cont, ref_level='landmark1_to_landmark2')
print(ols.summary())

print(lm_pair_contrasts(ols, ref='landmark1_to_landmark2'))
print(lm_LL_vs_avg_LC_CC(ols))
