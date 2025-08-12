#%% imports

import sys
import yaml

import utils

import pandas as pd
import statsmodels.formula.api as smf

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
group_session_order = config["experiment"]["session_order"][0]

optimized_fname = config["filenames"]["cell_data_opt_template"]
pooled_cells_fname = config["filenames"]["pooled_cells"]

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [13,1]]

intensity_cols = [f'int_optimized_{x}' for x in group_session_order]
intensity_cols13 = [f'int_optimized_{x}' for x in group_session_order[:-1]]
#%%
dfs = []
for mouse, region in regions:
    df = utils.read_pooled_with_background(mouse, region, config)
    if mouse == 13:
        df = df[intensity_cols13+["detected_in_sessions"]]
    else:
        df = df[intensity_cols+["detected_in_sessions"]]
    df["Mouse"] = mouse
    dfs += [df]
    
df_concat = pd.concat(dfs)

#%%
df_long = (
    df_concat
    .reset_index()                          # ensure CellID is a column
    .rename(columns={'index':'CellID'})
    .melt(
        id_vars=['Mouse','CellID', 'detected_in_sessions'],
        value_vars=intensity_cols,
        var_name='Session',
        value_name='Intensity'
    )
)
#%%
def mark_for_removal(row):
    sid = row["Session"].split("_")[-1]
    return not sid in row['detected_in_sessions']

df_long["to_remove"] = df_long.apply(mark_for_removal, axis=1)
#%%
backup = df_long.copy()
df_long = df_long.loc[~df_long["to_remove"]].drop(columns=['to_remove', 'detected_in_sessions'])

#%%
df_long['Session'] = df_long['Session'].str.replace('int_optimized_','')
mapping = {
    's0':'s0',
    'landmark1':'landmark',
    'landmark2':'landmark',
    'ctx1':'ctx',
    'ctx2':'ctx'
}
df_long['SessionType'] = df_long['Session'].map(mapping)

df_long['Session'] = pd.Categorical(df_long['Session'],
                                    categories=['s0','landmark1','landmark2','ctx1','ctx2'],
                                    ordered=True)
df_long['SessionType'] = pd.Categorical(df_long['SessionType'],
                                        categories=['s0','landmark','ctx'],
                                        ordered=True)

#%%

model_A = smf.mixedlm("Intensity ~ Session",
                      data=df_long,
                      groups=df_long["Mouse"])

res_A = model_A.fit(reml=False)
print("AIC (Session):", res_A.aic, "BIC:", res_A.bic)

print(res_A.summary())

# 5. Model B: Intensity ~ SessionType + (1|Mouse)
model_B = smf.mixedlm("Intensity ~ SessionType",
                      data=df_long,
                      groups=df_long["Mouse"])
res_B = model_B.fit(reml=False)
print("AIC (SessionType):", res_B.aic, "BIC:", res_B.bic)
print(res_B.summary())

# print("Log‐likelihood A:", res_A.llf, "   converged:", res_A.mle_retvals['converged'])
# print("Log‐likelihood B:", res_B.llf, "   converged:", res_B.mle_retvals['converged'])


# 6. Compare AIC (lower is better) or do a likelihood‐ratio test
print("AIC Session:", res_A.aic, "AIC SessionType:", res_B.aic)
#%%
# Likelihood ratio test
lr_stat = 2 * (res_A.llf - res_B.llf)
df_diff = res_A.df_modelwc - res_B.df_modelwc
from scipy.stats import chi2
p_value = chi2.sf(lr_stat, df_diff)
print(f"LR test p-value: {p_value:.3g} (df={df_diff})")
