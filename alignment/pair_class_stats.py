#%% imports

import sys
import yaml

# custom modules
import intersession as its
import cell_preprocessing as cp
import cell_classification as cc
import utils
import plotting

import pandas as pd
import pingouin as pg

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
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1], [13,1]]

#%%
dfs = []
for mouse, region in regions:
    dfs += [utils.read_pooled_with_background(mouse, region, config)]
    
#%%
pairs = list(zip(group_session_order, group_session_order[1:]))

#%%
group_pct_df = cc.gather_group_percentages_across_mice(regions, pairs, config, groups=["on", "off", "const"], dfs = dfs)


#%%
aov = pg.rm_anova(dv='percentage',
                  within=['session_pair','group'],
                  subject='mouse_id',
                  data=group_pct_df,
                  detailed=True,
                  correction=True)  # adds GG/HF corrections
print(aov)
aov.to_csv("/mnt/data/fos_gfp_tmaze/results/multisession_nc/aov.csv")
#%%
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt

m = smf.mixedlm("pct ~ C(Pair)*C(Class)", data=group_pct_df, groups=group_pct_df["mouse_id"]).fit(reml=False)
resid = m.resid

sm.qqplot(resid, line="45")
plt.title("QQ plot of residuals (MixedLM)")
plt.show()

plt.figure()
plt.scatter(m.fittedvalues, resid, s=10, alpha=0.6)
plt.axhline(0, ls="--")
plt.xlabel("Fitted")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted (MixedLM)")
plt.show()