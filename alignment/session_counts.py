import sys
import yaml

# custom modules
import cell_classification as cc
import utils
import numpy as np, patsy as pt


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

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1], [13,1]]

#%%
dfs = []
for mouse, region in regions:
    dfs += [utils.read_pooled_with_background(mouse, region, config)]
    
#%%
pairs = list(zip(SESSIONS, SESSIONS[1:]))
#%%
def make_presence_long(df_cells):
    rows=[]
    for s in SESSIONS:
        rows.append(pd.DataFrame({
            "Mouse": df_cells["Mouse"],
            #"Cell":  df_cells.index if "Cell" not in df_cells.columns else df_cells["Cell"],
            "Session": s,
            "present": df_cells["detected_in_sessions"].apply(lambda v: utils.in_s(v, s)).astype(int)
        }))
    long = pd.concat(rows, ignore_index=True)
    return long
#%%
pres = make_presence_long(df_cells)                  # one row per cell×session
# per mouse × session: successes = #present, trials = total cells for that mouse
counts = (pres.groupby(["Mouse","Session"])["present"]
               .agg(count="sum", n_total="size")
               .reset_index())
counts["prop"] = counts["count"]/counts["n_total"]