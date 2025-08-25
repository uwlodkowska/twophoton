import sys
import yaml
import utils
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
from scipy.stats import f as fdist


import numpy as np, patsy as pt, re
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Autoregressive, Exchangeable
#%% config

res = [utils.get_concatenated_df_from_config("config_files/multisession.yaml", 0, "_multi"),
utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", 0, "_cll"),
utils.get_concatenated_df_from_config("config_files/ctx_landmark.yaml", 1, "_lcc")]

#%%
res
#%%
def parse_setlike(s: str) -> set:
    """Parse strings like "{'s0','landmark1'}" into a Python set."""
    try:
        return set(ast.literal_eval(str(s)))
    except Exception:
        s = str(s).strip().strip("{}")
        if not s:
            return set()
        return set(t.strip().strip("'").strip('"') for t in s.split(","))

def identity_from_session(s: str) -> str:
    """Map session name to identity code."""
    if s == "s0": return "S0"
    if s.startswith("landmark"): return "L"
    if s.startswith("ctx"): return "C"
    return s.upper()


def build_long_from_df(df: pd.DataFrame,
                       sessions_in_order: list[str],
                       mouse_col: str = "mouse",
                       cell_col: str  = "cell_id",
                       z_col: str     = "Center Z (px)",
                       ) -> pd.DataFrame:
    """
    Minimal: wide -> long (3 sessions), with session/CurrID/Prev2ID as L/C only.
    Uses existing *_rstd if present; no detrending/fallback is done here.
    """
    req_cols = {mouse_col, cell_col, z_col, "detected_in_sessions"}
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Map session -> available intensity columns (keep original names as-is)
    int_raw_map  = {s: (f"int_optimized_{s}" if f"int_optimized_{s}" in df.columns else None)
                    for s in sessions_in_order}
    int_rstd_map = {s: (f"int_optimized_{s}_rstd" if f"int_optimized_{s}_rstd" in df.columns else None)
                    for s in sessions_in_order}

    rows = []
    for idx, row in df.iterrows():
        sset = parse_setlike(row["detected_in_sessions"])
        z    = row.get(z_col, np.nan)
        mouse = row.get(mouse_col)
        cell  = row.get(cell_col)

        for p, s in enumerate(sessions_in_order):
            active   = int(s in sset)
            raw_col  = int_raw_map[s]
            rstd_col = int_rstd_map[s]

            intensity_raw = row.get(raw_col, np.nan) if raw_col else np.nan
            int_z_pre     = row.get(rstd_col, np.nan) if rstd_col else np.nan

            curr_id = identity_from_session(s)              # 'L' or 'C' (S0 never used here)
            rows.append({
                "mouse": mouse,
                "cell_id": cell,
                "session": curr_id,                         # <-- canonical L/C
                "Period": p,
                "CurrID": curr_id,                          # keep for convenience
                "active": active,
                "intensity_raw": intensity_raw,             # original naming preserved in value source
                "int_z": int_z_pre,                         # use precomputed *_rstd if present
                "z_px": z,
            })

    long = pd.DataFrame(rows).sort_values(["mouse","cell_id","Period"]).reset_index(drop=True)

    # First- and second-order lags within each cell (restricted to the 3-test window)
    g = long.groupby(["mouse","cell_id"], sort=False)
    long["prev_active"] = g["active"].shift(1).fillna(0).astype(int)
    long["PrevID"]      = g["CurrID"].shift(1)               # L/C for periods >=1, NaN at first
    long["Prev2ID"]     = g["CurrID"].shift(2)               # L/C for periods >=2, NaN otherwise

    # Keep only relevant columns
    keep = ["mouse","cell_id","session","Period","CurrID","PrevID","Prev2ID",
            "prev_active","active","intensity_raw","int_z","z_px"]
    return long[keep]




def fit_presence_triplet_gee_no_alias(d, include_intensity=False):
    base = d.copy()

    # carryover: L vs not-L (S0 and C both 0)
    base["prev_was_L"] = (base["PrevID"] == "L").astype(int)

    # pin categories for stable design
    base["Period"] = pd.Categorical(base["Period"], categories=[0,1,2], ordered=True)
    base["CurrID"] = pd.Categorical(base["CurrID"], categories=["C","L"])

    # cluster by mouse (your choice)
    groups = base["mouse"].astype(str)

    # model: order + current identity + carryover (L vs not-L) + prior state
    f = "active ~ C(Period) + C(CurrID) + prev_was_L + prev_active"
    if include_intensity and "int_z" in base.columns:
        base["cell_gid"] = base["mouse"].astype(str) + ":" + base["cell_id"].astype(str)
        base["int_z_c"] = base["int_z"] - base.groupby("cell_gid")["int_z"].transform("mean")
        f += " + int_z_c"

    y, X = pt.dmatrices(f, data=base, return_type="dataframe")
    gee = GEE(y.values.ravel(), X, groups=groups, family=Binomial(),
              cov_struct=Exchangeable()).fit(scale="X2")

    # Wald tests (order / identity / carryover)
    names = X.columns.tolist()
    idx_order = [i for i,n in enumerate(names) if n.startswith("C(Period")]
    idx_curr  = [i for i,n in enumerate(names) if n.startswith("C(CurrID")]
    idx_prevL = [names.index("prev_was_L")] if "prev_was_L" in names else []
    L_order   = np.eye(len(names))[idx_order]
    L_curr    = np.eye(len(names))[idx_curr]
    L_prevL   = np.eye(len(names))[idx_prevL] if idx_prevL else None

    out = {
        "gee": gee,
        "wald_order": gee.wald_test(L_order, use_f=False),
        "wald_identity": gee.wald_test(L_curr,  use_f=False),
    }
    if L_prevL is not None:
        out["wald_carryover_L"] = gee.wald_test(L_prevL, use_f=False)
    return out


#%%
dfs = []
for tmp in res:
    sessions, df = tmp[0]
    if "s0" in sessions:
        sessions = ['landmark1', 'landmark2', 'ctx1']
    dfs += [build_long_from_df(df,sessions)]
mega_df = pd.concat(dfs)
mega_df['PrevID'] = mega_df['PrevID'].fillna('S0')
#%%
gee_res = fit_presence_triplet_gee_no_alias(mega_df)
#%%

print(gee_res["gee"].summary())

for k, v in gee_res.items():
    if k != "gee":
        print(k, v , "\n")

#%%
mega_df.Period