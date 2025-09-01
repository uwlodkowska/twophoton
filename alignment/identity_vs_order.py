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
    base["prev_was_same"] = ((base["PrevID"].isin(["L","C"])) &
                         (base["PrevID"] == base["CurrID"])).astype(int)

    # pin categories for stable design
    base["Period"] = pd.Categorical(base["Period"], categories=[0,1,2], ordered=True)
    base["CurrID"] = pd.Categorical(base["CurrID"], categories=["C","L"])

    # cluster by mouse (your choice)
    groups = base["mouse"].astype(str)

    # model: order + current identity + carryover (L vs not-L) + prior state
    f = "active ~ C(Period) + C(CurrID) + prev_was_same + prev_active"
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
    idx_prev_same = [names.index("prev_was_same")] if "prev_was_same" in names else []
    L_order   = np.eye(len(names))[idx_order]
    L_curr    = np.eye(len(names))[idx_curr]
    L_prev_same   = np.eye(len(names))[idx_prev_same] if idx_prev_same else None

    out = {
        "gee": gee,
        "wald_order": gee.wald_test(L_order, use_f=False),
        "wald_identity": gee.wald_test(L_curr,  use_f=False),
    }
    if L_prev_same is not None:
        out["wald_carryover_same"] = gee.wald_test(L_prev_same, use_f=False)
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

#%% results translated
from gee_utils import (emm_period_currid_tables, wald_for_prefixes,
                       per_mouse_equalizing_weights)

# 0) prepare data (what you already have)
base = mega_df.copy()
base["prev_was_L"] = (base["PrevID"] == "L").astype(int)
base["Period"] = pd.Categorical(base["Period"], categories=[0,1,2], ordered=True)
base["CurrID"] = pd.Categorical(base["CurrID"], categories=["C","L"])

# 1) formula
f = "1 + C(Period) + C(CurrID) + prev_was_L + prev_active"

# 2) fit GEE (optionally equalize mice)
w = None#per_mouse_equalizing_weights(base, mouse_col="mouse")  # or None
RHS = "C(Period) + C(CurrID) + prev_was_L + prev_active"   # intercept is implicit
y, X = pt.dmatrices("active ~ " + RHS, data=base, return_type="dataframe")
gee = GEE(y.values.ravel(), X, groups=base["mouse"].astype(str),
          family=Binomial(), cov_struct=Exchangeable(),
          weights=w  # or omit to use unweighted
         ).fit(scale="X2")

print(gee.summary())

# 3) joint Walds (order, identity, carryover)
wald_order    = wald_for_prefixes(gee, ("C(Period",))
wald_identity = wald_for_prefixes(gee, ("C(CurrID",))
wald_carry    = wald_for_prefixes(gee, ("prev_was_L",))  # single-term test
print("Order:", wald_order)
print("Identity:", wald_identity)
print("Carryover L:", wald_carry)

# 4) EMM tables (mouse-balanced means with delta-method CIs)
tab_period, tab_curr = emm_period_currid_tables(gee, base, f,
                                                averaging="mouse",
                                                mouse_col="mouse",
                                                weights=w)
print(tab_period)  # P0,P1,P2 probs + CI + delta_vs_ref (P0)
print(tab_curr)    # C,L probs + CI + delta_vs_ref (C)
#%%
from gee_utils import emm_table_for_factor
f = "1 + C(Period) + C(CurrID) + prev_was_L + prev_active"
tab_p0 = emm_table_for_factor(
    gee, base.assign(prev_active=0), f,
    factor="Period", levels=[0,1,2], ref=0,
    averaging="mouse", mouse_col="mouse"
)
print(tab_p0)  # shows the ~10 pp drop cleanly

#%%
names = gee.model.exog_names
i1 = names.index("C(Period)[T.1]") if "C(Period)[T.1]" in names else names.index("C(Period, levels=[0, 1, 2])[T.1]")
i2 = names.index("C(Period)[T.2]") if "C(Period)[T.2]" in names else names.index("C(Period, levels=[0, 1, 2])[T.2]")
import numpy as np
R = np.zeros((1, len(names))); R[0, i2] = 1; R[0, i1] = -1
print("P2 vs P1:", gee.wald_test(R, use_f=False))
sub = base[(base["prev_active"]==0)].copy()
# keep Period 0/1/2; fit the same model (omit prev_active since it’s 0)
RHS = "C(Period) + C(CurrID) + prev_was_L"
y, X = pt.dmatrices("active ~ " + RHS, data=sub, return_type="dataframe")
gee_sub = GEE(y.values.ravel(), X, groups=sub["mouse"].astype(str),
              family=Binomial(), cov_struct=Exchangeable()).fit(scale="X2")
print(gee_sub.summary())
#%%

import numpy as np, pandas as pd
from scipy.special import expit
import matplotlib.pyplot as plt

# names, params, cov from your fitted GEE
names  = gee.model.exog_names
beta   = pd.Series(gee.params, index=names)
V      = pd.DataFrame(gee.cov_params(), index=names, columns=names)

# helper to get eta,se and p,CI for a given row
def pred_row(**vals):
    x = pd.Series(0.0, index=names); x["Intercept"]=1.0
    for k,v in vals.items():
        if k in names: x[k]=v
    eta = float(x @ beta)
    se  = float(np.sqrt(x @ V @ x))
    p   = expit(eta)
    lo, hi = expit(eta - 1.96*se), expit(eta + 1.96*se)
    return p, lo, hi

# build rows for P0,P1,P2 at Curr=C, prev_was_L=0, prev_active=0
rows = []
base = {"Intercept":1.0}
p0,lo0,hi0 = pred_row()
p1,lo1,hi1 = pred_row(**{"C(Period)[T.1]":1})
p2,lo2,hi2 = pred_row(**{"C(Period)[T.2]":1})
rows = pd.DataFrame({
    "Period":["P0","P1","P2"],
    "p":[p0,p1,p2],
    "lo":[lo0,lo1,lo2],
    "hi":[hi0,hi1,hi2],
})
# plot
fig, ax = plt.subplots(figsize=(4,3))
ax.bar(rows["Period"], rows["p"], width=0.6)
ax.vlines(rows["Period"], rows["lo"], rows["hi"])
for i,r in rows.iterrows():
    ax.text(i, r["p"]+0.02, f"{r['p']:.2f}", ha="center", va="bottom")
ax.set_ylim(0,1); ax.set_ylabel("Pr(presence)")
ax.set_title("Presence probability by period (GEE)")
plt.tight_layout(); plt.show()
