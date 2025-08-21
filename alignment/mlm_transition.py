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
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import stats

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


TRANSITIONS = [
    ("s0",        "landmark1", "S0_L1"),
    ("landmark1", "landmark2", "L1_L2"),
    ("landmark2", "ctx1",      "L2_C1"),
    ("ctx1",      "ctx2",      "C1_C2"),
]

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2], [5,2,], [6,1], [7,1], [13,1]]


#%%
# ---------- imports ----------

def _parse_detected(x):
    """Return a Python set of session names from different stored formats."""
    if isinstance(x, (set, list, tuple)):
        return set(map(str, x))
    if pd.isna(x):
        return set()
    s = str(x).strip()
    # strip braces/[] and quotes, split by comma
    s = s.strip("{}[]()")
    if not s:
        return set()
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    return set(p for p in parts if p)

def _in_session(det_field, session):
    try:
        return session in _parse_detected(det_field)
    except Exception:
        return False

# ---------- long builder (per mouse) ----------
def wide_to_long_one_mouse(df_mouse, mouse_id, mode="sub", eps=1e-6,
                           add_depth_metric=True, depth_col="z", lowess_frac=0.3):
    """
    Build long format with:
      - Fluor: intensity - background (mode='sub') or dF/F (mode='dff')
      - BG_z:  session-wise z-score of background within Mouse×Session
      - metric: depth-adjusted, session-standardized residual of Fluor (relative metric)
    Columns expected per session: int_optimized_{s}, background_{s}
    Optional columns: detected_in_sessions (string/set), z (depth)
    """
    dfm = df_mouse.reset_index(drop=True).copy()
    dfm["Cell"] = dfm.index.astype(int)
    dfm["Mouse"] = mouse_id

    # parsed detection
    det = dfm.get("detected_in_sessions", pd.Series([None]*len(dfm)))

    rows = []
    for s in SESSIONS:
        i_col = f"int_optimized_{s}"
        b_col = f"background_{s}"
        if i_col not in dfm.columns or b_col not in dfm.columns:
            continue

        I = dfm[i_col].astype(float)
        BG = dfm[b_col].astype(float)

        if mode == "sub":
            Fluor = I - BG
        elif mode == "dff":
            Fluor = (I - BG) / np.maximum(BG, eps)
        else:
            raise ValueError("mode must be 'sub' or 'dff'")

        present = pd.Series(det).apply(lambda x: _in_session(x, s)) | (~I.isna())

        sub = pd.DataFrame({
            "Mouse": mouse_id,
            "Cell": dfm.index.astype(int),
            "Session": s,
            "Fluor": Fluor,
            "BG": BG,
            "present": present.astype(bool),
            depth_col: dfm[depth_col] if depth_col in dfm.columns else np.nan
        })

        # standardize BG within this Mouse×Session on valid Fluor rows
        valid = sub["Fluor"].notna()
        mu = sub.loc[valid, "BG"].mean()
        sd = sub.loc[valid, "BG"].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            sub["BG_z"] = 0.0
        else:
            sub["BG_z"] = (sub["BG"] - mu) / sd

        rows.append(sub)

    long = pd.concat(rows, ignore_index=True)
    long = long[~long["Fluor"].isna()].copy()

    # depth-adjusted, session-standardized metric (relative, depth-safe)
    if add_depth_metric:
        def _depth_standardize(sub):
            y = sub["Fluor"].astype(float)
            z = sub[depth_col].astype(float)
            # symmetric transform that tolerates negatives; log1p on positives only
            y_trans = np.where(y >= 0, np.log1p(y), -np.log1p(-y))
            if len(sub) >= 10 and z.notna().sum() >= 8 and np.nanstd(z) > 0:
                fit = lowess(y_trans, z, frac=lowess_frac, return_sorted=False)
                resid = y_trans - fit
            else:
                resid = y_trans - np.nanmean(y_trans)
            # z-score within Mouse×Session
            m = np.nanmean(resid); s = np.nanstd(resid, ddof=1)
            sub["metric"] = (resid - m) / s if (s and np.isfinite(s)) else resid*0.0
            return sub

        long = long.groupby(["Mouse","Session"], group_keys=False).apply(_depth_standardize)
    else:
        long["metric"] = long["Fluor"].astype(float)

    return long

# ---------- pairwise Δ builder ----------
def build_deltas(long, transitions=TRANSITIONS, require_present_any=True, use_metric=True):
    """
    Return pairwise DataFrame with:
      dF: metric_B - metric_A (or Fluor_B - Fluor_A)
      bg_avg, bg_diff: from BG_z
    """
    out=[]
    val_col = "metric" if use_metric else "Fluor"

    for a,b,name in transitions:
        A = long.query("Session == @a")[["Mouse","Cell",val_col,"BG_z"]].rename(
            columns={val_col:"mA","BG_z":"BGz_A"})
        B = long.query("Session == @b")[["Mouse","Cell",val_col,"BG_z"]].rename(
            columns={val_col:"mB","BG_z":"BGz_B"})
        M = pd.merge(A, B, on=["Mouse","Cell"], how="inner")
        if require_present_any:
            M = M.dropna(subset=["mA","mB"], how="all")
        M["Pair"] = name
        M["dF"] = M["mB"] - M["mA"]
        M["bg_avg"]  = (M["BGz_A"] + M["BGz_B"]) / 2.0
        M["bg_diff"] =  M["BGz_B"] - M["BGz_A"]
        out.append(M[["Mouse","Cell","Pair","dF","bg_avg","bg_diff"]])
    return pd.concat(out, ignore_index=True)

# ---------- classify up/down/stable ----------
def add_trend_label(pairwise_df, k=1.0, by="Mouse"):
    """
    Label each row as down/stable/up using |dF| < tau as 'stable',
    tau = 1.4826 * MAD(dF) * k computed per 'by' group (default per-Mouse).
    """
    df = pairwise_df.copy()
    if by == "Mouse":
        grp = df.groupby("Mouse")["dF"]
        mad = grp.transform(lambda x: np.median(np.abs(x - np.median(x))))
    elif by == "MousePair":
        grp = df.groupby(["Mouse","Pair"])["dF"]
        mad = grp.transform(lambda x: np.median(np.abs(x - np.median(x))))
    else:
        raise ValueError("by must be 'Mouse' or 'MousePair'")
    tau = 1.4826 * mad * k
    lab = np.where(np.abs(df["dF"]) < tau, "stable",
                   np.where(df["dF"] > 0, "up", "down"))
    df["trend"] = pd.Categorical(lab, categories=["down","stable","up"], ordered=True)
    return df

# ---------- models & summaries ----------
def run_gee_models(pairwise_labeled):
    d = pairwise_labeled.copy()
    d["CellKey"] = d["Mouse"].astype(str) + "_" + d["Cell"].astype(str)

    gee_up = sm.GEE.from_formula(
        "I(trend=='up') ~ C(Pair) + bg_avg + bg_diff",
        groups="CellKey",
        data=d,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    ).fit()

    gee_down = sm.GEE.from_formula(
        "I(trend=='down') ~ C(Pair) + bg_avg + bg_diff",
        groups="CellKey",
        data=d,
        family=sm.families.Binomial(),
        cov_struct=sm.cov_struct.Exchangeable()
    ).fit()

    return gee_up, gee_down

def summary_tables(pairwise_labeled):
    # per transition fractions (per mouse and overall)
    fractions_mouse = (pairwise_labeled
        .groupby(["Mouse","Pair","trend"]).size()
        .groupby(level=[0,1]).apply(lambda s: s / s.sum())
        .reset_index(name="frac"))
    fractions_overall = (pairwise_labeled
        .groupby(["Pair","trend"]).size()
        .groupby(level=0).apply(lambda s: s / s.sum())
        .reset_index(name="frac"))
    # mixedness per cell
    cell_signs = (pairwise_labeled
        .assign(sign=lambda d: d["trend"].map({"down":-1,"stable":0,"up":+1}))
        .groupby(["Mouse","Cell"])
        .agg(n_up=("sign", lambda s: (s>0).sum()),
             n_down=("sign", lambda s: (s<0).sum()),
             n_stable=("sign", lambda s: (s==0).sum()))
        .assign(mixed=lambda r: (r.n_up>0) & (r.n_down>0))
        .reset_index())
    per_mouse_mixed = cell_signs.groupby("Mouse")["mixed"].mean()

    return fractions_mouse, fractions_overall, per_mouse_mixed

# ---------- USAGE ----------
# If you already have `long`, skip to build_deltas(...) below.
# Otherwise, prepare a dict of your per-mouse wide DataFrames:
# MICE = {"M1": df_mouse1, "M2": df_mouse2, ...}
#%%
#%%
dfs = []
for mouse, region in regions:
    df_mouse = utils.read_pooled_with_background(mouse, region, config)
    long_mouse = wide_to_long_one_mouse(df_mouse, mouse, mode="sub", add_depth_metric=True, depth_col='Center Z (px)')
    dfs += [long_mouse]
#%%    
long = pd.concat(dfs, ignore_index=True)
print("Built long:", long.shape, long.columns.tolist())
# else: assume `long` already exists in your workspace

# Build deltas on the depth-adjusted metric
pairwise = build_deltas(long, transitions=TRANSITIONS, require_present_any=True, use_metric=True)

# Label trends (k=1.0 is a good starting point; try 0.7 and 1.3 for sensitivity)
pairwise_lab = add_trend_label(pairwise, k=1.0, by="Mouse")

# Run GEE: Up vs not, Down vs not (clustered by Cell)
gee_up, gee_down = run_gee_models(pairwise_lab)
print("\n=== GEE: Up vs not (by transition) ===")
print(gee_up.summary())
print("\n=== GEE: Down vs not (by transition) ===")
print(gee_down.summary())

# Fractions & mixedness quick looks
fra_mouse, fra_overall, per_mouse_mixed = summary_tables(pairwise_lab)
print("\n=== Fraction up/down/stable per transition (overall) ===")
print(fra_overall.pivot(index="Pair", columns="trend", values="frac").fillna(0))

print("\n=== Mixedness (fraction of cells with both Up and Down across transitions), per mouse ===")
print(per_mouse_mixed.describe())
print("Wilcoxon vs 0.5 (H1 two-sided):", stats.wilcoxon(per_mouse_mixed - 0.5))
