#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 00:39:28 2025

@author: ula
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
import re

def _prep(df, direction):
    d = df[df['direction'] == direction].copy()
    d = d.sort_values('sd cutoff')
    return d

def plot_effects_vs_cutoff(csv_path, save_path=None):
    """
    Expects the CSV produced by sweep_gee_to_csv (probability scale).
    Produces a 2-row figure: UP on top, DOWN below.
    Plots per-pair effects with 95% CI ribbons and the LC/CC average.
    """
    df = pd.read_csv(csv_path)
    pairs = [
        ("s0_l1_coeff","s0_l1_CI_low","s0_l1_CI_hi","s0_l1"),
        ("l2_c1_coeff","l2_c1_CI_low","l2_c1_CI_hi","l2_c1"),
        ("c1_c2_coeff","c1_c2_CI_low","c1_c2_CI_hi","c1_c2"),
    ]
    # average curve
    avg_cols = ("avg_coeff","avg_CI_low","avg_CI_hi","avg")

    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
    for ax, direction, title in zip(
        axes, ["UP","DOWN"], ["UP (fraction of cells)", "DOWN (fraction of cells)"]
    ):
        d = _prep(df, direction)
        x = d["sd cutoff"].values

        # per-pair curves
        for ycol, lcol, hcol, label in pairs:
            y = d[ycol].values
            lo = d[lcol].values
            hi = d[hcol].values
            ax.plot(x, y, label=label)
            ax.fill_between(x, lo, hi, alpha=0.2, linewidth=0)

        # avg(LC,CC) curve (dashed)
        y = d[avg_cols[0]].values
        lo = d[avg_cols[1]].values
        hi = d[avg_cols[2]].values
        ax.plot(x, y, linestyle="--", label=avg_cols[3])
        ax.fill_between(x, lo, hi, alpha=0.15, linewidth=0)

        # reference band (optional): visualize ref_prob CI as a horizontal region
        if {"ref_prob","ref_CI_low","ref_CI_hi"}.issubset(d.columns):
            # take the value at the *lenient* cutoff as anchor (first row)
            r_lo = d["ref_CI_low"].iloc[0]
            r_hi = d["ref_CI_hi"].iloc[0]
            ax.axhspan(r_lo, r_hi, alpha=0.08)
            ax.axhline(d["ref_prob"].iloc[0], linestyle=":", linewidth=1)

        # cosmetics
        ax.set_title(title)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        # annotate sample size per point (optional)
        if "size" in d.columns:
            for xi, yi, n in zip(x, y, d["size"].values):
                ax.annotate(str(int(n)), (xi, yi), xytext=(0, 8), textcoords="offset points",
                            fontsize=8, ha="center")

        ax.legend(loc="best", frameon=False)

    axes[-1].set_xlabel("SD cutoff")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, axes

def plot_rho_vs_cutoff(csv_path, save_path=None):
    """
    Optional panel if you also store 'rho_up' and 'rho_down' in your CSV.
    If not present, this function will no-op.
    """
    df = pd.read_csv(csv_path)
    if not {"rho_up","rho_down","sd cutoff"}.issubset(df.columns):
        print("rho columns not found; skipping rho plot.")
        return None, None

    d = df.drop_duplicates(subset=["sd cutoff"]).sort_values("sd cutoff")
    x = d["sd cutoff"].values
    fig, ax = plt.subplots(figsize=(6, 3))
    if "rho_up" in d.columns:
        ax.plot(x, d["rho_up"].values, marker="o", label="rho_up")
    if "rho_down" in d.columns:
        ax.plot(x, d["rho_down"].values, marker="s", label="rho_down")
    ax.set_xlabel("SD cutoff")
    ax.set_ylabel("Estimated within-mouse correlation (ρ̂)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/mnt/data/fos_gfp_tmaze/results/gee_transition/active_in_both.csv"
LENIENT_CUTOFF = 0.5  # pick your “main” cutoff

PAIR_SPECS = [
    ("s0_l1", "s0→L1"),
    ("l2_c1", "L2→C1"),
    ("ref", "L1→L2"),
    ("c1_c2", "C1→C2"),
]

def _rows_at_cutoff(df, direction, cutoff):
    d = df[(df["direction"] == direction) & (df["sd cutoff"] == cutoff)].copy()
    if d.empty:
        raise ValueError(f"No rows for {direction} at cutoff {cutoff}")
    # if multiple rows per cutoff (e.g., multiple thresholds per direction), take first
    return d.iloc[0]

def plot_forest_at_cutoff(df, direction, cutoff, save_path=None, title_suffix=""):
    row = _rows_at_cutoff(df, direction, cutoff)

    ylabels, probs, lo, hi, pvals, pholm = [], [], [], [], [], []
    for key, label in PAIR_SPECS:
        ylabels.append(label)
        probs.append(row[f"{key}_coeff"])
        lo.append(row[f"{key}_CI_low"])
        hi.append(row[f"{key}_CI_hi"])
        pvals.append(row[f"{key}_p"])
        pholm.append(row[f"{key}_Holm"])

    y = np.arange(len(ylabels))[::-1]  # top-to-bottom
    probs = np.array(probs); lo = np.array(lo); hi = np.array(hi)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.hlines(y, lo, hi)           # CI bars
    ax.plot(probs, y, "o")         # point estimates
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Probability")
    ax.set_xlim(0, 0.1)
    ax.set_title(f"{direction} @ {cutoff} SD {title_suffix}".strip())

    # annotate p and Holm
    for yi, pv, ph in zip(y, pvals, pholm):
        txt = f"p={pv:.3f}, Holm={ph:.3f}"
        ax.annotate(txt, xy=(1.0, yi), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8)

    # optional: reference probability band
    if all(k in row for k in ["ref_coeff","ref_CI_low","ref_CI_hi"]):
        ax.axvspan(row["ref_CI_low"], row["ref_CI_hi"], alpha=0.08)
        ax.axvline(row["ref_prob"], linestyle=":", linewidth=1)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

def plot_p_vs_cutoff(df, direction, save_path=None):
    d = df[df["direction"] == direction].copy().sort_values("sd cutoff")
    x = d["sd cutoff"].values

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for key, label in PAIR_SPECS:
        y = d[f"{key}_p"].values
        ax.plot(x, y, marker="o", label=f"{label} (raw p)")
        y_h = d[f"{key}_Holm"].values
        ax.plot(x, y_h, linestyle="--", marker=".", label=f"{label} (Holm)")

    # avg block
    if {"avg_p","avg_Holm"}.issubset(d.columns):
        ax.plot(x, d["avg_p"].values, marker="s", label="avg (raw p)")
        ax.plot(x, d["avg_Holm"].values, linestyle="--", marker="s", label="avg (Holm)")

    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xlabel("SD cutoff")
    ax.set_ylabel("p-value")
    ax.set_title(f"{direction}: p vs cutoff")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

def plot_prob_vs_cutoff(df, direction, save_path=None):
    d = df[df["direction"] == direction].copy().sort_values("sd cutoff")
    x = d["sd cutoff"].values
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    for key, label in PAIR_SPECS:
        y  = d[f"{key}_coeff"].values
        lo = d[f"{key}_CI_low"].values
        hi = d[f"{key}_CI_hi"].values
        ax.plot(x, y, label=label)
        ax.fill_between(x, lo, hi, alpha=0.2, linewidth=0)

    # avg curve (dashed)
    if {"avg_coeff","avg_CI_low","avg_CI_hi"}.issubset(d.columns):
        y  = d["avg_coeff"].values
        lo = d["avg_CI_low"].values
        hi = d["avg_CI_hi"].values
        ax.plot(x, y, linestyle="--", label="avg(LC,CC)")
        ax.fill_between(x, lo, hi, alpha=0.15, linewidth=0)

    ax.set_xlabel("SD cutoff")
    ax.set_ylabel("Probability")
    ax.set_title(f"{direction}: probability vs cutoff")
    ax.set_ylim(0, 0.31)
    ax.legend(frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------- helpers ---------
PAIR_SPECS = [("s0_l1","s0→L1"), ("l2_c1","L2→C1"), ("c1_c2","C1→C2"), ("ref", "REF")]

def _row_at_cutoff(df, direction, cutoff):
    print(df["sd cutoff"])
    d = df[(df["direction"] == direction) & (df["sd cutoff"] == cutoff)]
    if d.empty:
        raise ValueError(f"No rows for direction={direction} at cutoff={cutoff}")
    return d.iloc[0]

# --------- Panel A: pair effect at chosen cutoff ---------
def _row_at_cutoff(df: pd.DataFrame, direction: str, cutoff: float) -> pd.Series:
    d = df[df["direction"] == direction]
    if d.empty:
        raise ValueError(f"No rows for direction='{direction}' in {set(df['direction'])}")
    if "sd cutoff" not in d.columns:
        # try legacy column name variations
        for alt in ["sd_cutoff", "sd", "cutoff"]:
            if alt in d.columns:
                d = d.rename(columns={alt: "sd cutoff"})
                break
    idx = (d["sd cutoff"] - cutoff).abs().idxmin()
    return d.loc[idx]

def fig_pair_effect_at_cutoff(
    csv_path: str,
    direction: str,
    cutoff: float,
    per_mouse_overlay: dict = None,   # keys by canonical label (e.g., 'landmark_to_ctx') or legacy keys
    per_mouse_props: dict  = None,    # optional: same keying as overlay, values are lists per mouse
    ref_label: str  = None,           # e.g. 'landmark_to_landmark' (canonical) or a legacy key
    title: str  = None,
    save_path: str  = None,
    xlim: tuple[float,float]  = None,
    canonical: bool = True,           # use canonical display by default for pooled experiments
    label_order: list[str] = None,    # explicit order of pair labels (canonical or legacy)
    pretty_map: dict[str,str] = None  # optional pretty names per label
):
    """
    Makes a horizontal forest-style plot for all pair levels present in the CSV row:
      - red dot + thick bar: GEE marginal prob ±95% CI
      - thin blue bar + hollow square: per-mouse mean ±95% CI (if provided)
      - grey dots: per-mouse proportions (if provided)
      - shaded band + dotted line: reference level’s CI and mean (if ref_label given)
    """
    df = pd.read_csv(csv_path)
    row = _row_at_cutoff(df, direction, cutoff)

    # ---- discover pair labels from CSV columns ----
    # columns like: 'pair_<label>_coeff', 'pair_<label>_CI_low', 'pair_<label>_CI_hi'
    pair_labels = []
    for c in row.index:
        m = re.match(r"^pair_(.+)_coeff$", str(c))
        if m:
            pair_labels.append(m.group(1))
    pair_labels = sorted(set(pair_labels))

    # for legacy CSVs that used fixed short keys like 's0_l1_coeff'
    if not pair_labels:
        m2 = [re.match(r"^(s0_l1|l2_c1|c1_c2)_coeff$", str(c)) for c in row.index]
        pair_labels = [m.group(1) for m in m2 if m]

    if not pair_labels:
        raise ValueError("Could not find any 'pair_<label>_coeff' columns in the CSV row.")

    # optional label canonicalization (display side only)
    if canonical:
        disp_labels = [utils.canonical_from_pair_label(lab) for lab in pair_labels]
    else:
        disp_labels = pair_labels[:]  # keep as-is

    # order labels for plotting
    if label_order:
        # keep only those present, in the requested order
        ordered = [lab for lab in label_order if lab in disp_labels]
    else:
        ordered = sorted(disp_labels)

    # build mapping from display label -> slug column prefix and vice versa
    # slugs in CSV are *raw* labels (pre-canonical) under 'pair_<raw>_...'
    disp_to_slug = {}
    for raw in pair_labels:
        disp = utils.canonical_from_pair_label(raw) if canonical else raw
        disp_to_slug[disp] = utils.slug_for_cols(raw)  # 'pair_<raw>'

    # ---- collect GEE layer from CSV (by display order) ----
    pair_labels_pretty = []
    gee_prob, gee_lo, gee_hi = [], [], []
    for disp in ordered:
        slug = disp_to_slug[disp]  # 'pair_<raw>'
        pair_labels_pretty.append(pretty_map.get(disp, disp) if pretty_map else disp.replace("_", "→") if "_to_" in disp else disp)
        gee_prob.append(float(row.get(f"{slug}_coeff", np.nan)))
        gee_lo.append(float(row.get(f"{slug}_CI_low", np.nan)))
        gee_hi.append(float(row.get(f"{slug}_CI_hi",  np.nan)))


    pair_labels_pretty.append("ref")
    slug="ref"
    gee_prob.append(float(row.get(f"{slug}_coeff", np.nan)))
    gee_lo.append(float(row.get(f"{slug}_CI_low", np.nan)))
    gee_hi.append(float(row.get(f"{slug}_CI_hi",  np.nan)))

    y_positions = np.arange(len(ordered)+1)[::-1] 
    gee_prob = np.asarray(gee_prob, float)
    gee_lo   = np.asarray(gee_lo, float)
    gee_hi   = np.asarray(gee_hi, float)
    print(y_positions)
    print(gee_prob)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # Reference overlay (if provided and present)
    if ref_label is not None:
        ref_disp = utils.canonical_from_pair_label(ref_label) if canonical else ref_label

        rslug = "ref"
        ref_p  = float(row.get(f"{rslug}_coeff", np.nan))
        ref_lo = float(row.get(f"{rslug}_CI_low", np.nan))
        ref_hi = float(row.get(f"{rslug}_CI_hi",  np.nan))
        if np.isfinite(ref_p) and np.isfinite(ref_lo) and np.isfinite(ref_hi):
            ax.axvspan(ref_lo, ref_hi, color="0.85", alpha=0.6, zorder=0, label="Ref 95% CI")
            ax.axvline(ref_p, linestyle=":", color="0.4", linewidth=1.2, zorder=1)

    # Optional: per-mouse raw points
    if per_mouse_props:
        for yi, disp in zip(y_positions, ordered):
            vals = per_mouse_props.get(disp) or per_mouse_props.get(disp_to_slug[disp])  # tolerate either keying
            if vals:
                vals = np.asarray(vals, float)
                yjit = yi + (np.random.rand(len(vals)) - 0.5) * 0.16
                ax.plot(vals, yjit, ".", alpha=0.45, color="0.5",
                        label="Mice (props)" if yi == y_positions[0] else None, zorder=2)

    # Optional: per-mouse mean ± CI
    if per_mouse_overlay:
        for yi, disp in zip(y_positions, ordered):
            v = per_mouse_overlay.get(disp) or per_mouse_overlay.get(disp_to_slug[disp])
            if v:
                mean_pm, lo_pm, hi_pm = v
                if np.isfinite(lo_pm) and np.isfinite(hi_pm):
                    ax.hlines(yi, lo_pm, hi_pm, linewidth=1.4, color="C0",
                              label="Across-mouse mean ±95% CI" if yi == y_positions[0] else None, zorder=3)
                if np.isfinite(mean_pm):
                    ax.plot([mean_pm], [yi], marker="s", mfc="white", mec="C0", mew=1.2, zorder=4)

    # GEE marginal: thick bar + red dot
    ax.hlines(y_positions, gee_lo, gee_hi, linewidth=3.0, color="C3", zorder=5)
    ax.plot(gee_prob, y_positions, "o", color="C3", ms=6, label="GEE EMM ±95% CI", zorder=6)

    # Cosmetics
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pair_labels_pretty)

    # x-lims
    if xlim is None:
        x_all = np.concatenate([gee_lo, gee_hi, gee_prob])
        if ref_label is not None and (ref_disp in ordered):
            x_all = np.append(x_all, [ref_p, ref_lo, ref_hi])
        if per_mouse_overlay:
            for (m, l, h) in per_mouse_overlay.values():
                x_all = np.append(x_all, [m, l, h])
        if per_mouse_props:
            for v in per_mouse_props.values():
                x_all = np.append(x_all, v)
        xmin = max(0.0, np.nanmin(x_all) - 0.02)
        xmax = min(1.0, np.nanmax(x_all) + 0.02)
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            xmin, xmax = 0.0, 1.0
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Probability")
    ax.set_title(title or f"EMM — {direction}, cutoff {cutoff} rSD")

    # Simple legend (dedupe, fixed order)
    handles, labels = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for pref in ["Ref 95% CI", "Mice (props)", "Across-mouse mean ±95% CI", "GEE EMM ±95% CI"]:
        for h, lab in zip(handles, labels):
            if lab == pref and lab not in seen:
                seen.add(lab); H.append(h); L.append(lab)
    for h, lab in zip(handles, labels):
        if lab and lab not in seen:
            seen.add(lab); H.append(h); L.append(lab)
    if H:
        ax.legend(H, L, frameon=False, loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax





# --------- Panel B: fragility — p vs cutoff (per pair + global Wald) ---------
def fig_fragility_p_vs_cutoff(csv_path, direction, y_max = 1, save_path=None, title=None, ax=None):
    df = pd.read_csv(csv_path)
    d = df[df["direction"] == direction].copy().sort_values("sd cutoff")
    x = d["sd cutoff"].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 3.6))
        owns_fig = True
    else:
        fig = ax.figure
    PAIRS = [("s0_l1","s0→L1"), ("l2_c1","L2→C1"), ("c1_c2","C1→C2")]
    # per-pair raw p and Holm
    for key, lab in PAIRS:
        #ax.plot(x, d[f"{key}_p_prob"].values, marker="o", label=f"{lab} (p)")
        ax.plot(x, d[f"{key}_Holm_prob"].values,  marker=".", label=f"{lab} (Holm)")

    # global (Wald) p across all pairs (helps readers see overall sensitivity)
    if "group wald p" in d.columns:
        ax.plot(x, d["group wald p"].values, marker="s", linewidth=2, label="omnibus")

    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xlabel("Próg rSD")
    ax.set_ylabel("p-wartość")
    if title is None:
        title = f"{direction} — wrażliwość na próg (p vs cutoff)"
    ax.set_title(title)
    ax.set_ylim(0, y_max)
    ax.legend(frameon=False, ncol=2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax

import numpy as np
import pandas as pd
from scipy.stats import t

def build_per_mouse_overlay(
    long_df: pd.DataFrame,
    mouse_col: str,           # e.g. "mouse"
    pair_col: str,            # e.g. "pair" (values like "landmark2_to_ctx1", ...)
    outcome_col: str,         # e.g. "y_up" or "y_down" (binary 0/1)
    pair_key_map: dict = None,# legacy: map raw pair values -> short keys for 5-session plots
    canonical: bool = False,  # if True, aggregate by canonical pairs ('landmark_to_ctx', ...)
    pair_order: list[str] = None  # optional order of labels to return
):
    """
    Returns: dict {plot_key: (mean_prop_across_mice, CI_low, CI_high)}.

    Behavior:
      - If canonical=True: pairs are collapsed to canonical labels automatically.
      - Else if pair_key_map is provided: keys come from pair_key_map (legacy 5-session).
      - Else: keys are the unique raw pair values in pair_col.
    """
    d = long_df[[mouse_col, pair_col, outcome_col]].dropna().copy()

    if canonical:
        d["_plot_key"] = d[pair_col].astype(str).map(utils.canonical_from_pair_label)
    elif pair_key_map is not None:
        d["_plot_key"] = d[pair_col].map(pair_key_map).astype("object")
    else:
        d["_plot_key"] = d[pair_col].astype(str)

    # remove rows we couldn't map
    d = d[~d["_plot_key"].isna()].copy()

    # per-mouse mean within each (pair key)
    pm = (
        d.groupby(["_plot_key", mouse_col], observed=True)[outcome_col]
         .mean()
         .rename("prop")
    )

    agg = (
        pm.reset_index()
          .groupby("_plot_key", observed=True)["prop"]
          .agg(["mean", "std", "count"])
    )

    # t-based CI across mice
    dfree = agg["count"] - 1
    se = agg["std"] / np.sqrt(agg["count"])
    tcrit = t.ppf(0.975, dfree.replace(0, np.nan))
    ci_lo = agg["mean"] - tcrit * se
    ci_hi = agg["mean"] + tcrit * se

    # choose output order
    keys = list(agg.index.astype(str))
    if pair_order:
        # keep only those we have, in requested order
        keys = [k for k in pair_order if k in keys]
    else:
        keys.sort()

    out = {}
    for key in keys:
        m = float(agg.loc[key, "mean"])
        lo = float(ci_lo.loc[key]) if np.isfinite(ci_lo.loc[key]) else np.nan
        hi = float(ci_hi.loc[key]) if np.isfinite(ci_hi.loc[key]) else np.nan
        out[key] = (m, lo, hi)
    return out



#%%


PAIRS = [('s0_l1','S0→L1'), ('l2_c1','L2→C1'), ('c1_c2','C1→C2')]

def _load_table(csv_path, direction, what_cells='active_in_both'):
    d = pd.read_csv(csv_path)
    m = (d['direction'] == direction)
    if 'what cells' in d.columns and what_cells is not None:
        m &= (d['what cells'] == what_cells)
    d = d.loc[m].copy().sort_values('sd cutoff')
    if d.empty:
        raise ValueError("No rows for the requested direction/what_cells.")
    return d

def _detect_count_columns(df, explicit=None):
    """Return (median, min, max) column names or None if not found."""
    if explicit is not None:
        if all(c in df.columns for c in explicit):
            return explicit
        raise ValueError(f"Count columns {explicit} not found. Available: {list(df.columns)}")
    candidates = [
        ('eligible_cells_median','eligible_cells_min','eligible_cells_max'),
        ('eligible_median','eligible_min','eligible_max'),
        ('cells_median','cells_min','cells_max'),
        ('per_mouse_median','per_mouse_min','per_mouse_max'),
        ('n_med','n_min','n_max'),
        ('med_cells','min_cells','max_cells'),
    ]
    for trio in candidates:
        if all(c in df.columns for c in trio):
            return trio
    return None

def plot_p_vs_cutoff_with_counts(csv_path, direction, what_cells='active_in_both',
                                 use_holm=True, y_max=1.0, ref_vlines=(0.75, 1.0),
                                 count_cols=None, title=None):
    """
    Left axis: p-values across cutoffs for each pair (+ omnibus).
    Right axis: eligible cells per mouse (median with min–max band).
    """
    d = _load_table(csv_path, direction, what_cells)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))

    # ---- p-value lines (left axis)
    suffix = 'Holm_prob' if use_holm else 'p_prob'
    for key, label in PAIRS:
        col = f'{key}_{suffix}'
        if col in d:
            ax.plot(d['sd cutoff'], d[col], marker='o', linestyle='-', label=label)

    if 'group wald p' in d:
        ax.plot(d['sd cutoff'], d['group wald p'], marker='s', linestyle='--',
                linewidth=2, label='Omnibus Wald')

    for v in (ref_vlines or []):
        ax.axvline(float(v), linestyle='--', linewidth=1, alpha=0.6)

    ax.axhline(0.05, linestyle=':', linewidth=1)
    ax.set_ylim(0, y_max)
    ax.set_xlabel('Próg (rSD)')
    ax.set_ylabel('p-wartość')

    # ---- counts panel (right axis)
    cols = _detect_count_columns(d, explicit=count_cols)
    ax2 = None
    if cols is not None:
        med_col, min_col, max_col = cols
        counts = (d[['sd cutoff', med_col, min_col, max_col]]
                  .groupby('sd cutoff', as_index=False).first())  # if repeated rows
        ax2 = ax.twinx()
        ax2.plot(counts['sd cutoff'], counts[med_col], marker='D', linestyle='--',
                 alpha=0.9, label='Mediana #kom./mysz')
        ax2.fill_between(counts['sd cutoff'], counts[min_col], counts[max_col],
                         alpha=0.12, label='[min, max]')
        ax2.set_ylabel('Liczba kwalif. komórek / mysz')
        ax2.set_ylim(bottom=0)

        # merged legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1+h2, l1+l2, frameon=False, ncol=2)

    if title is None:
        title = f'{direction}: p vs próg ({"Holm" if use_holm else "surowe"})'
    ax.set_title(title)
    plt.tight_layout()
    return (fig, ax, ax2)



def plot_emm_vs_cutoff(csv_path, direction, what_cells='active_in_both',
                       ref_vlines=(0.75, 1.0), title=None):
    """
    EMM probabilities with 95% CIs across cutoffs for each pair.
    """
    print(csv_path, what_cells)
    d = _load_table(csv_path, direction, what_cells)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    for key, label in PAIRS:
        coeff = f'{key}_coeff'; lo = f'{key}_CI_low'; hi = f'{key}_CI_hi'
        if coeff in d and lo in d and hi in d:
            x = d['sd cutoff'].to_numpy()
            y = d[coeff].to_numpy()
            ylo = d[lo].to_numpy(); yhi = d[hi].to_numpy()
            err = [y - ylo, yhi - y]
            ax.errorbar(x, y, yerr=err, fmt='o-', capsize=3, label=label)

    # (optional) show reference pair (L1→L2) if present
    if {'ref_coeff','ref_CI_low','ref_CI_hi'}.issubset(d.columns):
        ax.plot(d['sd cutoff'], d['ref_coeff'], linestyle=':', linewidth=1.5, label='REF (L1→L2)')

    for v in ref_vlines or []:
        ax.axvline(float(v), linestyle='--', linewidth=1, alpha=0.6)

    ax.set_xlabel('Próg (rSD)')
    ax.set_ylabel('Prawdopodobieństwo (EMM)')
    ax.set_ylim(0, 0.3)
    if title is None:
        title = f'{direction}: EMM vs próg'
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    return fig, ax



def plot_counts_by_cutoff_per_pair(
    csv_path: str,
    direction: str = "UP",
    pairs: list = None,
    title: str = None,
    show_bands: bool = True
):
    """
    Build one plot with a line per session pair across all cutoffs.
    For each pair & cutoff:
      n_min = CI_low * cluster_min (floored)
      n_mid = prob   * cluster_mean (rounded)
      n_max = CI_hi  * cluster_max (ceiled)

    Expected columns in CSV:
      'sd cutoff', 'direction', 'cluster_min/mean/max' and per-pair fields:
        s0_l1_p_prob, s0_l1_CI_low, s0_l1_CI_hi
        c1_c2_p_prob, c1_c2_CI_low, c1_c2_CI_hi
        l2_c1_p_prob, l2_c1_CI_low, l2_c1_CI_hi
        ref_coeff,   ref_CI_low,   ref_CI_hi
    """
    df = pd.read_csv(csv_path)

    if pairs is None:
        pairs = ["ref","s0l1","c1c2","l2c1"]

    stem_map = {
        "s0l1": ("s0_l1_coeff", "s0_l1_CI_low", "s0_l1_CI_hi"),
        "c1c2": ("c1_c2_coeff", "c1_c2_CI_low", "c1_c2_CI_hi"),
        "l2c1": ("l2_c1_coeff", "l2_c1_CI_low", "l2_c1_CI_hi"),
        "ref":  ("ref_coeff",    "ref_CI_low",   "ref_CI_hi"),  # prob-scale mid from ref_coeff
    }

    # Coerce numerics (robust against string-typed numbers)
    num_cols = ['sd cutoff','cluster_min','cluster_mean','cluster_max',
                'ref_coeff','ref_CI_low','ref_CI_hi',
                's0_l1_p_prob','s0_l1_CI_low','s0_l1_CI_hi',
                'c1_c2_p_prob','c1_c2_CI_low','c1_c2_CI_hi',
                'l2_c1_p_prob','l2_c1_CI_low','l2_c1_CI_hi']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filter by UP/DOWN
    d = df[df["direction"] == direction].copy()
    if d.empty:
        raise ValueError(f"No rows for direction={direction!r}")

    # Build long-form table with counts per pair×cutoff
    rows = []
    for key in pairs:
        if key not in stem_map:
            raise ValueError(f"Unknown pair key: {key}")
        prob_col, lo_col, hi_col = stem_map[key]
        for c in [prob_col, lo_col, hi_col]:
            if c not in d.columns:
                raise ValueError(f"Missing column for pair {key}: {c}")
        sub = d[["sd cutoff","cluster_min","cluster_mean","cluster_max",
                 prob_col, lo_col, hi_col]].copy()
        sub = sub.rename(columns={
            "sd cutoff": "cutoff", prob_col: "prob", lo_col: "ci_low", hi_col: "ci_hi"
        })
        sub["n_min"] = np.floor(sub["ci_low"].clip(lower=0) * sub["cluster_min"]).astype("Int64")
        sub["n_mid"] = (sub["prob"].clip(lower=0) * sub["cluster_mean"]).round().astype("Int64")
        sub["n_max"] = np.ceil(sub["ci_hi"].clip(lower=0) * sub["cluster_max"]).astype("Int64")
        sub["pair"]  = key
        rows.append(sub)

    long = pd.concat(rows, ignore_index=True).sort_values(["pair","cutoff"])

    # Plot
    fig, ax = plt.subplots(figsize=(9, 6))
    color_map = {"ref":"#1f77b4","s0l1":"#ff7f0e","c1c2":"#2ca02c","l2c1":"#d62728"}  # distinct colors by request

    for key, grp in long.groupby("pair"):
        grp = grp.sort_values("cutoff")
        x = grp["cutoff"].astype(float).values
        y = grp["n_mid"].astype(float).values
        if show_bands:
            ylo = grp["n_min"].astype(float).values
            yhi = grp["n_max"].astype(float).values
            ax.fill_between(x, ylo, yhi, alpha=0.18, label=f"{key} min..max", color=color_map.get(key))
        ax.plot(x, y, marker="o", linewidth=2, label=f"{key} mid", color=color_map.get(key))

    ax.set_xlabel("Cutoff (SD)")
    ax.set_ylabel("Estimated # cells in class")
    ax.set_title(title or f"Counts vs cutoff per pair — direction: {direction}")
    ax.grid(True, alpha=0.3)

    # Compact legend with only pair names (mid lines)
    handles, labels = ax.get_legend_handles_labels()
    mid_entries = [(h, l) for h, l in zip(handles, labels) if l.endswith(" mid")]
    if mid_entries:
        handles2, labels2 = zip(*mid_entries)
        ax.legend(handles2, [l.replace(" mid","") for l in labels2], title="Session pair", loc="best")
    else:
        ax.legend(loc="best")

    fig.tight_layout()


    return long

