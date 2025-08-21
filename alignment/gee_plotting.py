#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 00:39:28 2025

@author: ula
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
def fig_pair_effect_at_cutoff(
    csv_path: str,
    direction: str,
    cutoff: float,
    per_mouse_overlay: dict = None,   # {"s0_l1": (mean, lo, hi), ...}
    per_mouse_props: dict  = None,     # {"s0_l1": [p_mouse1, ...], ...}
    ref_key: str  = "ref",              # e.g. "l1_l2" or "s0_l1" — row highlight
    title: str  = None,
    save_path: str  = None,
    xlim: tuple[float,float]  = None
):
    """
    Horizontal forest-style plot per pair:
      - red dot + thick bar = GEE marginal prob ±95% CI
      - thin blue bar + hollow square = per-mouse mean ±95% CI
      - grey jittered dots = per-mouse proportions (optional)
      - faint row highlight for reference pair (optional)
    """
    df = pd.read_csv(csv_path)
    print(df)
    row = _row_at_cutoff(df, direction, cutoff)

    # ----- collect data from CSV (GEE layer) -----
    pair_labels_pretty = []
    gee_prob = []
    gee_lo = []
    gee_hi = []
    keys_in_plot_order = []
    for pair_key, pretty in PAIR_SPECS:
        keys_in_plot_order.append(pair_key)
        pair_labels_pretty.append(pretty)
        gee_prob.append(row[f"{pair_key}_coeff"])
        gee_lo.append(row[f"{pair_key}_CI_low"])
        gee_hi.append(row[f"{pair_key}_CI_hi"])

    y_positions = np.arange(len(pair_labels_pretty))[::-1]  # top→bottom
    gee_prob = np.asarray(gee_prob, float)
    gee_lo   = np.asarray(gee_lo, float)
    gee_hi   = np.asarray(gee_hi, float)

    # ----- figure -----
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    # Reference overlay: vertical band (CI) + dotted line (point)
    if ref_key is not None and ref_key in keys_in_plot_order:
        ref_p  = float(row[f"{ref_key}_coeff"])
        ref_lo = float(row[f"{ref_key}_CI_low"])
        ref_hi = float(row[f"{ref_key}_CI_hi"])
        ax.axvspan(ref_lo, ref_hi, color="0.85", alpha=0.6, zorder=0, label="Para referencyjna ±95% CI")
        ax.axvline(ref_p, linestyle=":", color="0.4", linewidth=1.2, zorder=1)

    # Optional: per-mouse raw points (grey jitter)
    if per_mouse_props:
        for yi, pair_key in zip(y_positions, keys_in_plot_order):
            vals = per_mouse_props.get(pair_key)
            if vals:
                vals = np.asarray(vals, float)
                yjit = yi + (np.random.rand(len(vals)) - 0.5) * 0.16
                ax.plot(vals, yjit, ".", alpha=0.45, color="0.5",
                        label="Osobniki (proporcje)" if yi == y_positions[0] else None, zorder=2)

    # Optional: per-mouse mean ± CI (thin blue bar + hollow square)
    if per_mouse_overlay:
        for yi, pair_key in zip(y_positions, keys_in_plot_order):
            if pair_key in per_mouse_overlay:
                mean_pm, lo_pm, hi_pm = per_mouse_overlay[pair_key]
                if np.isfinite(lo_pm) and np.isfinite(hi_pm):
                    ax.hlines(yi, lo_pm, hi_pm, linewidth=1.4, color="C0",
                              label="Śr. po osobnikach ±95% CI" if yi == y_positions[0] else None, zorder=3)
                if np.isfinite(mean_pm):
                    ax.plot([mean_pm], [yi], marker="s", mfc="white", mec="C0", mew=1.2, zorder=4)

    # GEE marginal: thick bar + red dot
    ax.hlines(y_positions, gee_lo, gee_hi, linewidth=3.0, color="C3", zorder=5)
    ax.plot(gee_prob, y_positions, "o", color="C3", ms=6, label="GEE: est. marginalna (95% CI)", zorder=6)

    # Cosmetics
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pair_labels_pretty)

    # tight x-lims from everything drawn (if not provided)
    if xlim is None:
        x_all = np.concatenate([gee_lo, gee_hi, gee_prob])
        if ref_key is not None and ref_key in keys_in_plot_order:
            x_all = np.append(x_all, [ref_lo, ref_hi, ref_p])
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

    ax.set_xlabel("Prawdopodobieństwo")
    if title is None:
        title = f"{direction} @ {cutoff} SD — efekt par"
    ax.set_title(title)

    # Legend: unique labels, consistent order
    handles, labels = ax.get_legend_handles_labels()
    seen, h_clean, l_clean = set(), [], []
    # prefer order: ref band, per-mouse points, per-mouse mean, GEE
    preferred = ["Para referencyjna ±95% CI", "Osobniki (proporcje)", "Śr. po osobnikach ±95% CI", "GEE: est. marginalna (95% CI)"]
    for pref in preferred:
        for h, lab in zip(handles, labels):
            if lab == pref and lab not in seen:
                seen.add(lab); h_clean.append(h); l_clean.append(lab)
    # add any remaining (unlikely)
    for h, lab in zip(handles, labels):
        if lab and lab not in seen:
            seen.add(lab); h_clean.append(h); l_clean.append(lab)

    if h_clean:
        ax.legend(h_clean, l_clean, frameon=False, loc="lower right")

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
        ax.plot(x, d["group wald p"].values, marker="s", linewidth=2, label="omnibus Wald")

    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xlabel("Próg SD")
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
    mouse_col: str,          # e.g. "mouse"
    pair_col: str,           # e.g. "pair"  (values like "s0_landmark1", "landmark2_ctx1", "ctx1_ctx2")
    outcome_col: str,        # binary 0/1 column for the direction you’re plotting, e.g. "y_up" or "y_down"
    pair_key_map: dict       # maps values in pair_col -> keys used in your plots, e.g.
                             # {"s0_landmark1":"s0_l1","landmark2_ctx1":"l2_c1","ctx1_ctx2":"c1_c2"}
):
    """
    Returns: dict {plot_key: (mean_prop_across_mice, CI_low, CI_high)} for each pair,
    where the CI is a t-based 95% CI across mice (mean of *per-mouse* proportions).
    """
    # keep only rows where outcome is 0/1
    d = long_df[[mouse_col, pair_col, outcome_col]].dropna().copy()

    # per-mouse proportion within each pair
    # (mean of the binary outcome per mouse per pair)
    pm = (
        d.groupby([pair_col, mouse_col], observed=True)[outcome_col]
         .mean()
         .rename("prop")
    )

    # aggregate across mice (per pair): mean, sd, n
    agg = (
        pm.reset_index()
          .groupby(pair_col, observed=True)["prop"]
          .agg(["mean", "std", "count"])
    )
    # t-based CI across mice (df = n-1)
    # handle sd=NaN (single mouse) gracefully → CI becomes NaN
    dfree = agg["count"] - 1
    se = agg["std"] / np.sqrt(agg["count"])
    tcrit = t.ppf(0.975, dfree.replace(0, np.nan))  # avoid df=0
    ci_lo = agg["mean"] - tcrit * se
    ci_hi = agg["mean"] + tcrit * se

    # map to your plotting keys
    out = {}
    for pair_val, row in agg.iterrows():
        key = pair_key_map.get(pair_val)
        if key is not None:
            out[key] = (float(row["mean"]),
                        float(ci_lo.loc[pair_val]) if np.isfinite(ci_lo.loc[pair_val]) else np.nan,
                        float(ci_hi.loc[pair_val]) if np.isfinite(ci_hi.loc[pair_val]) else np.nan)
    return out



#%%
df = pd.read_csv(CSV_PATH)
LENIENT = 0.5
df["sd cutoff"] == 0.5
#%%
# MAIN FIGURE
# plot_forest_at_cutoff(df, "DOWN", 0.75, save_path="/mnt/data/fig1A_down_forest.png")
# plot_forest_at_cutoff(df, "UP",   0.75, save_path="/mnt/data/fig1B_up_forest.png")

# # SUPPLEMENT
# plot_p_vs_cutoff(df, "DOWN", save_path="/mnt/data/figS1_down_p.png")
# plot_p_vs_cutoff(df, "UP",   save_path="/mnt/data/figS1_up_p.png")
# plot_prob_vs_cutoff(df, "DOWN", save_path="/mnt/data/figS2_down_prob.png")
# plot_prob_vs_cutoff(df, "UP",   save_path="/mnt/data/figS2_up_prob.png")

# #%%
# fig_pair_effect_at_cutoff(CSV_PATH, "DOWN", LENIENT,
#                           per_mouse_overlay=None,
#                           save_path="/mnt/data/fig_pair_down.png")

# # Panel A’ (UP at the same cutoff, to show it’s absent/fragile)
# fig_pair_effect_at_cutoff(CSV_PATH, "UP", LENIENT,
#                           per_mouse_overlay=None,
#                           save_path="/mnt/data/fig_pair_up.png")

# # Panel B (fragility/sensitivity)
# fig_fragility_p_vs_cutoff(CSV_PATH, "DOWN", save_path="/mnt/data/fig_frag_down.png")
# fig_fragility_p_vs_cutoff(CSV_PATH, "UP",   save_path="/mnt/data/fig_frag_up.png")
