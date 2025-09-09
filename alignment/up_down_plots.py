import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=["#0000a2",  "#bc272d","#e9c716", "#50ad9f", "#4a2377"]) 
from pathlib import Path
import re


import matplotlib.ticker as ticker
import utils
#%%
# ---------- USER SETTINGS ----------
outdir = Path("gee_plots")
outdir.mkdir(parents=True, exist_ok=True)



#%%
experiments = ["lcc", "cll", "multi"]
thresholds = [0.50, 0.75, 1.00, 1.25]
ref = "ctx_to_ctx"
alpha_sig = 0.05
alpha_marg = 0.10

colors = ["#0d1b2a", " 	#68b0de"]
FONT = dict(suptitle=24, title=20, label=18, ticks=14, legend=18)
mpl.rcParams["legend.fontsize"] = 15
#%%

#%%


def _ensure_sorted_cutoffs(df):
    cols = [c for c in ["cutoff","direction","pair","mode"] if c in df.columns]
    return df.sort_values(cols, kind="mergesort")

def _abbr_session(s: str) -> str:
    # keep first letter (uppercased) + any trailing digits
    m = re.match(r"([A-Za-z]+)(\d*)$", s)
    #if not m: 
    ret = s[:1].upper()
    if ret == "S":
        return r'S$_{0}$'
    return s[:1].upper()
    base, num = m.groups()
    return f"{base[0].upper()}{num}"

def _abbr_pair(pair: str) -> str:
    # expected like "ctx_to_landmark2" or "s1_to_s2"
    if "_to_" in pair:
        a, b = pair.split("_to_", 1)
        return f"{_abbr_session(a)}{_abbr_session(b)}"
    return pair  # fallback

def _abbr_mode(mode: str) -> str:
    # optional: shorten your modes
    if mode.lower() in ("set_fixed","set-fixed","fixed"):
        return "stałe kowariaty"
    if mode.lower() in ("mouse_avg","mouse-average","mouse"):
        return "uśrednione po myszach"
    return mode



def _get_directions(df, preferred=("UP","DOWN")):
    if "direction" not in df.columns:
        return ["all"]
    dirs = list(df["direction"].dropna().unique())
    # order as preferred when present
    order = [d for d in preferred if d in dirs]
    order += [d for d in dirs if d not in order]
    return order



def _legend_outside_wrap(fig, handles, labels, max_cols=3):
    if not handles: 
        return
    ncol = min(max_cols, max(1, len(labels)))
    fig.legend(handles, labels, ncol=ncol,
               loc="upper center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=FONT["legend"])

def _safe(s):  # filenames
    return str(s).replace("/", "-").replace(" ", "_")

def emm_forest_pair_ref(
    res_df, cutoff, pair_label, title_prefix, savepath,
    direction=None,
    overlay_mode="mouse_avg",           # which mode's ref CI to shade if available
    colors=None,                        # {"mouse_avg":"#1f77b4","set_fixed":"#d62728"}
    jitter=0.08
):
    """
    Two-row forest (NO recalculation):
      top   : Δp (pair − ref) ±95% CI from res_df['delta','ci_lo','ci_hi']
      bottom: P(ref) ±95% CI if res_df has 'p_ref_lo','p_ref_hi' (else points)
    Plots both modes on each row with small vertical jitter and distinct colors.
    Optionally shades a grey band for overlay_mode's ref CI if present.
    """
    d = _ensure_sorted_cutoffs(res_df).copy()
    d = d[(d["cutoff"] == cutoff) & (d["pair"] == pair_label)]
    
    direction = d["direction"].loc[0]
    

    colors = colors or {"mouse_avg": "#1f77b4", "set_fixed": "#d62728"}
    modes = [m for m in ["mouse_avg", "set_fixed"] if m in d["mode"].unique()]
    rows = {m: d[d["mode"] == m].iloc[0] for m in modes if not d[d["mode"] == m].empty}
    if not rows:
        return

    # ----- x-limits from what's already in res_df -----
    # Top uses Δp CI:
    top_los = [float(r["ci_lo"]) for r in rows.values()]
    top_his = [float(r["ci_hi"]) for r in rows.values()]
    # Bottom uses ref CI ONLY if present:
    def _has_ref_ci(r):
        return ("p_ref_lo" in r.index and "p_ref_hi" in r.index
                and np.isfinite(r["p_ref_lo"]) and np.isfinite(r["p_ref_hi"]))
    bottom_los = [float(r["p_ref_lo"]) for r in rows.values() if _has_ref_ci(r)]
    bottom_his = [float(r["p_ref_hi"]) for r in rows.values() if _has_ref_ci(r)]

    xlo = min(top_los + bottom_los) if bottom_los else min(top_los)
    xhi = max(top_his + bottom_his) if bottom_his else max(top_his)
    pad = 0.08 * (xhi - xlo) if xhi > xlo else 0.01

    # ----- figure -----
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.set_xlim(xlo - pad, xhi + pad)

    # Grey ref band if that CI exists in res_df for overlay_mode
    if overlay_mode in rows and _has_ref_ci(rows[overlay_mode]):
        r = rows[overlay_mode]
        ax.axvspan(float(r["p_ref_lo"]), float(r["p_ref_hi"]),
                   color="0.85", alpha=0.6, zorder=0)
        ax.axvline(float(r["p_ref"]), color="0.5", linestyle=":", linewidth=1)

    # y positions: 0=ref row, 1=Δp row; jitter per mode
    base_y = {"ref": 0.0, "contrast": 1.0}
    offs   = {"mouse_avg": -jitter, "set_fixed": +jitter}

    # draw helpers (using ONLY columns already in res_df)
    def _draw_ref_row():
        for m, r in rows.items():
            y = base_y["ref"] + offs.get(m, 0.0)
            c = colors.get(m, "C0")
            # ref CI if available
            if _has_ref_ci(r):
                ax.hlines(y, float(r["p_ref_lo"]), float(r["p_ref_hi"]),
                          color=c, linewidth=3, alpha=0.95)
            # ref point
            ax.scatter([float(r["p_ref"])], [y], color=c, s=40, zorder=3)

    def _draw_contrast_row():
        for m, r in rows.items():
            y = base_y["contrast"] + offs.get(m, 0.0)
            c = colors.get(m, "C0")
            ax.hlines(y, float(r["ci_lo"]), float(r["ci_hi"]),
                      color=c, linewidth=3, alpha=0.95)
            ax.scatter([float(r["delta"])], [y], color=c, s=40, zorder=3)

    _draw_contrast_row()
    _draw_ref_row()

    # cosmetics
    ax.set_yticks([base_y["ref"], base_y["contrast"]])
    ax.set_yticklabels(["ref", _abbr_pair(pair_label)])
    ax.set_xlabel("Probability (ref)  /  Δp (pair − ref)")
    title = f"{title_prefix} — {_abbr_pair(pair_label)} at cutoff {cutoff}"
    if direction: title += f" ({direction})"
    ax.set_title(title)
    ax.grid(axis="x", color="0.9", linewidth=0.8)
    ax.spines["right"].set_visible(False); ax.spines["top"].set_visible(False)
    ax.set_ylim(-0.6, 1.6)

    # legend (wrap outside)
    handles, labels = [], []
    for m in modes:
        h, = ax.plot([], [], color=colors.get(m,"C0"), marker="o", linestyle="-", linewidth=3)
        handles.append(h); labels.append(_abbr_mode(m))
    if overlay_mode in rows and _has_ref_ci(rows[overlay_mode]):
        h_band, = ax.plot([], [], color="0.6", linewidth=6, alpha=0.4)
        handles = [h_band] + handles
        labels  = ["Ref 95% CI (shaded)"] + labels
    _legend_outside_wrap(fig, handles, labels, max_cols=3)

    fig.tight_layout()
    fig.savefig(savepath, dpi=240, bbox_inches="tight")
    plt.close(fig)



# -----------------------------
# Panels: p-values vs cutoff
# -----------------------------
def plot_pvals_vs_cutoff_panels(
    res_df,omni_df, pair_labels,title_prefix, savepath,
    alpha_sig=0.05, alpha_marg=0.10
):
    d = _ensure_sorted_cutoffs(res_df).copy()
    if "direction" not in d.columns:
        d["direction"] = "all"
    directions = _get_directions(d)

    n = len(directions)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5*ncols, 6.3*nrows), squeeze=False, sharey=True)

    # main title (prefix only)
    fig.suptitle(title_prefix, fontsize=FONT["suptitle"])
    multi = (len(pair_labels) > 1)
    # collect legend entries across panels
    legend_handles = {}
    for idx, direct in enumerate(directions):
        ax = axes[idx // ncols][idx % ncols]
        d_dir = d[d["direction"] == direct]
        if d_dir.empty:
            ax.set_visible(False)
            continue
        #ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.tick_params(axis='y', labelsize=FONT["ticks"])
        ax.tick_params(axis='x', labelsize=FONT["ticks"])
        ref = d_dir["ref"].unique()[0] if "ref" in d_dir.columns else None
        #pairs = d_dir[["pair", "ref"]].drop_duplicates().to_numpy()
        
        pairs = d_dir["pair"].unique() if "pair" in d_dir.columns else None

        for pair,ref in pair_labels:
            print(pair,ref)
            # if '0' not in ref:
            #     continue
            # pair = paird[0]
            # ref = paird[1]
            if multi:
                modes = ["mouse_avg"]
            else:
                modes = d_dir["mode"].unique()
            for mode in modes:
                dd = d_dir[(d_dir["pair"] == pair)&(d_dir["ref"] == ref) & (d_dir["mode"] == mode)]
                if dd.empty:
                    continue
                base_label = (
                    f"{_abbr_pair(pair)} — {_abbr_pair(ref)}"#" · {_abbr_mode(mode)}"
                    if multi else
                    f"{_abbr_pair(pair)} — {_abbr_pair(ref)} · {_abbr_mode(mode)}"
                )
                
                
                h_raw, = ax.plot(dd["cutoff"], dd["p_raw"], marker="x",linestyle="--", alpha=0.4,
                                  linewidth=1.8, label=f"{base_label}")
                legend_handles[f"{base_label}"] = h_raw
                c = h_raw.get_color()
                # --- HOLM: same color, different style
                if len(pairs) > 1 and "p_holm" in dd:
                    ax.plot(dd["cutoff"], dd["p_holm"], marker="o", linewidth=1.8, color=c,
                                      label=f"{base_label} (Holm)")
                else:
                    h_raw, = ax.plot(dd["cutoff"], dd["p_raw"], marker="o",
                                      linewidth=1.5, label=f"{base_label}")
                    legend_handles[f"{base_label}"] = h_raw

        ax.axhline(alpha_sig, linestyle=":", linewidth=1)
        ax.axhline(alpha_marg, linestyle=":", linewidth=1)
        # --- legends: colors for pairs×modes, styles for statistics
        
        
        if omni_df is not None and len(pair_labels)>1:
            do_panel = omni_df[omni_df["direction"] == direct]
            for m in ["mouse_avg"]:#sorted(do_panel["mode"].dropna().unique()):
                dom = do_panel[do_panel["mode"] == m].sort_values("cutoff")
                if not dom.empty and {"cutoff","p"}.issubset(dom.columns):
                    h_omni, = ax.plot(dom["cutoff"], dom["p"], linestyle="-.", marker="s",
                                      linewidth=2.2, label=f"Omnibus Wald")
                    legend_handles[f"Omnibus Wald"] = h_omni

        
        
        
        if idx % ncols == 0:
            ax.set_ylabel("Wartość p", fontsize=FONT["label"])
        ax.set_title(f"{direct}", fontsize=FONT["title"])
        ax.grid(True, alpha=0.25)
    
    
    from matplotlib.lines import Line2D
    # Color legend = unique RAW handles (one per pair×mode)
    handles = list(dict(legend_handles).values())
    leg_colors = fig.legend(handles=handles, loc="lower left")#, title="Pary")#" × tryb")
    fig.add_artist(leg_colors)
    # Style legend (black proxies)
    style_handles = [
        Line2D([0],[0], linestyle="--", marker="x", linewidth=1.8, color="black", label="Przed poprawką"),
        Line2D([0],[0], linestyle="-", marker="o", linewidth=1.8, color="black", label="Po korekcie Holma"),
        Line2D([0],[0], linestyle=":", marker="s", linewidth=2.2, color="black", label="Test omnibusowy"),
    ]
    # if omni_df is not None:
    #     style_handles.append(Line2D([0],[0], linestyle="-.", marker="s", linewidth=1.6,
    #                                 color="black", label="Omnibus"))
    if multi:
        fig.legend(handles=style_handles, loc="lower right")
    # legend below panels, wrapped so it doesn't shrink plots
    # if legend_handles:
    #     labels, handles = zip(*legend_handles.items())
    #     ncol = min(4, max(1, len(labels)))  # wrap into up to 4 columns
    #     fig.legend(handles, labels, ncol=ncol,
    #                loc="upper center", bbox_to_anchor=(0.5, -0.03),
    #                frameon=False, fontsize=FONT["legend"])

    # one shared x-axis label for the whole figure
    try:
        fig.supxlabel("Próg (rSD)", fontsize=FONT["label"])
    except Exception:
        fig.text(0.5, 0.02, "Próg (rSD)", ha="center", va="center", fontsize=FONT["label"])
    for ax in axes.ravel():
        ax.set_ylim(0, 1.02)   # or (-0.02, 1.02) if you want a tiny margin


    # leave space for suptitle (top) and legend/xlabel (bottom)
    fig.tight_layout(rect=[0.02, 0.12, 1.00, 0.92])
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Panels: Δp vs cutoff
# -----------------------------
def plot_effectsizes_vs_cutoff_panels(
    res_df,pair_labels, title_prefix, savepath
):
    """
    One figure with side-by-side panels (by direction) showing Δp across cutoffs.
    Plots each pair × mode as a line over cutoff, with shaded CIs from ci_lo/ci_hi.
    """
    d = _ensure_sorted_cutoffs(res_df).copy()
    if "direction" not in d.columns:
        d["direction"] = "all"
    directions = _get_directions(d)

    n = len(directions)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5*ncols, 6.3*nrows), squeeze=False, sharey=True)

    # main title from prefix
    fig.suptitle(title_prefix, fontsize=FONT["suptitle"])
    multi = (len(pair_labels) >1)
    legend_handles = {}
    for idx, direct in enumerate(directions):
        ax = axes[idx // ncols][idx % ncols]
        
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
        ax.tick_params(axis='y', labelsize=FONT["ticks"])
        ax.tick_params(axis='x', labelsize=FONT["ticks"])
        d_dir = d[d["direction"] == direct]
        if d_dir.empty:
            ax.set_visible(False)
            continue
        ref = d_dir["ref"].unique()[0] if "ref" in d_dir.columns else None
        #pairs = d_dir[["pair", "ref"]].drop_duplicates().to_numpy()
        
        pairs = d_dir["pair"].unique() if "pair" in d_dir.columns else None

        for pair,ref in pair_labels:
            # if '0' not in ref:
            #     continue
            # pair = paird[0]
            # ref = paird[1]
            modes =  ["mouse_avg"]
            if not multi:
                modes = d_dir["mode"].unique()
            for mode in modes:
                dd = d_dir[(d_dir["pair"] == pair)&(d_dir["ref"] == ref) & (d_dir["mode"] == mode)]
                if dd.empty:
                    continue
                dd = dd.sort_values("cutoff")
                if multi:
                    label = f"{_abbr_pair(pair)} — {_abbr_pair(ref)}"
                else:
                    label = f"{_abbr_pair(pair)} — {_abbr_pair(ref)} · {_abbr_mode(mode)}"

                # line (for legend color)
                h, = ax.plot(dd["cutoff"], dd["delta"], marker="o", linewidth=1.8, label=label, zorder=3)
                legend_handles[label] = h

                # CI shadow (if columns present)
                if {"ci_lo","ci_hi"}.issubset(dd.columns):
                    if dd["ci_lo"].notna().any() and dd["ci_hi"].notna().any():
                        ax.fill_between(dd["cutoff"], dd["ci_lo"], dd["ci_hi"],
                                        color=h.get_color(), alpha=0.12, linewidth=0, zorder=2)

        ax.axhline(0, linestyle=":", linewidth=1)
        # shared x-label → don't set per-axes xlabel
        if idx % ncols == 0:
            ax.set_ylabel("Δp (para − referencja)", fontsize=FONT["label"])
        ax.set_title(f"{direct}", fontsize=FONT["title"])
        ax.grid(True, alpha=0.25)

    # legend below, wrapped (so it doesn't shrink the plots)
    if legend_handles:
        labels, handles = zip(*legend_handles.items())
        ncol = min(4, max(1, len(labels)))
        fig.legend(handles, labels, ncol=ncol,
                   loc="upper center", bbox_to_anchor=(0.5, -0.03),
                   frameon=False, fontsize=FONT["legend"])
    cols = [c for c in ("diff","ci_lo","ci_hi") if c in d.columns]
    M = float(np.nanmax(np.abs(d[cols].to_numpy()))) if cols else 0
    for ax in axes.ravel():
        ax.set_ylim(-M, M)
    # one shared x-axis label
    try:
        fig.supxlabel("Próg (rSD)", fontsize=FONT["label"])
    except Exception:
        fig.text(0.5, 0.02, "Próg (rSD)", ha="center", va="center", fontsize=FONT["label"])

    fig.tight_layout(rect=[0.02, 0.08, 1.00, 0.92])
    fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- tiny helpers ----
def _pval_stars(p, alpha_sig=0.05, alpha_marg=0.10):
    if p is None or not np.isfinite(p): return ""
    return "**" if p < alpha_sig else ("*" if p < alpha_marg else "")

def _safe(s):  # for filenames
    return str(s).replace("/", "-").replace(" ", "_")




def flagged_cutoffs(res_df):
    d = res_df.groupby("cutoff", as_index=False)["p_holm"].min()
    return d.loc[d["p_holm"] < alpha_marg, "cutoff"].tolist()
# ---------- DRIVER ----------
def run_all_groups(df, pairs, SESSIONS, ref,pair_labels,sweep_func, exp, suff, to_exclude):
    # if exp == "LCC" or exp == "CLL":
    #     canonical = True
    # else: canonical = False
    canonical = False
    res_df, omni_df = sweep_func(
        df, pairs=pairs, SESSIONS=SESSIONS,
        thresholds=thresholds, ref=ref,
        pair_labels=pair_labels, mouse_col="mouse",
        alpha=alpha_sig, canonical = canonical,
        cells_to_exclude = to_exclude,
    )
    res_df["experiment"] = exp
    omni_df["experiment"] = exp

    # --- 1) p-value curves
    plt.yticks(fontsize=FONT["ticks"])
    plot_pvals_vs_cutoff_panels(
        res_df,
        omni_df,
        pair_labels,
        title_prefix=f"Zależność p od progu odcięcia dla komórek aktywnych {suff} - grupa {exp}",
        savepath=outdir / f"{exp}_{to_exclude}_pvals_panels.png",
        alpha_sig=0.05, alpha_marg=0.10
    )
    
    # Δp (both directions in one PNG)
    plot_effectsizes_vs_cutoff_panels(
        res_df,
        pair_labels,
        title_prefix=f"Wielkość efektu dla komórek aktywnych {suff} - grupa {exp}",
        savepath=outdir / f"{exp}_{to_exclude}_delta_panels.png"
    )

    # --- 3) forest plots at marginal/significant cutoffs
    # for c in flagged_cutoffs(res_df):
    #     emm_forest_pair_ref(
    #         res_df,
    #         cutoff=c,
    #         pair_label="landmark_to_ctx",
    #         title_prefix="lcc — UP",
    #         savepath="lcc__emm_forest_UP_cut1.00.png",
    #         direction="UP",                  # or None
    #         overlay_mode="mouse_avg",        # shaded ref CI uses this mode if present
    #         colors={"mouse_avg":"#1f77b4","set_fixed":"#d62728"},
    #         jitter=0.06
    #         )
    res_df.to_csv(outdir / f"{exp}_{to_exclude}_results.csv", index=False)
    omni_df.to_csv(outdir / f"{exp}_{to_exclude}_omnibus.csv", index=False)

#%%
