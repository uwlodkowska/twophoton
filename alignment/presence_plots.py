import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
import pandas as pd

def presence_boxplot_grouped(
    df,
    mouse_col="mouse",
    pair_col="Pair",
    on_col="on", off_col="off", const_col="const", n_col="n",
    prop_on_col=None, prop_off_col=None, prop_const_col=None,
    classes_order=("on","off","const"),
    pair_order=None,
    jitter=0.12,
    title="Proporcje komórek – boxplot",
    ax_=None,
    rename_map = None
):
    d = df.copy()

    # proportions
    if prop_on_col and prop_off_col and prop_const_col:
        d = d.rename(columns={prop_on_col:"prop_on",
                              prop_off_col:"prop_off",
                              prop_const_col:"prop_const"})
    else:
        n = d[n_col].replace(0, np.nan)
        d["prop_on"]    = d[on_col] / n
        d["prop_off"]   = d[off_col] / n
        d["prop_const"] = d[const_col] / n

    # tidy
    long = d.melt(id_vars=[mouse_col,pair_col],
                  value_vars=["prop_on","prop_off","prop_const"],
                  var_name="class", value_name="prop").dropna()
    long["class"] = long["class"].map({"prop_on":"on",
                                       "prop_off":"off",
                                       "prop_const":"const"})
    if classes_order:
        long["class"] = pd.Categorical(long["class"], classes_order, ordered=True)
    if pair_order is None:
        pair_order = list(pd.unique(long[pair_col]))

    # palette
    if len(pair_order)==2:
        palette = {pair_order[0]:"#8dc2e4", pair_order[1]:"#778da9"}
        palette_dots = {pair_order[0]:"#68b0de", pair_order[1]:"#415a77"}
    else:
        palette = sns.color_palette("Set2", len(pair_order))

    fig, ax = plt.subplots(figsize=(7,4))
    
    if ax_ is not None:
        ax = ax_
    # --- boxplots with seaborn dodge ---
    sns.boxplot(
        data=long, x="class", y="prop",
        hue=pair_col, hue_order=pair_order,
        showfliers=False, dodge=True,
        palette=palette, ax=ax,
        boxprops=dict(alpha=0.8), whiskerprops=dict(alpha=0.8), capprops=dict(alpha=0.8)
    )

    # --- jittered points with manual dodge ---
    n_hues = len(pair_order)
    hue_offsets = np.linspace(-0.2,0.2,n_hues)  # horizontal shifts for dots

    for i,h in enumerate(pair_order):
        sub = long[long[pair_col]==h]
        # numeric base positions (0,1,2 for on/off/const)
        base_x = sub["class"].cat.codes.values
        # add offset for this hue
        jittered_x = base_x + hue_offsets[i] + np.random.uniform(-jitter,jitter,len(sub))
        ax.scatter(jittered_x, sub["prop"],
        s=28, alpha=0.9,
        facecolors=palette_dots[h],   # <- use facecolors
        # thin white rim for contrast
        linewidths=0.6,
        zorder=10,                    # <- draw above boxes/whiskers
        label=None )

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    #ax.set_xlabel("Rodzaj sesji")
    ax.set_ylabel("Proporcja")
    ax.set_title(title)
    ax.legend(title="", frameon=False)
    #plt.tight_layout()
    
    handles, labels = ax.get_legend_handles_labels()

    if rename_map is not None:
        new_labels = [rename_map.get(l, l) for l in labels]
        
        ax.legend(handles, new_labels, title="", frameon=False)
    return ax
