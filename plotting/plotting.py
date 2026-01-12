#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:43:04 2023

@author: ula
"""
import matplotlib.pyplot as plt
import constants, intersession
import numpy as np
import pandas as pd
import seaborn as sns
import upsetplot

import cell_classification as cc
import utils

def plot_intensity_change_histogram(d1, label1, d2, label2, title):
    plt.title(title)
    plt.hist(d1, bins=150, range=(-140,140), label = label1)
    plt.hist(d2, bins=150, range=(-140,140), alpha=0.5, label = label2)
    plt.legend()
    plt.savefig(constants.dir_path+title+".png")
    plt.close()
    
def show_transfer_rate_heatmaps(df, from_col, to_col, title):
    dc_arr = intersession.distribution_change_precalculated(df, from_col, to_col, 5)
    plt.imshow(dc_arr)
    plt.title(title)
    plt.show()
    
def plot_feature_through_sessions(regions, session_order, feature_func, plt_title, **kwargs):
    print(kwargs)
    all_mice = []
    for m,r in regions:    
        all_mice += [feature_func(m,r, session_order, **kwargs)]
    all_mice = np.array(all_mice)
    plt.title(plt_title)   
    for i, fval in enumerate(all_mice):
        plt.plot(session_order,fval, marker='.', linestyle="None", color = "C"+str(regions[i][0]))
    
    plt.plot(session_order, np.mean(all_mice, axis=0), linestyle='None', marker='_', markersize=20)
    print(np.mean(all_mice, axis=0))
    plt.show()
    
    
def plot_feature_averaged_for_mouse(regions, session_order, feature_func, plt_title):
    df = pd.DataFrame(regions, columns=['m', 'r'])
    
    df[session_order] = df.apply(lambda row: feature_func(row.m, row.r, session_order), axis=1 )
    df = df.groupby(['m']).mean()
    
    plt.title(plt_title)
    all_mice = df[session_order].to_numpy()
    for i, fval in enumerate(all_mice):
        plt.plot(session_order,fval, marker='.', linestyle="None", color = str(regions[i][0]))
    
    plt.plot(session_order, np.mean(all_mice, axis=0), linestyle='None', marker='_', markersize=20)
    print(np.mean(all_mice, axis=0))
    plt.show()
    
def session_detection_vs_background(df, session_ids, show_hist=False, sub_bgr=False):
    for sid in session_ids:
        df[f"intensity_{sid}"] = df[f"int_optimized_{sid}"]
        if sub_bgr:
            df[f"intensity_{sid}"] -= df[f"background_{sid}"]
        detected_in_session = df["detected_in_sessions"].apply(lambda s: sid in s)
        detected = df[detected_in_session]
        not_detected = df[~detected_in_session]
    
        
        if show_hist:
            plt.hist(detected[f"int_optimized_{sid}"], bins=50, alpha=0.6,range=(0, max(df[f"int_optimized_{sid}"])), label=f"Detected in {sid}")
            plt.hist(not_detected[f"int_optimized_{sid}"], bins=50, alpha=0.6,range=(0, max(df[f"int_optimized_{sid}"])), label=f"Not detected in {sid}")
            plt.legend()
            plt.show()
        
        sns.scatterplot(data=detected, x="Center Z (px)", y=f"intensity_{sid}", s=10)
        sns.scatterplot(data=not_detected, x="Center Z (px)", y=f"intensity_{sid}", s=10)
        plt.show()
        
        
        
def show_lmplots_comparison(df, session_ids):
    long_df = make_longform_df(df, session_ids)
    for sid in session_ids:
        df_sid = long_df[long_df["session"] == sid]
    
        g = sns.lmplot(
            data=df_sid,
            x="z",
            y="intensity",
            hue="subtracted",
            lowess=True,
            aspect=1.5,
            height=4,
            scatter_kws={"s": 10, "alpha": 0.4}
        )
        g.set(title=f"Session {sid}: Intensity vs Z (before/after subtraction)", xlabel="Center Z (px)", ylabel="Intensity")
        plt.tight_layout()
        plt.show()
    

def avg_row_detections(row, all_sessions):
    detected = []
    undetected = []
    for sid in all_sessions:
        if sid in row["detected_in_sessions"]:
            detected.extend([row[f"int_optimized_{sid}"]])
        else:
            undetected.extend([row[f"int_optimized_{sid}"]])

    row["cell_avg_detected"] = np.mean(detected) if len(detected) else 0
    row["cell_avg_undetected"] = np.mean(undetected) if len(undetected) else 0
    return row
        
def cell_detection_vs_background(df, session_ids):
    '''
    plots intensity of each cell shaped roi at specified coordinates in df in sessions 
    it has been identified in and not

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    session_ids : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    df["cell_avg_detected"] = 0.0
    df["cell_avg_undetected"] = 0.0
    df = df.apply(avg_row_detections, all_sessions = session_ids, axis=1)
    plt.hist(df["cell_avg_detected"], bins=50, label="Avg cell int in detected session")
    plt.hist(df["cell_avg_undetected"], bins=50, alpha=0.5, label="Avg cell int in undetected session")
    plt.legend()
    plt.show()
    lims = [0, max(df["cell_avg_detected"])]
    plt.plot(lims, lims, '--', color='red')
    sns.scatterplot(data=df, x="cell_avg_undetected", y="cell_avg_detected")
    plt.show()
    
def plot_tendency_groups(df, pairs):
    
    groups = ["on", "off","const"]
    labels = [f'{a}→{b}' for a, b in pairs]

    for g in groups:
        plt.plot(labels, cc.calculate_cells_per_tendency_group(df, pairs, g), label=g)
    plt.legend()
    plt.show()
    
    
def generic_class_plot(df, xcol, ycol, title, hue="Mouse"):
    ylabel = "Fraction of cells"

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=df,
        x=xcol,
        y=ycol,
        hue=hue,
        palette="Set2"
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Session")
    plt.xticks(rotation=45)
    plt.legend(title=hue, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
    agg_df = (
    df.groupby(xcol)[ycol]
    .agg(["mean", "sem"])  # SEM = standard error of the mean
    .reset_index()
    )
    
    plt.figure(figsize=(8, 5))
    
    sns.barplot(
    data=agg_df,
    x=xcol,
    y="mean",
    palette="Set2",
    errorbar=None,
    zorder=1  # plot underneath points
    )
    
    # Add SEM as manual error bars
    plt.errorbar(
        x=range(len(agg_df)),
        y=agg_df["mean"],
        yerr=agg_df["sem"],
        fmt='none',
        ecolor='black',
        capsize=5,
        zorder=2
    )
    
    # Add raw data points as stripplot
    sns.stripplot(
        data=df,
        x=xcol,
        y=ycol,
        color='black',
        jitter=True,
        alpha=0.6,
        dodge=False,
        size=4,
        zorder=3
    )
    
    plt.title("Mean Fraction of Transient Cells per Session")
    plt.ylabel("Fraction of cells (mean ± SEM)")
    plt.xlabel("Session")
    plt.xticks(ticks=range(len(agg_df)), labels=agg_df[xcol], rotation=75)
    plt.tight_layout()
    plt.show()

        

def plot_cohort_tendencies(regions, id_pairs, config, groups=["on", "off", "const"],ttype="presence", dfs=None):
    df_plot = cc.gather_group_percentages_across_mice(regions, id_pairs, config, groups, dfs=dfs, ttype=ttype)

    df_plot["transition"] = df_plot["session_from"] + "_to_" + df_plot["session_to"]
    
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_plot,
        x="transition",
        y="percentage",
        hue="group",
        marker="o",
        errorbar="se",  # mean ± SEM
    )
    plt.title("Group percentages over session transitions")
    plt.ylabel("Percentage of cells")
    plt.xlabel("Session transition")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
def plot_class_distribution(
    regions,
    config,
    classes,
    filterby = None,
    class_column="spec_class",
    count_column="percentage",
    normalize=True,
    title="Cell Class Distribution per Mouse",
    dfs = None
):
    """
    Create a bar plot comparing cell class distributions across mice.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with one row per (mouse, class) containing raw counts.
    class_column : str
        Column name indicating the class label (e.g., 'landmark_specific').
    mouse_column : str
        Column name indicating the mouse ID.
    count_column : str
        Column name containing raw counts.
    normalize : bool
        Whether to normalize counts per mouse to fractions.
    filterby : str
        Classes to filter df by before caculating percentage
    """

    df = cc.gather_cells_specificity_percentages_across_mice(regions, 
                                                          config, 
                                                          classes, 
                                                          True,
                                                          filterby=filterby,
                                                          dfs=dfs)

    
    yval = count_column
    ylabel = "Fraction of cells"

    plt.figure(figsize=(8,5))
    sns.barplot(
        data=df,
        x=class_column,
        y=yval,
        hue=class_column,
        palette="Set2"
    )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Mouse")
    plt.xticks(rotation=45)
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    
def convert_to_upsetplot_fmt(regions, config, sessions):
    for mouse,region in regions:
        df = utils.read_pooled_cells(mouse, region, config)
        df['detected_in_sessions'] = df['detected_in_sessions'].apply(frozenset)
        for session in sessions:
            df[session] = df['detected_in_sessions'].apply(lambda x: session in x).astype(bool)
        return df.groupby(sessions).size()
        
def plot_upsetplot(regions, config, sessions):
    df = convert_to_upsetplot_fmt(regions, config, sessions)

    print(df)
    upsetplot.plot(df) 
    plt.show() 
    
def make_longform_df(df, session_ids, suffix="int_optimized", background_suffix="background"):
    records = []
    
    for sid in session_ids:
        raw_col = f"{suffix}_{sid}"
        bgr_col = f"{background_suffix}_{sid}"
        zcol = constants.ICY_COLNAMES["zcol"]

        detected_in_session = df["detected_in_sessions"].apply(lambda s: sid in s)
        detected = df[detected_in_session]
        detected = df.copy()
        for _, row in detected.iterrows():
            records.append({
                "session": sid,
                "intensity": row[raw_col],
                "subtracted": False,
                "z": row[zcol]
            })

        if bgr_col in detected.columns:
            for _, row in detected.iterrows():
                records.append({
                    "session": sid,
                    "intensity": row[raw_col] - row[bgr_col],
                    "subtracted": True,
                    "z": row[zcol]
                })

    return pd.DataFrame(records)


def compare_session_intensities(df, session_ids, mouse, suffix="int_optimized", background_suffix="background", figsize=(10, 5)):
    """
    Plot per-session intensity distributions before and after background subtraction.

    Parameters:
    - df: DataFrame with intensity and background columns
    - session_ids: list of session IDs
    - suffix: suffix used for raw intensity columns (default: 'int_optimized')
    - background_suffix: suffix used for background columns (default: 'background')
    - figsize: tuple specifying figure size
    """
    long_df = make_longform_df(df, session_ids, suffix=suffix, background_suffix=background_suffix)

    plt.figure(figsize=figsize)
    sns.violinplot(data=long_df, x="session", y="intensity", hue="subtracted", split=True)
    plt.title(f"Intensity Distributions Before and After Background Subtraction mouse {mouse}")
    plt.xlabel("Session")
    plt.ylabel("Intensity")
    plt.legend(title="Subtracted", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()


def plot_cv_by_depth(df, session_ids, mouse):

    plt.figure(figsize=(10, 5))
    df["z_bin"] = (df["Center Z (px)"] // 5).astype(int)
    for sid in session_ids:
        z_col = "z_bin"
        intensity_col = f"int_optimized_{sid}"
        detected_in_session = df["detected_in_sessions"].apply(lambda s: sid in s)
        detected = df[detected_in_session]
        grouped = detected.groupby(z_col)[intensity_col]
        mean = grouped.mean()
        std = grouped.std()
        cv = std / mean

        plt.plot(cv.index, cv, label=sid)

    plt.xlabel("Z (depth, px)")
    plt.ylabel("CV (std/mean)")
    plt.title(f"Coefficient of Variation by Depth mouse {mouse}")
    plt.legend()
    plt.tight_layout()
    plt.show()