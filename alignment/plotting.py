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

import cell_classification as cc

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
    
def session_detection_vs_background(df, session_ids):
    for sid in session_ids:
        detected_in_session = df["detected_in_sessions"].apply(lambda s: sid in s)
        detected = df[detected_in_session]
        not_detected = df[~detected_in_session]
    
        
        print(f"session {sid}")
        print("detected: ", detected[f"int_optimized_{sid}"].describe())
        print("not detected: ", not_detected[f"int_optimized_{sid}"].describe())
        plt.hist(detected[f"int_optimized_{sid}"], bins=50, alpha=0.6,range=(0, max(df[f"int_optimized_{sid}"])), label=f"Detected in {sid}")
        plt.hist(not_detected[f"int_optimized_{sid}"], bins=50, alpha=0.6,range=(0, max(df[f"int_optimized_{sid}"])), label=f"Not detected in {sid}")
        plt.legend()
        plt.show()
        
        sns.scatterplot(data=detected, x="Center Z (px)", y=f"int_optimized_{sid}")
        sns.scatterplot(data=not_detected, x="Center Z (px)", y=f"int_optimized_{sid}")
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
    
    groups = ["up", "down", "stable"]
    labels = [f'{a}→{b}' for a, b in pairs]

    for g in groups:
        plt.plot(labels, cc.calculate_cells_per_tendency_group(df, pairs, g), label=g)
    plt.legend()
    plt.show()
    
def plot_cohort_tendencies(regions, id_pairs, config):
    df_plot = cc.gather_group_percentages_across_mice(regions, id_pairs, config)

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
    class_column="class_label",
    count_column="count",
    normalize=True,
    title="Cell Class Distribution per Mouse"
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
    title : str
        Title for the plot.
    """

    df = cc.gather_cells_specificity_percentages_across_mice(regions, 
                                                          config, 
                                                          classes, 
                                                          True)

    
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
    
        
