#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 21:07:33 2025

@author: ula
"""
import utils

import numpy as np
import matplotlib.pyplot as plt
import cell_preprocessing as cp
import pandas as pd

import sys
import yaml


import seaborn as sns
import matplotlib.pyplot as plt

import visualization as vis
import constants
import intersession

dfghj
#%%
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/multisession.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

#%%

SOURCE_DIR_PATH = config["experiment"]["dir_path"]
ICY_PATH = SOURCE_DIR_PATH + config["experiment"]["path_for_icy"]

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
OPT_PATH = config["experiment"]["optimized_path"]


regions = config["experiment"]["regions"][0]
group_session_order = config["experiment"]["session_order"][0]#[1:-1]



#%%

def plot_z_decay_curve(stack, ax, label="Session", use_median=False):
    if use_median:
        decay = np.median(stack, axis=(1, 2))
    else:
        decay = np.mean(stack, axis=(1, 2))
    
    ax.plot(np.arange(len(decay)), decay, marker='o', label=label)
    ax.set_xlabel("Z-slice")
    ax.set_ylabel("Mean Intensity")
    ax.set_title(f"Z-decay Curve: {label}")
    ax.legend()
    ax.grid(True)

def plot_scatter(df, cols, ax, session, hue_col = None):
    sns.scatterplot(data=df, x=cols[0], y=cols[1], s=10, alpha=0.6, ax=ax, hue=hue_col)
    ax.set_title(f"{cols[0]} vs {cols[1]} session {session}")
    ax.set_xlabel(cols[0])
    ax.set_ylabel(cols[1])

def plot_histogram(df, col):
    #plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=30, kde=False)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    


#%% plots and napari call for visualization

# for i, df in enumerate(sessions):
#     # img = utils.read_image(mouse, region, group_session_order[i], config)


#     fig, axs = plt.subplots(1, 1, figsize=(6, 5))  # side-by-side plots
#     df = df.loc[df["Interior (px)"]>150]
#     plot_scatter(df, ["Center Z (px)",f"int_optimized{i}"], axs, group_session_order[i], hue_col="Interior (px)")
    # plt.tight_layout()
    
    # sns.histplot(df[f"int_optimized{i}"]/(df["Center Z (px)"]+1), bins=30,binrange=[0,10], kde=False)


#     
#     vis.visualize_df_centroids(mouse, region, group_session_order[i], df, config)
    # fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    # plot_z_decay_curve(img, ax=axs, label=f"{group_session_order[i]}")
#     #plot_scatter(df, ["Interior (px)", "Mean Intensity (ch 0)"], ax=axs)
    #plot_histogram(df, "Interior (px)")
    # plot_histogram(df, f"int_optimized{i}")
    # plot_histogram(df, "Mean Intensity (ch 0)")
    plt.show()
#     # fig, axs = plt.subplots(1, 1, figsize=(6, 5))  # side-by-side plots
    
#     # plot_scatter(df, ["Interior (px)", "Center Z (px)"], ax=axs)
#     # plt.tight_layout()
#     # plt.show()
#%%
# for i, df in enumerate(sessions):
#     print(df.columns)
    
#%%
mouse =1 
region = 1
tstx = intersession.pooled_cells(mouse,region, group_session_order, config, test=False)
tstx['detected_in_sessions'] = tstx['detected_in_sessions'].apply(frozenset)
classes = tstx.groupby(by=["detected_in_sessions"], dropna=False).count()

tstx2 = intersession.pooled_cells(mouse,region, group_session_order[::-1], config, test=False)
tstx2['detected_in_sessions'] = tstx2['detected_in_sessions'].apply(frozenset)
classes2 = tstx2.groupby(by=["detected_in_sessions"], dropna=False).count()

#%%
regions = [[1,1]]#, [14,1]]#, [6,1], [13,1]]
# classes = []
# for mouse, region in regions:
#     df, component_sizes, suspicious_components = intersession.pool_cells_globally(mouse, region, group_session_order, config)
#     df['detected_in_sessions'] = df['detected_in_sessions'].apply(frozenset)
#     classes.extend([ df.groupby(by=["detected_in_sessions"], dropna=False).count()])
#     plt.hist(component_sizes)
#     plt.show()

#%%
df_reseg = intersession.pool_cells_globally(mouse, region, group_session_order, config, 7)
#%%

df_noreseg = intersession.pool_cells_globally(mouse, region, group_session_order, config, 7)


#%% Code for centroid optimization
regions = [[1,1]]#[[6,1], [14,1], [13,1]]
for mouse, region in regions:
    # sessions = utils.read_single_session_cell_data(
    #                               mouse, region, group_session_order, config, test=False, optimized=False)
    
    # for i, s in enumerate(sessions):
    #     s = s.loc[s["Interior (px)"]>150].copy()
    for sid in group_session_order[:1]:
        img = utils.read_image(mouse, region, sid, config)
        df_reseg = cp.optimize_centroids(df_reseg, img, suff=f"_{sid}", tolerance=3)
        df_noreseg = cp.optimize_centroids(df_noreseg, img, suff=f"_{sid}")

#%%
df.to_csv("/mnt/data/fos_gfp_tmaze/results/multisession/m1r1_pooled.csv")

#%%
df_exploded = df.explode("detected_in_sessions")

#%%
def session_detection_vs_background(df, session_ids):
    for sid in session_ids:
        detected_in_session = df["detected_in_sessions"].apply(lambda s: sid in s)
        detected = df[detected_in_session]
        not_detected = df[~detected_in_session]
    
        
        print(f"session {sid}")
        print("detected: ", detected.describe())
        print("not detected: ", not_detected.describe())
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
    row["cell_avg_detected"] = np.mean(detected)
    row["cell_avg_undetected"] = np.mean(undetected)
    return row

def cell_detection_vs_background(df, session_ids):
    df = df.append(["cell_avg_detected", "cell_avg_undetected"])
    df.apply(avg_row_detections, all_sessions = session_ids, axis=1)
    plt.hist(df["cell_avg_detected"], bins=50, label="Avg cell int in detected session")
    plt.hist(df["cell_avg_undetected"], bins=50, label="Avg cell int in detected session")
    plt.legend()
    plt.show()
        
#%%
session_detection_vs_background(df_reseg, group_session_order[:1])
session_detection_vs_background(df_noreseg, group_session_order[:1])