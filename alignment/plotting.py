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
    
#%%
