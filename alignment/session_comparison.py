#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:24:59 2025

@author: ula
"""


import numpy as np

import utils
import intersession
import cell_preprocessing as cp
import pandas as pd
import matplotlib.pyplot as plt
import constants

from pandarallel import pandarallel

#%%
regions = constants.LANDMARK_REGIONS#[[2,1],[3,2],[4,2],[6,2], [7,1], [11,1], [14,1]]
#%% plot number of active cells in each session

session_total = []
any_session = []
for m,r in regions:
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
    st = []
    for i in range(3):
        st+= [df.loc[df[f"active{i}"]].shape[0]]
    st = np.array(st)/st[0]
    session_total += [st]        
    plt.plot(st, marker='o', markersize=2, linestyle='None')

means = np.array(session_total).mean(axis=0)
stds = np.array(session_total).std(axis=0)



x_vals = np.array([0, 1, 2])  # x-axis positions


plt.plot(x_vals, means, marker='o', linestyle='None')
plt.fill_between(x_vals, means - stds, means + stds, alpha=0.2)
plt.title("Session cell counts, mean+6*std criterium")
plt.xticks(x_vals, ["landmark", "ctx1", "ctx2"])

plt.show()


#%% plot overlap sizes

session_total = []
any_session = []
for m,r in regions:
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
    st = []
    all_active =  df.loc[(df["active0"]) | (df["active1"])| (df["active2"])].shape[0]
    for i in range(2):
        st+= [df.loc[(df[f"active{i}"]) & (df[f"active{i+1}"])].shape[0]]
    st = np.array(st)/all_active
    session_total += [st]        
    plt.plot(st, marker='o', markersize=2, linestyle='None')

means = np.array(session_total).mean(axis=0)
stds = np.array(session_total).std(axis=0)



x_vals = np.array([0, 1])  # x-axis positions


plt.plot(x_vals, means, marker='o', linestyle='None')
plt.fill_between(x_vals, means - stds, means + stds, alpha=0.2)
plt.title("Session overlap, mean+6*std criterium")
plt.xticks(x_vals, ["landmark-ctx", "ctx-ctx"])

plt.plot()