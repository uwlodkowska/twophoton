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
import sys
import yaml


from scipy.stats import ttest_rel 
#%%
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

#%%
regions = constants.LANDMARK_REGIONS#[[2,1],[3,2],[4,2],[6,2], [7,1], [11,1], [14,1]]
regions = constants.CTX_REGIONS

#%%

DIR_PATH = config["experiment"]["full_path"]
group_session_order = ["landmark", "ctx1", "ctx2"]
group_session_order = ["ctx", "landmark1", "landmark2"]

#%% plot number of active cells in each session



session_total = []

for m, r in regions:
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    st = []
    for i in range(3):
        st.append(df.loc[df[f"active{i}"]].shape[0])
    st = np.array(st) / st[0]  # Normalize to first session
    session_total.append(st)

session_total = np.array(session_total)

# Compute means and standard deviations
means = session_total.mean(axis=0)
stds = session_total.std(axis=0) / np.sqrt(session_total.shape[0])

x_vals = np.array([0, 1, 2])  # X-axis positions
labels = group_session_order

# Create bar plot with error bars
plt.figure(figsize=(6, 4))
bars = plt.bar(x_vals, means, yerr=stds, capsize=5, color=['C0', 'C1', 'C2'], alpha=0.7)

# Perform paired t-tests (e.g., ctx1 vs landmark, ctx2 vs landmark)
p_values = [ttest_rel(session_total[:, 0], session_total[:, i]).pvalue for i in range(1, 3)]

# Add asterisks for significance
for i, p in enumerate(p_values, start=1):
    if p < 0.05:  # Significance threshold
        plt.text(x_vals[i], means[i] + stds[i] + 0.05, "*", fontsize=14, ha='center')

# Formatting
plt.xticks(x_vals, labels)
plt.ylabel("Normalized cell count")
plt.title("Session cell counts (mean ± std)")
plt.ylim(0, max(means + stds) * 1.2)  # Adjust y-limit for readability
plt.show()


#%% plot overlap sizes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel  # For paired statistical test

session_total = []

for m, r in regions:
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    st = []
    all_active = df.loc[df["active0"] | df["active1"] | df["active2"]].shape[0]
    
    for i in range(2):
        st.append(df.loc[df[f"active{i}"] & df[f"active{i+1}"]].shape[0])
    
    st = np.array(st) / all_active
    session_total.append(st)

# Convert list to NumPy array
session_total = np.array(session_total)

# Compute means and standard errors
means = session_total.mean(axis=0)
sems = session_total.std(axis=0) / np.sqrt(session_total.shape[0])  # SEM

# X-axis labels
x_labels = [f'{group_session_order[0]}-{group_session_order[1]}', 
            f'{group_session_order[1]}-{group_session_order[2]}']
x_vals = np.arange(len(x_labels))

# Create bar plot
plt.bar(x_vals, means, yerr=sems, capsize=5, color=['royalblue', 'tomato'], alpha=0.7)

# Statistical significance test (paired t-test)
if session_total.shape[0] > 1:  # Ensure at least two samples for t-test
    t_stat, p_val = ttest_rel(session_total[:, 0], session_total[:, 1])
    if p_val < 0.05:
        plt.text(0.5, max(means + sems) * 1.1, "*", fontsize=14, ha='center', color='black')

# Formatting
plt.xticks(x_vals, x_labels)
plt.title("Session Overlap (Mean ± SEM)")
plt.ylim(0, max(means + sems) * 1.2)  # Adjust Y-axis to fit significance marker

plt.show()
