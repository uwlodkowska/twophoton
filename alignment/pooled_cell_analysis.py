#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 21:44:57 2025

@author: ula
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import itertools
import cell_preprocessing as cp
#import ace_tools as tools  # For displaying tables
import sys
import yaml
#%%
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark_rev.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


#%%

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]


regions = config["experiment"]["regions"][0]
group_session_order = config["experiment"]["session_order"][0]
#%%

# Store data for each mouse
mouse_data = {}

bgr_data = pd.read_csv(BGR_DIR)
# Read data for each mouse
for m,r in regions:
    bgrv = []
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    df = cp.find_active_cells(df, bgrv, 6)
    mouse_data[m] = df  # Store DataFrame in dictionary

# List of intensity columns
intensity_columns = ["int_optimized0", "int_optimized1", "int_optimized2"]

# Store results
skewness_kurtosis_before = {}
skewness_kurtosis_after = {}

# Small constant to avoid log(0)
epsilon = 1e-6  
ks_results = []

# Compute skewness & kurtosis for each mouse
for mouse, df_or in mouse_data.items():
    skewness_kurtosis_before[mouse] = {}
    skewness_kurtosis_after[mouse] = {}
    for i,col in enumerate(intensity_columns):
        df = df_or.loc[df_or[f'active{i}']]
        skew_before = stats.skew(df[col].dropna())
        kurt_before = stats.kurtosis(df[col].dropna(), fisher=True)
        # Apply log transformation
        df[col + "_log"] = np.log(df[col] + epsilon)
        df_or[col + "_log"] = np.log(df_or[col] + epsilon)

        # Compute skewness & kurtosis after transformation
        skew_after = stats.skew(df[col + "_log"].dropna())
        kurt_after = stats.kurtosis(df[col + "_log"].dropna(), fisher=True)
        #plt.hist(df[col], bins=40)

        skewness_kurtosis_before[mouse][col] = {"skewness": skew_before, "kurtosis": kurt_before}
        skewness_kurtosis_after[mouse][col] = {"skewness": skew_after, "kurtosis": kurt_after}


# Convert results to DataFrames for visualization
skew_kurt_before_df = pd.DataFrame.from_dict({(i, j): skewness_kurtosis_before[i][j] 
                                              for i in skewness_kurtosis_before 
                                              for j in skewness_kurtosis_before[i]}, orient="index")

skew_kurt_after_df = pd.DataFrame.from_dict({(i, j): skewness_kurtosis_after[i][j] 
                                             for i in skewness_kurtosis_after 
                                             for j in skewness_kurtosis_after[i]}, orient="index")


#%%

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------
# Existing code up to log transformation
# ---------------------------
mouse_data = {}

bgr_data = pd.read_csv(BGR_DIR)
# Read data for each mouse (regions is assumed to be defined elsewhere)
for m, r in regions:
    bgrv = []
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m'] == m) & (bgr_data['r'] == r), [f'mean{i}', f'sdev{i}']].values)
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    df = cp.find_active_cells(df, bgrv, 6)
    mouse_data[m] = df  # Store DataFrame in dictionary

# List of intensity columns
intensity_columns = ["int_optimized0", "int_optimized1", "int_optimized2"]

# Compute log transformation and additional stats
epsilon = 1e-6  
for mouse, df_or in mouse_data.items():
    for i, col in enumerate(intensity_columns):
        # Only use rows with active cells for stats
        df_active = df_or.loc[df_or[f'active{i}']]
        # Optionally compute skewness & kurtosis here if needed
        df_or[col + "_log"] = np.log(df_or[col] + epsilon)

# ---------------------------
# New code: Reshape and z-score per mouse before pooling
# ---------------------------

# Create a list to store the reshaped and z-scored DataFrames
pooled_list = []

for mouse, df in mouse_data.items():
    # Reshape the data into long format using the log-transformed columns
    melted = pd.melt(
        df,
        value_vars=[f'{col}_log' for col in intensity_columns],
        var_name='session',
        value_name='intensity'
    )
    # Extract the session number from column names (e.g., 'int_optimized0_log' -> 0)
    melted['session'] = melted['session'].str.extract('(\d)').astype(int)
    
    # Compute the z-score for this mouse individually
    mean_intensity = melted['intensity'].mean()
    std_intensity = melted['intensity'].std()
    melted['z_intensity'] = (melted['intensity'] - mean_intensity) / std_intensity
    
    melted['mouse'] = mouse  # Tag the data with the mouse identifier
    pooled_list.append(melted)

# Concatenate the z-scored data from all mice
pooled_df = pd.concat(pooled_list, ignore_index=True)

# ---------------------------
# Analysis: Compare z-scored intensity across sessions
# ---------------------------
session_stats = pooled_df.groupby('session')['z_intensity'].agg(['mean', 'std', 'count'])
print("Summary statistics by session:")
print(session_stats)

# Optional: Plot the z-scored intensity changes between sessions
fig, ax = plt.subplots()
session_stats_plot = session_stats.reset_index()
ax.errorbar(session_stats_plot['session'], session_stats_plot['mean'],
            yerr=session_stats_plot['std'], fmt='o-', capsize=5)
ax.set_xlabel('Session')
ax.set_ylabel('Z-scored intensity (per mouse)')
ax.set_title('Z-scored Cell Intensity Across Sessions')
plt.show()




#%%
skew_kurt_before_df.to_csv('/media/ula/DATADRIVE1/fos_gfp_tmaze/results/skewness_kurtosis_before.csv')
skew_kurt_after_df.to_csv('/media/ula/DATADRIVE1/fos_gfp_tmaze/results/skewness_kurtosis_after.csv')
#%% Perform KS tests for each session between pairs of mice
mice = np.array(regions)[:,0]
#%%
mouse_pairs = list(itertools.combinations(mice, 2))
ks_results = []
for i,col in enumerate(intensity_columns):
    for m1, m2 in mouse_pairs:
        df1 = mouse_data[m1]
        df1 = df1.loc[df1[f'active{i}']]
        data1 = df1[col + "_log"].dropna()
        df2 = mouse_data[m2]
        df2 = df2.loc[df2[f'active{i}']]
        data2 = df2[col + "_log"].dropna()
        ks_stat, p_value = stats.ks_2samp(data1, data2)
        ks_results.append({"session": col, "mouse1": m1, "mouse2": m2, "KS_stat": ks_stat, "p_value": p_value})

# Convert results to DataFrames for visualization
skew_kurt_df = pd.DataFrame.from_dict({(i, j): skewness_kurtosis[i][j] 
                                       for i in skewness_kurtosis for j in skewness_kurtosis[i]}, 
                                      orient="index")

ks_results_df = pd.DataFrame(ks_results)

#%% Display results
tools.display_dataframe_to_user(name="Skewness & Kurtosis per Mouse", dataframe=skew_kurt_df)
tools.display_dataframe_to_user(name="Kolmogorov-Smirnov Test Results", dataframe=ks_results_df)
