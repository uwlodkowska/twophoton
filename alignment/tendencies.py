#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 22:27:01 2025

@author: ula
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:01:16 2025

@author: ula
"""

import numpy as np

import utils
import intersession
import cell_preprocessing as cp
import pandas as pd
import matplotlib.pyplot as plt
import constants
# from constants import ICY_COLNAMES, CTX_REGIONS
import sys
import yaml

#%%
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

#%%
regions = constants.LANDMARK_REGIONS#[[2,1],[3,2],[4,2],[6,2], [7,1], [11,1], [14,1]]
regions = constants.CTX_REGIONS

#%%

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]



#%%
regions = [[2,1],[3,2],[4,2],[6,2], [7,1], [11,1], [14,1]]
#regions = constants.LANDMARK_REGIONS + CTX_REGIONS

#%%

tendencies = []
tendencies_on_off = []
all_pooled = []
ovlaps = []
for m,r in regions:
    bgrv = []
    imgs = [utils.read_image(m, r, i) for i in constants.LANDMARK_FIRST_SESSIONS]#[1,2,3]]
    tstx = intersession.pooled_cells(m,r, constants.LANDMARK_FIRST_SESSIONS)#[1,2,3]]
    #all_pooled += [tstx]
    prev = tstx
    for i,img in enumerate(imgs):
        df = cp.optimize_centroids(prev, img, suff=str(i))
        bgr,cprod = cp.calculate_background_intensity(df, img)
        b_mean = np.mean(bgr.bg_intensity)
        b_std = np.std(bgr.bg_intensity)
        threshold = b_mean + 6 * b_std
        df[f'active{i}'] = df[f"int_optimized{i}"] > threshold#s pohl
        bgrv += [[b_mean, b_std]]
        prev = df
    idx_arr = []
    for i in ['0','1', '2']:
        q1 = df["int_optimized"+i].quantile(0.9)
        idx_arr += [df[df["int_optimized"+i] <= q1].index]
        plt.hist(df["int_optimized"+i], bins = 40, alpha=0.5)
    df.to_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
   
    
    bgr_data = pd.read_csv("/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/background.csv")
    
    d = dict.fromkeys(bgr_data.columns, 0)
    d['m'] = m
    d['r'] = r 
    for i, bgr in enumerate(bgrv):
        d[f'mean{i}'] = bgr[0]
        d[f'sdev{i}'] = bgr[1]

    bgr_data.loc[len(bgr_data)] = d

    bgr_data.drop(columns="Unnamed: 0", inplace=True)
    
    
    bgr_data.to_csv("/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/background.csv")
    s12 = len(set(idx_arr[0]) & set(idx_arr[1])) / len(idx_arr[0])
    s23 = len(set(idx_arr[1]) & set(idx_arr[2])) / len(idx_arr[1])
    ovlaps+=[[s12, s23]]    
    plt.title("m"+str(m))
    plt.show()
    tendencies += [intersession.find_intersession_tendencies_raw(df, sessions=[0,1,2])]
    tendencies_on_off += [intersession.find_intersession_tendencies_on_off(df, sessions=[0,1,2])]

#%% HELPER FUNCTIONS FOR PLOTTING TENDENCIES

def plot_tendencies(tendencies, title):
    colors = ['blue', 'orange', 'green']

    means = np.array([tendencies[:, :3].mean(axis=0), tendencies[:, 3:].mean(axis=0)])
    stds = np.array([tendencies[:, :3].std(axis=0), tendencies[:, 3:].std(axis=0)])
    print(means)
    print(stds)
    x_vals = np.array([0, 1])  # x-axis positions

    plt.figure(figsize=(6, 4))
    groups = ['up', 'down', 'stable']
    for i in range(3):
        plt.plot(x_vals, means[:, i], marker='o', linestyle='-', color=colors[i], label=groups[i])
        plt.fill_between(x_vals, means[:, i] - stds[:, i], means[:, i] + stds[:, i], color=colors[i], alpha=0.2)

    plt.xlabel('n='+str(len(tendencies)))
    plt.ylabel('Value')
    plt.title(title)
    plt.xticks([0, 1], ['different', 'same'])
    plt.legend()
    plt.show()



#%% TENDENCIES FOR CELLS MARKED AS ACTIVE/NOT ACTIVE
from scipy.stats import ttest_rel

tendencies_on_off = []
session_total = []
any_session = []
for m,r in regions:
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/m{m}r{r}.csv")
    st = []
    for i in range(3):
        st+= [df.loc[df[f"active{i}"]].shape[0]]
    session_total += [st]
    any_session += [df.loc[df["active0"] | df["active1"] | df["active2"]].shape[0]]
    df = df.loc[~df["active0"] | ~df["active1"] | ~df["active2"]]
    tendencies_on_off += [intersession.find_intersession_tendencies_on_off(df, sessions=[0,1,2])]

tendencies = np.array(tendencies_on_off)
tendencies = np.array([tendencies[i]/st for i, st in enumerate(any_session)])

plot_tendencies(tendencies, "On/off cells, threshold mean+6*std background - ctx/landmark pooled")


data = tendencies

# Separate conditions
condition_A = data[:, :3]  # First 3 columns
condition_B = data[:, 3:]  # Last 3 columns

# Perform paired t-tests
p_values = [ttest_rel(condition_A[:, i], condition_B[:, i]).pvalue for i in range(3)]

# Define significance levels
def significance_marker(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'

# Compute means and standard errors
means_A = np.mean(condition_A, axis=0)
means_B = np.mean(condition_B, axis=0)
std_A = np.std(condition_A, axis=0) / np.sqrt(condition_A.shape[0])
std_B = np.std(condition_B, axis=0) / np.sqrt(condition_B.shape[0])

# Plot
x = np.arange(3)
width = 0.35  # Bar width

fig, ax = plt.subplots()
bars_A = ax.bar(x - width/2, means_A, width, yerr=std_A, label="Same", alpha=0.7, capsize=5)
bars_B = ax.bar(x + width/2, means_B, width, yerr=std_B, label="Different", alpha=0.7, capsize=5)

# Add significance markers
for i in range(3):
    x_pos = x[i]
    y_max = max(means_A[i] + std_A[i], means_B[i] + std_B[i])  # Position above the highest bar
    marker = significance_marker(p_values[i])
    
    if marker != "n.s.":
        ax.text(x_pos, y_max + 0.05, marker, ha='center', fontsize=12, color='red')
ax.set_ylim([0, 0.6])  

# Labels and legend
ax.set_xlabel("Value Index")
ax.set_ylabel("Mean Value")
ax.set_title("Comparison of transition between same sessions vs different")
ax.set_xticks(x)
ax.set_xticklabels(["Up", "Down", "Stable"])
ax.legend(loc='upper left', bbox_to_anchor=(-0.05, 1))

plt.show()



#%% TENDENCIES FOR RAW CELL DATA

tendencies = []
session_total = []
any_session = []
for m,r in regions:
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
    tendencies += [intersession.find_intersession_tendencies_raw(df, sessions=[0,1,2])]

tendencies = np.array(tendencies)
tendencies = np.array([tend/np.sum(tend[:3]) for tend in tendencies])

plot_tendencies(tendencies, "Raw value")

#%% TENDENCIES WITH BACKGROUND SUBTRACTION

bgr_data = pd.read_csv("/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/background.csv")
#%%
tendencies = []
for m,r in regions:
    bgrv = []
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
    tendencies += [intersession.find_intersession_tendencies_bgr(df, bgr = np.array(bgrv), sessions=[0,1,2], k=0.4)]


tendencies = np.array(tendencies)
tendencies = np.array([tend/np.sum(tend[:3]) for tend in tendencies])

plot_tendencies(tendencies, "Background subtraction")


#%% CLASSES

bgr_data = pd.read_csv("/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/background.csv")

tendencies = []
any_session = []
for m,r in regions:
    bgrv = []
    df = pd.read_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/ctx_landmark/m{m}r{r}.csv")
    tendencies += [intersession.cell_classes(df, sessions=[0,1,2])]
    any_session += [df.loc[df["active0"] | df["active1"] | df["active2"]].shape[0]]


tendencies = np.array(tendencies)

means = np.mean(tendencies, axis=0)  # Mean for each class
stds = np.std(tendencies, axis=0)    # Standard deviation for each class
num_classes = tendencies.shape[1]

# Plot
x = np.arange(num_classes)  # Class indices
plt.bar(x, means, yerr=stds, capsize=5, alpha=0.7, label="Mean ± Std")
plt.xlabel("Class Index")
plt.ylabel("Amount")
plt.title("Class Statistics Across Subjects")
plt.xticks(x, [f"Class {i}" for i in x])  # Label classes
plt.legend()
plt.show()

#%%
#bgr_data.to_csv(f"/media/ula/DATADRIVE1/fos_gfp_tmaze/results/background.csv")
#%%
for i, st in enumerate(session_total):
    plt.plot(np.array(st)/any_session[i])

#%%
for i in range(3):
    print(df.loc[df['active'+str(i)]].shape[0])
#%%
res = intersession.find_intersession_tendencies_on_off(df, sessions=[0,1,2])

#%%

imgs = [utils.read_image(m, r, i) for i in [1,2,3]]
for img in imgs:
    bgr_vec, cprod = cp.calculate_background_intensity(df, img)
    print(np.mean(bgr_vec.bg_intensity), np.std(bgr_vec.bg_intensity))
#%%
print(np.mean(bgr_vec.bg_intensity), np.std(bgr_vec.bg_intensity))

#%%
plt.scatter(bgr[ICY_COLNAMES['ycol']],bgr.bg_intensity)
#%%
plt.hist(df["int_optimized0"], range=(0,125), bins = 40)
plt.hist(bgr["bg_intensity"], range=(0,125), alpha=0.5, bins = 40)
plt.plot()
'''
#%%
plt.ion()
for m,r in regions:
    imgs = [utils.read_image(m, r, i) for i in [1,2,3]]
    plt.figure()
    for i in imgs:
        bins = 100
        counts, bin_edges = np.histogram(i, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts)
    plt.title("m"+str(m))
    plt.show()
    
    


#%%

for i, t in enumerate(tendencies):
    #if regions[i][0] in [2,3,6]:
    plt.plot(t[:3]/np.sum(t[:3]), label = regions[i][0])
plt.legend()
plt.show()


for i, t in enumerate(tendencies):
    #if regions[i][0] in [2,3,6]:
    plt.plot(t[3:]/np.sum(t[:3]), label = regions[i][0])
plt.legend()
plt.show()

#%%
tendecies_np  = np.array(tendencies)
tendecies_np=tendecies_np/np.sum(tendecies_np[:3])
'''