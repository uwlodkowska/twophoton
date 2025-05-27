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

import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import constants
# from constants import ICY_COLNAMES, CTX_REGIONS
import sys
import yaml
import gc
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark_rev.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


#%%

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
OPT_PATH = config["experiment"]["optimized_path"]


regions = config["experiment"]["regions"][0]
group_session_order = config["experiment"]["session_order"][0]
#%%
import psutil, os
def print_mem_usage():
    print("RAM used (MB):", psutil.Process(os.getpid()).memory_info().rss // 1024 // 1024)


#%%
#regions = [[2,1],[3,2],[4,2],[6,2], [7,1], [11,1], [14,1]]
#regions = constants.LANDMARK_REGIONS + CTX_REGIONS

#%% OPTIMIZE CENTROIDS AND MARK ON/OFF

tendencies = []
tendencies_on_off = []
all_pooled = []
ovlaps = []

regions=[[0,1]]
for m,r in regions:
    bgrv = []
    #imgs = [utils.read_image(m, r, i, config) for i in group_session_order]
    tstx = intersession.pooled_cells(m,r, group_session_order, config, test=True)
    #all_pooled += [tstx]
    prev = tstx
    for i, session_id in enumerate(group_session_order):
        img = utils.read_image(m, r, session_id, config)
        df = cp.optimize_centroids(prev, img, suff=str(i))
        print("centroids optimized for session ", session_id)
        bgr,_ = cp.calculate_background_intensity(df, img)
        print("background calculated for ", session_id)
        b_mean = np.mean(bgr.bg_intensity)
        b_std = np.std(bgr.bg_intensity)
        threshold = b_mean + 6 * b_std
        df[f'active{i}'] = df[f"int_optimized{i}"] > threshold#s pohl
        bgrv += [[b_mean, b_std]]
        prev = df
        print_mem_usage()
        
        gc.collect()
    idx_arr = []
    for i in ['0','1', '2']:
        q1 = df["int_optimized"+i].quantile(0.9)
        idx_arr += [df[df["int_optimized"+i] <= q1].index]
        plt.hist(df["int_optimized"+i], bins = 40, alpha=0.5)
    df.to_csv(OPT_PATH+f"m{m}r{r}.csv")
   
    bgr_data = pd.read_csv(BGR_DIR)
    
    d = dict.fromkeys(bgr_data.columns, 0)
    d['m'] = m
    d['r'] = r 
    for i, bgr in enumerate(bgrv):
        d[f'mean{i}'] = bgr[0]
        d[f'sdev{i}'] = bgr[1]

    bgr_data.loc[len(bgr_data)] = d

    bgr_data.drop(columns="Unnamed: 0", inplace=True)
    
    
    bgr_data.to_csv(BGR_DIR)
    s12 = len(set(idx_arr[0]) & set(idx_arr[1])) / len(idx_arr[0])
    s23 = len(set(idx_arr[1]) & set(idx_arr[2])) / len(idx_arr[1])
    ovlaps+=[[s12, s23]]    
    plt.title("m"+str(m))
    plt.show()
    tendencies += [intersession.find_intersession_tendencies_raw(df, sessions=[0,1,2])]
    tendencies_on_off += [intersession.find_intersession_tendencies_on_off(df, sessions=[0,1,2])]

#%%
sdfghj
for m,r in regions:
    bgrv = []
    #imgs = [utils.read_image(m, r, i, config) for i in group_session_order]
    tstx = intersession.pooled_cells(m,r, group_session_order, config)
    #all_pooled += [tstx]
    prev = tstx
    for i, session_id in enumerate(group_session_order):
        img = utils.read_image(m, r, session_id, config)

#%% BACKGROUND DATA CALCULATION
for m,r in regions:
    if m == 2:
        continue
    bgrv = []
    imgs = [utils.read_image(m, r, i, config) for i in group_session_order]#[1,2,3]]
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    for i,img in enumerate(imgs):
        bgr,cprod = cp.calculate_background_intensity(df, img)
        b_mean = np.mean(bgr.bg_intensity)
        b_std = np.std(bgr.bg_intensity)
        bgrv += [[b_mean, b_std]]
    idx_arr = []
    bgr_data = pd.read_csv(BGR_DIR)
    
    d = dict.fromkeys(bgr_data.columns, 0)
    d['m'] = m
    d['r'] = r 
    for i, bgr in enumerate(bgrv):
        d[f'mean{i}'] = bgr[0]
        d[f'sdev{i}'] = bgr[1]

    bgr_data.loc[len(bgr_data)] = d

    bgr_data.drop(columns="Unnamed: 0", inplace=True)
    bgr_data.to_csv(BGR_DIR)


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
#%% BARPLOT TENDENCIES


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def plot_tendencies_b(tendencies, title):
    # Define groups and colors
    groups = ['up', 'down', 'stable']
    conditions = ['different', 'same']
    colors = ["#66c2a5",  # teal-ish
              sns.color_palette("colorblind")[1]]  # orange-ish

    # Compute mean and SEM (standard error of the mean)
    means = np.array([tendencies[:, :3].mean(axis=0), tendencies[:, 3:].mean(axis=0)])
    sems = np.array([tendencies[:, :3].std(axis=0, ddof=1) / np.sqrt(tendencies.shape[0]),
                     tendencies[:, 3:].std(axis=0, ddof=1) / np.sqrt(tendencies.shape[0])])
    
    print("Means:\n", means)
    print("SEMs:\n", sems)

    # X positions
    n_groups = len(groups)
    bar_width = 0.35
    x = np.arange(n_groups)

    # Create figure
    plt.figure(figsize=(6, 4))

    # Plot bars with SEM
    bars1 = plt.bar(x - bar_width/2, means[0], bar_width, yerr=sems[0],
                    label='different', color=colors[0], capsize=5)
    bars2 = plt.bar(x + bar_width/2, means[1], bar_width, yerr=sems[1],
                    label='same', color=colors[1], capsize=5)

    # Paired t-tests and significance stars
    offset = 0.05 * np.max(means + sems)
    for i in range(n_groups):
        y1 = means[0, i] + sems[0, i]
        y2 = means[1, i] + sems[1, i]
        max_y = max(y1, y2) + offset
        x1 = x[i] - bar_width/2
        x2 = x[i] + bar_width/2

        # t-test
        stat, p = stats.ttest_rel(tendencies[:, i], tendencies[:, i + 3])
        print(p)
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        elif p < 0.1:
            stars = f'p-value={round(p, 2)}'


        # Draw line and annotation
        if p<0.1:
            plt.plot([x1, x1, x2, x2], [max_y, max_y + offset, max_y + offset, max_y], color='black', lw=1.5)
            plt.text((x1 + x2)/2, max_y + offset * 1.1, stars, ha='center', va='bottom')

    # Labels and legend
    plt.xticks(x, groups)
    plt.ylabel('Value')
    plt.xlabel('n = 7')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print()



#%% TENDENCIES FOR CELLS MARKED AS ACTIVE/NOT ACTIVE
from scipy.stats import ttest_rel

tendencies_on_off = []
session_total = []
any_session = []
bgr_data = pd.read_csv(BGR_DIR)

for m,r in [[0,1]]:#config["experiment"]["regions"][0]:
    bgrv = []
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    df = cp.find_active_cells(df, bgrv,3)
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
plot_tendencies_b(tendencies, "Cell classes for CLL group")
#%%

pooled_df = []  # List to collect sampled DataFrames

for m, r in config["experiment"]["regions"][1]:
    
    
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    bgrv = []
    for i in range(3):
        bgrv += list(
            bgr_data.loc[(bgr_data['m'] == m) & (bgr_data['r'] == r), [f'mean{i}', f'sdev{i}']].values
        )

    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    for i in range(3):
        df[f'int_optimized{i}'] = df[f'int_optimized{i}']-bgrv[i][0]

    # Randomly sample 200 rows (set replace=True if you want to allow duplicates if df has < 200 rows)
    sampled_df = df.sample(n=300, random_state=42)  # use a seed for reproducibility if desired

    # Add metadata if needed (like mouse or region)
    sampled_df["mouse"] = m
    sampled_df["region"] = r

    pooled_df.append(sampled_df)

# Combine all into one DataFrame
pooled_df = pd.concat(pooled_df, ignore_index=True)

#%%
pooled_df['s0s1'] = pooled_df["int_optimized1"] - pooled_df["int_optimized0"]
pooled_df['s1s2'] = pooled_df["int_optimized2"] - pooled_df["int_optimized1"]

plt.hist(pooled_df['s0s1'], bins=50)
plt.hist(pooled_df['s1s2'], bins=50, alpha=0.3)
plt.plot()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
from statsmodels.stats.multitest import multipletests

def plot_column_group_comparison(df, columns, title='', paired=True, colors=None, correction='fdr_bh'):
    """
    Plot mean ± SEM for multiple columns in a DataFrame and show significance between all pairs.
    Supports multiple comparisons correction.

    Parameters:
        df (pd.DataFrame): DataFrame containing the columns
        columns (list of str): Column names to compare
        title (str): Plot title
        paired (bool): Use paired t-tests (True) or independent (False)
        colors (list): Optional list of colors for bars
        correction (str): Correction method ('fdr_bh', 'bonferroni', 'holm', or None)
    """
    data = df[columns].dropna()
    n = len(data)
    means = data.mean().values
    sems = data.sem().values
    x = np.arange(len(columns))

    if colors is None:
        colors = plt.cm.tab10.colors[:len(columns)]

    # Plot
    plt.figure(figsize=(max(4, len(columns)*1.2), 4))
    bars = plt.bar(x, means, yerr=sems, capsize=5, color=colors, tick_label=columns)
    plt.ylabel('Mean value')
    plt.title(title)

    # Collect all pairwise p-values
    pvals = []
    pairs = []
    for i, j in combinations(range(len(columns)), 2):
        col1, col2 = columns[i], columns[j]
        vals1, vals2 = data[col1].values, data[col2].values

        if paired:
            stat, p = stats.ttest_rel(vals1, vals2)
        else:
            stat, p = stats.ttest_ind(vals1, vals2)

        pvals.append(p)
        pairs.append((i, j))

    # Apply correction
    if correction is not None:
        reject, pvals_corrected, _, _ = multipletests(pvals, method=correction)
    else:
        reject = [p < 0.05 for p in pvals]
        pvals_corrected = pvals

    # Annotate significance
    y_max = (means + sems).max()
    offset = 0.05 * y_max
    line_height = y_max + offset
    step = offset * 1.5

    for idx, (i, j) in enumerate(pairs):
        x1, x2 = x[i], x[j]
        y = line_height

        plt.plot([x1, x1, x2, x2], [y, y+offset, y+offset, y], color='black', lw=1.2)

        p = pvals_corrected[idx]
        if p < 0.001:
            stars = '***'
        elif p < 0.01:
            stars = '**'
        elif p < 0.05:
            stars = '*'
        else:
            stars = 'ns'

        plt.text((x1 + x2)/2, y + offset * 1.1, stars, ha='center', va='bottom', fontsize=10)
        line_height += step

    plt.tight_layout()
    plt.show()



#%%

plot_column_group_comparison(pooled_df,
                             ['int_optimized0', 'int_optimized1', 'int_optimized2'],
                             title='Z-Scored Intensities Across Sessions',
                             correction='fdr_bh')  # or 'bonferroni', 'holm', None



#%%
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

#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

def plot_on_off_tendencies(regions_ctx, regions_landmark, DIR_PATH):
    group_data = {}
    group_labels = {"ctx": regions_ctx, "landmark": regions_landmark}

    for group, regions in group_labels.items():
        tendencies_on_off = []
        session_total = []
        any_session = []

        for m,r in regions:
            df = pd.read_csv(DIR_PATH.format(m=m, r=r))
            st = []
            for i in range(3):
                st+= [df.loc[df[f"active{i}"]].shape[0]]
            session_total += [st]
            any_session += [df.loc[df["active0"] | df["active1"] | df["active2"]].shape[0]]
            df = df.loc[~df["active0"] | ~df["active1"] | ~df["active2"]]
            tendencies_on_off += [intersession.find_intersession_tendencies_on_off(df, sessions=[0,1,2])]

        tendencies = np.array(tendencies_on_off)
        tendencies = np.array([tendencies[i]/st for i, st in enumerate(any_session)])
        group_data[group] = tendencies

    # Prepare data for plotting
    categories = ["Up", "Down", "Stable"]
    data_list = []
    for group, data in group_data.items():
        for subj in data:
            data_list.extend([{"Group": group, "Category": cat, "Value": val} for cat, val in zip(categories, subj[:3])])
            data_list.extend([{"Group": group, "Category": cat, "Value": val} for cat, val in zip(categories, subj[3:])])

    df_plot = pd.DataFrame(data_list)
    df_plot["Condition"] = ["Same"] * (len(df_plot) // 2) + ["Different"] * (len(df_plot) // 2)

    # Calculate significance
    results = []
    for cat in categories:
        for group in group_data:
            data = group_data[group]
            A = data[:, categories.index(cat)]
            B = data[:, categories.index(cat) + 3]
            p = ttest_rel(A, B).pvalue
            results.append({"Group": group, "Category": cat, "p-value": p})

    # Plotting with seaborn
    sns.set(style="whitegrid")
    for group in group_data:
        subset = df_plot[df_plot.Group == group]
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x="Category", y="Value", hue="Condition", data=subset, errorbar="se", palette="Set2")

        for res in [r for r in results if r["Group"] == group]:
            p = res["p-value"]
            cat = res["Category"]
            idx = categories.index(cat)
            if p < 0.05:
                y1 = subset[(subset.Category == cat) & (subset.Condition == "Same")]["Value"].max()
                y2 = subset[(subset.Category == cat) & (subset.Condition == "Different")]["Value"].max()
                y = max(y1, y2) + 0.05
                x1, x2 = idx - 0.2, idx + 0.2
                plt.plot([x1, x2], [y, y], color='black')
                marker = "***" if p < 0.001 else "**" if p < 0.01 else "*"
                plt.text((x1 + x2) / 2, y + 0.01, marker, ha='center', color='black')

        plt.title(f"On/Off Transitions - {group}")
        plt.ylim(0, 0.6)
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)


plot_on_off_tendencies(config["experiment"]["regions"][0], config["experiment"]["regions"][1], DIR_PATH)

#%% TENDENCIES FOR RAW CELL DATA

tendencies = []
session_total = []
any_session = []
for m,r in regions:
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    tendencies += [intersession.find_intersession_tendencies_raw(df, sessions=[0,1,2])]

tendencies = np.array(tendencies)
tendencies = np.array([tend/np.sum(tend[:3]) for tend in tendencies])

plot_tendencies(tendencies, "Raw value")

#%% TENDENCIES WITH BACKGROUND SUBTRACTION

bgr_data = pd.read_csv(BGR_DIR)
#%%
tendencies = []
for m,r in regions:
    bgrv = []
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
    tendencies += [intersession.find_intersession_tendencies_bgr(df, bgr = np.array(bgrv), sessions=[0,1,2], k=0.4)]


tendencies = np.array(tendencies)
tendencies = np.array([tend/np.sum(tend[:3]) for tend in tendencies])

plot_tendencies(tendencies, "Background subtraction")


#%% CLASSES normality check

def check_normality(data):
    """
    Performs normality tests on the given dataset.
    - Uses Shapiro-Wilk test for small samples.
    - Uses Kolmogorov-Smirnov test as an alternative.
    - Plots Q-Q plot for visualization.
    
    :param data: 1D numpy array or list of values
    """
    print(data)
    shapiro_p = stats.shapiro(data)[1]  # p-value from Shapiro-Wilk test
    ks_p = stats.kstest(data, 'norm')[1]  # p-value from KS test

    # plt.figure(figsize=(6, 4))
    # sm.qqplot(data, line='s', fit=True)
    # plt.title("Q-Q Plot")
    # plt.show()

    #return {"Shapiro-Wilk p": shapiro_p, "Kolmogorov-Smirnov p": ks_p}
    return shapiro_p > 0.05

bgr_data = pd.read_csv(BGR_DIR)
for thre in range(1,6):
    for group in range(2):
        regions = config["experiment"]["regions"][group]
        group_session_order = config["experiment"]["session_order"][group]
        tendencies = []
        for m,r in regions:
            bgrv = []
            for i in range(3):
                bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
            df = pd.read_csv(DIR_PATH.format(m=m, r=r))
            df = cp.find_active_cells(df, bgrv, 2*thre)
            tendencies += [intersession.cell_classes(df, sessions=[0,1,2])]
        
        
        tendencies = np.array(tendencies)
        
        num_classes = tendencies.shape[1]
        for i in range(num_classes):
            class_p = check_normality(tendencies[:, i])
            if class_p < 0.05:
                print(f"Group {group_session_order[0][0]}, threshold {thre}, class {i}: {class_p}")



#%% in-group comparison


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def within_group_comparison(config, BGR_DIR, DIR_PATH):
    """
    Compares classes within each group.
    - Uses paired t-tests if normal.
    - Uses Wilcoxon signed-rank test if non-normal.
    - Plots class distributions per group.
    - Separates plots for classes active in one vs two sessions.
    - One plot per group per session count with significance markers between classes.
    - Plots in teal tone.
    """

    bgr_data = pd.read_csv(BGR_DIR)
    results = []
    all_data = []
    class_sessions = {
        0: {1}, 1: {2}, 2: {3},
        3: {1, 2, 3}, 4: {1, 2}, 5: {1, 3}, 6: {2, 3}
    }

    class_labels = {
        0: "S1", 1: "S2", 2: "S3",
        3: "S1&S2&S3", 4: "S1&S2", 5: "S1&S3", 6: "S2&S3"
    }

    group_labels = {0: "ctx", 1: "landmark"}
    group_tendencies = {0: [], 1: []}

    teal_color = "#66c2a5"  # Seaborn Set2 teal color

    for group in range(2):  # Two groups of mice
        regions = config["experiment"]["regions"][group]
        for m, r in regions:
            bgrv = []
            for i in range(3):
                bgrv += list(bgr_data.loc[(bgr_data['m'] == m) & (bgr_data['r'] == r), [f'mean{i}', f'sdev{i}']].values)

            df = pd.read_csv(DIR_PATH.format(m=m, r=r))
            df = cp.find_active_cells(df, bgrv, 3)
            class_fractions = intersession.cell_classes(df, sessions=[0, 1, 2])
            group_tendencies[group].append(class_fractions)

            for class_idx, fraction in enumerate(class_fractions):
                all_data.append({
                    "Group": group_labels[group],
                    "Class": class_labels[class_idx],
                    "ClassIndex": class_idx,
                    "Fraction": fraction,
                    "Sessions": len(class_sessions[class_idx])
                })

    df_plot = pd.DataFrame(all_data)

    for session_count in [1, 2]:
        for group in [0, 1]:
            group_name = group_labels[group]
            subset = df_plot[(df_plot['Sessions'] == session_count) & (df_plot['Group'] == group_name)]
            plt.figure(figsize=(10, 6))
            sns.boxplot(x="Class", y="Fraction", data=subset, showfliers=False, color=teal_color, linewidth=1, fliersize=0)
            sns.stripplot(x="Class", y="Fraction", data=subset, dodge=True, jitter=True, alpha=0.6, color="black")

            relevant_classes = sorted(subset['ClassIndex'].unique())
            tendencies = np.array(group_tendencies[group])
            for i in range(len(relevant_classes)):
                for j in range(i+1, len(relevant_classes)):
                    ci = relevant_classes[i]
                    cj = relevant_classes[j]
                    if len(class_sessions[ci]) == len(class_sessions[cj]) == session_count:
                        data_i = tendencies[:, ci]
                        data_j = tendencies[:, cj]
                        if check_normality(data_i) and check_normality(data_j):
                            stat, p = stats.ttest_rel(data_i, data_j)
                            test = "Paired t-test"
                        else:
                            stat, p = stats.wilcoxon(data_i, data_j)
                            test = "Wilcoxon"

                        results.append({
                            "Group": group_name,
                            "Comparison": f"{class_labels[ci]} vs {class_labels[cj]}",
                            "p-value": p,
                            "Test": test
                        })

                        if p < 0.05:
                            x1 = relevant_classes.index(ci)
                            x2 = relevant_classes.index(cj)
                            y = max(subset[subset['ClassIndex']==ci]['Fraction'].max(),
                                    subset[subset['ClassIndex']==cj]['Fraction'].max()) + 0.05
                            plt.plot([x1, x2], [y, y], color='black')
                            plt.text((x1 + x2) / 2, y + 0.01, '*', ha='center', va='bottom', fontsize=16)

            plt.title(f"{group_name} - Classes Active in {session_count} Session(s)")
            plt.xlabel("Class")
            plt.ylabel("Fraction of Active Cells")
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)



results_df = within_group_comparison(config, BGR_DIR, DIR_PATH)

# Display the comparison results
#import ace_tools as tools  # If using a notebook with interactive display
#tools.display_dataframe_to_user("Within-Group Comparison Results", results_df)

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def between_group_comparison(config, BGR_DIR, DIR_PATH):
    """
    Compares each class between two groups of animals.
    - Uses independent t-tests if normal.
    - Uses Mann-Whitney U test if non-normal.
    - Plots distributions per class, grouped by animal group.
    """

    bgr_data = pd.read_csv(BGR_DIR)
    results = []
    all_data = []

    for group in range(2):  # Two groups of mice
        regions = config["experiment"]["regions"][group]

        for m, r in regions:            
            bgrv = []
            for i in range(3):
                bgrv += list(bgr_data.loc[(bgr_data['m'] == m) & (bgr_data['r'] == r), [f'mean{i}', f'sdev{i}']].values)
            df = pd.read_csv(DIR_PATH.format(m=m, r=r))
            df = cp.find_active_cells(df, bgrv, 4)
            class_fractions = intersession.cell_classes_diff_norm(df, sessions=[0, 1, 2])

            for class_idx, fraction in enumerate(class_fractions):
                all_data.append({"Group": f"Group {group}", "Class": f"Class {class_idx}", "Fraction": fraction})

    df_all = pd.DataFrame(all_data)

    # Perform between-group comparisons for each class
    classes = df_all['Class'].unique()

    for class_label in classes:
        class_data = df_all[df_all['Class'] == class_label]

        group0 = class_data[class_data['Group'] == 'Group 0']['Fraction'].values
        group1 = class_data[class_data['Group'] == 'Group 1']['Fraction'].values

        if check_normality(group0) and check_normality(group1):
            stat, p = stats.ttest_ind(group0, group1)
            test_used = "Independent t-test"
        else:
            stat, p = stats.mannwhitneyu(group0, group1)
            test_used = "Mann-Whitney U"

        results.append({
            "Class": class_label,
            "p-value": p,
            "Test": test_used
        })

    results_df = pd.DataFrame(results)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Class", y="Fraction", hue="Group", data=df_all, showfliers=False, palette="Set2")
    sns.stripplot(x="Class", y="Fraction", hue="Group", data=df_all, dodge=True, jitter=True, alpha=0.6, linewidth=0.5)

    plt.title("Between-Group Comparison of Class Fractions")
    plt.xlabel("Class Index")
    plt.ylabel("Fraction of Active Cells")
    plt.legend(title="Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return results_df

# Usage
results_df = between_group_comparison(config, BGR_DIR, DIR_PATH)
print(results_df)







#%%


def plot_class_distributions(config, BGR_DIR, DIR_PATH):
    """
    Plots the fraction of cells in each class for both groups.

    - Extracts cell classes dynamically for each group.
    - Normalizes by total active cells.
    - Uses boxplots to visualize group-wise distribution per class.
    """

    bgr_data = pd.read_csv(BGR_DIR)
    all_data = []

    for group in range(2):  # Two groups of mice
        regions = config["experiment"]["regions"][group]

        for m, r in regions:
            bgrv = []
            for i in range(3):
                bgrv += list(bgr_data.loc[(bgr_data['m'] == m) & (bgr_data['r'] == r), [f'mean{i}', f'sdev{i}']].values)

            df = pd.read_csv(DIR_PATH.format(m=m, r=r))
            class_fractions = intersession.cell_classes(df, sessions=[0, 1, 2])

          
            for class_idx, fraction in enumerate(class_fractions):
                all_data.append({"Group": f"Group {group}", "Class": f"Class {class_idx}", "Fraction": fraction})

    # Convert to DataFrame for plotting
    df_plot = pd.DataFrame(all_data)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Class", y="Fraction", hue="Group", data=df_plot, showfliers=False, palette="Set2")
    sns.stripplot(x="Class", y="Fraction", hue="Group", data=df_plot, dodge=True, jitter=True, alpha=0.6)

    plt.title("Fraction of Cells per Class Across Groups")
    plt.xlabel("Class Index")
    plt.ylabel("Fraction of Active Cells")
    plt.legend(title="Group")
    plt.show()

plot_class_distributions(config, BGR_DIR, DIR_PATH)

#%% CLASSES 1 threshold
bgr_data = pd.read_csv(BGR_DIR)



tendencies = []
    
for group in range(1):
    regions = config["experiment"]["regions"][group]
    group_session_order = config["experiment"]["session_order"][group]
    for m,r in regions:
        bgrv = []
        for i in range(3):
            bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
        df = pd.read_csv(DIR_PATH.format(m=m, r=r))
        #df = cp.find_active_cells(df, bgrv, 2)
        tendencies += [intersession.cell_classes_diff_norm(df, sessions=[0,1,2])]


tendencies = np.array(tendencies)

means = np.mean(tendencies, axis=0)  # Mean for each class
stds = np.std(tendencies, axis=0)    # Standard deviation for each class
n = tendencies.shape[0]  # Number of subjects
sem = stds / np.sqrt(n)  # Standard error of the mean

# Plot
x = np.arange(means.shape[0])  # Class indices
plt.bar(x, means, yerr=sem, capsize=5, alpha=0.7, label="Mean ± SEM")
plt.xlabel("Class Index")
plt.ylabel("Amount")
plt.title("Class Statistics Across Subjects (Mean ± SEM) LC pooled" )#+ " ".join(group_session_order))
plt.xticks(x, [f"Class {i}" for i in x])  # Label classes
plt.legend()
plt.show()


#%% CLASSES across thresholds

bgr_data = pd.read_csv(BGR_DIR)

tendencies = []

thresholds  = np.arange(1,6)
for m,r in regions:
    
    subject_tendencies = []  # Stores tendencies per threshold for one subject
    bgrv = []
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    
    for i in range(3):
        bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)
    for j in thresholds:
        df = cp.find_active_cells(df, bgrv, j*2)
        subject_tendencies.append(intersession.cell_classes_diff_norm(df, sessions=[0,1,2]))
    tendencies.append(subject_tendencies)
    
    
    
tendencies = np.array(tendencies)  # Shape: (num_subjects, num_thresholds, num_classes)


# Compute mean and std across subjects
means = np.mean(tendencies, axis=0)  # Mean for each class
stds = np.std(tendencies, axis=0)    # Standard deviation for each class
print(stds)
# Plot
plt.figure(figsize=(8, 6))

num_classes = tendencies.shape[2]

for class_idx in range(3):
    plt.errorbar(2*thresholds, means[:, class_idx], yerr=stds[:, class_idx],
                 marker='o', linestyle='-', capsize=5, label=f"Class {class_idx}")

plt.xlabel("Threshold")
plt.ylabel("Fraction of cells in class")
plt.title("Fraction of cells per class across thresholds "+ " ".join(group_session_order))
plt.legend()
plt.show()


for class_idx in range(4,7):
    plt.errorbar(2*thresholds, means[:, class_idx], yerr=stds[:, class_idx],
                 marker='o', linestyle='-', capsize=5, label=f"Class {class_idx}")

plt.xlabel("Threshold")
plt.ylabel("Fraction of cells in class")
plt.title("Fraction of cells per class across thresholds "+ " ".join(group_session_order))
plt.legend()
plt.show()


#%% check qualities of cells active in two sessions
for m,r in regions:
    colname = 'active'
    sessions = [0,1,2]
    df = pd.read_csv(DIR_PATH.format(m=m, r=r))
    df = df.loc[df[colname+str(sessions[2])]]
    df['class6'] = (~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])& (df[colname+str(sessions[2])])
    #plt.scatter(df['int_optimized1'], df['int_optimized2'],s=1, c = df['class6'])
    plt.hist(df['int_optimized1'], range=[0,200], bins = 30)
    df = df.loc[df['class6']]
    plt.hist(df['int_optimized1'], range=[0,200], bins = 30)
    plt.show()
    
    
    
#%% check normality of intensity for differ3ent thresholds

bgr_data = pd.read_csv(BGR_DIR)

tendencies = []

thresholds  = np.arange(1,6)
thresholds=[0]
for j in thresholds:
    for m,r in regions:
        subject_tendencies = []  # Stores tendencies per threshold for one subject
        bgrv = []
        df = pd.read_csv(DIR_PATH.format(m=m, r=r))
        
        for i in range(3):
            bgrv += list(bgr_data.loc[(bgr_data['m']==m)&(bgr_data['r']==r), [f'mean{i}',f'sdev{i}']].values)

        df = cp.find_active_cells(df, bgrv, j*2)
        #df = df.loc[df['active0']]
        plt.hist(df.int_optimized0, bins=40)
        plt.title(f'm{m} r{r} thre {j} normal {check_normality(df.int_optimized0)}')
        #print(f'm{m} r{r} thre {j} normal {check_normality(np.array(df.int_optimized0))}')
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