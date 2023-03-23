#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 16:47:27 2023

@author: ula
"""

import pandas as pd
import math
import constants
import numpy as np
import intersession
import matplotlib.pyplot as plt
from scipy import stats
#%%
def top_intensity_from_file(mouse, region):
    top_df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    return top_df

def assign_quantile(df, colname, step=0.2):
    df[colname+"_q"] = (df[colname].rank(pct=True)/step).apply(math.floor)
    df.loc[df[colname]==math.floor(1/step)] = math.floor(1/step)-1
    
def show_transfer_rate_heatmaps(df, from_col, to_col, title):
    dc_arr = intersession.distribution_change_precalculated(df, from_col, to_col, 5)
    plt.imshow(dc_arr)
    plt.title(title)
    plt.show()
    
    
#%%
df01_all = []
df12_all = []
for m,r in constants.CTX_REGIONS:
    df = top_intensity_from_file(m,r)
    for i in range(3):
        assign_quantile(df, 'int_optimized'+str(i))
        df.loc[df["int_optimized"+str(i)+"_q"]==5] = 4

    df_filtered = df.loc[((df.int_optimized0_q != df.int_optimized1_q) | (df.int_optimized1_q != df.int_optimized2_q))]

    
    title_pre = "m{}r{}".format(m,r)
    nbins = 50
    histrange=(-175, 175)
    
    diff_arr1 = np.array(df_filtered.int_optimized0-df_filtered.int_optimized1)
    diff_arr2 = np.array(df_filtered.int_optimized1-df_filtered.int_optimized2)
    
    df01 = df.loc[(df.int_optimized0_q == df.int_optimized1_q)]
    df12 = df.loc[(df.int_optimized2_q == df.int_optimized1_q)]
    
    df01_arr = [df01.loc[df01.int_optimized0_q==i].shape[0]/df01.shape[0] for i in range(5)]
    df12_arr = [df12.loc[df12.int_optimized1_q==i].shape[0]/df12.shape[0] for i in range(5)]
    
    df01_all += [df01_arr]
    df12_all += [df12_arr]
    
bwidth = 0.25   
br1 = np.arange(5)
br2 = [x + bwidth for x in br1]
for i in range(len(df01_all)):
    plt.plot(br1, df01_all[i], marker='o', linestyle="None", color='b')
    plt.plot(br2, df12_all[i], marker='o', linestyle="None", color='g')
plt.title("cells stable between sessions - quantile distribution")
plt.legend(['CL','LL'])
plt.show()
#%%    
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    
    colname_temp = 'int_optimized{}_q'
    for i in range(2):
        axes[i].scatter(df_filtered.int_optimized0, diff_arr1, s=1, c=df_filtered[colname_temp.format(i+1)])
    plt.title(title_pre)
    plt.show()
    #stats_arr += [[np.mean(diff_arr1), np.mean(diff_arr2), np.std(diff_arr1), np.std(diff_arr2)]]
    #plt.hist(df_filtered.int_optimized1-df_filtered.int_optimized2, range=histrange, bins = nbins, alpha=0.5)
    df_filtered.sort_values(by='int_optimized1', inplace=True)
    df_filtered.sort_values(by='int_optimized0', inplace=True)
    plt.title("m" + str(m)+"_r"+str(r)+"_intensity LC")
    plt.plot(np.array(df_filtered.int_optimized0_q))
    plt.plot(np.array(df_filtered.int_optimized1_q), alpha=0.5)
    plt.show()
    
    df_filtered.sort_values(by='int_optimized2', inplace=True)
    df_filtered.sort_values(by='int_optimized1', inplace=True)
    plt.title("m" + str(m)+"_r"+str(r)+"_intensity CC")
    plt.plot(np.array(df_filtered.int_optimized1_q))
    plt.plot(np.array(df_filtered.int_optimized2_q), alpha=0.5)
    plt.show()
    
    # show_transfer_rate_heatmaps(df, 'int_optimized0_q', 'int_optimized1_q', title_pre + " top 10% ctx landmark")
    # show_transfer_rate_heatmaps(df, 'int_optimized1_q', 'int_optimized2_q', title_pre + " top 10% LL")
    # show_transfer_rate_heatmaps(df_filtered, 'int_optimized0_q', 'int_optimized1_q', title_pre + "top 10% ctx landmark filtered")
    # show_transfer_rate_heatmaps(df_filtered, 'int_optimized1_q', 'int_optimized2_q', title_pre + "top 10% LL filtered")
    
# for reg in stats_arr:
#     plt.plot(['CL', 'LL'],reg[:2]/reg[0], marker='o')
# plt.title("Mean cell intensity change CLL")
# plt.show()

#%%

for m,r in constants.CTX_REGIONS:
    df = top_intensity_from_file(m,r)
    all_mice += [calculate_integrated_value_by_column(df,[])]
all_mice = np.array(all_mice)   
plt.title("Integrated intensity CLL")   
for i, reg in enumerate(all_mice):
    plt.plot(reg, marker='o', label = str(constants.CTX_REGIONS[i]))
ax = plt.subplot(111)