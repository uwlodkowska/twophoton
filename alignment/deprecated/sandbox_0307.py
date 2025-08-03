#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:34:25 2023

@author: ula
"""
import utils
import constants
import matplotlib.pyplot as plt
import pandas as pd
import cell_preprocessing as cp
import intersession
import numpy as np
from scipy.stats.stats import pearsonr
#%%
def optimize_last_session(mouse, region, session_code):
    df = utils.read_single_session_cell_data(mouse, region, [session_code])
    img = utils.read_image(mouse, region, session_code)
    df['in_calculated'] = df.apply(cp.calculate_intensity,img = img, axis = 1)
    df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
    df = cp.optimize_centroids(df, img)
    df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+session_code+"_optimized.csv")
#%%
def calculate_integrated_int(mouse, region, sessions):
    intensity_sum_icy = []
    intensity_sum = []
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    for i in range(len(sessions)):
        #df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized_n.csv")
        df['lower_lim'] = df.int_optimized*0.8
        df['upper_lim'] = df.int_optimized*1.2
        increase = df[df.int_optimized_n>df.upper_lim]
        decrease = df[df.int_optimized_n<df.lower_lim]
        intensity_sum += [1-(increase.shape[0]+decrease.shape[0])/df.shape[0]]
        #intensity_sum_icy += [df[constants.ICY_COLNAMES["mean_intensity"]].sum()]
    return np.array([intensity_sum[i] for i in [0,1]])
    #return np.array(intensity_sum)/intensity_sum[0]

#%%
def calculate_integrated_value_by_column(df, columns, title):
    sum_arr = []
    for col in columns:
        sum_arr += [df[col].sum()]
    return np.array(sum_arr)/sum_arr[0]
#%%
def distribution_change_all_sessions(mouse, region, sessions):
    trs = []
    
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]
    dfs = utils.read_single_session_cell_data(mouse, region, sessions)

    for i in range(len(sessions)-1):
        df = dfs[i]
        df['in_calculated'] = df.apply(cp.calculate_intensity,img = imgs[i], axis = 1)
        df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
        cp.optimize_centroids(df, imgs[i])
        df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+sessions[i] +"_optimized.csv")
        df_n = df.copy()
        cp.optimize_centroids(df_n, imgs[i+1], "_n")
        df_n.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i+1] +"_optimized_n.csv")
        trs += [intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
    df = utils.read_single_session_cell_data(mouse, region, ['landmark2'])
    img = utils.read_image(mouse, region, 'landmark2')
    df['in_calculated'] = df.apply(cp.calculate_intensity,img = img, axis = 1)
    df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
    cp.optimize_centroids(df, img)
    df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_landmark2_optimized.csv")
    return trs
    
#%%
all_mice = pd.DataFrame(columns = ['int', 'behav'])
behav = utils.read_behav_data()
for m,r in constants.CTX_REGIONS:    
    performance = behav.loc[m, 'ctx_avg']
    all_mice.loc[len(all_mice)] = np.append([calculate_integrated_int(m,r,constants.CTX_FIRST_SESSIONS)[1]],
                          performance)
# for m,r in constants.LANDMARK_REGIONS:    
#     performance = behav.loc[m, 'landmark_avg']
#     all_mice.loc[len(all_mice)] = np.append(calculate_integrated_int(m,r,constants.LANDMARK_FIRST_SESSIONS)[1],
#                           performance)
plt.scatter(all_mice.behav, all_mice.int)
plt.title("Success rate vs fraction of stable cells from L1 to L2")
plt.ylabel("Stable cells") 
plt.xlabel("Ctx score")
plt.show() 
pearsonr(all_mice.behav, all_mice.int)
#%%

plt.title("Integrated int CLL")   
for i, reg in enumerate(all_mice):
    plt.plot(['C', 'L1', 'L2'],reg, marker='o', label = str(constants.CTX_REGIONS[i]))
ax = plt.subplot(111)

# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0,3)
plt.show()
print(np.mean(all_mice, axis = 0))

all_mice = []
for m,r in constants.LANDMARK_REGIONS:
    all_mice += [calculate_integrated_int(m,r,constants.LANDMARK_FIRST_SESSIONS)]
all_mice = np.array(all_mice)
plt.title("Integrated int LCC")     
    
for i, reg in enumerate(all_mice):
    plt.plot(['L', 'C1', 'C2'], reg, marker='o', label = str(constants.LANDMARK_REGIONS[i]))
ax = plt.subplot(111)


# Shrink current axis by 20%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.ylim(0,3)
plt.show()
print(np.mean(all_mice, axis = 0))