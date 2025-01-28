#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 13:16:21 2023

@author: ula
"""

import plotting
import constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%%

def compare_cumulative_intensity(mouse, region, sessions, percentage=0.1):
    ret = []
    for i in range(len(sessions)):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized.csv")
        df.sort_values(by="int_optimized", inplace=True, ascending=False)
        intensity_distrib = np.cumsum(df.int_optimized.to_numpy())/df.int_optimized.sum()
        crit_idx = np.argmax(intensity_distrib>=percentage)
        #plt.plot(intensity_distrib)
        #plt.plot([crit_idx],[intensity_distrib[crit_idx]], marker='o')
        ret += [crit_idx/df.shape[0]]
    #plt.show()
    return ret
     

#%%

def compare_cumulative_intensity_by_cells_p(mouse, region, sessions, percentage=0.1):
    ret = []
    for i in range(len(sessions)):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized.csv")
        df.sort_values(by="int_optimized", inplace=True, ascending=False)
        df_sub = df.nlargest(int(df.shape[0]*percentage), 'int_optimized')
            #plt.plot(np.cumsum(df.int_optimized.to_numpy()/df.int_optimized.sum()))
        ret += [df_sub.int_optimized.sum()/df.int_optimized.sum()]
        #plt.show()
    return ret
#%%
def compare_cell_numbers(mouse, region, sessions):
    ret = []
    for i in range(len(sessions)):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized.csv")
        ret += [df.shape[0]]
    return np.array(ret)/ret[0]

#%%
def compare_intensity_mean(mouse, region, sessions):
    ret = []
    for i in range(len(sessions)):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized.csv")
        ret += [df.int_optimized.mean()]
    return np.array(ret)/ret[0]
#%%
def compare_intensity_ratio(mouse, region, sessions):
    ret = []
    for i in range(1,len(sessions)):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i] +"_optimized_n.csv")
        ret += [(df.int_optimized_n/df.int_optimized-1).mean()]
    return pd.Series([0] + ret, index=sessions)

#%%
def compare_top_cells_intensity(mouse, region, sessions):
    intensity_sum = []
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    
    for i in [0,1]:
        df['lower_lim'+str(i)] = df["int_optimized"+str(i)]*0.8
        df['upper_lim'+str(i)] = df["int_optimized"+str(i)]*1.2
    
    # print(df.shape[0])
    df = df.loc[((df.int_optimized1 > df.upper_lim0) 
                  | (df.int_optimized1 < df.lower_lim0)
                  | (df.int_optimized2 > df.upper_lim1)
                  | (df.int_optimized2 < df.lower_lim1))]    

    # print(df.shape[0])

    intensity_sum += [df["int_optimized"+str(i)].mean() for i in range(3)]
    ret = [1]+ [intensity_sum[i+1]/intensity_sum[i] for i in [0,1]]

    # if ret[1] < 1:
    #     print("decreased ", mouse, ret[1])
    # elif ret[1] > 1:
    #     print("increased ", mouse, ret[1])
    return ret 

#%%
def compare_increase_number(mouse, region, sessions):
    increase = []
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    for i in [0,1]:
        df['lower_lim'+str(i)] = df["int_optimized"+str(i)]*0.8
        df['upper_lim'+str(i)] = df["int_optimized"+str(i)]*1.2
    
    df = df.loc[((df.int_optimized1 > df.upper_lim0) 
                  | (df.int_optimized1 < df.lower_lim0)
                  | (df.int_optimized2 > df.upper_lim1)
                  | (df.int_optimized2 < df.lower_lim1))]    

    
    increase += [df.loc[df["int_optimized"+str(i+1)]<df["lower_lim"+str(i)]].shape[0]/df.shape[0] for i in [0,1]]

    ret = [1]+increase

    return ret 
#%%

#%%

plotting.plot_feature_through_sessions(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS, 
                                       compare_cumulative_intensity, "Fraction of cells that make up 10% of total intensity")
plotting.plot_feature_through_sessions(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS, 
                                       compare_cumulative_intensity, "Fraction of cells that make up 10% of total intensity")

#%%
for i in range(10):
    plotting.plot_feature_through_sessions(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS, 
                                       compare_cumulative_intensity_by_cells_p, "Fraction of intensity in x% cells", percentage=0.1*i)
plotting.plot_feature_through_sessions(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS, 
                                       compare_cumulative_intensity_by_cells_p, "Fraction of intensity in x% cells")
#%%

plt.ylim(0.4,2)
plotting.plot_feature_through_sessions(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS,
                                       compare_intensity_mean, "Mean intensity for CLL sessions") 
plt.ylim(0.4,2)                     
plotting.plot_feature_through_sessions(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS,
                                       compare_intensity_mean, "Mean intensity for LCC sessions")

#%%
plt.ylim(-0.25,0.25)
plotting.plot_feature_averaged_for_mouse(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS,
                                       compare_intensity_ratio, "Intensity ratio for CLL sessions") 
plt.ylim(-0.25,0.25)
plotting.plot_feature_averaged_for_mouse(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS,
                                       compare_intensity_ratio, "Intensity ratio for LCC sessions")
#%%
plotting.plot_feature_through_sessions(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS,
                                       compare_top_cells_intensity, "Mean intensity for CLL sessions") 
plotting.plot_feature_through_sessions(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS,
                                       compare_top_cells_intensity, "Mean intensity for LCC sessions") 

#%%
plt.ylim(0, 1)
plotting.plot_feature_through_sessions(constants.CTX_REGIONS, constants.CTX_FIRST_SESSIONS,
                                       compare_increase_number, "Mean intensity diff for CLL sessions") 

plt.ylim(0, 1)
plotting.plot_feature_through_sessions(constants.LANDMARK_REGIONS, constants.LANDMARK_FIRST_SESSIONS,
                                       compare_increase_number, "Mean intensity diff for LCC sessions") 