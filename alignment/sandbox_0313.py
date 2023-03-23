#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:28:16 2023

@author: ula
"""

import utils
import constants
import matplotlib.pyplot as plt
import pandas as pd
import cell_preprocessing as cp
import plotting
import numpy as np
#%%

def assign_type(row):
    if row.int_optimized1 > row.int_optimized0:
        if row.int_optimized2 > row.int_optimized1:
            return'A'
        else:
            return 'B'
    else:
        if row.int_optimized2 < row.int_optimized1:
            return 'C'
        else:
            return 'D'
        
def get_brightest_cells(mouse, region, session, percentage):
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ 
                     session +"_optimized.csv")
    df = df.loc[df.int_optimized > df.int_optimized.quantile(1-percentage)]
    df = df[constants.COORDS_3D]
    return df

def calculate_r2(df, col1, col2):
    int_avg = df[col2].mean()
    sse = ((df[col1] - df[col2])**2).sum()
    stot = ((df[col1] - int_avg)**2).sum()
    return (1-sse/stot)
    

def top_cells_intensity_change(mouse, region, sessions, percentage):
    top_df = pd.DataFrame(columns=constants.COORDS_3D)
    for s in sessions:
        df = get_brightest_cells(mouse, region, s, percentage)
        top_df =pd.concat([top_df,df], ignore_index=True)
    print(top_df.shape[0])
    for i, s in enumerate(sessions):
        img = utils.read_image(mouse, region, s)
        top_df = cp.optimize_centroids(top_df, img, str(i))
    print(top_df[['int_optimized0', 'int_optimized1', 'int_optimized2']])
    top_df = top_df.drop_duplicates(subset=['int_optimized0', 'int_optimized1', 'int_optimized2'])
    print(top_df.shape[0])
    top_df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    r2_cl = calculate_r2(top_df, 'int_optimized1', 'int_optimized0')
    r2_ll = calculate_r2(top_df, 'int_optimized2', 'int_optimized1')

    #top_df['type'] = top_df.apply(assign_type, axis = 1)
    #type_frac = [ top_df[top_df.type== i].shape[0]/top_df.shape[0]  for i in ['A', 'B', 'C', 'D'] ]
    top_df.sort_values(by='int_optimized1', inplace=True)
    top_df.sort_values(by='int_optimized0', inplace=True)
    # plt.plot(np.array(top_df.int_optimized0))
    # plt.plot(np.array(top_df.int_optimized1), alpha=0.5)
    # plt.title(constants.INTENSITY_PLT.format(mouse, region, 'CL'))
    # plt.savefig(constants.dir_path+constants.INTENSITY_PLT.format(mouse, region, 'CL')+".png")
    # plt.close()
    top_df.sort_values(by='int_optimized2', inplace=True)
    top_df.sort_values(by='int_optimized1', inplace=True)
    # plt.plot(np.array(top_df.int_optimized1))
    # plt.plot(np.array(top_df.int_optimized2), alpha=0.5)
    # plt.title(constants.INTENSITY_PLT.format(mouse, region, 'LL'))
    # plt.savefig(constants.dir_path+constants.INTENSITY_PLT.format(mouse, region, 'LL')+".png")
    # plt.close()
    # plot_title = constants.INTENSITY_HIST.format(mouse,region)
    # plotting.plot_intensity_change_histogram(np.array(top_df.int_optimized0-top_df.int_optimized1), 
    #                                          "C-L",
    #                                          np.array(top_df.int_optimized1-top_df.int_optimized2),
    #                                          "L-L",
    #                                          plot_title)
    return r2_cl, r2_ll


    

#%%
r2_arr = []
for m,r in constants.LANDMARK_REGIONS:
    r2_vals = top_cells_intensity_change(m,r,constants.LANDMARK_FIRST_SESSIONS, 0.1)
    r2_arr += [r2_vals]
#%%
r2_arr = np.array(r2_arr)
for reg in r2_arr:
    plt.plot(reg, marker='o')
plt.title('R2 LCC')
plt.show()
#%%
