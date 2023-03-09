#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 15:21:29 2023

@author: ula
"""
import utils
import constants
import matplotlib.pyplot as plt
import pandas as pd
#%%
def plot_histograms(mouse, region, sessions):
    
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]

    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[1] +"_optimized.csv")
    df_n = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[1] +"_optimized_n.csv")
    print(df.shape[0], df_n.shape[0])
    plt.hist(df.int_optimized, bins = 100, alpha = 0.5)
    plt.hist(df_n.int_optimized_n, bins = 100, alpha = 0.5)
    plt.show()

    shapes_list = []
    shifted_list = []
    # for idx, row in df_n.iterrows():
        
    #     z = int(row[constants.ICY_COLNAMES['zcol']])
    #     x = int(row[constants.ICY_COLNAMES['xcol']])
    #     y = int(row[constants.ICY_COLNAMES['ycol']])
    #     shapes_list += draw_normalized_area(x,y,z,img)
        
    #     z = int(row[constants.ICY_COLNAMES['zcol']]) + row['shift_z']
    #     x = int(row[constants.ICY_COLNAMES['xcol']]) + row['shift_x']
    #     y = int(row[constants.ICY_COLNAMES['ycol']]) + row['shift_y']
    #     shifted_list += draw_normalized_area(x,y,z,img)

    # blob_points = np.vstack(shapes_list)

    # blob_points_shifted = np.vstack(shifted_list)
        


#%%
for i in [2,5,8,10, 11,13,14,16]:
    plot_histograms(i,1,['ctx', 'landmark1', 'landmark2'])
        