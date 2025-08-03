#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:39:24 2023

@author: ula
"""

import utils
import cell_preprocessing as cp
import constants
import intersession
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

def distribution_change_all_sessions(mouse, region, sessions):
    trs = []
    
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]
    dfs = utils.read_single_session_cell_data(mouse, region, sessions)

    # for i in range(3):
    #      dfs[i] = dfs[i].sample(20)

    for i in range(len(sessions)-1):
        df = dfs[i]
        df['in_calculated'] = df.apply(cp.calculate_intensity,img = imgs[i], axis = 1)
        df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
        df = cp.optimize_centroids(df, imgs[i])
        df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+sessions[i] +"_optimized.csv")
        df_n = df.copy()
        df_n = cp.optimize_centroids(df_n, imgs[i+1], "_n")
        df_n.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i+1] +"_optimized_n.csv")
        trs += [intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
        print(i)
    return trs

def distribution_change_short(mouse, region, sessions):
    trs = []
    
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]

    for i in range(len(sessions)-1):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+sessions[i] +"_optimized.csv")
        df_n = df.copy()
        cp.optimize_centroids(df_n, imgs[i+1], "_n")
        df_n.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i+1] +"_optimized_n.csv")
        trs += [intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
        print(i)
    return trs
        
#%%
for m,r in [[12,1], [12,2], [12,3], [15,1], [15,2], [15,3], [17,1], [17,2], [18,1], [18,2]]:
    trs = distribution_change_all_sessions(m,r,constants.LANDMARK_FIRST_SESSIONS)
    print(trs)
    print("")

