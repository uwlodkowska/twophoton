#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:39:24 2023

@author: ula
"""

import utils
import cell_preprocessing as cp
import constants

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
        df_n = df.copy()
        cp.optimize_centroids(df_n, imgs[i+1], "_n")
        trs += [cp.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
        print(i)
    return trs
        
#%%

trs = distribution_change_all_sessions(10,1,['ctx', 'landmark1', 'landmark2'])

#%%