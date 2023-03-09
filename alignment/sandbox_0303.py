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
        cp.optimize_centroids(df, imgs[i])
        df.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+sessions[i] +"_optimized.csv")
        df_n = df.copy()
        cp.optimize_centroids(df_n, imgs[i+1], "_n")
        df_n.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i+1] +"_optimized_n.csv")
        trs += [intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
        print(i)
    return trs

def distribution_change_short(mouse, region, sessions):
    trs = []
    
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]
    dfs = utils.read_single_session_cell_data(mouse, region, sessions)

    for i in range(len(sessions)-1):
        df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+sessions[i] +"_optimized.csv")
        df_n = df.copy()
        cp.optimize_centroids(df_n, imgs[i+1], "_n")
        df_n.to_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[i+1] +"_optimized_n.csv")
        trs += [intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n')]
        print(i)
    return trs
        
#%%
for i in [2,5,8,10, 11,13,14,16]:
    trs = distribution_change_short(i,1,['ctx', 'landmark1', 'landmark2'])
    print(trs)
    print("")

#%%
m2_cl = [0.01226994, 0.06748466, 0.15132924, 0.23312883, 0.53578732]
m2_ll = [0.01117318, 0.03351955, 0.11918063, 0.25139665, 0.58472998]
m5_cl = [0.0037594 , 0.02443609, 0.06954887, 0.22744361, 0.67481203]
m5_ll = [0.00375235, 0.01876173, 0.06941839, 0.21575985, 0.69230769]
m8_cl = [0.00469484, 0.01643192, 0.05399061, 0.24178404, 0.68309859]
m8_ll = [0.00652174, 0.01956522, 0.07391304, 0.22173913, 0.67826087]
m10_cl = [0.00840336, 0.03781513, 0.05042017, 0.25630252, 0.64705882]
m10_ll = [0.01666667, 0.01875   , 0.08125   , 0.19166667, 0.69166667]
m11_cl = [0.00549451, 0.02564103, 0.06776557, 0.23809524, 0.66300366]
m11_ll = [0.00550459, 0.0293578 , 0.10275229, 0.23486239, 0.62752294]
m13_cl = [0.00425532, 0.0212766 , 0.0893617 , 0.26382979, 0.6212766 ]
m13_ll = [0.01195219, 0.01992032, 0.07569721, 0.27888446, 0.61354582]
m14_cl = [0.01502146, 0.027897  , 0.10944206, 0.2360515 , 0.61158798]
m14_ll = [0.02293578, 0.02752294, 0.09174312, 0.21559633, 0.64220183]
m16_cl = [0.01030928, 0.02061856, 0.06185567, 0.20274914, 0.70446735]
m16_ll = [0.00296736, 0.02967359, 0.06824926, 0.21661721, 0.68249258]

all_mice = np.array([m2_cl, m2_ll, m5_cl, m5_ll, m8_cl, m8_ll, m10_cl, m10_ll, m11_cl, m11_ll, m13_cl, m13_ll, m14_cl, m14_ll, m16_cl, m16_ll ])

#%%
for i in range(8):
    plt.plot([all_mice[2*i, 4], all_mice[2*i+1, 4]], marker='o')
plt.show()