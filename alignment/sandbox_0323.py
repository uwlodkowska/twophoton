#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:43:06 2023

@author: ula
"""
import utils
import constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#%%
types = ['A', 'B', 'C', 'D']

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
        
#%%

all_mice = []        
for m,r in constants.CTX_REGIONS:
    df = utils.top_intensity_from_file(m,r)
    for i in range(3):
        utils.assign_quantile(df, 'int_optimized'+str(i))
        df.loc[df["int_optimized"+str(i)+"_q"]==5] = 4
    df['type'] = df.apply(assign_type, axis=1)
    all_mice += [[df.loc[df.type==t].shape[0]/df.shape[0] for t in types]]
    
    
    #df_filtered = df.loc[((df.int_optimized0_q != df.int_optimized1_q) | (df.int_optimized1_q != df.int_optimized2_q))]
    
for reg in all_mice:
    plt.plot(types, reg, marker='o', linestyle="None")
plt.title("Cell types CLL")
plt.ylim(0,1)
plt.show()

all_mice = []        
for m,r in constants.LANDMARK_REGIONS:
    df = utils.top_intensity_from_file(m,r)
    for i in range(3):
        utils.assign_quantile(df, 'int_optimized'+str(i))
        df.loc[df["int_optimized"+str(i)+"_q"]==5] = 4
    df['type'] = df.apply(assign_type, axis=1)
    all_mice += [[df.loc[df.type==t].shape[0]/df.shape[0] for t in types]]
    
    
    #df_filtered = df.loc[((df.int_optimized0_q != df.int_optimized1_q) | (df.int_optimized1_q != df.int_optimized2_q))]
    
for reg in all_mice:
    plt.plot(types, reg, marker='o', linestyle=None)
plt.title("Cell types LCC")
plt.ylim(0,1)
plt.show()