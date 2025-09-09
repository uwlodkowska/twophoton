#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:56:28 2025

@author: ula
"""
import numpy as np


import pandas as pd
import matplotlib.pyplot as plt


#%%
regions = [[2,1], [3,2], [4,2], [6,2], [7,1], [11,1], [14,1]]

#%%
behav = pd.read_csv("/media/ula/DATADRIVE1/fos_gfp_tmaze/fos_gfp_tmaze2/behav/behavior.csv", 
                    skiprows=3,usecols=np.arange(1,16))
behav.fillna(0, inplace=True)
for m in behav.columns:
    if m=='2':
        continue
    accuracy = behav[str(m)]
    
    first_nonzero = np.argmax(accuracy != 0)
    accuracy = accuracy[first_nonzero:]
    
    # last_nonzero = np.nonzero(accuracy.to_numpy())[0][-1]
    # print(accuracy.to_numpy(), last_nonzero)
    # accuracy = accuracy[:last_nonzero]
    
    first_zero = np.argmax(accuracy == 0)
    accuracy = accuracy[:first_zero]
    
    plt.plot(np.arange(len(accuracy)),accuracy/24, marker='o', markersize='2')#, label = "m"+str(m)) 
    plt.axhline(0.8, linestyle='--')
    #plt.ylim(0.4,1)
#plt.xticks(np.arange(19))
plt.title("Accuracy over days")
plt.legend()
#%%

dict_df = pd.read_csv("/home/ula/Downloads/rstudio_Ula/ABA_brain_regions.csv")