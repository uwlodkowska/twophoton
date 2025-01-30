#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:01:16 2025

@author: ula
"""

import numpy as np

import utils
import intersession
import cell_preprocessing as cp
import pandas as pd
import matplotlib.pyplot as plt

from pandarallel import pandarallel


#%%
regions = [[2,1]]#, [3,2], [4,2], [6,2], [7,1], [11,1], [14,1]]


#%%


#%%


pandarallel.initialize()
tendencies = []
all_pooled = []
ovlaps = []
for m,r in regions:
    imgs = [utils.read_image(m, r, i) for i in [1,2,3]]
    tstx = intersession.pooled_cells(m,r, [1,2,3])
    all_pooled += [tstx]
    prev = tstx
    for i,img in enumerate(imgs):
        df = cp.optimize_centroids(prev, img, suff=str(i))
        bgr = cp.calculate_background_intensity(df, img)
        prev = df
    idx_arr = []
    for i in ['0','1', '2']:
        q1 = df["int_optimized"+i].quantile(0.9)
        idx_arr += [df[df["int_optimized"+i] <= q1].index]
        plt.hist(df["int_optimized"+i], bins = 40, alpha=0.5)
    s12 = len(np.intersect1d(idx_arr[0], idx_arr[1]))/len(idx_arr[0])
    s23 = len(np.intersect1d(idx_arr[1], idx_arr[2]))/len(idx_arr[1])
    ovlaps+=[[s12, s23]]    
    plt.title("m"+str(m))
    plt.show()
    tendencies += [intersession.find_intersession_tendencies(df, sessions=[0,1,2])]


#%%
bgr = cp.calculate_background_intensity(df, img)
#%%
plt.ion()
for m,r in regions:
    imgs = [utils.read_image(m, r, i) for i in [1,2,3]]
    plt.figure()
    for i in imgs:
        bins = 100
        counts, bin_edges = np.histogram(i, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(bin_centers, counts)
    plt.title("m"+str(m))
    plt.show()
    
    


#%%

for i, t in enumerate(tendencies):
    #if regions[i][0] in [2,3,6]:
    plt.plot(t[:3]/np.sum(t[:3]), label = regions[i][0])
plt.legend()
plt.show()


for i, t in enumerate(tendencies):
    #if regions[i][0] in [2,3,6]:
    plt.plot(t[3:]/np.sum(t[:3]), label = regions[i][0])
plt.legend()
plt.show()

#%%
tendecies_np  = np.array(tendencies)
tendecies_np=tendecies_np/np.sum(tendecies_np[:3])