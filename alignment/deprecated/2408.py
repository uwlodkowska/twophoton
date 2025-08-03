#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 17:41:53 2024

@author: ula
"""
import numpy as np

import utils
import intersession
from scipy import stats
import cell_preprocessing as cp
import time

import matplotlib.pyplot as plt

#%%
mouse = 4
region = 2

regions = [[2,1], [3,2], [4,2], [6,2], [7,1]]#, [11,1], [14,1]]
#%%
'''
print('a',merged.shape[0])
print('b',len(np.unique(np.array(merged.index_s2))))
print('c',len(np.unique(np.array(merged.index_s1))))
'''
#%%
diff = []
same = []

# for m,r in regions:
#     img = utils.read_image(m, r, 1)
#     tstx = intersession.pooled_cells(m,r, [1,2,3])
#     print(time.time())
#     df1 = cp.optimize_centroids_old(tstx, img)
#%%
#%%
tendencies = []
for m,r in regions:
    imgs = [utils.read_image(m, r, i) for i in [1,2,3]]
    tstx = intersession.pooled_cells(m,r, [1,2,3])
    prev = tstx
    for i,img in enumerate(imgs):
        df = cp.optimize_centroids(prev, img, suff=str(i))
        prev = df    
    tendencies += [intersession.find_intersession_tendencies(df, sessions=[0,1,2])]

#%%   
prev = tstx
for m,r in regions:
    for i,img in enumerate(imgs):
        df2 = cp.optimize_centroids_old(prev, img, suff=str(i))
        prev = df2
#%%

# plt.hist(df.int_optimized0, bins = 20, range=(0, 150))
# plt.hist(df.int_optimized1, bins = 20, alpha=0.5, range=(0, 150))
# plt.hist(df.int_optimized2, bins = 20, alpha=0.5, range=(0, 150))

#%%
#%%

for t in tendencies:
    plt.plot(t/np.sum(t[:3]))
plt.show()
