#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:49:09 2023

@author: ula
"""

import intersession
import pandas as pd
import constants
import matplotlib.pyplot as plt
import cell_preprocessing as cp
import utils
import timeit
#%%
mouse = 10
region = 1
st = 0.2
sessions = constants.CTX_FIRST_SESSIONS
df_n = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[1] +"_optimized_n.csv")
res = intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n', step=st)
plt.imshow(res)
plt.show()

df_n = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ sessions[2] +"_optimized_n.csv")
res = intersession.distribution_change(df_n, 'int_optimized', 'int_optimized_n', step = st)
plt.imshow(res)
plt.show()
#%%

df = utils.read_single_session_cell_data(mouse, region, ['landmark2'])
img = utils.read_image(mouse, region, 'landmark2')
df['in_calculated'] = df.apply(cp.calculate_intensity,img = img, axis = 1)
df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
df=df[df.in_calculated>df.in_calculated.quantile(0.9)]
print(timeit.timeit('cp.optimize_centroids_new(df, img)', globals=globals(), number=20))
print(timeit.timeit('cp.optimize_centroids(df, img)', globals=globals(), number=20))

#%%

