#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 16:45:43 2023

@author: ula
"""

import intersession
import visualization
import utils
import constants
import cell_preprocessing as cp
import numpy as np
import matplotlib.pyplot as plt
#%%

thresholded = intersession.identify_persistent_cells_w_thresholding(10, 1, ["ctx", "landmark1"])
original_way = intersession.identify_persistent_cells(10, 1, ["ctx", "landmark1"])

#%%
original_way.groupby(by = "index_s1")



#%%
df = utils.read_single_session_cell_data(10, 1, ["ctx"])
plt.plot(df)

diff = [i for i in thresholded.index if i not in original_way["index_s1"].to_numpy()]
print(len(diff))

df_outside_orig = df.iloc[diff]



visualization.visualize_with_centroids_custom(utils.read_image(10, 1, "ctx"), 
                                              df_outside_orig[constants.COORDS_3D])
#%%
img = utils.read_image(10, 1, "ctx")
df = utils.read_single_session_cell_data(10, 1, ["ctx"])
df['in_calculated'] = df.apply(cp.calculate_intensity,img = img, axis = 1)
df.sort_values('in_calculated', inplace=True)

df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
plt.hist(df['in_calculated'], bins=50)
#%%
df = df[df[constants.ICY_COLNAMES["mean_intensity"]]>3*df['in_calculated']]
point_properties = {
    'color' : df[constants.ICY_COLNAMES["mean_intensity"]]
    }
img_w = utils.read_image(10, 1, "ctx", watershed = True)
visualization.visualize_with_centroids_custom(np.stack([img, img_w]), 
                                              df[constants.COORDS_3D], point_properties)

#%%
mouse = 11
img = utils.read_image(mouse, 1, "ctx")
img_w = utils.read_image(mouse, 1, "landmark1")
df = utils.read_single_session_cell_data(mouse, 1, ["ctx"])
visualization.visualize_with_centroids_custom(np.stack([img, img_w]), 
                                              df[constants.COORDS_3D])
#%%
img = utils.read_image(10, 1, "ctx")
img_w = utils.read_image(10, 1, "landmark1")
df = utils.read_single_session_cell_data(10, 1, ["ctx"])
df['in_calculated'] = df.apply(cp.calculate_intensity,img = img, axis = 1)
df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
print(df[df['in_calculated']>50].shape[0]/df.shape[0])
#%%
df['in_calculated_w'] = df.apply(cp.calculate_intensity,img = img_w, axis = 1)
fig, ax = plt.subplots()
ax.set_box_aspect(1)
plt.scatter(df['in_calculated'], df['in_calculated_w'], s=1)
plt.xlim(0,250)
plt.ylim(0,250)

#%%
intersession.distribution_change(df, 'in_calculated', 'in_calculated_w')