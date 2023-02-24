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

#%%

thresholded = intersession.identify_persistent_cells_w_thresholding(10, 1, ["ctx", "landmark1"])
original_way = intersession.identify_persistent_cells(10, 1, ["ctx", "landmark1"])

#%%

df = utils.read_single_session_cell_data(10, 1, ["ctx"])
diff = [i for i in thresholded.index if i not in original_way["index_s1"].to_numpy()]
print(len(diff))

df_outside_orig = df.iloc[diff]



#visualization.visualize_with_centroids_custom(utils.read_image(10, 1, "ctx"), 
#                                              df_outside_orig[constants.COORDS_3D])