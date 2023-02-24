#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:27:56 2023

@author: ula
"""
import numpy as np
import pandas as pd
import utils
import constants
from cell_preprocessing import calculate_intensity
import visualization

def identify_persistent_cells(mouse, region, session_ids):
    #tu dodac standaryzacje intensity?
    sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    for i, s in enumerate(sessions):
        sessions[i] = s[constants.COORDS_3D].reset_index()
    cross_prod = sessions[0].merge(sessions[1], how = "cross", suffixes=("_s1", "_s2"))
    cross_prod['distance'] = np.linalg.norm(cross_prod[[n+"_s1" for n in constants.COORDS_3D]].values
                                            -cross_prod[[n+"_s2" for n in constants.COORDS_3D]].values, axis=1)
    cross_prod = cross_prod[cross_prod.distance < constants.TOLERANCE]

    return cross_prod

# identify_persistent_cells(10,1, utils.read_single_session_cell_data(
#                               10, 1, constants.CTX_FIRST_SESSIONS[:2]))


def identify_persistent_cells_w_thresholding(mouse, region, session_ids):
    """
    Parameters
    ----------
    mouse : int
        mouse id
    region : int
        region id
    session_ids : array of string
        at idx 0 - id of session from which we take coordinates and see if there
        are cells at these coordinates in an image from the session at idx 1
        

    Returns
    -------
    None.
    
    Takes coords identified in one session, projects them onto image from 
    anothee one and assumes that wherever the mean intensity is higher than
    threshold computed for coordinates and image from that other session, there
    is a cell there

    """
    coord_dfs = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    img = utils.read_image(mouse, region, session_ids[1])
    
    for df in coord_dfs:    
        df['in_calculated'] = df.apply(calculate_intensity, 
                                                 img = img, axis = 1)
    threshold = coord_dfs[1]['in_calculated'].quantile(0.2)
    persistent_df = coord_dfs[0][coord_dfs[0].in_calculated > threshold]
    
    
    visualization.visualize_with_centroids_custom(img, persistent_df[constants.COORDS_3D])
    return persistent_df
    