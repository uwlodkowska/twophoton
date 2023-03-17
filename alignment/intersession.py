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
import cell_preprocessing as cp
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
        df['in_calculated'] = df.apply(cp.calculate_intensity, 
                                                 img = img, axis = 1)
    threshold = coord_dfs[1]['in_calculated'].quantile(0.3)
    persistent_df = coord_dfs[0][coord_dfs[0].in_calculated > threshold]
    
    
    visualization.visualize_with_centroids_custom(img, persistent_df[constants.COORDS_3D])
    return persistent_df

def distribution_change(df, from_column, to_column, step=0.2):
    number_of_classes = int(1/step)
    transfer_rate = np.zeros((number_of_classes,number_of_classes))
    df.loc[:,'dest'] = number_of_classes - 1
    for i in range(1, number_of_classes):
        idx = df[df[to_column]<df[to_column].quantile(1-i*step)].index
        df.loc[idx,'dest'] = number_of_classes - i - 1
    
    for i in range(number_of_classes):
        lower_lim = df[from_column].quantile(i*step)
        upper_lim = df[from_column].quantile((i+1)*step)
        df_top_from = df[df[from_column]>lower_lim & df[from_column]<upper_lim]
        transfer_rate[i] = np.array([df_top_from[df_top_from['dest'] == i].shape[0] for i in range(number_of_classes)])
    transfer_rate = transfer_rate/df.shape[0]
    return transfer_rate

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
        
def top_cells_intensity_change(mouse, region, sessions, percentage):
    top_df = pd.DataFrame(columns=constants.COORDS_3D)
    for s in sessions:
        df = cp.get_brightest_cells(mouse, region, s, percentage)
        top_df =pd.concat([top_df,df], ignore_index=True)
    for i, s in enumerate(sessions):
        img = utils.read_image(mouse, region, s)
        cp.optimize_centroids(top_df, img, str(i))

    top_df['type'] = top_df.apply(assign_type, axis = 1)
    type_frac = [ top_df[top_df.type== i].shape[0]/top_df.shape[0]  for i in ['A', 'B', 'C', 'D'] ]

    return top_df

def distribution_change_all_sessions(mouse, region, sessions):
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]
    dfs = utils.read_single_session_cell_data(mouse, region, sessions)
    
    for i in range(len(sessions)-1):
        df = dfs[i]
        df['in_calculated'] = df.apply(cp.calculate_intensity,img = imgs[i], axis = 1)
        df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
        df['in_calculated_n'] = df.apply(cp.calculate_intensity,img = imgs[i+1], axis = 1)
        tr = distribution_change(df, 'in_calculated', 'in_calculated_n')
        print(tr)