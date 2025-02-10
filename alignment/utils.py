#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:19:59 2023

@author: ula
"""
import pandas as pd
import constants
from skimage import io
import numpy as np
import math

def read_single_session_cell_data(mouse, region, sessions):
    ret = []
    for s in sessions:
        df = pd.read_csv(constants.path_for_icy + constants.FILENAMES['cell_data_fn_template']
                             .format(mouse, region, s), "\t", header=1)
        if len(sessions) == 1:
            return df
        ret += [df]
    return ret
    
        
def read_image(mouse, region, session, watershed = False):
    if watershed:
        return io.imread(constants.dir_path + constants.FILENAMES['watershed_img_fn_template']
                         .format(mouse, region, session))
    '''
    print("img path", constants.path_for_icy + constants.FILENAMES['img_fn_template']
                     .format(mouse, region, session))
    '''
    return io.imread(constants.path_for_icy + constants.FILENAMES['img_fn_template']
                     .format(mouse, region, session))

def top_intensity_from_file(mouse, region):
    top_df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    return top_df

def assign_quantile(df, colname, step=0.2):
    df[colname+"_q"] = (df[colname].rank(pct=True)/step).apply(math.floor)
    df.loc[df[colname]==math.floor(1/step)] = math.floor(1/step)-1
    
def read_behav_data():
    return pd.read_excel(constants.BEHAV_DIR, sheet_name="summary",header=0, index_col=1)

def filter_close_matches(df1, df2, coord_cols, tolerance):
    """
    Merges two dataframes, computes distances based on 3D coordinates, 
    filters by tolerance, and removes duplicates.
    
    Args:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        coord_cols (list): List of coordinate column names (e.g., ['x', 'y', 'z'])
        tolerance (float): Distance threshold for filtering
        
    Returns:
        pd.DataFrame: Filtered dataframe with averaged coordinates
        np.ndarray: Indices from df1 to drop
        np.ndarray: Indices from df2 to drop
    """
    cross_prod = df1.merge(df2, how="cross", suffixes=("_1", "_2"))
    
    # Compute Euclidean distance
    cross_prod["distance"] = np.linalg.norm(
        cross_prod[[f"{col}_1" for col in coord_cols]].values 
        - cross_prod[[f"{col}_2" for col in coord_cols]].values, axis=1
    )

    # Filter by tolerance
    cross_prod = cross_prod[cross_prod["distance"] < tolerance]
    cross_prod = cross_prod.sort_values(by="distance", ascending=True)

    # Deduplicate
    cross_prod = cross_prod.drop_duplicates(subset=["index_2"]).drop_duplicates(subset=["index_1"])

    # Compute averaged coordinates
    for col in coord_cols:
        cross_prod[col] = (cross_prod[f"{col}_1"] + cross_prod[f"{col}_2"]) / 2
        cross_prod = cross_prod.drop(columns=[f"{col}_1", f"{col}_2"])

    return cross_prod.drop(columns=["distance"]), cross_prod["index_1"].values, cross_prod["index_2"].values
    
