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
import os.path
import cell_preprocessing as cp
import ast

def read_single_session_cell_data(mouse, region, sessions, config, test=False, optimized=False):
    DIR_PATH = config["experiment"]["dir_path"]
    ICY_PATH = DIR_PATH + config["experiment"]["path_for_icy"]
    ret = []
    
    if optimized:
        fname = config['filenames']['cell_data_opt_template']
        header_start = 0
        sep = ","
    else:
        fname = config['filenames']['cell_data_fn_template']
        header_start = 1
        sep = "\t"
    
    for s in sessions:
        df = pd.read_csv(
            ICY_PATH + fname.format(mouse, region, s), 
            sep=sep, 
            header=header_start
        )
        if test:
            df = df.loc[(df[constants.ICY_COLNAMES['zcol']]>10) & (df[constants.ICY_COLNAMES['zcol']]<20)]

        if len(sessions) == 1:
            return df
        ret += [df]
    return ret

def read_pooled_cells(mouse, region, config):
    DIR = config["experiment"]["result_path"]
    fname = config["filenames"]["pooled_cells"].format(mouse, region)
    df = pd.read_csv(DIR+fname)
    df["detected_in_sessions"] = df["detected_in_sessions"].apply(lambda x: 
                                                                  ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def read_images(mouse, region, session_ids, config):
    imgs = []
    for sid in session_ids:
        imgs.extend([read_image(mouse, region, sid, config)])
    return imgs
      
def read_image(mouse, region, session, config, watershed = False):
    if watershed:
        return io.imread(constants.dir_path + constants.FILENAMES['watershed_img_fn_template']
                         .format(mouse, region, session))
    '''
    print("img path", constants.path_for_icy + constants.FILENAMES['img_fn_template']
                     .format(mouse, region, session))
    '''
    
    img_path = config["experiment"]["dir_path"] + config["experiment"]["path_for_icy"] + config["filenames"]['img_fn_template'].format(mouse, region, session)
    print(img_path)
    return io.imread(img_path)

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
    
def get_optimized_cell_data(mouse, region, session_id,img, config):
    fname = config['filenames']['cell_data_opt_template']
    if os.path.isfile(fname):
        return read_single_session_cell_data(mouse, 
                                           region, 
                                           session_id, 
                                           config, 
                                           test=False, 
                                           optimized=True)[0]
    print("No such file, calculating optimization")
    df = read_single_session_cell_data(mouse, 
                                       region, 
                                       session_id, 
                                       config, 
                                       test=False, 
                                       optimized=False)[0]
    return cp.optimize_centroids(df, img, suff="", tolerance = 2)