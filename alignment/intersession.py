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


def identify_persistent_cells(mouse, region, sessions, session_ids):
    #tu dodac standaryzacje intensity?
    if sessions is None:
        sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    cell_count = 0
    for i, s in enumerate(sessions):
        cell_count += s.shape[0]
        sessions[i] = s[constants.COORDS_3D].reset_index()
    print("summed ", sessions[0].shape[0]+sessions[1].shape[0])
        
    
    cross_prod = sessions[0].merge(sessions[1], how = "cross", suffixes=("_s1", "_s2"))
    cross_prod['distance'] = np.linalg.norm(cross_prod[[n+"_s1" for n in constants.COORDS_3D]].values
                                            -cross_prod[[n+"_s2" for n in constants.COORDS_3D]].values, axis=1)
    cross_prod = cross_prod[cross_prod.distance < constants.TOLERANCE]
    cross_prod = cross_prod.sort_values(by='distance', ascending=True)
    cross_prod = cross_prod.drop_duplicates(subset=["index_s2"]).drop_duplicates(subset=["index_s1"])
    
    for cname in constants.COORDS_3D:
        cross_prod[cname] = (cross_prod[cname+"_s1"]+cross_prod[cname+"_s2"])/2
        cross_prod = cross_prod.drop(columns = [cname+"_s1",cname+"_s2"])
    
    print("duplicates ", cross_prod.shape[0])
    #print(np.array(cross_prod.index_s1))
    
    sessions[0] = sessions[0].drop(np.array(cross_prod.index_s1))
    summed = pd.concat(sessions)
    print("summed ", summed.shape[0])
    print("duplicates removed ", summed.shape[0])
    cross_prod = cross_prod.drop(columns = ["index_s1","index_s2", "distance"])
    count = cross_prod.shape[0]/(cell_count-cross_prod.shape[0])
    return summed, count 


def pooled_cells(mouse, region, session_ids):
    sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    persistent, count = identify_persistent_cells(mouse, region, sessions[:2], [])
    print("first ct ", count)
    pooled, count = identify_persistent_cells(mouse, region, [persistent,sessions[-1]], [])
    print("second ct ",count)
    return pooled



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
    quantile_thre = 0
    
    coord_dfs = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    imgs = [utils.read_image(mouse, region, session_ids[0]), 
            utils.read_image(mouse, region, session_ids[1])]
    
    over_thre_count = np.array([0,0])
    thre=[0,0]
    for idx, df in enumerate(coord_dfs): 
        df['in_calculated'] = df.apply(cp.calculate_intensity, 
                                                 img = imgs[idx], axis = 1)
        
        thre[idx] = df['in_calculated'].quantile(quantile_thre)
        over_thre_count[idx] = df.loc[df.in_calculated > thre[idx]].shape[0]
    
    #threshold = coord_dfs[1]['in_calculated'].quantile(quantile_thre)
    
    coord_dfs[0]['in_calculated'] = coord_dfs[0].apply(cp.calculate_intensity, 
                                                 img = imgs[1], axis = 1)
    
    coord_dfs[0] = cp.optimize_centroids(coord_dfs[0], imgs[1])
    
    persistent_df = coord_dfs[0].loc[coord_dfs[0].int_optimized > thre[1]]
    print('summary ', persistent_df.shape[0], over_thre_count)
    overlap_size = persistent_df.shape[0]/(np.sum(over_thre_count)-persistent_df.shape[0])
    #visualization.visualize_with_centroids_custom(imgs[0], persistent_df[constants.COORDS_3D])
    return overlap_size, persistent_df

 
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
        df_top_from = df[(df[from_column]>lower_lim) & (df[from_column]<upper_lim)]
        transfer_rate[i] = np.array([df_top_from[df_top_from['dest'] == i].shape[0] for i in range(number_of_classes)])
    transfer_rate = transfer_rate/df.shape[0]
    print(transfer_rate)
    return transfer_rate

def distribution_change_precalculated(df, from_column, to_column, number_of_classes):
    transfer_rate = np.zeros((number_of_classes,number_of_classes))
    for i in range(number_of_classes):
        for j in range(number_of_classes):
            transfer_rate[i,j] = df.loc[((df[from_column] == i)&(df[to_column] == j))].shape[0]
    transfer_rate = transfer_rate/df.shape[0]
    print(transfer_rate)
    return transfer_rate

def assign_type(row):
    if row.int_optimized1 > row.int_optimized0:
        if row.int_optimized2 > row.int_optimized0:
            return'A'
        else:
            return 'B'
    else:
        if row.int_optimized0 < row.int_optimized2:
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

    return type_frac

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
        
        
def find_intersession_tendencies_raw(df,sessions=[1,2,3], colname='int_optimized'):
    tendencies = []
    for i in range(len(sessions)-1):
        df[colname+str(sessions[i])+'_low'] = df[colname+str(sessions[i])]*0.8
        df[colname+str(sessions[i])+'_high'] = df[colname+str(sessions[i])]*1.2
        
        condition_down = df[colname+str(sessions[i+1])] < df[colname+str(sessions[i])+'_low']
        condition_up = df[colname+str(sessions[i+1])] > df[colname+str(sessions[i])+'_high']
        down = df.loc[condition_down]
        stable = df.loc[ (~condition_down) & (~condition_up)]
        up = df.loc[condition_up]
        print("up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies

def find_intersession_tendencies_bgr(df,bgr,k=1, sessions=[1,2,3], colname='int_optimized'):
    tendencies = []
    
    print("shape ", df.shape[0])
    for i in range(len(sessions)):
        df[colname+str(sessions[i])] = df[colname+str(sessions[i])]-bgr[i,0]
    for i in range(len(sessions)-1):
        df[colname+str(sessions[i])+'_low'] = df[colname+str(sessions[i])] - k*(bgr[i,1]+bgr[i+1,1])
        df[colname+str(sessions[i])+'_high'] = df[colname+str(sessions[i])] + k*(bgr[i,1]+bgr[i+1,1])
        
        condition_down = df[colname+str(sessions[i+1])] < df[colname+str(sessions[i])+'_low']
        condition_up = df[colname+str(sessions[i+1])] > df[colname+str(sessions[i])+'_high']
        down = df.loc[condition_down]
        stable = df.loc[ (~condition_down) & (~condition_up)]
        up = df.loc[condition_up]
        print("up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies      
            
def find_intersession_tendencies_on_off(df, sessions=[1,2,3], colname='active'):
    tendencies = []
    for i in range(len(sessions)-1):
        condition_down = (df[colname+str(sessions[i])]) & (~df[colname+str(sessions[i+1])])
        condition_stable = (df[colname+str(sessions[i+1])]) & (df[colname+str(sessions[i])])
        condition_up = (~df[colname+str(sessions[i])]) & (df[colname+str(sessions[i+1])])
        down = df.loc[condition_down]
        stable = df.loc[condition_stable]
        up = df.loc[condition_up]
        print("on off up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies
            
def cell_classes(df, sessions=[1,2,3], colname='active'):
    
    any_session = len(df.loc[(df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])])
                    | (df[colname+str(sessions[2])])])
    class1 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class2 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class3 = len(df.loc[(~df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class4 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class5 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class6 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class7 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    ret = np.array([class1,class2,class3,class4,class5,class6,class7])/any_session
    print(ret)
    return ret
