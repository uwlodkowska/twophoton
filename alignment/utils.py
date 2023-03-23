#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:19:59 2023

@author: ula
"""
import pandas as pd
import constants
from skimage import io
import intersession
import math

def read_single_session_cell_data(mouse, region, sessions):
    ret = []
    for s in sessions:
        df = pd.read_csv(constants.dir_path + constants.FILENAMES['cell_data_fn_template']
                             .format(mouse, region, s), "\t", header=1)
        if len(sessions) == 1:
            return df
        ret += [df]
    return ret
    
        
def read_image(mouse, region, session, watershed = False):
    if watershed:
        return io.imread(constants.dir_path + constants.FILENAMES['watershed_img_fn_template']
                         .format(mouse, region, session))
    return io.imread(constants.dir_path + constants.FILENAMES['img_fn_template']
                     .format(mouse, region, session))

def top_intensity_from_file(mouse, region):
    top_df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    return top_df

def assign_quantile(df, colname, step=0.2):
    df[colname+"_q"] = (df[colname].rank(pct=True)/step).apply(math.floor)
    df.loc[df[colname]==math.floor(1/step)] = math.floor(1/step)-1
    
