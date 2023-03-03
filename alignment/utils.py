#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 17:19:59 2023

@author: ula
"""
import pandas as pd
from constants import dir_path, FILENAMES
from skimage import io

def read_single_session_cell_data(mouse, region, sessions):
    ret = []
    for s in sessions:
        df = pd.read_csv(dir_path + FILENAMES['cell_data_fn_template']
                             .format(mouse, region, s), "\t", header=1)
        if len(sessions) == 1:
            return df
        ret += [df]
    return ret
    
        
def read_image(mouse, region, session, watershed = False):
    if watershed:
        return io.imread(dir_path + FILENAMES['watershed_img_fn_template']
                         .format(mouse, region, session))
    return io.imread(dir_path + FILENAMES['img_fn_template']
                     .format(mouse, region, session))