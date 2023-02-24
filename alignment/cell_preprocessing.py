# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.draw import disk
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

from constants import dir_path, ICY_COLNAMES, ROI_DIAMETER, FILENAMES, XY_SCALE, Z_SCALE



def calculate_disk(coords, radius, disk_no, img):
    center_z = coords[ICY_COLNAMES['zcol']]+disk_no
    if (center_z <0 or center_z>=img.shape[0]):
        return [0,0] #spr czy to potrzebne
    disk_ = disk((coords[ICY_COLNAMES['ycol']],coords[ICY_COLNAMES['xcol']]), radius,shape = img[0].shape) 
    sum_int = np.sum(img[center_z][disk_])
    area_int = len(img[center_z][disk_])
    return [sum_int, area_int]

def calculate_intensity(coords, img):
    center_coords_df = coords[[ICY_COLNAMES['xcol'], ICY_COLNAMES['ycol'],ICY_COLNAMES['zcol']]]
    center_coords_df = center_coords_df.round().astype(int)
    sum_int = 0
    area_int = 0
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        diameter = ROI_DIAMETER[abs(i)]
        rad = diameter//2
        res = calculate_disk(center_coords_df,rad, i, img)
        sum_int += res[0]
        area_int += res[1]
    if area_int == 0:
        return 0
    return sum_int/area_int


def standarize_intensityd(df, img):
    df["intensity_standarized"] = calculate_intensity([df[ICY_COLNAMES['xcol']],
                                                               df[ICY_COLNAMES['ycol']],
                                                               df[ICY_COLNAMES['zcol']]],
                                                               img)
    return df#[df['Mean Intensity (ch 0)']/df['intensity_standarized']<=1.5]


def find_quantile_threshold(df, img):
    df = standarize_intensityd(df, img)
    return df["intensity_standarized"]

def pixels_to_um(df):
    df[ICY_COLNAMES['xcol']] = df[ICY_COLNAMES['xcol']]*XY_SCALE
    df[ICY_COLNAMES['ycol']] = df[ICY_COLNAMES['ycol']]*XY_SCALE
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]*Z_SCALE
    return df

    
def test_fun(mouse, region, s_idxses, session_order):
    old_method_df = pd.read_csv(dir_path + FILENAMES["cell_data_fn_template"]
                .format(mouse, region, session_order[s_idxses[1]]+"_"+session_order[s_idxses[0]]))
    df = pd.read_csv(dir_path + FILENAMES["cell_data_fn_template"]
                     .format(mouse, region, session_order[s_idxses[0]]), "\t", header=1)

    img_ref = io.imread(dir_path + FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[0]])).astype("uint8")

    img_comp = io.imread(dir_path + FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[1]])).astype("uint8")
    df["int1"] = df.apply(calculate_intensity, img = img_ref, axis = 1)
    df["int2"] = df.apply(calculate_intensity, img = img_comp, axis = 1)
    joined = old_method_df.join(df, on='idx2', how='right')
    return joined
    #plt.plot(joined.intensity2)
    #plt.plot(joined.int1, alpha=0.5)
    

    