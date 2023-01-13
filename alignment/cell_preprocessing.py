# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.draw import disk
import numpy as np


ROI_DIAMETER = [8,7,4]
ICY_COLNAMES = {'mean_intensity' : 'Mean Intensity (ch 0)',
                'xcol' : 'Intensity center X (px) (ch 0)',
                'ycol' : 'Intensity center Y (px) (ch 0)',
                'zcol' : 'Intensity center Z (px) (ch 0)'}
XY_SCALE = 1.19;
Z_SCALE = 2;

single_session_cell_data_fn = "m{}r{}_{}_output.txt"#output from icy


def calculate_intensity(center_coords, img):
    center_coords = [int(round(cc)) for cc in center_coords]
    sum_int = 0
    area_int = 0
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        if(center_coords[2]+i >= 0 and center_coords[2]+i < img.shape[0]):
            diameter = ROI_DIAMETER[abs(i)]
            rad = diameter//2
            disk_ = disk((center_coords[0], center_coords[1]), rad, shape=img[0].shape)
            sum_int += np.sum(img[center_coords[2]+i][disk_])
            area_int += len(img[center_coords[2]+i][disk_])
    return sum_int/area_int

def filter_unstable_intensity(df, img):
    df["intensity_standarized"] = df.apply(lambda row : 
                                           calculate_intensity([row[ICY_COLNAMES['xcol']],
                                                               row[ICY_COLNAMES['ycol']],
                                                               row[ICY_COLNAMES['zcol']]],
                                                               img), axis = 1)
    return df[df['Mean Intensity (ch 0)']/df['intensity_standarized']<=1.5]


def pixels_to_um(df):
    df[ICY_COLNAMES['xcol']] = df[ICY_COLNAMES['xcol']]*XY_SCALE
    df[ICY_COLNAMES['ycol']] = df[ICY_COLNAMES['ycol']]*XY_SCALE
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]*Z_SCALE
    return df

def find_session_ovlap(mouse, region, s1_idx, s2_idx):
    