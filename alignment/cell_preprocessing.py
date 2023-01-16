# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.draw import disk
import numpy as np
import pandas as pd
from skimage import io

dir_path = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/alignment_result/aligned_despeckle/"


ROI_DIAMETER = [8,7,4]
ICY_COLNAMES = {'mean_intensity' : 'Mean Intensity (ch 0)',
                'xcol' : 'Intensity center X (px) (ch 0)',
                'ycol' : 'Intensity center Y (px) (ch 0)',
                'zcol' : 'Intensity center Z (px) (ch 0)'}
XY_SCALE = 1.19;
Z_SCALE = 2;

single_session_cell_data_fn = "m{}r{}_{}_output.txt"#output from icy
cell_data_fn_template = "m{}r{}_{}_output.txt"
img_fn_template = "m{}r{}_{}.tif"


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
    df["intensity_standarized"] = calculate_intensity([df[ICY_COLNAMES['xcol']],
                                                               df[ICY_COLNAMES['ycol']],
                                                               df[ICY_COLNAMES['zcol']]],
                                                               img)
    return df[df['Mean Intensity (ch 0)']/df['intensity_standarized']<=1.5]


def pixels_to_um(df):
    df[ICY_COLNAMES['xcol']] = df[ICY_COLNAMES['xcol']]*XY_SCALE
    df[ICY_COLNAMES['ycol']] = df[ICY_COLNAMES['ycol']]*XY_SCALE
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]*Z_SCALE
    return df

def find_overlap(mouse, region, s_idxses, session_order):
    dfs = dict.fromkeys(s_idxses)
    imgs = dict.fromkeys(s_idxses)
    for i in s_idxses:
        dfs[i] = pd.read_csv(dir_path + cell_data_fn_template
                             .format(mouse, region, session_order[i]), "\t", header=1)
        imgs[i] = io.imread(dir_path + img_fn_template
                             .format(mouse, region, session_order[i])).astype("uint8")
    #dokonczyc przepisywanie oryginalnego porownania z nb
    #wektoryzacja
    
def test_fun(mouse, region, s_idxses, session_order):
    pd.read_csv(dir_path + cell_data_fn_template
                .format(mouse, region, session_order[s_idxses[0]]+"_"+session_order[s_idxses[1]]))
    df = pd.read_csv(dir_path + cell_data_fn_template
                     .format(mouse, region, session_order[s_idxses[0]]), "\t", header=1)
    img = io.imread(dir_path + img_fn_template
                    .format(mouse, region, session_order[s_idxses[1]])).astype("uint8")
    calculate_intensity([df[ICY_COLNAMES['xcol']],
                                                               df[ICY_COLNAMES['ycol']],
                                                               df[ICY_COLNAMES['zcol']]],
                                                               img)
    

    
    