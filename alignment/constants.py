#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:28:08 2023

@author: ula
"""

dir_path = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/alignment_result/aligned_despeckle/"


ROI_DIAMETER = [8,7,4]
ICY_COLNAMES = {'mean_intensity' : 'Mean Intensity (ch 0)',
                'xcol' : 'Intensity center X (px) (ch 0)',
                'ycol' : 'Intensity center Y (px) (ch 0)',
                'zcol' : 'Intensity center Z (px) (ch 0)'}
XY_SCALE = 1.19
Z_SCALE = 2

COORDS_3D = [ICY_COLNAMES['zcol'], ICY_COLNAMES['ycol'], ICY_COLNAMES['xcol']]

SCALE = [Z_SCALE, XY_SCALE, XY_SCALE]

FILENAMES = {'single_session_cell_data_fn' : "m{}r{}_{}_output.txt",#output from icy,
             'cell_data_fn_template' : "m{}r{}_{}_output.txt",
             'img_fn_template' : "m{}r{}_{}.tif",
             'watershed_img_fn_template' : "m{}r{}_{}_watershed.tif"}
INTENSITY_HIST = "Intensity_diff_m{}_r{}"
INTENSITY_PLT = "Cell_intensity_m{}_r{}_s{}"

TOLERANCE = 3

CTX_FIRST_SESSIONS = ['ctx', 'landmark1', 'landmark2']
LANDMARK_FIRST_SESSIONS = ['landmark', 'ctx1', 'ctx2']

CTX_REGIONS = [[2,1],[2,2],[5,1],[5,2],[5,3],[8,1],[8,2],[10,1],[10,2], [10,3],
               [13,1], [13,2], [13,3], [14,1], [14,2], [14,3], [16,1], [16,2],
               [11,1], [20,2], [20,3]]

LANDMARK_REGIONS = [[1,1],[1,2],[3,1],[3,2],[4,1],[4,2],[6,1],[6,2], [7,1], [7,2], 
               [7,3], [9,1],[9,2]]