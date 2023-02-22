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

SCALE = [Z_SCALE, XY_SCALE, XY_SCALE]

FILENAMES = {'single_session_cell_data_fn' : "m{}r{}_{}_output.txt",#output from icy,
             'cell_data_fn_template' : "m{}r{}_{}_output.txt",
             'img_fn_template' : "m{}r{}_{}.tif"}

TOLERANCE = 5

CTX_FIRST_SESSIONS = ['ctx', 'landmark1', 'landmark2']