#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:28:08 2023

@author: ula
"""
import numpy as np

dir_path = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/alignment_result/aligned_despeckle/"
dir_path = "/media/ula/DATADRIVE1/fos_gfp_tmaze/fos_gfp_tmaze2/sca_tifs/base/ready_to_process/despeckle/"
res_dir_path = dir_path + "alignment_result/"
path_for_icy = res_dir_path + "aligned_despeckle/"

ROI_DIAMETER = [8,6,4]
ICY_COLNAMES = {'mean_intensity' : 'Mean Intensity (ch 0)',
                'xcol' : 'Intensity center X (px) (ch 0)',
                'ycol' : 'Intensity center Y (px) (ch 0)',
                'zcol' : 'Intensity center Z (px) (ch 0)'}
XY_SCALE = 1.19
Z_SCALE = 2

ALIGNMENT_SUBSTACK_SIZE = 5
ALIGNMENT_SEARCH_WINDOW = 25

COORDS_3D = [ICY_COLNAMES['zcol'], ICY_COLNAMES['ycol'], ICY_COLNAMES['xcol']]

SCALE = [Z_SCALE, XY_SCALE, XY_SCALE]

FILENAMES = {'single_session_cell_data_fn' : "m{}r{}_{}_output.txt",#output from icy,
             #'cell_data_fn_template' : "m{}r{}_{}_output.txt",
             'cell_data_fn_template' : "m{0}s{2}_r{1}_output.txt", #for 2p ctx exp from 2021
             #'img_fn_template' : "m{}r{}_{}.tif",
             'img_fn_template' : "m{0}s{2}_r{1}.tif",
             'watershed_img_fn_template' : "m{}r{}_{}_watershed.tif"}
INTENSITY_HIST = "Intensity_diff_m{}_r{}"
INTENSITY_PLT = "Cell_intensity_m{}_r{}_s{}"

TOLERANCE = 5

CTX_FIRST_SESSIONS = ['ctx', 'landmark1', 'landmark2']
LANDMARK_FIRST_SESSIONS = ['landmark', 'ctx1', 'ctx2']

CTX_REGIONS = np.array([[5,1],[5,2],[5,3],[8,1],[8,2],[10,1],[10,2], [10,3],
               [13,1], [13,2], [13,3], [14,1], [14,2], [14,3], [16,1], #[16,2],
               [11,1], [20,2], [20,3], [11,2],[11,3],[13,2], [13,3], [14,4], 
               #[19,1], [19,2], [19,3],
               [2,1],[2,2], 
               [20,1]])

LANDMARK_REGIONS = [[1,1],[1,2],[3,1],[3,2],[4,1],[4,2],[6,1],[6,2], [7,1], [7,2], 
               [7,3], [9,1],[9,2], 
               [12,1], [12,2], [12,3], [15,1], [15,2], [15,3], 
               [17,1], [17,2], [18,1], [18,2]]

CTX_REGS_ROSTRAL = [[2,2], [5,1], [5,3], [10,1], [10,2], [11,1], [11,3], [13,2],
                    [14,2], [14,3], [16,1], [20,2], [20,3]]
LANDMARK_REGS_ROSTRAL = [[1,1], [3,1],[3,2], [4,2], [6,1], [6,2], [7,2], [7,3], [9,1],[9,2], 
                         [12,1], [12,2], [15,2], [15,3], [17,2], [18,2]] 

BEHAV_DIR = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/behav/Tmaze_2p_landmark_ctx.xlsx"

IMG_EXT = ".tif"

ALIGNMENT_FNAMES = {'thresh' : "m{}_r{}_{}_spots",
'raw' : "m{}_r{}_{}"}

ALIGNMENT_FNAMES = {'thresh' : "m{0}s{2}_r{1}_spots",
'raw' : "m{0}s{2}_r{1}"}


STACK_WINDOW = 11