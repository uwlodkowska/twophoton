#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:48:04 2025

@author: ula
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml
import utils
from skimage import io
#%%
def plot_z_decay_curve(stack, label="Session", use_median=False):
    if use_median:
        decay = np.median(stack, axis=(1, 2))  # Median intensity per Z-slice
    else:
        decay = np.mean(stack, axis=(1, 2))    # Mean intensity per Z-slice

    #plt.figure(figsize=(6, 4))
    plt.plot(np.arange(len(decay)), decay, marker='o', label=label)
    plt.xlabel("Z-slice")
    plt.ylabel("Mean Intensity")
    plt.title(f"Z-decay Curve: {label}")
    plt.legend()
    plt.grid(True)
    #plt.tight_layout()
    #plt.show()



#%%

config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/ctx_landmark_rev.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
OPT_PATH = config["experiment"]["optimized_path"]


regions = config["experiment"]["regions"][0]
group_session_order = config["experiment"]["session_order"][0]#[1:-1]
#%%
m,r = (1,1)
img_arr = []
img_arr_sca = []
for i, session_id in enumerate(group_session_order[-1:]):
    img = io.imread(f"/mnt/data/fos_gfp_tmaze/2025_2p/despeckle/m{m}r{r}_{session_id}.tif")
    img_arr += [img]
    img_sca = io.imread(f"/mnt/data/fos_gfp_tmaze/2025_2p/despeckle_not_sca/m{m}r{r}_{session_id}.tif")
    img_arr_sca += [img_sca]

    plot_z_decay_curve(img, f"{session_id} unadjusted")
    plot_z_decay_curve(img_sca, session_id)
    plt.show()