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

def identify_persistent_cells(mouse, region, sessions):
    #tu dodac standaryzacje intensity
    for i, s in enumerate(sessions):
        sessions[i] = s[constants.COORDS_3D].reset_index()
    cross_prod = sessions[0].merge(sessions[1], how = "cross", suffixes=("_s1", "_s2"))
    cross_prod['distance'] = np.linalg.norm(cross_prod[[n+"_s1" for n in constants.COORDS_3D]].values
                                            -cross_prod[[n+"_s2" for n in constants.COORDS_3D]].values, axis=1)
    cross_prod = cross_prod[cross_prod.distance < constants.TOLERANCE]

    return cross_prod

identify_persistent_cells(10,1, utils.read_single_session_cell_data(
                              10, 1, constants.CTX_FIRST_SESSIONS[:2]))
