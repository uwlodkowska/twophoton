#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:27:56 2023

@author: ula
"""
import numpy as np
import pandas as pd
from constants import ICY_COLNAMES

def identify_persistent_cells(mouse, region, sessions):
    res_df = pd.DataFrame(columns=["idx1", "idx2", "intensity1", "intensity2"])
    dfs = 
    '''
    multiple_algs = 0
    for idx1, row1 in df1.iterrows():
        rep_counter = 0
        for idx2, row2 in df2.iterrows():
            dist = np.linalg.norm(row1[xcolname:zcolname]-row2[xcolname:zcolname])
            if dist < tolerance:
                rep_counter += 1
                if rep_counter == 2:
                    multiple_algs+=1
                elif rep_counter == 1:
                    res_df = res_df.append({"idx1":idx1, "idx2" :idx2, 
                                            "intensity1":row1['intensity_standarized'], 
                                            "intensity2":row2['intensity_standarized']},
                                           ignore_index=True)
                    '''
    return res_df

identify_persistent_cells(10,1,[])