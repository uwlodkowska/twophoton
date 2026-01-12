#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:54:53 2025

@author: ula
"""

import pandas as pd

#%%

df = pd.read_csv("/home/ula/Downloads/tmaze_hybrids - idisco.csv")
#%%
print(df.columns)
#%%
df = df.drop(columns=['0', 'orientation', 'threshold_max', 'threshold_shape',
       'Unnamed: 5', 'final_landmark', 'final_ctx','training_length', 'training_test_ratio'])

#%%
df['final_training'] = pd.to_numeric(df['final_training'].str.replace(',', '.'), errors='coerce')



#%%
df.to_csv("/home/ula/Downloads/tmaze_hybrids_behav.csv")
#%%