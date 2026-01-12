#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 11:47:12 2025

@author: ula
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

mouse_name = ['m1', 'm3', 'm4', 'm5', 'm6', 'm8', 'm11', 'm12', 'm14', 'm15', 'm16']
pre_test = [84, 88, 70, 74, 86, 90, 70, 86, 82, 70, 70]
test = [62, 52, 70, 48, 68, 48, 66, 48, 60, 62,54]
avg = [76.66666667,78.66666667,69,76.66666667,72.7,71.66666667,69.66666667,59.16666667,80,64.33333333,70.66666667]

#%%
df = pd.DataFrame(data={'name':mouse_name, 'pre_test' : pre_test, 'test': test, 'avg':avg})
#%% 
df['diff'] = df['pre_test'] - df['test']
df_sorted = df.sort_values(by=['test'], ascending = False)

plt.plot(df_sorted.name, df_sorted.pre_test, label='non-conflict test')
plt.plot(df_sorted.name, df_sorted.test, label='conflict test - landmark choices')
plt.ploy
plt.legend()
plt.show()

plt.plot(df_sorted.name, df_sorted.avg, label='avg performance 3 days before test')
plt.plot(df_sorted.name, df_sorted.test, label='conflict test - landmark choices')
plt.legend()
plt.show()