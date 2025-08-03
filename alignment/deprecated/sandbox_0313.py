#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:28:16 2023

@author: ula
"""

import constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import intersession as ints
from scipy import signal
#%%

def assign_type_orig(row):
    if row.int_optimized1 > row.int_optimized0:
        if row.int_optimized2 > row.int_optimized0:
            return'A'
        else:
            return 'B'
    else:
        if row.int_optimized2 < row.int_optimized0:
            return 'C'
        else:
            return 'D'
        
def assign_type(row):
    if row.int_optimized1 > row.int_optimized0:
        if row.int_optimized2 > row.int_optimized0:
            return'A'
    else:
        if row.int_optimized2 < row.int_optimized0:
            return 'C'
        


def calculate_r2(df, col1, col2):
    int_avg = df[col2].mean()
    sse = ((df[col1] - df[col2])**2).sum()
    stot = ((df[col1] - int_avg)**2).sum()
    return (1-sse/stot)
    

def top_cells_intensity_change(mouse, region, sessions, percentage):
    # top_df = pd.DataFrame(columns=constants.COORDS_3D)
    # for s in sessions:
    #     df = get_brightest_cells(mouse, region, s, percentage)
    #     top_df =pd.concat([top_df,df], ignore_index=True)
    # print(top_df.shape[0])
    # for i, s in enumerate(sessions):
    #     img = utils.read_image(mouse, region, s)
    #     top_df = cp.optimize_centroids(top_df, img, str(i))
    # print(top_df[['int_optimized0', 'int_optimized1', 'int_optimized2']])
    # top_df = top_df.drop_duplicates(subset=['int_optimized0', 'int_optimized1', 'int_optimized2'])
    # print(top_df.shape[0])
    top_df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_top.csv")
    # r2_cl = calculate_r2(top_df, 'int_optimized1', 'int_optimized0')
    # r2_ll = calculate_r2(top_df, 'int_optimized2', 'int_optimized1')

    top_df['type'] = top_df.apply(assign_type, axis = 1)
    type_frac = [ top_df[top_df.type== i].shape[0]/top_df.shape[0]  for i in ['A', 'B', 'C', 'D'] ]
    top_df.sort_values(by='int_optimized1', inplace=True)
    top_df.sort_values(by='int_optimized0', inplace=True)
    nbins = 50
    plt.plot(top_df.int_optimized0.to_numpy(), linestyle="None", marker='.')
    plt.plot(top_df.int_optimized1.to_numpy(), linestyle="None", marker='.')
    plt.show()
    # plt.title(constants.INTENSITY_PLT.format(mouse, region, 'CL'))
    #plt.show()
    # plt.savefig(constants.dir_path+constants.INTENSITY_PLT.format(mouse, region, 'CL')+".png")
    # plt.close()
    top_df.sort_values(by='int_optimized2', inplace=True)
    top_df.sort_values(by='int_optimized1', inplace=True)
    plt.plot(top_df.int_optimized1.to_numpy(), linestyle="None", marker='.')
    plt.plot(top_df.int_optimized2.to_numpy(), linestyle="None", marker='.')
    plt.title("LL")
    plt.show()
    # plt.hist(np.array(abs(top_df.int_optimized1-top_df.int_optimized2)), bins=nbins, alpha=0.5)
    # plt.title(constants.INTENSITY_PLT.format(mouse, region, 'LL'))
    # plt.show()
    # plt.savefig(constants.dir_path+constants.INTENSITY_PLT.format(mouse, region, 'LL')+".png")
    # plt.close()
    # plot_title = constants.INTENSITY_HIST.format(mouse,region)
    # plotting.plot_intensity_change_histogram(np.array(top_df.int_optimized0-top_df.int_optimized1), 
    #                                          "C-L",
    #                                          np.array(top_df.int_optimized1-top_df.int_optimized2),
    #                                          "L-L",
    #                                          plot_title)
    #return r2_cl, r2_ll

#%%

def top_cells_intensity_change_(mouse, sessions, percentage, tit):
    tot_size = 0
    type_frac = np.array([0,0,0,0])
    for reg in mouse[1]:
        top_df = pd.read_csv(constants.dir_path +"m" + str(mouse[0])+"_r"+str(reg)+"_top.csv")
        top_df['type'] = top_df.apply(assign_type, axis = 1)
        type_frac += np.array([top_df[top_df.type== i].shape[0]  for i in ['A', 'B', 'C', 'D'] ])
        tot_size += top_df.shape[0]
    return type_frac/tot_size
    #plt.plot(['A', 'B', 'C', 'D'],type_frac, linestyle="None", marker="." )
    #plt.title(tit)
    # plt.hist(np.array(abs(top_df.int_optimized1-top_df.int_optimized2)), bins=nbins, alpha=0.5)
    # plt.title(constants.INTENSITY_PLT.format(mouse, region, 'LL'))
    # plt.show()
    # plt.savefig(constants.dir_path+constants.INTENSITY_PLT.format(mouse, region, 'LL')+".png")
    # plt.close()
    # plot_title = constants.INTENSITY_HIST.format(mouse,region)
    # plotting.plot_intensity_change_histogram(np.array(top_df.int_optimized0-top_df.int_optimized1), 
    #                                          "C-L",
    #                                          np.array(top_df.int_optimized1-top_df.int_optimized2),
    #                                          "L-L",
    #                                          plot_title)
    #return r2_cl, r2_ll


from scipy import stats
#%%
cmice = {}
for k, v in constants.CTX_REGS_ROSTRAL:
    cmice.setdefault(k, []).append(v)
print(cmice)
lmice = {}
for k, v in constants.LANDMARK_REGS_ROSTRAL:
    lmice.setdefault(k, []).append(v)
print(lmice)

#%%
r2_arr = []
for m in cmice.items():
    print(m)
    r2_arr += [top_cells_intensity_change_(m, constants.CTX_FIRST_SESSIONS, 0.1, 'CLL')]
res = np.array(r2_arr).T[0::2]
fig, ax = plt.subplots()
ax.set_ylim(0,0.74)
ax.bar([0,1],np.mean(res, axis = 1), yerr=np.std(res, axis = 1), capsize=10)
ax.set_xticks([0,1])
ax.set_xticklabels(['LANDMARK specific','CTX specific'])
#ax.plot(res, marker='o')
#ax.hlines(y=1, linewidth=2, color='r', linestyle='-', xmin=-0.05, xmax=1.05)
ax.set_ylabel('Fraction of cells')
print(np.mean(np.array(r2_arr).T, axis=1))
plt.title("CTX-LANDMARK-LANDMARK")
plt.show()
print(stats.ttest_ind(res[0], res[1], equal_var=False))

#%%
r2_arr = []
for m in lmice.items():
    r2_arr += [top_cells_intensity_change_(m, constants.LANDMARK_FIRST_SESSIONS, 0.1, 'LCC')]
res = np.array(r2_arr).T[0::2]

fig, ax = plt.subplots()
ax.set_ylim(0,0.74)
ax.bar([0,1],np.mean(res, axis = 1), yerr=np.std(res, axis = 1), capsize=10)
ax.set_xticks([0,1])
ax.set_xticklabels(['CTX specific', 'LANDMARK specific'])
#ax.plot(res, marker='o')
#ax.hlines(y=1, linewidth=2, color='r', linestyle='-', xmin=-0.05, xmax=1.05)
print(np.mean(np.array(r2_arr).T, axis=1))
plt.title("LANDMARK-CTX-CTX")
plt.show()
print(stats.ttest_ind(res[0], res[1], equal_var=False))
#%%
r2_arr = np.array(r2_arr)
for reg in r2_arr:
    plt.plot(reg, marker='o')
plt.title('R2 LCC')
plt.show()
#%%
plotting.plot_intensity_change_histogram(np.array(top_df.int_optimized0-top_df.int_optimized1), 
                                          "C-L",
                                          np.array(top_df.int_optimized1-top_df.int_optimized2),
                                          "L-L",
                                          plot_title)