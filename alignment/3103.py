#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 10:13:32 2023

@author: ula
"""

import constants
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import utils
#%%
def plot_consecutive_sessions(m,r, session):
    df = pd.read_csv(constants.dir_path +"m" + str(m)+"_r"+str(r)+"_"+session +"_optimized_n.csv")
    df.sort_values(by='int_optimized_n', inplace=True)
    df.sort_values(by='int_optimized', inplace=True)
    df['lower_lim'] = df.int_optimized*0.8
    df['upper_lim'] = df.int_optimized*1.2
    increase = df[df.int_optimized_n>df.upper_lim]
    decrease = df[df.int_optimized_n<df.lower_lim]
    # plt.plot(df.int_optimized.to_numpy(), linestyle='None', marker='.', markersize=2)
    # plt.plot(df.int_optimized_n.to_numpy(), linestyle='None', marker='.', markersize=2)
    # plt.plot(df.lower_lim.to_numpy(), markersize=2, color='red')
    # plt.plot(df.upper_lim.to_numpy(), markersize=2, color='red')
    # plt.show()
    behav = utils.read_behav_data()
    print(behav)
    performance = behav.loc[m, 'performance_diff']
    cell_data = np.array([df.shape[0]-decrease.shape[0]-increase.shape[0],
                     decrease.shape[0], increase.shape[0]])/df.shape[0]
    return np.append(cell_data, performance)



#%%
cell_population_data = pd.DataFrame(columns = ['stable', 'decrease', 'increase', 'behav'])
for m,r in constants.CTX_REGIONS:
    cell_population_data.loc[len(cell_population_data)] = plot_consecutive_sessions(
        m,r, constants.CTX_FIRST_SESSIONS[1])
    #class_sizes_ll += [plot_consecutive_sessions(m,r, constants.CTX_FIRST_SESSIONS[2])]

cell_population_data.sort_values(by='behav', inplace=True)
plt.plot(cell_population_data.behav.to_numpy(), 
         cell_population_data.increase.to_numpy(),
         linestyle="None",
         marker='o')
plt.show()
# class_sizes_cl = np.array(class_sizes_cl)
# class_sizes_ll = np.array(class_sizes_ll)

# cl_range = np.arange(0,6,2)
# ll_range = np.arange(1,6,2)

# plt.plot(cl_range,class_sizes_cl.mean(axis=0), linestyle='None', marker='_', markersize=20)
# plt.plot(ll_range,class_sizes_ll.mean(axis=0), linestyle='None', marker='_', markersize=20)
# plt.title("Mean int of stable, decreasing and increasing cells")
# plt.xticks(range(6), ["CL", "LL"]*3)
# for i in range(len(class_sizes_cl)):
#     plt.plot(cl_range, class_sizes_cl[i], linestyle='None', marker='.', markersize=2)
#     plt.plot(ll_range, class_sizes_ll[i], linestyle='None', marker='.', markersize=2)
# plt.show()