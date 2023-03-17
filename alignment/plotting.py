#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:43:04 2023

@author: ula
"""
import matplotlib.pyplot as plt
import constants

def plot_intensity_change_histogram(d1, label1, d2, label2, title):
    plt.title(title)
    plt.hist(d1, bins=150, range=(-140,140), label = label1)
    plt.hist(d2, bins=150, range=(-140,140), alpha=0.5, label = label2)
    plt.legend()
    plt.savefig(constants.dir_path+title+".png")
    plt.close()