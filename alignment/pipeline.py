#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 17:35:42 2024

@author: ula
"""

import intersession as ints
from scipy import stats

#%%

mice = [[3,2]]
diff = []
same = []


for m,r in mice:
    ov_size, df = ints.identify_persistent_cells_w_thresholding(m,r,[2,1])
    diff+=[ov_size]
    print(ov_size)
    
    ov_size, df = ints.identify_persistent_cells_w_thresholding(m,r,[3,2])
    same+=[ov_size]
    print(ov_size)
    
#%%

#stats.ttest_ind(diff, same, equal_var=True)
#%%