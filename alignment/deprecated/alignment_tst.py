#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:53:21 2023

@author: ula
"""

import alignment_search as align

session_order = ['1','2','3']

regions = [[4,1],[4,2],[4,3],[5,1],[5,2],[6,2],[7,1],[7,2],[8,1],[8,2],[11,1],[11,2]]
regions = [[14,1],[15,1],[15,2],[15,3]]

for m, r in regions:
    align.align_all_sessions(m,r, session_order)
