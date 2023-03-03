#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:17:24 2023

@author: ula
"""
import intersession
import utils
import constants
import cell_preprocessing as cp
import napari
import numpy as np
from skimage.draw import disk, circle_perimeter
#%%

img = utils.read_image(10, 1, "ctx")
img_w = utils.read_image(10, 1, "landmark1")
img3 = utils.read_image(10, 1, "landmark2")
df = utils.read_single_session_cell_data(10, 1, ["ctx"])
df['in_calculated'] = df.apply(cp.calculate_intensity_wrapper,img = img, axis = 1)
df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]

#%%
df['in_calculated_w'] = df.apply(cp.calculate_intensity_wrapper,img = img_w, axis = 1)


#%%
tr = intersession.distribution_change(df, 'in_calculated', 'in_calculated_w')
print(tr)

df = utils.read_single_session_cell_data(10, 1, ["landmark1"])
df['in_calculated'] = df.apply(cp.calculate_intensity_wrapper,img = img_w, axis = 1)
df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]

#%%
df['in_calculated_w'] = df.apply(cp.calculate_intensity_wrapper,img = img3, axis = 1)


#%%
tr = intersession.distribution_change(df, 'in_calculated', 'in_calculated_w')
print(tr)

#%%
import warnings
warnings.filterwarnings("ignore")
#%%
for i in [2,5,8,10, 11,13,14,16]:
    intersession.distribution_change_all_sessions(i,1,['ctx', 'landmark1', 'landmark2'])
    print("")
#%%

df = utils.read_single_session_cell_data(10, 1, ["ctx"])
img = utils.read_image(10, 1, "ctx")
df = cp.standarize_intensity(df, img)
for col in ['shift_x','shift_y','shift_z', 'int_optimized']:
    df[col] = 0
    
#df = df.iloc
#%%
cp.optimize_centroids(df, img)
#%%

def draw_normalized_area(x,y,z, img):
    shapes = []
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        center_z = z + i
        if not (center_z <0 or center_z>=img.shape[0]):
            diameter = constants.ROI_DIAMETER[abs(i)]
            rad = diameter//2
           
            disk_ = disk((y,x), rad, shape = img[0].shape)
    
            disk_shape = np.stack([np.full(len(disk_[0]), center_z), 
                                                          disk_[0], disk_[1]])
            shapes += [disk_shape.T]
    return shapes

#%%


sample_df = df.sample(100, axis = 0)

shapes_list = []
shifted_list = []
for idx, row in sample_df.iterrows():
    
    z = int(row[constants.ICY_COLNAMES['zcol']])
    x = int(row[constants.ICY_COLNAMES['xcol']])
    y = int(row[constants.ICY_COLNAMES['ycol']])
    shapes_list += draw_normalized_area(x,y,z,img)
    
    z = int(row[constants.ICY_COLNAMES['zcol']]) + row['shift_z']
    x = int(row[constants.ICY_COLNAMES['xcol']]) + row['shift_x']
    y = int(row[constants.ICY_COLNAMES['ycol']]) + row['shift_y']
    shifted_list += draw_normalized_area(x,y,z,img)

blob_points = np.vstack(shapes_list)

blob_points_shifted = np.vstack(shifted_list)

#%%
viewer = napari.view_image(
        img, 
        ndisplay=3, 
        scale=constants.SCALE
        ) 


viewer.add_points(
        blob_points,
        size=1,
        symbol='square',
        scale=constants.SCALE,
        face_color= 'green'
        )
viewer.add_points(
    blob_points_shifted,
    size=1,
    symbol='square',
    scale=constants.SCALE,
    face_color= 'red'
    )
'''
viewer.add_points(
        df[constants.COORDS_3D],
        size=5,
        # shading='spherical',
        symbol='ring',
        out_of_slice_display=True,
        scale=constants.SCALE,
        face_color= 'red'
        )

point_properties = {
    'color' : df["shift_x"]*100
    }
shifted = np.array([df[constants.ICY_COLNAMES['zcol']]+df["shift_z"],
 df[constants.ICY_COLNAMES['ycol']]+df["shift_y"],
 df[constants.ICY_COLNAMES['xcol']]+df["shift_x"]
 ])

pts_layer = viewer.add_points(
        shifted.T,
        properties = point_properties,
        size=5,
        # shading='spherical',
        symbol='ring',
        scale=constants.SCALE,
        face_color='color'
)      

napari.run()
'''
