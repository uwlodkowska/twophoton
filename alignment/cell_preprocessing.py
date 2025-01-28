# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from skimage.draw import disk
import numpy as np
import pandas as pd
from skimage import io
from constants import ICY_COLNAMES
import constants


def calculate_disk(coords, radius, disk_no, img):
    center_z = coords[ICY_COLNAMES['zcol']]+disk_no
    if (center_z <0 or center_z>=img.shape[0]):
        return [0,0] #spr czy to potrzebne
    disk_ = disk((coords[ICY_COLNAMES['ycol']],coords[ICY_COLNAMES['xcol']]), radius,shape = img[0].shape) 
    sum_int = np.sum(img[center_z][disk_])
    area_int = len(img[center_z][disk_])
    return [sum_int, area_int]

def calculate_intensity_old(coords, img):
    center_coords_df = coords[constants.COORDS_3D]
    center_coords_df = center_coords_df.round().astype(int)
    sum_int = 0
    area_int = 0
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        diameter = constants.ROI_DIAMETER[abs(i)]
        rad = diameter//2
        res = calculate_disk(center_coords_df,rad, i, img)
        sum_int += res[0]
        area_int += res[1]
    if area_int == 0:
        return 0
    return sum_int/area_int

def calculate_circ_slice(x,y,rad, img_s):
    xmin = max(x-rad, 0)
    ymin = max(y-rad, 0)
    xmax = min(x+rad, img_s.shape[1]-1)
    ymax = min(y+rad, img_s.shape[0]-1)
    brightness = int(img_s[ymax][xmax])+int(img_s[ymin][xmin])-int(img_s[ymin][xmax])-int(img_s[ymax][xmin])
    area =  (xmax-xmin)*(ymax-ymin)
    return brightness, area

def calculate_intensity_row(row, img):
    x_center = int(row[ICY_COLNAMES['xcol']].round())
    y_center = int(row[ICY_COLNAMES['ycol']].round())
    z_center = int(row[ICY_COLNAMES['zcol']].round())
    
    if not (0 <= x_center < img.shape[2] and 0 <= y_center < img.shape[1]
            and 0 <= z_center < img.shape[0]):
        print(row)
        return np.NaN
    return calculate_intensity(x_center, y_center, z_center, img)

def calculate_intensity(x,y,z, img):
    sum_int = 0
    area_int = 0
    for i in range(-2,3): #going through 5 flat slices making up the 3d cell
        diameter = constants.ROI_DIAMETER[abs(i)]
        rad = diameter//2
        if (z+i <0 or z+i>=img.shape[0]):
            continue
        img_s = img[z+i]
        res = calculate_circ_slice(x,y,rad,img_s) 
        sum_int += res[0]
        area_int += res[1]
    if area_int == 0:
        return 0
    return sum_int/area_int

def get_brightest_cells(mouse, region, session, percentage):
    df = pd.read_csv(constants.dir_path +"m" + str(mouse)+"_r"+str(region)+"_"+ 
                     session +"_optimized.csv")
    df = df.loc[df.int_optimized > df.int_optimized.quantile(1-percentage)]
    df = df[constants.COORDS_3D]
    return df
 
def optimize_centroid_position_old(row, img, suff):
    current_max = 0
    best_coords = [0,0,0]
    for x in range(-constants.TOLERANCE, constants.TOLERANCE+1):
        for y in range(-constants.TOLERANCE, constants.TOLERANCE+1):
            for z in range(-1, 2):
                tst_coords = {
                    ICY_COLNAMES['xcol'] : int(row[ICY_COLNAMES['xcol']].round())+x,
                    ICY_COLNAMES['ycol'] : int(row[ICY_COLNAMES['ycol']].round())+y,
                    ICY_COLNAMES['zcol'] : int(row[ICY_COLNAMES['zcol']].round())+z,
                    }
                mean_calculated = calculate_intensity_old(pd.Series(tst_coords), img)
                if mean_calculated > current_max:
                    current_max = mean_calculated
                    best_coords = [x,y,z]
    row['shift_x'+suff] = best_coords[0]
    row['shift_y'+suff] = best_coords[1]
    row['shift_z'+suff] = best_coords[2]
    row['int_optimized'+suff] = current_max
    
    return row

def optimize_centroids_old(df, img, suff=""):
    for col in ['shift_x','shift_y','shift_z', 'int_optimized']:
        df[col+suff] = 0
    df = df.apply(optimize_centroid_position_old,img =img, suff=suff, axis=1)
    return df


from scipy.ndimage import uniform_filter

def calculate_integral_image(stack):
    """Compute integral image for efficient brightness calculations."""
    integral_volume = np.array([img.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64) for img in stack])
    return integral_volume

def optimize_centroid_position(row, img, suff, tolerance):
    """Optimize the position of a single centroid."""
    x_center = int(row[ICY_COLNAMES['xcol']].round())
    y_center = int(row[ICY_COLNAMES['ycol']].round())
    z_center = int(row[ICY_COLNAMES['zcol']].round())
    
    current_max = -np.inf
    best_coords = (0, 0, 0)
    
    for x in range(-tolerance, tolerance + 1):
        for y in range(-tolerance, tolerance + 1):
            for z in range(-1, 2):
                new_x = x_center + x
                new_y = y_center + y
                new_z = z_center + z
                
                # Check bounds if needed
                if not (0 <= new_x < img.shape[2] and 0 <= new_y < img.shape[1]
                        and 0 <= new_z < img.shape[0]):
                    continue
                
                mean_calculated = calculate_intensity(new_x, new_y, new_z, img)  # Update with integral method
                if mean_calculated > current_max:
                    current_max = mean_calculated
                    best_coords = (x, y, z)
    row[f'shift_x{suff}'] = best_coords[0]
    row[f'shift_y{suff}'] = best_coords[1]
    row[f'shift_z{suff}'] = best_coords[2]
    row[f'int_optimized{suff}'] = current_max
    return row

def optimize_centroids(df, img, suff="", tolerance=constants.TOLERANCE):
    """Optimize the positions of centroids in a DataFrame."""
    integral_img = calculate_integral_image(img)
    #df = df.parallel_apply(optimize_centroid_position, img=integral_img, suff=suff, tolerance=tolerance, axis=1)
    df = df.apply(optimize_centroid_position, img=integral_img, suff=suff, tolerance=tolerance, axis=1)
    return df



def standarize_intensity(df, img):
    df["intensity_standarized"] = df.apply(calculate_intensity,img = img, axis = 1)
    return df[df['Mean Intensity (ch 0)']/df['intensity_standarized']<=1.5]


def find_quantile_threshold(df, img):
    df = standarize_intensity(df, img)
    return df["intensity_standarized"]

def pixels_to_um(df):
    df[ICY_COLNAMES['xcol']] = df[ICY_COLNAMES['xcol']]*constants.XY_SCALE
    df[ICY_COLNAMES['ycol']] = df[ICY_COLNAMES['ycol']]*constants.XY_SCALE
    df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']]*constants.Z_SCALE
    return df

def calculate_background_intensity(df, img):
    bg_df = pd.DataFrame()
    for cname in ICY_COLNAMES:
        if 'col' in cname:
            bg_df[ICY_COLNAMES[cname]] = df[ICY_COLNAMES[cname]]-5#magic string! z-axis size of cell
    bg_df['bg_intensity'] = df.apply(calculate_intensity_row ,img = img, axis = 1)
    #bg_df['bg_intensity'] = df.parallel_apply(calculate_intensity_row ,img = img, axis = 1)
    return bg_df
    
def test_fun(mouse, region, s_idxses, session_order):
    old_method_df = pd.read_csv(constants.dir_path + constants.FILENAMES["cell_data_fn_template"]
                .format(mouse, region, session_order[s_idxses[1]]+"_"+session_order[s_idxses[0]]))
    df = pd.read_csv(constants.dir_path + constants.FILENAMES["cell_data_fn_template"]
                     .format(mouse, region, session_order[s_idxses[0]]), "\t", header=1)

    img_ref = io.imread(constants.dir_path + constants.FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[0]])).astype("uint8")

    img_comp = io.imread(constants.dir_path + constants.FILENAMES["img_fn_template"]
                    .format(mouse, region, session_order[s_idxses[1]])).astype("uint8")
    df["int1"] = df.apply(calculate_intensity, img = img_ref, axis = 1)
    df["int2"] = df.apply(calculate_intensity, img = img_comp, axis = 1)
    joined = old_method_df.join(df, on='idx2', how='right')
    return joined
    #plt.plot(joined.intensity2)
    #plt.plot(joined.int1, alpha=0.5)
    

    