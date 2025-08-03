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
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree

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

# def calculate_intensity(x,y,z, img):
#     sum_int = 0
#     area_int = 0
#     for i in range(-2,3): #going through 5 flat slices making up the 3d cell
#         diameter = constants.ROI_DIAMETER[abs(i)]
#         rad = diameter//2
#         if (z+i <0 or z+i>=img.shape[0]):
#             continue
#         img_s = img[z+i]
#         res = calculate_circ_slice(x,y,rad,img_s) 
#         sum_int += res[0]
#         area_int += res[1]
#     if area_int == 0:
#         return 0
#     return sum_int/area_int

def calculate_intensity(x, y, z, img):
    sum_int = 0
    area_int = 0
    z_slices = np.arange(z - 2, z + 3)  # 5 slices centered at z
    valid_slices = z_slices[(z_slices >= 0) & (z_slices < img.shape[0])]  # Keep valid z-indices

    for z_slice in valid_slices:
        diameter = constants.ROI_DIAMETER[abs(z_slice - z)]
        rad = diameter // 2

        xmin = max(x - rad, 0)
        xmax = min(x + rad, img.shape[2] - 1)
        ymin = max(y - rad, 0)
        ymax = min(y + rad, img.shape[1] - 1)

        img_s = img[z_slice]

        # Efficient NumPy sum within the bounding box
        brightness = int(img_s[ymax][xmax])+int(img_s[ymin][xmin])-int(img_s[ymin][xmax])-int(img_s[ymax][xmin])
        area =  (xmax-xmin)*(ymax-ymin)

        sum_int += brightness
        area_int += area

    return 0 if area_int == 0 else sum_int / area_int



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


def calculate_integral_image(stack):
    """Compute integral image for efficient brightness calculations."""
    integral_volume = np.array([img.cumsum(axis=0, dtype=np.int64).cumsum(axis=1, dtype=np.int64) for img in stack])
    return integral_volume

def find_active_cells(df, bgr, k_std):
    for i in range(3):
        b_mean = bgr[i][0]
        b_std = bgr[i][1]
        threshold = b_mean + k_std * b_std
        df[f'active{i}'] = df[f"int_optimized{i}"] > threshold
    return df

# def optimize_centroid_position(args):
#     """Optimize the position of a single centroid."""
#     row, img, suff, tolerance = args
#     x_center = int(np.round(row[ICY_COLNAMES['xcol']]))
#     y_center = int(np.round(row[ICY_COLNAMES['ycol']]))
#     z_center = int(np.round(row[ICY_COLNAMES['zcol']]))
    
#     current_max = -np.inf
#     best_coords = (0, 0, 0)

#     for x in range(-tolerance, tolerance + 1):
#         for y in range(-tolerance, tolerance + 1):
#             for z in range(-1, 2):
#                 new_x = x_center + x
#                 new_y = y_center + y
#                 new_z = z_center + z
                
#                 # Check bounds
#                 if not (0 <= new_x < img.shape[2] and 0 <= new_y < img.shape[1] and 0 <= new_z < img.shape[0]):
#                     continue
                
#                 mean_calculated = calculate_intensity(new_x, new_y, new_z, img)  
#                 if mean_calculated > current_max:
#                     current_max = mean_calculated
#                     best_coords = (x, y, z)

#     return (best_coords[0], best_coords[1], best_coords[2], current_max)


def optimize_centroid_position(args):
    """Optimize the position of a single centroid using vectorized operations."""
    row, img, suff, tolerance = args
    if "close_neighbour" in row.index and row["close_neighbour"]:
        tolerance = tolerance//2
    
    x_center = int(np.round(row[ICY_COLNAMES['xcol']]))
    y_center = int(np.round(row[ICY_COLNAMES['ycol']]))
    z_center = int(np.round(row[ICY_COLNAMES['zcol']]))
    
    # Generate all possible shifts
    x_shifts = np.arange(-tolerance, tolerance + 1)
    y_shifts = np.arange(-tolerance, tolerance + 1)
    z_shifts = np.array([-1, 0, 1])

    # Generate all possible coordinate combinations
    x_offsets, y_offsets, z_offsets = np.meshgrid(x_shifts, y_shifts, z_shifts, indexing='ij')
    
    new_x = x_center + x_offsets.ravel()
    new_y = y_center + y_offsets.ravel()
    new_z = z_center + z_offsets.ravel()

    # Keep only valid coordinates
    valid_mask = (0 <= new_x) & (new_x < img.shape[2]) & \
                 (0 <= new_y) & (new_y < img.shape[1]) & \
                 (0 <= new_z) & (new_z < img.shape[0])

    new_x, new_y, new_z = new_x[valid_mask], new_y[valid_mask], new_z[valid_mask]

    # Compute brightness in one batch
    brightness_values = np.array([calculate_intensity(x, y, z, img) for x, y, z in zip(new_x, new_y, new_z)])
    # added for the purpose of method validation tests
    if len(brightness_values) == 0:
        return (0,)*4
    
    
    # Find the best shift
    best_idx = np.argmax(brightness_values)
    best_coords = (new_x[best_idx] - x_center, new_y[best_idx] - y_center, new_z[best_idx] - z_center)

    return (*best_coords, brightness_values[best_idx])

def update_df_coords(df, suff):
    shift_arr = ["shift_z", 'shift_y', 'shift_x']
    for i, cname in enumerate(constants.COORDS_3D):
        df[cname] = df[cname] + df[shift_arr[i]+suff]
    return df

def optimize_centroids(df, img, suff="", tolerance=constants.TOLERANCE, update_coords=True):
    """Optimize the positions of centroids using parallel processing."""
    integral_img = calculate_integral_image(img)

    args_list = [(row, integral_img, suff, tolerance) for _, row in df.iterrows()]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(optimize_centroid_position, args_list))

    # Convert results back to DataFrame
    df[[f'shift_x{suff}', f'shift_y{suff}', f'shift_z{suff}', f'int_optimized{suff}']] = results

    if update_coords:
        df = update_df_coords(df, suff)

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

def calculate_intensity_row(args):
    """Calculate intensity for a given row and image."""
    row, img = args  # Unpack arguments
    integral_img = calculate_integral_image(img)
    x_center = int(row[ICY_COLNAMES['xcol']].round())
    y_center = int(row[ICY_COLNAMES['ycol']].round())
    z_center = int(row[ICY_COLNAMES['zcol']].round())

    if not (0 <= x_center < img.shape[2] and 0 <= y_center < img.shape[1] 
            and 0 <= z_center < img.shape[0]):
        return np.NaN
    return calculate_intensity(x_center, y_center, z_center, integral_img)

def calculate_background_intensity(df, img):
    """Calculate background intensity for a DataFrame using parallel processing."""
    
    bg_df = df[[ICY_COLNAMES[cname] for cname in ICY_COLNAMES if 'col' in cname]].copy()
    bg_df[ICY_COLNAMES['zcol']] = df[ICY_COLNAMES['zcol']] - constants.TOLERANCE - 0.1



    df_coords = df[constants.COORDS_3D].values
    bg_coords = bg_df[constants.COORDS_3D].values

    tree = cKDTree(df_coords)

    # Identify background points too close to foreground
    close_indices = tree.query_ball_point(bg_coords, r=constants.TOLERANCE)
    too_close = [i for i, neighbors in enumerate(close_indices) if len(neighbors) > 0]

    bg_df.drop(index=bg_df.index[too_close], inplace=True)

    # Prepare data for parallel processing
    args_list = [(row, img) for _, row in bg_df.iterrows()]


    with ProcessPoolExecutor() as executor:
        bg_df['bg_intensity'] = list(executor.map(calculate_intensity_row, args_list))



    bg_df.dropna(inplace=True)

    return bg_df, None
    





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
    

    