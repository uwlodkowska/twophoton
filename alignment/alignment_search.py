# -*- coding: utf-8 -*-

import numpy as np
import constants
from skimage import io
import tifffile, csv
from os import path
import sys
import yaml

#%%
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/test.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

#%%
DIR_PATH = config["experiment"]["dir_path"]
RESULT_PATH = DIR_PATH + config["experiment"]["res_dir_path"]
ICY_PATH = DIR_PATH + config["experiment"]["path_for_icy"]


regions = config["experiment"]["regions"][0]
group_session_order = config["experiment"]["session_order"][0]

alignment_filenames = config["alignment_filenames"]

#%%
def calculate_cropped_diff(substack1, substack2, x, y):
    xdif = min(substack1.shape[1], substack2.shape[1])-abs(x);
    ydif = min(substack1.shape[2], substack2.shape[2])-abs(y);
    sub1_tmp = substack1[:,max(0,x):max(0,x)+xdif,max(0,y):max(0,y)+ydif]
    sub2_tmp = substack2[:,max(0,-x):max(0,-x)+xdif,max(0,-y):max(0,-y)+ydif]
    dif_tmp = abs(sub1_tmp - sub2_tmp)
    return dif_tmp

def find_optimal_crop(substack1, substack2):
    expansions = 0
    step = 10
    start_range = 25
    x_start, y_start, x_end, y_end = -start_range, -start_range, start_range, start_range
    
    while True:
        xrange = range(x_start, x_end)
        yrange = range(y_start, y_end)

        minv = float('inf')
        minx, miny = None, None

        for x in xrange:
            for y in yrange:
                dif_cropped = calculate_cropped_diff(substack1, substack2, x, y)
                tmp_avg = np.average(dif_cropped)
                if tmp_avg < minv:
                    minv = tmp_avg
                    minx = x
                    miny = y

        edge_x = None
        edge_y = None

        if minx == x_start:
            edge_x = 'left'
        elif minx == x_end - 1:
            edge_x = 'right'
        if miny == y_start:
            edge_y = 'down'
        elif miny == y_end - 1:
            edge_y = 'up'

        if not edge_x and not edge_y:
            break  # Done: min is not on any edge

        # Expand only where needed
        if edge_x == 'left':
            x_start -= step
        elif edge_x == 'right':
            x_end += step

        if edge_y == 'down':
            y_start -= step
        elif edge_y == 'up':
            y_end += step

        expansions += 1


    
    return np.array([[minx],[miny]]), minv

def find_crop_coords(stack1, stack2):
    translations = np.array([[],[]])
    stack_size = stack1.shape[0]
    sum_of_avgs = 0
    start_idx = 0
    
    while start_idx + constants.ALIGNMENT_SUBSTACK_SIZE < stack_size:
        stop_idx = start_idx + constants.ALIGNMENT_SUBSTACK_SIZE
        if stop_idx + constants.ALIGNMENT_SUBSTACK_SIZE > stack_size:
            stop_idx = stack_size+1
        coords, sub_avg = find_optimal_crop(stack1[start_idx:stop_idx], 
                                            stack2[start_idx:stop_idx])
        sum_of_avgs += sub_avg
        translations = np.append(translations, coords, axis = 1)
        start_idx = stop_idx
    print("trans", translations)
    return translations, sum_of_avgs

def find_dims_post_alignment(orig_dim1, orig_dim2, translation_arr):
    max_trans = int(max(translation_arr, key = abs))
    if (np.sign(min(translation_arr)) * np.sign(max(translation_arr)) == -1):
        return max_trans, int(min(orig_dim1, orig_dim2) 
                              - (max(translation_arr) - min(translation_arr)))
    return max_trans, min(orig_dim1, orig_dim2) - abs(max_trans)

def calc_adj_translation(max_tr, curr_trans, direction):
    #direction - jesli sesja subsequent to -1
    #do sprawdzenia nawiasy cuda wianki
    return int(max(0, (max(0, max_tr * direction) + max_tr-curr_trans)))

def truncate_along_z(stack1, stack2, prev_stacks, minz):
    print("minz ", minz, "stack shape ", stack1.shape)
    z1 = stack1.shape[0]
    z2 = stack2.shape[0]
    res1 = stack1[max(0,minz) : min(z1, z2+minz)]
    res2 = stack2[max(0,-minz) : min(z1-minz, z2)]
    
    if prev_stacks is not None:
        for i in range(len(prev_stacks)):
            prev_stacks[i] = prev_stacks[i][max(0,minz) : min(z1, z2+minz)];
    return [res1, res2, prev_stacks]


def align_stacks(orig1, orig2, optimal_translations, minz, start_img_id, prev_stacks = None):
    z_truncated = truncate_along_z(orig1, orig2, prev_stacks, minz)

    print("start image id ", start_img_id)


    max_transx, difx = find_dims_post_alignment(z_truncated[0].shape[1], 
                                                z_truncated[1].shape[1], 
                                                optimal_translations[0])
    max_transy, dify = find_dims_post_alignment(z_truncated[0].shape[2], 
                                                z_truncated[1].shape[2], 
                                                optimal_translations[1])

    ret1 = np.empty((z_truncated[0].shape[0], difx, dify))
    ret2 = np.empty_like(ret1)

    prev_ret = []
    prev_stacks = z_truncated[-1]
    
    if prev_stacks is not None:
        for prev_stack in prev_stacks:
            prev_ret += [np.empty_like(ret1)]

        
    x0 = [calc_adj_translation(max_transx,x, -1) for x in optimal_translations[0]]
    y0 = [calc_adj_translation(max_transy,y, -1) for y in optimal_translations[1]]
    
    x0_b = max(0, max_transx)
    y0_b = max(0, max_transy)
    
    for idx, (r1, r2) in enumerate(zip(z_truncated[0],z_truncated[1])):
        tr_idx = idx//constants.ALIGNMENT_SUBSTACK_SIZE
        if tr_idx >= len(x0):
            tr_idx = len(x0) - 1
        ret1[idx] =  r1[x0_b:x0_b+difx,y0_b:y0_b+dify]
        if prev_stacks is not None:
            for i in range(len(prev_stacks)):
                prev_ret[i][idx] = prev_stacks[i][idx][x0_b:x0_b+difx,y0_b:y0_b+dify]
        ret2[idx] =  r2[x0[tr_idx]:x0[tr_idx]+difx,y0[tr_idx]:y0[tr_idx]+dify]

    return ret1, ret2, prev_ret

#%%
def set_filepaths(mouse, region, start_session_id, session_order):
    sn = session_order[:start_session_id+1]
    
    empty_pre = len(session_order)-1-start_session_id
    
    fn = []
    raw = []

    try:    
        for i in range(start_session_id-1):
            fn += [RESULT_PATH+alignment_filenames['thresh'].format(mouse, region,sn[i]) +constants.IMG_EXT]
            raw += [ICY_PATH+alignment_filenames['raw'].format(mouse, region,sn[i]) +constants.IMG_EXT]
        if start_session_id == 1:
            fn += [DIR_PATH+alignment_filenames['thresh'].format(mouse, region,sn[start_session_id-1]) +constants.IMG_EXT]
            raw += [DIR_PATH+alignment_filenames['raw'].format(mouse, region,sn[start_session_id-1]) +constants.IMG_EXT]  
        else:
            fn += [RESULT_PATH+alignment_filenames['thresh'].format(mouse, region,sn[start_session_id-1]) +constants.IMG_EXT]
            raw += [ICY_PATH+alignment_filenames['raw'].format(mouse, region, sn[start_session_id-1]) +constants.IMG_EXT]
        fn += [DIR_PATH+alignment_filenames['thresh'].format(mouse, region,sn[start_session_id]) +constants.IMG_EXT]
        raw += [DIR_PATH+alignment_filenames['raw'].format(mouse, region, sn[start_session_id]) +constants.IMG_EXT]
    except:
        raise Exception("Wrong filenames for: ", sn)

    return fn, raw

#%%
def read_images(name_list, convert=True, bounds=[0,0]):
    ret = []
    for fn in name_list:
        if fn is not None:
            new_img = io.imread(fn)
            new_img = new_img[bounds[0]:len(new_img)-bounds[1]]
            
            if convert:
                ret += [new_img.astype("uint8")]
            else:
                ret += [new_img]
    return ret

#%%
def prepare_for_tif_save(image):
    newimg = np.empty((image.shape[0], 1, image.shape[1], image.shape[2]))
    for idx, im in enumerate(image):
        newimg[idx] = np.array([im])
    return newimg.astype("uint8")
#%%

def save_results(ret1, ret2, optimal_translations, minz, start_img_id, mouse, region, 
                 res_dir,fname_template, file_suffix,session_order,  to_csv=True):
    
    ret1 = prepare_for_tif_save(ret1)
    ret2 = prepare_for_tif_save(ret2)
    findif = abs(ret1-ret2)
    
    sn = session_order[start_img_id-1:start_img_id+1]

    tifffile.imwrite(res_dir+fname_template.format(mouse, region,sn[0])
                     + constants.IMG_EXT,ret1,imagej=True, 
                     metadata={'unit': 'pixels','axes': 'ZCYX'})
    tifffile.imwrite(res_dir+fname_template.format(mouse, region,sn[1])
                     + constants.IMG_EXT, ret2,imagej=True, 
                     metadata={'unit': 'pixels','axes': 'ZCYX'})
    tifffile.imwrite(res_dir+"m"+str(mouse)+"r"+str(region)+"_"+file_suffix
                     + constants.IMG_EXT,findif,imagej=True, metadata={'unit': 'pixels','axes': 'ZCYX'})

    if to_csv:
        with open(res_dir+"m"+str(mouse)+"r"+str(region)+"_"+str(start_img_id)+file_suffix+ '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(optimal_translations[0])
            writer.writerow(optimal_translations[1])
            writer.writerow(str(minz))
            
#%%
def save_single_img(img_path, image):
    if image is not None:
        image = prepare_for_tif_save(image)
        tifffile.imwrite(img_path, image, imagej=True, metadata={'unit': 'pixels','axes': 'ZCYX'})

#%%

def align_sessions(mouse, region, start_session_id, file_suffix, session_order, bounds=[0,0]):
    thresh_names, raw_names = set_filepaths(
        mouse, region, start_session_id, session_order)
    print("align_session_params ", mouse, region, start_session_id, file_suffix)
    
    orig = read_images(thresh_names[-2:], bounds = bounds)
    raw = read_images(raw_names[-2:], convert = False, bounds = bounds)

    prev_imgs = None
    raw_prev_imgs = None
    #TODO wczytywac raz
    if start_session_id > 1:
        prev_imgs = read_images(thresh_names[:start_session_id-1])
        raw_prev_imgs = read_images(raw_names[:start_session_id-1])

    z1 = orig[0].shape[0]
    z2 = orig[1].shape[0]
    
    
    optimal_sum_of_avgs = 50000
    optimal_translations = []
    minz = 0
    for z in range(-constants.STACK_WINDOW, constants.STACK_WINDOW):
        tst1 = orig[0][max(0,z) : min(z1, z2+z)];
        tst2 = orig[1][max(0,-z) : min(z1-z, z2)];

        coords_list, sum_of_avgs = find_crop_coords(tst1, tst2)
        if sum_of_avgs < optimal_sum_of_avgs:
            optimal_sum_of_avgs = sum_of_avgs
            optimal_translations = coords_list
            minz = z
    aligned1, aligned2, aligned_prev = align_stacks(orig[0], orig[1], optimal_translations, 
                                               minz, start_session_id, prev_stacks = prev_imgs)
    save_results(aligned1, aligned2, optimal_translations, minz, 
                 start_session_id, mouse, region, RESULT_PATH, 
                 alignment_filenames['thresh'], file_suffix, session_order)
    
    for i, previous_img in enumerate(aligned_prev):
        save_single_img(RESULT_PATH
                        +alignment_filenames['thresh'].format(mouse, region,session_order[i])
                        +constants.IMG_EXT,
                        previous_img)
        
    aligned1_raw, aligned2_raw, pev_raw_aligned = align_stacks(raw[0], raw[1], 
                                                               optimal_translations, minz, 
                                                     start_session_id, prev_stacks = raw_prev_imgs)

    for  i, previous_img in enumerate(pev_raw_aligned):
        save_single_img(ICY_PATH
                        +alignment_filenames['raw'].format(mouse, region,session_order[i])
                        +constants.IMG_EXT,
                        previous_img)
    
    save_results(aligned1_raw, aligned2_raw, optimal_translations, minz, 
                 start_session_id, mouse, region, ICY_PATH,
                 alignment_filenames['raw'], file_suffix,session_order, to_csv = False)
    
#%%

def align_all_sessions(m,r, session_order, bounds=[0,0]):
    #print(m, r, session_order)
    for start_img_id in range(1, len(session_order)):
        print("starting ", start_img_id, session_order)
        suff = "_" + str(session_order[start_img_id-1]) + "_" + str(session_order[start_img_id])
        align_sessions(m, r, start_img_id, suff, session_order, bounds=bounds)
        print("-------------------------------------")
        
#%%
align_all_sessions(6,1,group_session_order, bounds=[0,0])
#%%

