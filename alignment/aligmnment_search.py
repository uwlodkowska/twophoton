# -*- coding: utf-8 -*-

import numpy as np

SEARCH_WINDOW = 15
SUBSTACK_SIZE = 5

def calculate_cropped_diff(substack1, substack2, x, y):
    xdif = min(substack1.shape[1], substack2.shape[1])-abs(x);
    ydif = min(substack1.shape[2], substack2.shape[2])-abs(y);
    sub1_tmp = substack1[:,max(0,x):max(0,x)+xdif,max(0,y):max(0,y)+ydif]
    sub2_tmp = substack2[:,max(0,-x):max(0,-x)+xdif,max(0,-y):max(0,-y)+ydif]
    dif_tmp = abs(sub1_tmp - sub2_tmp)
    return dif_tmp

def find_optimal_crop(substack1, substack2):
    minv = 5000
    minx = 0
    miny = 0
    for x in range(-SEARCH_WINDOW, SEARCH_WINDOW):
        for y in range(-SEARCH_WINDOW, SEARCH_WINDOW):
            dif_cropped = calculate_cropped_diff(substack1, substack2, x, y)
            tmp_avg = np.average(dif_cropped)
            if tmp_avg < minv:
                minv = tmp_avg
                minx = x
                miny = y
    return np.array([[minx],[miny]]), minv

def find_crop_coords(stack1, stack2):
    translations = np.array([[],[]])
    stack_size = stack1.shape[0]
    sum_of_avgs = 0
    start_idx = 0
    
    while start_idx + SUBSTACK_SIZE < stack_size:
        stop_idx = start_idx + SUBSTACK_SIZE
        if stop_idx + SUBSTACK_SIZE > stack_size:
            stop_idx = stack_size+1
        coords, sub_avg = find_optimal_crop(stack1[start_idx:stop_idx], 
                                            stack2[start_idx:stop_idx])
        sum_of_avgs += sub_avg
        translations = np.append(translations, coords, axis = 1)
        start_idx = stop_idx
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

def truncate_along_z(stack1, stack2, minz):
    z1 = stack1.shape[0]
    z2 = stack2.shape[0]
    res1 = stack1[max(0,minz) : min(z1, z2+minz)];
    res2 = stack2[max(0,-minz) : min(z1-minz, z2)];
    return [res1, res2]

def align_stacks(orig1, orig2, optimal_translations, minz, start_img_id, legacy_stack = None):
    z_truncated = truncate_along_z(orig1, orig2, minz)

    print("optimal trans ", optimal_translations)
    print("minz ", minz)


    max_transx, difx = find_dims_post_alignment(res1.shape[1], res2.shape[1], optimal_translations[0])
    max_transy, dify = find_dims_post_alignment(res1.shape[2], res2.shape[2], optimal_translations[1])

    ret1 = np.empty((res1.shape[0], difx, dify))
    ret2 = np.empty_like(ret1)
    
    legacy_ret = None
    if legacy_stack is not None:
        legacy_stack = legacy_stack[max(0,minz) : min(z1, z2+minz)];
        legacy_ret = np.empty_like(ret1)
        
    x0 = [calc_adj_translation(max_transx,x, -1) for x in optimal_translations[0]]
    y0 = [calc_adj_translation(max_transy,y, -1) for y in optimal_translations[1]]
    
    x0_b = max(0, max_transx)
    y0_b = max(0, max_transy)
    for idx, (r1, r2) in enumerate(zip(res1,res2)):
        tr_idx = idx//substack_size
        if tr_idx >= len(x0):
            tr_idx = len(x0) - 1
        ret1[idx] =  r1[x0_b:x0_b+difx,y0_b:y0_b+dify]
        if legacy_stack is not None:
            legacy_ret[idx] = legacy_stack[idx][x0_b:x0_b+difx,y0_b:y0_b+dify]
        ret2[idx] =  r2[x0[tr_idx]:x0[tr_idx]+difx,y0[tr_idx]:y0[tr_idx]+dify]
    return ret1, ret2, legacy_ret










def align_sessions(mouse, region, start_session_id, file_suffix, session_order):
    sn = session_order[start_session_id-1:start_session_id+1]
    fn0 = None
    legacy = None
    raw_legacy = None
    
    if starting_session_id == 1:
        fn1 = dir_path+filename_template.format(mouse, region,sn[0]) +ext
        raw1 = dir_path+filename_raw_temp.format(mouse, region,sn[0]) +ext
    elif starting_session_id == 2:
        raw_legacy = path_for_icy+filename_raw_temp.format(mouse, region,session_order[0]) + ext
        fn0 = res_dir_path+filename_template.format(mouse, region,session_order[0]) +ext
        fn1 = res_dir_path+filename_template.format(mouse, region,sn[0]) +ext
        raw1 = path_for_icy+filename_raw_temp.format(mouse, region,sn[0]) +ext
    
    fn2 = dir_path+filename_template.format(mouse, region,sn[1]) +ext
    raw2 = dir_path+filename_raw_temp.format(mouse, region,sn[1]) +ext
    print(fn1)
    if path.exists(fn1) and path.exists(fn2):
        orig1 = io.imread(fn1).astype("uint8")
        orig2 = io.imread(fn2).astype("uint8")
        raw1 = io.imread(raw1)
        raw2 = io.imread(raw2)
        
        
        if starting_session_id == 2:
            legacy = io.imread(fn0).astype("uint8") 
            raw_legacy = io.imread(raw_legacy).astype("uint8") 
            
        
        z1 = orig1.shape[0]
        z2 = orig2.shape[0]
        optimal_sum_of_avgs = 50000
        optimal_translations = []
        minz = 0
        for z in range(-stack_window, stack_window):
            tst1 = orig1[max(0,z) : min(z1, z2+z)];
            tst2 = orig2[max(0,-z) : min(z1-z, z2)];

            coords_list, sum_of_avgs = find_stack_translations(tst1, tst2)
            if sum_of_avgs < optimal_sum_of_avgs:
                optimal_sum_of_avgs = sum_of_avgs
                optimal_translations = coords_list
                minz = z
        aligned1, aligned2, legacy_ = align_stacks(orig1, orig2, optimal_translations, 
                                                   minz, starting_session_id, legacy_stack = legacy)
        save_results(aligned1, aligned2, optimal_translations, minz, 
                     starting_session_id, mouse, region, res_dir_path, 
                     filename_template, file_suffix, session_order)
        
        save_single_img(res_dir_path+filename_template.format(mouse, region,session_order[0])+ext,
                        legacy_)
        


        aligned1_raw, aligned2_raw, raw_legacy_ = align_stacks(raw1, raw2,optimal_translations, minz, 
                                                     starting_session_id, legacy_stack = raw_legacy)

        save_single_img(path_for_icy+filename_raw_temp.format(mouse, region,session_order[0])+ext,
                        raw_legacy_)
        save_results(aligned1_raw, aligned2_raw, optimal_translations, minz, starting_session_id,
                     mouse, region, path_for_icy,filename_raw_temp, file_suffix,session_order, 
                     to_csv = False)