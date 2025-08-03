#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 20:27:56 2023

@author: ula
"""
import numpy as np
import pandas as pd
import utils
import constants
import cell_preprocessing as cp
import visualization
import networkx as nx
from scipy.spatial import cKDTree

def identify_persistent_cells(mouse, region, sessions, session_ids):
    cols_to_save = constants.COORDS_3D + ["detected_in_sessions"]
    
    if sessions is None:
        print("no sessions provided, reading")
        sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    cell_count = 0
    for i, s in enumerate(sessions):
        #sprawdzić po co mi ten reset tutaj
        sessions[i] = s[cols_to_save].reset_index(drop=True)
        cell_count += s.shape[0]
    print("summed ", sessions[0].shape[0]+sessions[1].shape[0])
        
    
    coords_0 = sessions[0][constants.COORDS_3D].values
    coords_1 = sessions[1][constants.COORDS_3D].values
    
    # Build KDTree for fast spatial search
    tree_1 = cKDTree(coords_1)
    matches = tree_1.query_ball_point(coords_0, r=constants.TOLERANCE)
    
    # Prepare lists to build the result DataFrame
    rows_0, rows_1, distances = [], [], []
    for i, idxs in enumerate(matches):
        for j in idxs:
            rows_0.append(i)
            rows_1.append(j)
            # Compute the distance (for sorting/filtering later)
            dist = np.linalg.norm(coords_0[i] - coords_1[j])
            distances.append(dist)
    
    if not rows_0:
        print("No matching cells found within tolerance.")
        return pd.concat(sessions), 0.0

    # Build the result DataFrame
    cross_prod = pd.DataFrame({
        "index_s1": rows_0,
        "index_s2": rows_1,
        "distance": distances
    })
    
    cross_prod = cross_prod.sort_values(by="distance", ascending=True)
    cross_prod = cross_prod.drop_duplicates(subset=["index_s1"])
    cross_prod = cross_prod.drop_duplicates(subset=["index_s2"])
    
    
    for cname in constants.COORDS_3D:
        cross_prod[cname + "_s1"] = sessions[0].iloc[cross_prod["index_s1"]][cname].values
        cross_prod[cname + "_s2"] = sessions[1].iloc[cross_prod["index_s2"]][cname].values
        cross_prod[cname] = (cross_prod[cname + "_s1"] + cross_prod[cname + "_s2"]) / 2
        cross_prod = cross_prod.drop(columns=[cname + "_s1", cname + "_s2"])

    
    sessions[1] = sessions[1].drop(np.array(cross_prod.index_s2))


    sessions[0].loc[cross_prod["index_s1"], 'detected_in_sessions'] = \
    sessions[0].loc[cross_prod["index_s1"], 'detected_in_sessions'].apply(lambda s: s | {session_ids[-1]})


    summed = pd.concat(sessions)
    print("cprod ", cross_prod.shape[0])
    print("duplicates removed ", summed.shape[0])
    cross_prod = cross_prod.drop(columns = ["index_s1","index_s2", "distance"])
    count = cross_prod.shape[0]/(cell_count-cross_prod.shape[0])
    
    return summed, count 



def pooled_cells(mouse, region, session_ids, config, test=False):
    sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids, config, test=test, optimized=True)
    #to filtrowanie pewnie trzeba przenieśc gdzieś
    for i, session in enumerate(sessions):
#        session = shift_optimized(session, i)
        
        session = session.loc[session["Interior (px)"]>150].copy()
        session.loc[:,"detected_in_sessions"] = [{session_ids[i]} for _ in range(len(session))]
        sessions[i] = session
    
    persistent, count = identify_persistent_cells(mouse, region, sessions[:2], session_ids[:2])
    print("first ct ", count)
    for i in range(2, len(session_ids)):
        print("adding ", session_ids[i])
        persistent, count = identify_persistent_cells(mouse, region, [persistent,sessions[i]], session_ids[:i+1])
        print(f"{i}th ct ",count)
    return persistent



def identify_persistent_cells_w_thresholding(mouse, region, session_ids):
    """
    Parameters
    ----------
    mouse : int
        mouse id
    region : int
        region id
    session_ids : array of string
        at idx 0 - id of session from which we take coordinates and see if there
        are cells at these coordinates in an image from the session at idx 1
        

    Returns
    -------
    None.
    
    Takes coords identified in one session, projects them onto image from 
    anothee one and assumes that wherever the mean intensity is higher than
    threshold computed for coordinates and image from that other session, there
    is a cell there

    """
    quantile_thre = 0
    
    coord_dfs = utils.read_single_session_cell_data(
                                  mouse, region, session_ids)
    imgs = [utils.read_image(mouse, region, session_ids[0]), 
            utils.read_image(mouse, region, session_ids[1])]
    
    over_thre_count = np.array([0,0])
    thre=[0,0]
    for idx, df in enumerate(coord_dfs): 
        df['in_calculated'] = df.apply(cp.calculate_intensity, 
                                                 img = imgs[idx], axis = 1)
        
        thre[idx] = df['in_calculated'].quantile(quantile_thre)
        over_thre_count[idx] = df.loc[df.in_calculated > thre[idx]].shape[0]
    
    #threshold = coord_dfs[1]['in_calculated'].quantile(quantile_thre)
    
    coord_dfs[0]['in_calculated'] = coord_dfs[0].apply(cp.calculate_intensity, 
                                                 img = imgs[1], axis = 1)
    
    coord_dfs[0] = cp.optimize_centroids(coord_dfs[0], imgs[1])
    
    persistent_df = coord_dfs[0].loc[coord_dfs[0].int_optimized > thre[1]]
    print('summary ', persistent_df.shape[0], over_thre_count)
    overlap_size = persistent_df.shape[0]/(np.sum(over_thre_count)-persistent_df.shape[0])
    #visualization.visualize_with_centroids_custom(imgs[0], persistent_df[constants.COORDS_3D])
    return overlap_size, persistent_df

 
def distribution_change(df, from_column, to_column, step=0.2):
    number_of_classes = int(1/step)
    transfer_rate = np.zeros((number_of_classes,number_of_classes))
    df.loc[:,'dest'] = number_of_classes - 1
    for i in range(1, number_of_classes):
        idx = df[df[to_column]<df[to_column].quantile(1-i*step)].index
        df.loc[idx,'dest'] = number_of_classes - i - 1
    
    for i in range(number_of_classes):
        lower_lim = df[from_column].quantile(i*step)
        upper_lim = df[from_column].quantile((i+1)*step)
        df_top_from = df[(df[from_column]>lower_lim) & (df[from_column]<upper_lim)]
        transfer_rate[i] = np.array([df_top_from[df_top_from['dest'] == i].shape[0] for i in range(number_of_classes)])
    transfer_rate = transfer_rate/df.shape[0]
    print(transfer_rate)
    return transfer_rate

def distribution_change_precalculated(df, from_column, to_column, number_of_classes):
    transfer_rate = np.zeros((number_of_classes,number_of_classes))
    for i in range(number_of_classes):
        for j in range(number_of_classes):
            transfer_rate[i,j] = df.loc[((df[from_column] == i)&(df[to_column] == j))].shape[0]
    transfer_rate = transfer_rate/df.shape[0]
    print(transfer_rate)
    return transfer_rate

def assign_type(row):
    if row.int_optimized1 > row.int_optimized0:
        if row.int_optimized2 > row.int_optimized0:
            return'A'
        else:
            return 'B'
    else:
        if row.int_optimized0 < row.int_optimized2:
            return 'C'
        else:
            return 'D'
        
def top_cells_intensity_change(mouse, region, sessions, percentage):
    top_df = pd.DataFrame(columns=constants.COORDS_3D)
    for s in sessions:
        df = cp.get_brightest_cells(mouse, region, s, percentage)
        top_df =pd.concat([top_df,df], ignore_index=True)
    for i, s in enumerate(sessions):
        img = utils.read_image(mouse, region, s)
        cp.optimize_centroids(top_df, img, str(i))

    top_df['type'] = top_df.apply(assign_type, axis = 1)
    type_frac = [ top_df[top_df.type== i].shape[0]/top_df.shape[0]  for i in ['A', 'B', 'C', 'D'] ]

    return type_frac

def distribution_change_all_sessions(mouse, region, sessions):
    imgs = []
    for s in sessions:
        imgs += [utils.read_image(mouse, region, s)]
    dfs = utils.read_single_session_cell_data(mouse, region, sessions)
    
    for i in range(len(sessions)-1):
        df = dfs[i]
        df['in_calculated'] = df.apply(cp.calculate_intensity,img = imgs[i], axis = 1)
        df = df[df[constants.ICY_COLNAMES["mean_intensity"]]<1.5*df['in_calculated']]
        df['in_calculated_n'] = df.apply(cp.calculate_intensity,img = imgs[i+1], axis = 1)
        tr = distribution_change(df, 'in_calculated', 'in_calculated_n')
        print(tr)
        
        
def find_intersession_tendencies_raw(df,sessions=[1,2,3], colname='int_optimized'):
    tendencies = []
    for i in range(len(sessions)-1):
        df[colname+str(sessions[i])+'_low'] = df[colname+str(sessions[i])]*0.8
        df[colname+str(sessions[i])+'_high'] = df[colname+str(sessions[i])]*1.2
        
        condition_down = df[colname+str(sessions[i+1])] < df[colname+str(sessions[i])+'_low']
        condition_up = df[colname+str(sessions[i+1])] > df[colname+str(sessions[i])+'_high']
        down = df.loc[condition_down]
        stable = df.loc[ (~condition_down) & (~condition_up)]
        up = df.loc[condition_up]
        print("up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies

def find_intersession_tendencies_bgr(df,bgr,k=1, sessions=[1,2,3], colname='int_optimized'):
    tendencies = []
    
    print("shape ", df.shape[0])
    for i in range(len(sessions)):
        df[colname+str(sessions[i])] = df[colname+str(sessions[i])]-bgr[i,0]
    for i in range(len(sessions)-1):
        df[colname+str(sessions[i])+'_low'] = df[colname+str(sessions[i])] - k*(bgr[i,1]+bgr[i+1,1])
        df[colname+str(sessions[i])+'_high'] = df[colname+str(sessions[i])] + k*(bgr[i,1]+bgr[i+1,1])
        
        condition_down = df[colname+str(sessions[i+1])] < df[colname+str(sessions[i])+'_low']
        condition_up = df[colname+str(sessions[i+1])] > df[colname+str(sessions[i])+'_high']
        down = df.loc[condition_down]
        stable = df.loc[ (~condition_down) & (~condition_up)]
        up = df.loc[condition_up]
        print("up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies      
            
def find_intersession_tendencies_on_off(df, sessions=[1,2,3], colname='active'):
    tendencies = []
    for i in range(len(sessions)-1):
        condition_down = (df[colname+str(sessions[i])]) & (~df[colname+str(sessions[i+1])])
        condition_stable = (df[colname+str(sessions[i+1])]) & (df[colname+str(sessions[i])])
        condition_up = (~df[colname+str(sessions[i])]) & (df[colname+str(sessions[i+1])])
        down = df.loc[condition_down]
        stable = df.loc[condition_stable]
        up = df.loc[condition_up]
        print("on off up: ", up.shape[0],"down: ", down.shape[0],"stable: ", stable.shape[0])
        tendencies += [up.shape[0], down.shape[0], stable.shape[0]]
    return tendencies
            
def cell_classes(df, sessions=[1,2,3], colname='active'):
    
    any_session = len(df.loc[(df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])])
                    | (df[colname+str(sessions[2])])])
    class0 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class1 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class2 = len(df.loc[(~df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class3 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class4 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])
    class5 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    class6 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])
    ret = np.array([class0,class1,class2,class3,class4,class5,class6])/any_session
    #print(ret)
    return ret

def cell_classes_diff_norm_(df, sessions=[1,2,3], colname='active'):
    
    any_session = len(df.loc[(df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])])
                    | (df[colname+str(sessions[2])])])
    class0 = len(df.loc[(df[colname+str(sessions[0])])])/any_session
    class1 = len(df.loc[(df[colname+str(sessions[1])])])/any_session
    class2 = len(df.loc[(df[colname+str(sessions[2])])])/any_session
    
    class3 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/any_session
    
    either_session = len(df.loc[((df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])]))
                    & (~df[colname+str(sessions[2])])])
    class4 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])/either_session
    either_session = len(df.loc[((df[colname+str(sessions[0])]) | (df[colname+str(sessions[2])]))
                    & (~df[colname+str(sessions[1])])])
    class5 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/either_session
    either_session = len(df.loc[((df[colname+str(sessions[1])]) | (df[colname+str(sessions[2])]))
                    & (~df[colname+str(sessions[0])])])
    class6 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/either_session
    ret = np.array([class0,class1,class2,class3,class4,class5,class6])
    #print(ret)
    return ret


def cell_classes_diff_norm(df, sessions=[1,2,3], colname='active'):
    
    any_session = len(df.loc[(df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])])
                    | (df[colname+str(sessions[2])])])
    class0 = len(df.loc[(df[colname+str(sessions[0])]) & ~(df[colname+str(sessions[1])]
                    & df[colname+str(sessions[2])])])/any_session
    class1 = len(df.loc[(df[colname+str(sessions[1])]) & ~(df[colname+str(sessions[0])]
                    & df[colname+str(sessions[2])])])/any_session
    
    class2 = len(df.loc[(df[colname+str(sessions[2])]) & ~(df[colname+str(sessions[0])]
                    & df[colname+str(sessions[1])])])/any_session
    
    
    
    class3 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/any_session
    
    either_session = len(df.loc[((df[colname+str(sessions[0])]) | (df[colname+str(sessions[1])]))
                    & (~df[colname+str(sessions[2])])])
    class4 = len(df.loc[(df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (~df[colname+str(sessions[2])])])/either_session
    either_session = len(df.loc[((df[colname+str(sessions[0])]) | (df[colname+str(sessions[2])]))
                    & (~df[colname+str(sessions[1])])])
    class5 = len(df.loc[(df[colname+str(sessions[0])]) & (~df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/either_session
    either_session = len(df.loc[((df[colname+str(sessions[1])]) | (df[colname+str(sessions[2])]))
                    & (~df[colname+str(sessions[0])])])
    class6 = len(df.loc[(~df[colname+str(sessions[0])]) & (df[colname+str(sessions[1])])
                    & (df[colname+str(sessions[2])])])/either_session
    ret = np.array([class0,class1,class2,class3,class4,class5,class6])
    print(ret)
    return ret


from scipy.ndimage import gaussian_filter

def recluster_component(component_df, coords_cols, session_col="detected_in_sessions",
                        tolerance_um=5.0, voxel_size=np.array([1.18, 1.18, 2.0])):
    """
    Recluster a suspicious component using lower distance threshold.
    Returns a list of DataFrames, one per subcomponent.
    """
    coords = component_df[coords_cols].values
    scaled_coords = coords * voxel_size

    tree = cKDTree(scaled_coords)
    neighbors = tree.query_ball_tree(tree, r=tolerance_um)

    G = nx.Graph()
    G.add_nodes_from(component_df.index)

    for i, nlist in enumerate(neighbors):
        for j in nlist:
            if i >= j:
                continue
            if list(component_df.iloc[i][session_col])[0] != list(component_df.iloc[j][session_col])[0]:
                G.add_edge(component_df.index[i], component_df.index[j])

    subcomponents = []
    for nodes in nx.connected_components(G):
        sub_df = component_df.loc[list(nodes)].copy()
        subcomponents.append(sub_df)

    return subcomponents


def has_intensity_dip(region1, region2, image, threshold_ratio=0.7):
    """
    Check if there's a noticeable intensity dip between two centroids in an image.
    Returns True if dip is detected (i.e., they likely belong to separate peaks).
    """
    from skimage.draw import line
    c1 = np.round(region1).astype(int)
    c2 = np.round(region2).astype(int)
    rr, cc = line(c1[1], c1[0], c2[1], c2[0])
    intensities = image[rr, cc]
    min_i = intensities.min()
    peak_i = max(image[c1[1], c1[0]], image[c2[1], c2[0]])
    return min_i < threshold_ratio * peak_i


def recluster_with_intensity_check_multi_stack(
    component_df,
    coords_cols,
    session_col,
    image_dict,
    tolerance_um=5.0,
    voxel_size=np.array([1.18, 1.18, 2.0]),
    threshold_ratio=0.7
):
    #TODO fix this mess
    coords = component_df[coords_cols].values
    scaled_coords = coords * voxel_size
    tree = cKDTree(scaled_coords)
    neighbors = tree.query_ball_tree(tree, r=tolerance_um)

    G = nx.Graph()
    G.add_nodes_from(component_df.index)

    for i, nlist in enumerate(neighbors):
        for j in nlist:
            if i >= j:
                continue

            row_i = component_df.iloc[i]
            row_j = component_df.iloc[j]

            sess_i = list(row_i[session_col])[0]
            sess_j = list(row_j[session_col])[0]

            z_i = int(round(row_i[coords_cols[2]]))
            z_j = int(round(row_j[coords_cols[2]]))

            if sess_i not in image_dict or sess_j not in image_dict:
                continue
            if z_i >= image_dict[sess_i].shape[0] or z_j >= image_dict[sess_j].shape[0]:
                continue

            img_i = gaussian_filter(image_dict[sess_i][z_i], sigma=1.0)

            c1 = row_i[coords_cols[:2]].values
            c2 = row_j[coords_cols[:2]].values

            if sess_i == sess_j:
                # SAME session: only connect if there is NO intensity dip
                if not has_intensity_dip(c1, c2, img_i, threshold_ratio=threshold_ratio):
                    G.add_edge(component_df.index[i], component_df.index[j])
            else:
                # DIFFERENT session: connect directly (or optionally check dip here too)
                G.add_edge(component_df.index[i], component_df.index[j])

    subcomponents = []
    for nodes in nx.connected_components(G):
        sub_df = component_df.loc[list(nodes)].copy()
        subcomponents.append(sub_df)

    return subcomponents




def update_persistent_cells_with_reclustering(
    suspicious_components,
    coords_cols,
    session_col,
    image_dict=None,
    recluster_method="distance",
    tolerance_um=4,
    voxel_size=np.array([1.18, 1.18, 2.0]),
    threshold_ratio=0.7,
):
    """
    Recluster suspicious components and return new persistent cells.
    Supports both geometric distance-based and intensity-dip-based reclustering.
    """
    all_reclustered = []

    for component_df in suspicious_components:
        if recluster_method == "distance":
            subcomponents = recluster_component(
                component_df,
                coords_cols=coords_cols,
                session_col=session_col,
                tolerance_um=tolerance_um,
                voxel_size=voxel_size
            )

        elif recluster_method == "intensity":
            if image_dict is None:
                raise ValueError("image_stacks must be provided for intensity-based reclustering.")
            subcomponents = recluster_with_intensity_check_multi_stack(
                component_df,
                coords_cols=coords_cols,
                session_col=session_col,
                image_dict=image_dict,
                tolerance_um=tolerance_um,
                voxel_size=voxel_size,
                threshold_ratio=threshold_ratio
            )

        else:
            raise ValueError("Unknown recluster_method. Use 'distance' or 'intensity'.")

        for sub_df in subcomponents:
            mean_coords = sub_df[coords_cols].mean()
            session_set = set(sub_df['detected_in_sessions'].explode())
            all_reclustered.append({
                **{c: mean_coords[c] for c in coords_cols},
                "detected_in_sessions": session_set,
                "n_sessions": len(session_set),
                "close_neighbour" : True,
            })

    return all_reclustered


def pool_cells_globally(mouse, region, session_ids, config, tolerance=6):
    all_cells = []
    cell_to_session = []

    sessions = utils.read_single_session_cell_data(
                                  mouse, region, session_ids, config, test=False, optimized=True)

    # Assign global IDs to all cells
    global_idx = 0
    for sess_idx, df in enumerate(sessions):
        df = df.copy()
        df = df.loc[df["Interior (px)"]>150].copy()
        print("df shape ", df.shape[0])
        df.loc[:,"detected_in_sessions"] =[{session_ids[sess_idx]}] * len(df)
        df['global_id'] = np.arange(global_idx, global_idx + len(df))
        global_idx += len(df)
        all_cells.append(df)
        cell_to_session.extend([session_ids[sess_idx]] * len(df))

    all_cells_df = pd.concat(all_cells, ignore_index=True)
    
    coords = all_cells_df[constants.COORDS_3D_CENTER].values
    scaled_coords = coords * np.array([1.18, 1.18, 2.0])

    # Build KDTree and find all neighbors within tolerance
    tree = cKDTree(scaled_coords)
    neighbors = tree.query_ball_tree(tree, r=tolerance)

    # Build graph: connect only cells from different sessions
    G = nx.Graph()
    G.add_nodes_from(all_cells_df['global_id'])

    for i, neighbors_i in enumerate(neighbors):
        for j in neighbors_i:
            if i >= j:
                continue  # avoid duplicates and self-loops
            if cell_to_session[i] != cell_to_session[j]:
                G.add_edge(all_cells_df.at[i, 'global_id'], all_cells_df.at[j, 'global_id'])

    # Find connected components = persistent cells
    persistent_cells = []
    
    normal_components = []
    suspicious_components = []
    
    for component in nx.connected_components(G):
        component_df = all_cells_df[all_cells_df['global_id'].isin(component)].copy()
        session_counts = component_df['detected_in_sessions'].explode().value_counts()
        
        if any(session_counts > 1):
            suspicious_components.append(component_df)
            
            #normal_components.append(component_df)
        else:
            normal_components.append(component_df)
    
    
    for component_df in normal_components:
        #component_df = all_cells_df[all_cells_df['global_id'].isin(component)]

        session_counts = component_df['detected_in_sessions'].explode().value_counts()

        mean_coords = component_df[constants.COORDS_3D_CENTER].mean()
        session_set = set(component_df['detected_in_sessions'].explode())
        persistent_cells.append({
            **{c: mean_coords[c] for c in constants.COORDS_3D_CENTER},
            "detected_in_sessions": session_set,
            "n_sessions": len(session_set),
            "close_neighbour" : False,
        })

    # Add unmatched cells (not connected to any other)
    matched_ids = set.union(*[set(c) for c in nx.connected_components(G)])
    unmatched = all_cells_df[~all_cells_df['global_id'].isin(matched_ids)]
    for _, row in unmatched.iterrows():
        persistent_cells.append({
            **{c: mean_coords[c] for c in constants.COORDS_3D_CENTER},
            "detected_in_sessions": row["detected_in_sessions"],
            "n_sessions": len(row["detected_in_sessions"]),
            "close_neighbour" : False,
        })
    print(f'number of sus comps {len(suspicious_components)}')
    reclustered = update_persistent_cells_with_reclustering(
        suspicious_components,
        coords_cols=constants.COORDS_3D_CENTER,
        session_col="detected_in_sessions",
        image_dict=None,
        recluster_method="distance", 
        tolerance_um=5,
    )
    persistent_cells.extend(reclustered)

    return pd.DataFrame(persistent_cells)

