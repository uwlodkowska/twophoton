#%% imports

import sys
import yaml

# custom modules
import intersession as its
import cell_preprocessing as cp
import utils
import plotting



#%% config

config_file = sys.argv[1] if len(sys.argv) > 1 else "config_files/multisession.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)


SOURCE_DIR_PATH = config["experiment"]["dir_path"]
ICY_PATH = SOURCE_DIR_PATH + config["experiment"]["path_for_icy"]

DIR_PATH = config["experiment"]["full_path"]
BGR_DIR = config["experiment"]["background_dir"]
RESULT_PATH = config["experiment"]["result_path"]

regions = config["experiment"]["regions"]
group_session_order = config["experiment"]["session_order"][0]

optimized_fname = config["filenames"]["cell_data_opt_template"]
pooled_cells_fname = config["filenames"]["pooled_cells"]

#%% ready regions

regions = [[16,2], [8,1]]

#%% reading  and prepping detection results from icy

for mouse, region in regions:
    sessions = utils.read_single_session_cell_data(
        mouse, 
        region, 
        group_session_order, 
        config, 
        test=False, 
        optimized=False
        )
    
    imgs = utils.read_images(mouse, region, group_session_order, config)
    
    for i, s in enumerate(sessions):
        s = s.loc[s["Interior (px)"]>150].copy()
        # tolerance 2, because I don't expect icy to place the centroid outside 
        # of the cell, which is about 7 px in xy
        s = cp.optimize_centroids(s, imgs[i], suff="", tolerance = 2)
        s.to_csv(ICY_PATH + optimized_fname.format(
            mouse, 
            region, 
            group_session_order[i]))
        sessions[i] = s
#%% pooling
        
#!warning! here tolerance is not in pixels, but um
for mouse, region in regions[:1]:        
    df_reseg = its.pool_cells_globally(mouse, region, group_session_order, config, 7)
    for sid in group_session_order:
        img = utils.read_image(mouse, region, sid, config)
        df_reseg = cp.optimize_centroids(df_reseg, img, suff=f"_{sid}", tolerance=3, update_coords=False)
    df_reseg.to_csv(RESULT_PATH+pooled_cells_fname.format(mouse, region))
   
#%%
df = utils.read_pooled_with_background(1, 1, config)
plotting.session_detection_vs_background(df, group_session_order, sub_bgr = True)


#%%
df = utils.read_pooled_with_background(1, 1, config)
plotting.cell_detection_vs_background(df, group_session_order, sub_bgr=True)

#%%
regions = [[1,1], [14,1], [9,2],[8,1], [16,2]]
#%%
pairs = list(zip(group_session_order, group_session_order[1:]))
plotting.plot_cohort_tendencies(regions, pairs, config, groups=["on", "off", "const"])

#%%
classes = ["landmark_specific", "ctx_specific", "is_mixed", "test_specific"]
plotting.plot_class_distribution(regions, config, classes)
#%%
#%%
classes = ["is_transient", "is_intermediate", "is_persistent"]
plotting.plot_class_distribution(regions, config, classes)
#%%
for m, r in regions:
    df = utils.read_pooled_with_background(m,r, config)
    plotting.compare_session_intensities(df, group_session_order)
    #plotting.show_lmplots_comparison(df, group_session_order)
#%%
plotting.plot_upsetplot(regions, config, group_session_order[1:])
#%%
df = df_reseg.copy()
img = utils.read_image(16, 2, "s0", config)
df = cp.find_background(df, img, suff="_s0")

#%%
tst = utils.read_pooled_with_background(1, 1, config)
#%%
tst2 = utils.read_pooled_cells(1, 1, config)