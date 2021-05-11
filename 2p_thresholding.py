
# coding: utf-8

# In[1]:


from PIL import Image
from skimage import io
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')
plt.ioff()
import tifffile, csv, numpy as np
from skimage.morphology import closing, opening, square, disk, ball
from skimage.segmentation import watershed
from skimage.filters import threshold_otsu, sobel, threshold_local
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import mahotas as mh
import cv2, skimage
import os


# In[2]:


directory = "/media/ula/D/2p_fosgfp2020/"
path_comm = "m{}{}_tst_sa.tif"
out_name = "result_m{}{}.tif"
result_file = "result_label_m{}{}.csv"


field_names_coords = ['X', 'Y', 'Z']
scale_coeff = {
    'xy': 1.2, 
    'z': 2
}
radius_microns = 5
min_dist = 1.2*radius_microns


shift = {
    'm1s21' : [5,6,0],
    'm1s31' : [5,2,0],
    'm2_r1s31' : [1,11,0],
    'm2_r1s32' : [0,8,0],
    'm2_r2s31' : [2,3,0],
    'm2_r2s32' : [-2,4,0],
    'm2_r1s64' : [3,3,0],
    'm2_r1s65' : [-2,4,0],
    'm2_r2s64' : [-1,-11,0],
    'm2_r2s65' : [4,-16,0],
    'm3_r1s31' : [3,3,0],
    'm3_r1s32' : [1,3,0],
    'm12_r1s21' : [0,7,0],
    'm12_r1s31' : [32,-3,0],
    'm12_r2s32' : [4,3,0],
    'm12_r2s31' : [1,-4,0],
    'm9_r1s21' : [-4, -1, 0],
    'm9_r1s31' : [-9, 3, 0],
    'm9_r3s21' : [0, 2, 0],
    'm9_r3s31' : [0, 5, 0],
}

tolerance = radius_microns


# In[3]:


def dist(centroid1, centroid2):
    return np.sqrt((scale_coeff['xy']*(centroid1[0]-centroid2[0]))**2
                   +(scale_coeff['xy']*(centroid1[1]-centroid2[1]))**2
                   +(scale_coeff['z']*(centroid1[2]-centroid2[2]))**2)


# In[4]:


def calculate_markers(stack_slice):
    elev_map_ = np.array(sobel(stack_slice))
    
    thre_val = threshold_otsu(stack_slice)
    thre_val = 1.5*threshold_local(stack_slice, 75, offset=0)
    stack_slice[stack_slice > thre_val] = 255
    stack_slice[stack_slice <= thre_val] = 0
    
    thresh = stack_slice
    kernel = disk(1)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)
    sure_bg = cv2.dilate(opening,kernel,iterations=1)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    ret, markers_ = cv2.connectedComponents(sure_fg)
    markers_ = markers_+1
    markers_[unknown==255] = 0
    return markers_, opening, elev_map_


# In[5]:


def prepare_img_for_watershed(image_ts, out_name):
    elev_map = np.empty_like(image_ts)
    markers = np.empty_like(image_ts)

    for idx, im in enumerate(image_ts):
        markers_, opening, elev_map_ = calculate_markers(im)
        markers[idx] = markers_
        image_ts[idx] = opening
        elev_map[idx] = elev_map_

    with tifffile.TiffWriter(out_name) as tiff:   
        for im in image_ts:
            tiff.save(im)
    return elev_map, markers


# In[6]:


def watershed_label_img(elev_map, markers, image_ts):
    

    labels = label(watershed(-elev_map, markers, connectivity=2))
    #labels = label(image_ts)

    return labels



# In[7]:


def identify_cells(label_image):
    cells = []
    print(len(regionprops(label_image)))
    with open(result_file,"w") as result:
        wtr = csv.writer( result )
        wtr.writerow( field_names_coords )
        for region in regionprops(label_image):
            if region.area > 20:
                coords = np.flip(region.centroid).astype(int)+np.array([1,1,1])
                cells.append(coords)
                wtr.writerow(coords)
                
    unique_cells = []

    for idx_t, c1 in enumerate(cells):
        if idx_t% 10000 == 0:
            print(idx_t)
        idx = 0
        for c2 in unique_cells:
            if dist(c1, c2) < min_dist:
                break;
            idx += 1
        if idx == len(unique_cells):
            unique_cells.append(c1)
        else:
            unique_cells[idx] = ((c1+c2)/2).astype(int)
              
    return unique_cells


# In[8]:


def save_result(result_file, cells, header=None):
    with open(result_file,"w") as result:
        wtr = csv.writer( result )
        if header != None:
            wtr.writerow( header )
        for cell in cells:
            wtr.writerow(cell)


# In[9]:


def find_cells(mouse, scan_code):
    source_path = directory + path_comm.format(mouse, scan_code)
    result_img_path = directory + out_name.format(mouse, scan_code)
    result_csv_path = directory + result_file.format(mouse, scan_code)
    
    image_ts = io.imread(source_path)
    elev_map, markers = prepare_img_for_watershed(image_ts, result_img_path)
    print('?')
    label_image = watershed_label_img(elev_map, markers, image_ts)
    print(len(label_image))
    cells = identify_cells(label_image)
    print(mouse, scan_code, len(cells))
    save_result(result_csv_path, cells, field_names_coords)


# In[10]:


find_cells('2', 's4_r2')
find_cells('2', 's5_r2')
find_cells('2', 's6_r2')


# In[11]:


def read_coords(filename):
    ret = []
    with open(filename,"r") as source:
        rdr = csv.reader(source)
        next(rdr)
        for row in rdr:
            ret.append(row)
    return np.array(ret).astype(int)

def find_overlap(cell_list1, cell_list2, shift):
    idx_arr = []
    for idx, cell1 in enumerate(cell_list1):
        for idx2, cell2 in enumerate(cell_list2):
            distance = dist(cell1, cell2+shift)
            if distance < tolerance:
                idx_arr.append([idx, idx2])
                break
    return idx_arr

def save_overlap_indices(path, mouse, region_code, idx_arr, ov_code):
    name_parts = os.path.splitext(path)
    output = name_parts[0]+'_overlap' + ov_code + name_parts[1]
    output = output.format(mouse,region_code)
    save_result(output, idx_arr)


def find_3session_overlap(mouse, region=None):
    region_code = ""
    if region != None:
        region_code = "_"+region
    path = directory + result_file
    
    coords_arr1 = read_coords(path.format(mouse, 's4'+region_code))
    coords_arr2 = read_coords(path.format(mouse, 's5'+region_code))
    coords_arr3 = read_coords(path.format(mouse, 's6'+region_code))
    
    idx_arr1 = find_overlap(coords_arr1, coords_arr3, shift['m'+mouse+region_code+'s64'])
    idx_arr2 = find_overlap(coords_arr2, coords_arr3, shift['m'+mouse+region_code+'s65'])
    
    save_overlap_indices(path, mouse, region_code, idx_arr1, '64')
    save_overlap_indices(path, mouse, region_code, idx_arr2, '65')
    
    
    print(len(idx_arr1), len(idx_arr2))


# In[12]:


find_3session_overlap('2', 'r2')


# In[13]:


a=[6,4,5,2]
b=[1,0]

[a[i] for i in b]


# In[14]:


for idx, x in enumerate(a):
    print (idx)

