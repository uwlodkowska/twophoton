from ij.gui import Overlay, OvalRoi, Plot
from ij import IJ, ImagePlus
from java.awt import Color
from ij.plugin.frame import RoiManager
import csv, math
from collections import OrderedDict


shift_key_root = 'm{}s{}{}'

red = Color(255, 0, 0)
green = Color(0, 255, 0)
white = Color(255, 255, 255)
yellow = Color(255, 255, 0)
blue = Color(0, 0, 255)

roi_diameter = [8,7,4]
roi_area = [12,37,52]
area_scale_factor = pow(roi_diameter[0]/2,2) + 2*pow(roi_diameter[1]/2,2) + 2*pow(roi_diameter[2]/2,2)

field_names_coords = ['X', 'Y', 'Z', 'mean']
fieldnames_outfile = ['Size X (px)', 'Size Y (px)', 'Size Z (px)', 'Center X (px)', 'Center Y (px)','Center Z (px)','Interior (px)',
'Sphericity','Yaw (°)','Pitch (°)','Roll (°)','Min Intensity (ch 0)','Mean Intensity (ch 0)','Max Intensity (ch 0)',
'Intensity center X (px) (ch 0)', 'Intensity center Y (px) (ch 0)','Intensity center Z (px) (ch 0)']

scale_coeff = {
    'xy': 1.2, 
    'z': 2
}

directory = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/alignment_result/aligned_despeckle/"


mouse = 2
sole_ctx_session = 1
rep_ctx_sessions = [2,3]
img_source_file = "m{}s{}{}.tif"
img_source_file = "m{}r{}_{}.tif"
img_source_file = "m{}r{}_ctx.tif"
thresholded_img = "m{}s{}{}_spots.tif"
watershed_img = "m{}s{}{}_watershed.tif"
source_file = "m{}r{}_ctx_output.txt"

bgr_source_file = "result_bg_m{}s{}{}.csv"
overlap_file = "overlap_m{}{}s{}.csv"
result_file = "cells_thresholded_m{}s{}{}.csv"
bgr_file = "result_bg_m{}{}.csv"

radius_microns = 5
min_dist = 1.2*radius_microns

tolerance = radius_microns

overlap_idxs = []

rm = RoiManager.getInstance()
if not rm:
  rm = RoiManager()
rm.reset()

#funkcje pomocnicze, zastepujace niedostepne funkcje z numpy/skimage
def dot(arr1,arr2):
    arr = [i[0] * i[1] for i in zip(arr1, arr2)]
    return arr

def div(arr1,arr2):
    arr = [i[0] / i[1] for i in zip(arr1, arr2)]
    return arr

def subtract(arr1,arr2):
    arr = [i[0] - i[1] for i in zip(arr1, arr2)]
    return arr

def listsum(arr1,arr2):
    arr = [i[0] + i[1] for i in zip(arr1, arr2)]
    return arr

def dist(centroid1, centroid2):
    return round(math.sqrt((scale_coeff['xy']*(centroid1[0]-centroid2[0]))**2
                   +(scale_coeff['xy']*(centroid1[1]-centroid2[1]))**2
                   +(scale_coeff['z']*(centroid1[2]-centroid2[2]))**2),2)

def power(arr, n):
    return [i**n for i in arr]

def argmax(arr):
    max_val = max(arr)
    for i, el in enumerate(arr):
        if el == max_val:
            return i
    return 0

def cumulative_sum(arr):
    ret = []
    sum_so_far = 0
    for el in arr:
        sum_so_far += el
        ret.append(sum_so_far)
    return ret

def histogram(arr):
    min_ = min(arr)
    max_ = max(arr)
    step = (max_-min_)/256
    bins_ = []
    hist = []
    bins_.append(min_)
    for i in range(256):
        bins_.append(min_+(i+1)*step)
        hist.append(0)
    
    bin_centers = listsum(bins_[:-1], bins_[1:])
    
    bin_centers[:] = [x / 2 for x in bin_centers]
    
    for el in arr:
        if el == max_:
            idx = -1
        else:
            idx = int((el-min_)//step)
        hist[idx] += 1
    return hist, bin_centers

def otsu_thre(arr):
    hist, bin_centers = histogram(arr)
    
    weight1 = cumulative_sum(hist)
    weight2 = cumulative_sum(hist[::-1])[::-1]

    mean1 = div(cumulative_sum(dot(hist, bin_centers)), weight1)
    mean2 = (div(cumulative_sum(dot(hist, bin_centers)[::-1]), weight2[::-1]))[::-1]

    variance12 = dot(dot(weight1[:-1], weight2[1:]), power(subtract(mean1[:-1], mean2[1:]), 2))

    idx = argmax(variance12)
    threshold = bin_centers[:-1][idx]
    
    return threshold

def draw_histogram_with_thre(title, array, threshold):
	plt = Plot("histogram", title, "liczba komorek")
	plt.addHistogram(array)
	plt.setColor(red)
	plt.setLineWidth(5)
	plt_limits = plt.getLimits()
	plt.drawDottedLine(threshold, plt_limits[-2], threshold, len(array), 5)
	plt.show()

def create_roi_group(x, y, z, vol, img, color, stdev_i, roundness,mean_i, idx):
	roi_group = []
	mean = 0
	stdev = 0
	for i in range(-2,3):
		if(z+i+1 > 0):# and stdev_i < 41 and mean_i > 28):
			diameter = roi_diameter[abs(i)]
			rad = diameter/2
			img.setSlice(z+i)
			roi = OvalRoi(x-rad, y-rad, diameter, diameter)
			roi.setStrokeWidth(1)	
			roi.setPosition(z+i+1)
			#roi.setName(str(idx))
			img.setRoi(roi)
			stats = roi.getStatistics()
			mean += stats.mean * roi_area[abs(i)]
			#* pow(rad,2)
			#if (stdev_i/mean_i > 1):
			#	roi.setStrokeColor(red)
			#if i == 0:
			#	rm.add(roi, idx)
			roi_group.append(roi)
	mean /= 150#area_scale_factor
	for roi in roi_group:
		if mean_i/mean > 1.5
			roi.setStrokeColor(red)
	return roi_group, mean, stdev

def estimate_bgr(mouse_no, session_no, region):
	img_path = directory + thresholded_img.format(mouse_no, session_no, region)
	img = ImagePlus(img_path)
	
	sum_of_means = 0
	bgr_file = directory + bgr_source_file.format(mouse_no, session_no, region)

	with open(bgr_file,"r") as source:
	    rdr = csv.DictReader( source )
	    for ctr in rdr:
	    	x = int(float(ctr['X']))
	    	y = int(float(ctr['Y']))
	    	z = int(float(ctr['Z']))
	    	mean = 0
	    	for i in range(-2,3):
	    		if(z+i > 0):
					diameter = roi_diameter[abs(i)]
					rad = diameter/2
					img.setSlice(z+i)
					roi = OvalRoi(x-rad, y-rad, diameter, diameter)
					roi.setStrokeColor(green)	
					roi.setPosition(z+i)
					img.setRoi(roi)
					stats = roi.getStatistics()
					mean += stats.mean * roi_area[abs(i)]
					if i == 0:
						stdev = stats.stdDev
			mean /= 150
			sum_of_means += mean
	return sum_of_means/1000

def prepare_roi_stats_dict(mouse_no, session_no, img, filename, color, region="", ov_idxs=None):
	cells_file = directory + filename.format(mouse_no, region)#, session_no)
	print(cells_file)
	roi_dict = {
		'rois' : [],
		'means' : [],
		'stdev' : [],
		'vols' : [],
	}
	with open(cells_file,"r") as source:
		rdr = csv.DictReader(source, fieldnames = fieldnames_outfile, dialect="excel-tab")
		idx = -2
		meandiff = 0
		for r in rdr:
			if idx >= 0 and float(r['Interior (px)'])>10:
				pos = int(float(r['Intensity center Z (px) (ch 0)']))
				vol = float(r['Interior (px)'])
				x = int(float(r['Intensity center X (px) (ch 0)']))
				y = int(float(r['Intensity center Y (px) (ch 0)']))
				stdev_icy = 0#float(r['Standard Deviation (ch 0)'])
				mean_icy = float(r['Mean Intensity (ch 0)'])
				roundness = float(r['Size Z (px)'])
				sum_intensity = float(r[fieldnames_outfile[4]])
				##print(x,y, pos)
				roi_group, mean, stdev = create_roi_group(x, y, pos, vol, img, color, stdev_icy, roundness,mean_icy, idx)
				if mean > 0:
					roi_dict['means'].append(mean)
					roi_dict['stdev'].append(mean_icy/mean)
					if mean_icy/mean > 1.5:
						meandiff += 1
					roi_dict['rois'].append(roi_group)
					roi_dict['vols'].append(vol)

			idx += 1
		print(meandiff)
	return roi_dict


def selection_by_thresholding(mouse_no, session_no, region="", color = white, bgr = 0):
	img_path = directory + img_source_file.format(mouse_no, region)#, session_no)#thresholded_img#watershed_img#
	#img_path = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/trans_tst/masked_despeckle.tif"
	result_file_path = directory + result_file.format(mouse_no, session_no, region)
	imp = ImagePlus(img_path)
	roi_dict = prepare_roi_stats_dict(mouse_no, session_no, imp, source_file, color, region)
	#bgr_dict = prepare_roi_stats_dict(mouse_no, session_no, imp, bgr_source_file, green, region)
	mean_threshold = otsu_thre(roi_dict['means'])/2

	cell_list = []
	
	overlay = Overlay()
	with open(result_file_path,"w") as result:
		wtr = csv.writer( result )
		wtr.writerow( field_names_coords )
		for idx, roi in enumerate(roi_dict['rois']):
			if(roi_dict['means'][idx] > 0):#mean_threshold):
				vals = []
				for c in roi[int(len(roi)/2)].getContourCentroid():
					vals.append(int(c))
				vals.append(roi[int(len(roi)/2)].getPosition())
				vals.append(roi_dict['means'][idx])
				cell_list.append(vals)
				wtr.writerow(vals)
				
				for roi_p in roi:
					overlay.add(roi_p)

		'''for idx, roi in enumerate(bgr_dict['rois']):
			for roi_p in roi:
				overlay.add(roi_p)'''
				
	
	imp.setOverlay(overlay)
	
	imp.show()
	rm.runCommand(imp, "Show All with labels")
	rm.runCommand("Associate true")
	draw_histogram_with_thre('srednia intensywnosc w obrebie roi', roi_dict['means'], mean_threshold)
	draw_histogram_with_thre('stdev w obrebie roi', roi_dict['stdev'], mean_threshold)
	draw_histogram_with_thre('roi volumes', roi_dict['vols'], 0)
	
	return cell_list

def find_overlap(cell_list1, cell_list2, shift):
    ov_arr = []
    ov_idx1_arr = []
    ov_idx2_arr = []
    incount = 0
    for idx1, cell1 in enumerate(cell_list1):
        for idx2, cell2 in enumerate(cell_list2):
        	distance = dist(cell1[:3], listsum(cell2[:3], shift))
        	if distance < tolerance:
        		if idx2 in ov_idx2_arr:
        			incount += 1
        		else:
	        		ov_arr.append(cell1[:3])
	        		ov_idx1_arr.append(idx1)
	        		ov_idx2_arr.append(idx2)
        		break
    
    return ov_arr, ov_idx1_arr, ov_idx2_arr

def save_overlap_indices(path, mouse, region_code, idx_arr, ov_code):
    name_parts = os.path.splitext(path)
    output = name_parts[0]+'_overlap' + ov_code + name_parts[1]
    output = output.format(mouse,region_code)
    save_result(output, idx_arr)
    
def save_result(result_file, cells, header=None):
    with open(result_file,"w") as result:
        wtr = csv.writer( result )
        if header != None:
            wtr.writerow( header )
        for cell in cells:
            wtr.writerow(cell)	

def save_dict(overlap_dict, first = False):
		write_code = 'a'
		
		if first:
			write_code = 'w'
			
		with open(directory+"summary.csv", write_code) as csvfile:
			writer = csv.DictWriter(csvfile, overlap_dict.keys())
			writer.writerow(overlap_dict)



selection_by_thresholding(10, "1", region="1", color = green, bgr = 0)