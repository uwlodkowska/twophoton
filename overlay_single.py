from ij.gui import Overlay, OvalRoi, Plot
from ij import IJ, ImagePlus
from java.awt import Color
from ij.plugin.frame import RoiManager
import csv, math
from collections import OrderedDict

intersession_codes = {
	'1' : {
		'21' : 'b1b2',
		'32' : 'a1b2',
		'32' : 'ab_consec',
	},
	'other' : {
		'21' : 'ab_consec',
		'31' : 'a1b2',
		'32' : 'b1b2',
		'54' : 'ab_consec',
		'64' : 'a1b2',
		'65' : 'b1b2',
	},
}

shift = {
    'm3_r1s31' : [19,-9,0],
    'm3_r1s32' : [12,4,0],
    'm3_r1s21' : [7,-13,0],
    'm4_r1s31' : [-8, 61, 0],
    'm4_r1s32' : [-8, 56, 0],
    'm4_r1s21' : [7,5,0],
}

shift_key_root = 'm{}s{}{}'

red = Color(255, 0, 0)
green = Color(0, 255, 0)
white = Color(255, 255, 255)
yellow = Color(255, 255, 0)
blue = Color(0, 0, 255)

roi_diameter = [8,7,4]
area_scale_factor = pow(roi_diameter[0]/2,2) + 2*pow(roi_diameter[1]/2,2) + 2*pow(roi_diameter[2]/2,2)
field_names_coords = ['X', 'Y', 'Z', 'mean']

scale_coeff = {
    'xy': 1.2, 
    'z': 2
}

directory = "/media/ula/D/ppp/fos_gfp_tmaze2/processing/"


mouse = 2
sole_ctx_session = 1
rep_ctx_sessions = [2,3]
img_source_file = "m{}s{}{}_tst_sa.tif"
source_file = "result_label_m{}s{}{}.csv"
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

def create_roi_group(x, y, z, img, color, in_ovlap = False):
	roi_group = []
	mean = 0
	stdev = 0
	if in_ovlap:
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
				mean += stats.mean * pow(rad,2)
				if i == 0:
					stdev = stats.stdDev
				roi_group.append(roi)
		mean /= area_scale_factor
	else:
		for i in range(-2,3):
			if(z+i > 0):
				diameter = roi_diameter[abs(i)]
				rad = diameter/2
				img.setSlice(z+i)
				roi = OvalRoi(x-rad, y-rad, diameter, diameter)
				roi.setStrokeColor(color)		
				roi.setPosition(z+i)
				img.setRoi(roi)
				stats = roi.getStatistics()
				mean += stats.mean * pow(rad,2)
				if i == 0:
					stdev = stats.stdDev
				roi_group.append(roi)
		mean /= area_scale_factor
	return roi_group, mean, stdev

def estimate_bgr(mouse_no, session_no, region):
	img_path = directory + img_source_file.format(mouse_no, session_no, region)
	img = ImagePlus(img_path)
	
	sum_of_means = 0
	bgr_file = directory + source_file.format(mouse_no, session_no, region)

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
					mean += stats.mean * pow(rad,2)
					if i == 0:
						stdev = stats.stdDev
			mean /= area_scale_factor
			sum_of_means += mean
	return sum_of_means/1000

def prepare_roi_stats_dict(mouse_no, session_no, img, color, region="", ov_idxs=None):
	cells_file = directory + source_file.format(mouse_no, session_no, region)
	
	roi_dict = {
		'rois' : [],
		'means' : [],
		'stdev' : []
	}
	
	with open(cells_file,"r") as source:
	    rdr = csv.DictReader( source )
	    idx = 0
	    for r in rdr:
	    	pos = int(float(r['Z']))
	    	idx += 1
	    	roi_group, mean, stdev = create_roi_group(int(float(r['X'])), int(float(r['Y'])), int(float(r['Z'])), img, color)
	    	roi_dict['means'].append(mean)
	    	roi_dict['stdev'].append(stdev)
	    	roi_dict['rois'].append(roi_group)   
	return roi_dict

def overlap_imgs(mouse_no, session_no, ov_idxs, cells_list, region=""):
	img_path = directory + img_source_file.format(mouse_no, session_no, region)
	result_file_path = directory + result_file.format(mouse_no, session_no, region)
	
	imp = ImagePlus(img_path)


	cell_list = []
	overlay = Overlay()
	for ov_idx in ov_idxs:
		ov_cell = cells_list[ov_idx]
		pos = ov_cell[2]
		roi_group, mean, stdev = create_roi_group(ov_cell[0], ov_cell[1], pos, imp, color, True)
		for roi in roi_group:
			overlay.add(roi)
	imp.setOverlay(overlay)

	
	imp.show()

def selection_by_thresholding(mouse_no, session_no, region="", color = white, bgr = 0):
	img_path = directory + img_source_file.format(mouse_no, session_no, region)
	result_file_path = directory + result_file.format(mouse_no, session_no, region)
	
	imp = ImagePlus(img_path)
	roi_dict = prepare_roi_stats_dict(mouse_no, session_no, imp, color, region)
	mean_threshold = otsu_thre(roi_dict['means'])/2

	cell_list = []
	
	overlay = Overlay()
	with open(result_file_path,"w") as result:
		wtr = csv.writer( result )
		wtr.writerow( field_names_coords )
		for idx, roi in enumerate(roi_dict['rois']):
			if(roi_dict['means'][idx] > mean_threshold):
				vals = []
				for c in roi[2].getContourCentroid():
					vals.append(int(c))
				vals.append(roi[2].getPosition())
				vals.append(roi_dict['means'][idx])
				cell_list.append(vals)
				wtr.writerow(vals)
				
				for roi_p in roi:
					overlay.add(roi_p)
				
	
	imp.setOverlay(overlay)
	
	imp.show()
	
	draw_histogram_with_thre('srednia intensywnosc w obrebie roi', roi_dict['means'], mean_threshold)
	
	print(len(cell_list))
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

def calculate_overlaps_for_trial_group(starting_session, mouse, reg_code, first = False):
	
	shift_code_root = "m" + str(mouse) + reg_code 

	print(shift_code_root)
	shift_code_root += "s"

	all_cells_count = 0 

	cells_dict = {}
	ov_idx_dict = {0:{}, 1:{}}
	'''
	cells_idx_dict - klucz - nr komorki na oryginalnej liscie;
	value[i] jaki nr na liscie wszystkich komorek ma komórka nr <klucz> 
	(idx klucz) z sesji starting_session + i
	'''
	cells_idx_dict = {}
	intensity_dict = {}
	
	for i in range(3):
		session_code = str(starting_session + i)
		cells_dict[session_code] = selection_by_thresholding(str(mouse), session_code, region=reg_code)
		all_cells_count += len(cells_dict[session_code])
	
	overlap_dict = {}

	overlap = []
	threeway_overlap = []
	
	ov_idx1_arr = []
	for i in range(3):
		for j in range(i+1, 3):
			ov_code = str(starting_session+j)+str(starting_session+i)
						
			cells1 = cells_dict[str(starting_session+i)]
			cells2 = cells_dict[str(starting_session+j)]
			'''
			do ogarniecia
			if mouse == 1 and ov_code == "31":
				break
			'''
			
			prev_ov_1 = ov_idx1_arr
			overlap, ov_idx1_arr, ov_idx2_arr = find_overlap(cells1, cells2, shift[shift_code_root+ov_code])
			# jesli potrzebne - tu bedzie mozna policzyc threeway ovlap
			if i == 0 and j == 2:
				threeway_overlap = len(list(set(prev_ov_1).intersection(ov_idx1_arr)))
			ov_idx_dict[i][j] = []
			ov_idx_dict[i][j].append(ov_idx1_arr)
			ov_idx_dict[i][j].append(ov_idx2_arr)
			'''[]
			#ov_idx_dict[i][j].append(ov_idx1_arr)
			ov_idx_dict[i][j].append(ov_idx2_arr)'''
			
			len_intersection = len(overlap)
			all_cells_count -= len_intersection
			
			if mouse == 1:
				ov_code = intersession_codes['1'][ov_code]
			else:
				ov_code = intersession_codes['other'][ov_code]
			overlap_dict[ov_code] = []
			overlap_dict[ov_code].append(len_intersection)
			overlap_dict[ov_code].append(format(float(len_intersection)/(len(cells1)+len(cells2)-len_intersection),'.2f'))

	all_cells_count += threeway_overlap
	print(threeway_overlap)
	for i in range(3):
		cells = cells_dict[str(starting_session+i)]
		tst = [k for k in range(len(cells))]
			
		double_idx = 0
		for idx, cell in enumerate(cells):
			new_cell = True
			global_idx = -1
			for j in range(i-1, -1, -1):
				if idx in ov_idx_dict[j][i][1]:
					idx_pos = ov_idx_dict[j][i][1].index(idx)
					counterpart_idx = ov_idx_dict[j][i][0][idx_pos]
					counterpart_cell = cells_dict[str(starting_session+j)][counterpart_idx]
					
					global_idx = counterpart_cell[-1]
					
					if not new_cell:
						double_idx += 1
					
					new_cell = False
			if new_cell:
				curr_len = len(intensity_dict.keys())
				intensity_dict[curr_len] = [0]*3
				global_idx = curr_len
				
			intensity_dict[global_idx][i] = cell[3]
			cell.append(global_idx)
		#print("total? ", len(intensity_dict.keys()))
	for key in overlap_dict.keys():
		overlap_dict[key].append(format(float(overlap_dict[key][0])/all_cells_count,'.2f'))
	
	print(all_cells_count)

	overlap_dict = OrderedDict(sorted(overlap_dict.items(), key=lambda t: t[0]))

	save_dict(overlap_dict, first)
	'''
	_, o12_idx1, o12_idx2 = find_overlap(cells_dict["1"], cells_dict["2"], shift[shift_code_root+"21"])
	_, o13_idx1, o13_idx3 = find_overlap(cells_dict["1"], cells_dict["3"], shift[shift_code_root+"31"])
	_, o23_idx2, o23_idx3 = find_overlap(cells_dict["2"], cells_dict["3"], shift[shift_code_root+"32"])

	print(len(o12_idx2),len(o13_idx1),len(o23_idx2))
	print(len(list(set(o12_idx2).intersection(o23_idx2))))
	print(len(list(set(o13_idx3).intersection(o23_idx3))))
	print(len(list(set(o12_idx1).intersection(o13_idx1))))
	'''	
	#overlap_imgs("3", "3", o13_idx3, cells_dict["3"], region="_r1")
	#overlap_imgs("3", "3", o23_idx3, cells_dict["3"], region="_r1")
	
	a_spec = []
	b_spec1 = []
	b1_ex = []
	b2_ex = []
	b_spec2 = []
	b_growth = []
	increase = 0
	print(len(intensity_dict.keys()))
	for i in intensity_dict.keys():
		#if not (None in intensity_dict[i]):
		b1a = intensity_dict[i][1] - intensity_dict[i][0]
		b2a = intensity_dict[i][2] - intensity_dict[i][0]
		b1b2 = intensity_dict[i][2] - intensity_dict[i][1]
		
		if b1a > 30 and b2a > 30:
			b_spec1.append(intensity_dict[i][1])
			b_spec2.append(intensity_dict[i][2])
			b_g = intensity_dict[i][2] - intensity_dict[i][1]
			b_growth.append(b_g)
			if b_g > 0:
				increase += 1
		elif b1a < -30 and b2a < -30:
			a_spec.append(intensity_dict[i][0])
			

	print("a spec", len(a_spec), sum(a_spec)/len(a_spec))
	print("b spec1", len(b_spec1), sum(b_spec1)/len(b_spec1))
	print("b spec2", len(b_spec2), sum(b_spec2)/len(b_spec2))
	print("b growth", sum(b_growth)/len(b_growth), increase)

	print(overlap_dict)
#	return overlap_dict, o13_idx3, o23_idx3


#overlap_dict, o13_idx3, o23_idx3 = calculate_overlaps_for_trial_group(1, 3, '_r1')
'''
selection_by_thresholding(3, 1, region="_r1", color = yellow)
selection_by_thresholding(3, 2, region="_r1", color = red)
selection_by_thresholding(3, 3, region="_r1", color = red)
'''
bgr = estimate_bgr(3, 1, region="_r1")
selection_by_thresholding(3, 1, region="_r1", color = yellow, bgr = bgr)
estimate_bgr(3, 2, region="_r1")
selection_by_thresholding(3, 2, region="_r1", color = yellow, bgr = bgr)
estimate_bgr(3, 3, region="_r1")
selection_by_thresholding(3, 3, region="_r1", color = yellow, bgr = bgr)