from ij.gui import Overlay, OvalRoi, Plot
from ij import IJ, ImagePlus
from java.awt import Color
from ij.plugin.frame import RoiManager
import csv, math
from collections import OrderedDict

red = Color(255, 0, 0)
green = Color(0, 255, 0)

roi_diameter = [8,7,4]

directory = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/despeckle/alignment_result/aligned_despeckle/"

single_session_cell_data_fn = "m{}r{}_{}_output.txt"#output from icy
cell_data_fn_template = "m{}r{}_{}_output.txt"
img_fn_template = "m{}r{}_{}.tif"

def read_chosen_idxes(filename):
	idx_file = directory + filename
	ret = []
	with open(idx_file,"r") as source:
		source.next()
		rdr = csv.reader(source)
		for r in rdr:
			ret += [int(float(r[1]))]
	return ret

def draw_chosen_rois(mouse, region, sessions, idxes=[]):
	img_path = directory + img_fn_template.format(mouse, region, sessions[0])
	cells_file = directory + cell_data_fn_template.format(mouse, region, sessions[0])
	imp = ImagePlus(img_path)
	imp.show()
	overlay = Overlay()
	with open(cells_file,"r") as source:
		for i in range(1):
			source.next()
		rdr = csv.DictReader(source,dialect="excel-tab")
		idx = 0
		for r in rdr:
			if len(idxes) == 0 or idx in idxes :
				pos = int(float(r['Intensity center Z (px) (ch 0)']))
				x = int(float(r['Intensity center X (px) (ch 0)']))
				y = int(float(r['Intensity center Y (px) (ch 0)']))	
				roi_group = create_roi_group(x, y, pos, imp, overlay)
			idx +=1
	imp.setOverlay(overlay)



def create_roi_group(x, y, z, img, overlay, color=green):
	roi_group = []
	for i in range(-2,3):
		if(z+i+1 > 0 and z+i <= img.getNSlices()):
			diameter = roi_diameter[abs(i)]
			rad = diameter/2
			img.setSlice(z+i)
			roi = OvalRoi(x-rad, y-rad, diameter, diameter)
			roi.setStrokeWidth(1)	
			roi.setPosition(z+i+1)
			img.setRoi(roi)
			overlay.add(roi)
			stats = roi.getStatistics()
	return roi_group

tail_idxes = read_chosen_idxes("tail.csv")
draw_chosen_rois(10, 1, ['landmark1', 'ctx'], idxes=tail_idxes)