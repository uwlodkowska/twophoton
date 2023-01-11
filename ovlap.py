from ij.gui import Overlay, OvalRoi, Plot
from ij import IJ, ImagePlus
from java.awt import Color
from ij.plugin.frame import RoiManager
import csv

sessions={
'l' : ["landmark", "ctx1", "ctx2"],
'c' : ["ctx", "landmark1", "landmark2"]}

def visualize(mouse, region, s_ids, s_code):
	coords_path = res_dir_path + "aligned_despeckle/overlap_coords/"

	cell_data_fn_template = "m{}r{}_{}_output.txt"
	coords_fn_template = "m{}r{}_{}.txt"
	filename = coords_path+coords_fn_template.format(mouse, region, str(s_ids[0])+"_"+str(s_ids[1]))
	, colnames = get_names(mouse, region, sessions)