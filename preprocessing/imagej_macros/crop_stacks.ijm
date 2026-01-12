path = "/mnt/data/fos_gfp_tmaze/multisession/despeckle_nc/"
name = "m7r1_"
ext = ".tif"

sessions = newArray("s0", "landmark1", "landmark2", "ctx1", "ctx2");
for (i = 0; i < sessions.length; i++) {
	s_name = name + sessions[i];
	open(path+s_name+ext);
	selectWindow(s_name+ext);
	makeRectangle(100, 100, 250, 250);
	run("Crop");
	rename("Cropped_Image");
	selectWindow("Cropped_Image");
	//run("Make Substack...", "slices=10-68");
	saveAs("Tiff", path +s_name+"_cropped"+ext);
	selectWindow(s_name+"_cropped"+ext);
	close();
}
