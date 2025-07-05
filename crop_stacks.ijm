name = "m6r1_"
ext = ".czi"

sessions = newArray("s0", "landmark1", "landmark2", "ctx1", "ctx2");
for (i = 0; i < sessions.length; i++) {
	s_name = name + sessions[i];
	open("/home/ula/multisession/"+s_name+ext);
	selectWindow(s_name+ext);
	makeRectangle(125, 292, 174, 168);
	run("Crop");
	rename("Cropped_Image");
	selectWindow("Cropped_Image");
	run("Make Substack...", "slices=25-65");
	saveAs("Tiff", "/home/ula/multisession/"+s_name+"_cropped"+ext);
	selectWindow("Cropped_Image");
	close();
}
