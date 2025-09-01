dir = "/mnt/data/fos_gfp_tmaze/multisession/"

margin = 0;

scan_list = getFileList(dir);

for (n = 0; n < scan_list.length; n++){
	filename = scan_list[n];		
	print(filename);
	if(!File.isDirectory(dir+filename)){
		open(dir + filename);
		selectWindow(filename);
	
		noext = substring(filename, 0, filename.indexOf("."));
	
		refslice_no = round(nSlices*0.75);
		print(refslice_no);
		//setSlice(refslice_no);
		//run("Stack Contrast Adjustment", "is");
		
		saveAs("Tiff", dir + "despeckle_nc/orig/" + noext + ".tif");
		run("Despeckle", "stack");
		
		saveAs("Tiff", dir + "despeckle_nc/" + noext + ".tif");
		run("Close All");
	}
}
