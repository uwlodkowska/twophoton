dir = "/media/ula/DATADRIVE1/fos_gfp_tmaze/ctx_landmark/"

margin = 0;

scan_list = getFileList(dir);

for (n = 0; n < scan_list.length; n++){
	filename = scan_list[n];		
	if(!File.isDirectory(dir+filename)){
		open(dir + filename);
		selectWindow(filename);
	
		noext = substring(filename, 0, filename.indexOf("."));
	
		refslice_no = round(nSlices*0.75);
		setSlice(refslice_no);
		run("Stack Contrast Adjustment", "is");
		
		saveAs("Tiff", dir + "despeckle/orig/" + noext + ".tif");
		run("Despeckle", "stack");
		
		saveAs("Tiff", dir + "despeckle/" + noext + ".tif");
		run("Close All");
	}
}