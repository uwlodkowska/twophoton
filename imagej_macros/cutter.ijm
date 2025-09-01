path = "/mnt/data/fos_gfp_tmaze/context_only/despeckle_nc/icy/";
mouse = "m6";
region = "r2";
suffix = "_c";
ext = ".tif";
cut_start = 30;

sessions = newArray("s1", "s2", "s3", "ret1", "ret2", "ret3");
for (i = 0; i < sessions.length; i++) {
	start_name = mouse + sessions[i] + "_" + region + suffix + ext;
	save_name = mouse + sessions[i] + "_" + region + ext;
	open(path+start_name);
	selectWindow(start_name);
	cut_lim = nSlices;
	run("Make Substack...", "slices="+cut_start+"-"+cut_lim);
	rename(save_name);
	saveAs("Tiff", path+save_name);
	selectWindow(start_name);
	close();
	selectWindow(save_name);
	close();
}