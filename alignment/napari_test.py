#!/usr/bin/env python3
"""
Image points 3D
===============
Display points overlaid on a 3D image
.. tags:: visualization-nD
"""
from skimage import data, feature, filters
import utils, constants
import napari

cells = utils.read_image(10, 1, "ctx")
viewer = napari.view_image(
        cells, ndisplay=3, scale=constants.SCALE#, channel_axis=1, name=['membranes', 'nuclei']
        )

centroids_df = utils.read_single_session_cell_data(10, 1, ["ctx"])

pts = centroids_df[[constants.ICY_COLNAMES['zcol'], constants.ICY_COLNAMES['ycol'],
                    constants.ICY_COLNAMES['xcol']]]
pts_layer = viewer.add_points(
    pts,
    size=1,
    face_color=
    shading='spherical',
    scale=constants.SCALE,
    edge_width=0,
)




napari.run()
'''
viewer.add_points(pts)
viewer.camera.angles = (10, -20, 130)
cells = data.cells3d()
nuclei = cells[:, 1]
smooth = filters.gaussian(nuclei, sigma=10)
pts = feature.peak_local_max(smooth)
'''