from skimage import data, feature, filters
import utils, constants
import napari
import numpy as np

def visualize_with_centroids(mouse, region, session):
    cells = utils.read_image(mouse, region, session)


    centroids_df = utils.read_single_session_cell_data(mouse, region, [session])

    pts = centroids_df[constants.COORDS_3D]
    
    
    visualize_with_centroids_custom(cells, pts)
    
def visualize_df_centroids(mouse, region, session, centroids_df, config):
    cells = utils.read_image(mouse, region, session, config)


    pts = centroids_df[constants.COORDS_3D]
    
    
    visualize_with_centroids_custom(cells, pts)
    
def visualize_with_centroids_custom(img, pts, point_props = None):
        if len(img.shape) > 3:
            channel_axis = 0
            channel_names = np.arange(len(img.shape)).astype(str)
        else:
            channel_axis = None
            channel_names = None
        viewer = napari.view_image(
                img, 
                ndisplay=3, 
                scale=constants.SCALE,
                channel_axis=channel_axis, 
                name=channel_names
                )
        
        if point_props is None:
            pts_layer = viewer.add_points(
                pts,
                size=1,
                shading='spherical',
                scale=constants.SCALE,
                face_color= 'red'
                
            )
        else:
            pts_layer = viewer.add_points(
                pts,
                size=1,
                properties= point_props,
                shading='spherical',
                scale=constants.SCALE,
                face_color= 'color',
                face_colormap='viridis'
                
            )
     

        napari.run()
        