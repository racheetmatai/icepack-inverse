import numpy as np
import rasterio
import icepack
import geojson
import logging
import icepack.plot
from scipy.interpolate import griddata

# Set up logging configuration
logging.basicConfig(level=logging.ERROR)  # Adjust the level as needed

def get_min_max_coords(outline, δ=5e3):
    """
    Calculate the minimum and maximum coordinates based on the given outline.

    :param outline: GeoJSON outline data.
    :param δ: Buffer distance (default is 5e3).
    :return: Tuple of (xmin, xmax, ymin, ymax).
    """
    coords = np.array(list(geojson.utils.coords(outline)))
    xmin, xmax = coords[:, 0].min() - δ, coords[:, 0].max() + δ
    ymin, ymax = coords[:, 1].min() - δ, coords[:, 1].max() + δ
    return xmin, xmax, ymin, ymax



def plot_bounded_antarctica(outline, δ=5e3, *args, **kwargs):
    """
    Plot a bounded view of Antarctica.
    :param outline: GeoJSON outline data.
    :param δ: Buffer distance (default is 5e3).
    :param args: Additional arguments for subplots.
    :param kwargs: Additional keyword arguments for subplots.
    :return: Tuple of (figure, axes).
    """
    xmin, xmax, ymin, ymax = get_min_max_coords(outline, δ)
    image_filename =  icepack.datasets.fetch_mosaic_of_antarctica()
    with rasterio.open(image_filename, "r") as image_file:
        image_window = rasterio.windows.from_bounds(
            left=xmin,
            bottom=ymin,
            right=xmax,
            top=ymax,
            transform=image_file.transform,
        )
        image = image_file.read(indexes=1, window=image_window, masked=True)
    fig, axes = icepack.plot.subplots(*args, **kwargs)
    xmin, ymin, xmax, ymax = rasterio.windows.bounds(
        image_window, image_file.transform
    )
    kw = {
        "extent": (xmin, xmax, ymin, ymax),
        "cmap": "Greys_r",
        "vmin": 12e3,
        "vmax": 16.38e3,
    }
    try:
        axes.imshow(image, **kw)
    except AttributeError:
        for ax in axes:
            ax.imshow(image, **kw)

    return fig, axes

def interpolate_2d_array(array):
    rows, cols = np.indices(array.shape)
    points = np.column_stack((rows.ravel(), cols.ravel()))
    
    # Mask the NaN values
    mask = ~np.isnan(array).ravel()
    
    # Extract non-NaN values
    values = array[~np.isnan(array)]
    
    # Interpolate using griddata
    interpolated_values = griddata(points[mask], values, (rows, cols), method='linear')
    
    # Replace NaN values with interpolated values
    array[np.isnan(array)] = interpolated_values[np.isnan(array)]
    return array