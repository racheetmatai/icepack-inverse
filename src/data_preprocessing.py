import rasterio
import numpy as np
import firedrake
import os
import icepack
from src.helper_functions import interpolate_2d_array, get_min_max_coords


def take_average_where_nan(base_folder, current_folder, previous_folder, next_folder, total_folders, file_type = 'vx'):
    """
    Recursively processes TIF files in the given folders, replacing NaN values in the current file with the average
    of corresponding values from the next and previous files.

    Parameters:
    - base_folder: Base folder containing the data.
    - current_folder: Path to the current folder.
    - previous_folder: Path to the previous folder.
    - next_folder: Path to the next folder.
    - total_folders: Total number of folders.

    Returns:
    - None
    """
    print(previous_folder, current_folder, next_folder)
    # Check if previous_folder and next_folder exist
    if not os.path.exists(previous_folder):
        print(f"The folder '{previous_folder}' does not exist.")
        return

    if not os.path.exists(next_folder):
        print(f"The folder '{next_folder}' does not exist.")
        return

    # Get the current and next folder numbers
    current_folder_number = int(current_folder.split(os.path.sep)[-1])
    next_folder_number = int(next_folder.split(os.path.sep)[-1])
    previous_folder_number = int(previous_folder.split(os.path.sep)[-1])

    #['vx', 'vy', 'ex', 'ey']
    
    #for file_type in file_types:
    print("FILE TYPE:", file_type)
    current_files = [file for file in os.listdir(current_folder) if file.startswith("ASE") and file.endswith(f"{file_type}_v05.0_nan.tif")]
    next_files = [file for file in os.listdir(next_folder) if file.startswith("ASE") and file.endswith(f"{file_type}_v05.0_nan.tif")]
    previous_files = [file for file in os.listdir(previous_folder) if file.startswith("ASE") and file.endswith(f"{file_type}_v05.0_nan.tif")]

    for current_file, next_file, previous_file in zip(current_files, next_files, previous_files):
        current_file_path = os.path.join(current_folder, current_file)
        next_file_path = os.path.join(next_folder, next_file)
        previous_file_path = os.path.join(previous_folder, previous_file)

        # Read TIF files with rasterio
        with rasterio.open(current_file_path) as current_src:
            current_data = current_src.read(1)

        with rasterio.open(next_file_path) as next_src:
            next_data = next_src.read(1)

        with rasterio.open(previous_file_path) as previous_src:
            previous_data = previous_src.read(1)

        # Replace NaN values in current_data with values from next_data
        non_nan_mask_next = ~np.isnan(next_data)
        non_nan_mask_previous = ~np.isnan(previous_data)
        
        # Replace NaN values in current_data with the average of corresponding non-NaN values
        non_nan_mask = non_nan_mask_next & non_nan_mask_previous

        # Calculate weights based on the distances
        distance_to_previous = current_folder_number - previous_folder_number
        distance_to_next = next_folder_number - current_folder_number
        
        # Avoid division by zero
        weights_previous = 1.0 / (distance_to_previous + 1)
        weights_next = 1.0 / (distance_to_next + 1)
        
        # Replace NaN values in current_data with the weighted average
        current_data[np.isnan(current_data) & non_nan_mask] = (weights_previous * previous_data[np.isnan(current_data) & non_nan_mask] + 
                           weights_next * next_data[np.isnan(current_data) & non_nan_mask]) / (weights_previous + weights_next)
        
        
        
                    
        #current_data[np.isnan(current_data) & non_nan_mask] = (next_data[np.isnan(current_data) & non_nan_mask] + previous_data[np.isnan(current_data) & non_nan_mask]) / 2.0
        
        # If cannot average, replace NaN in current_data with values from next_data or previous_data
        current_data[np.isnan(current_data)] = next_data[np.isnan(current_data)]
        current_data[np.isnan(current_data)] = previous_data[np.isnan(current_data)]
        
        # Get metadata from the original file
        metadata = current_src.meta

        # Write the modified data back to the original file
        with rasterio.open(current_file_path, 'w', **metadata) as dst:
            dst.write(current_data, 1)

    

    next_folder_number = next_folder_number + 1
    previous_folder_number = previous_folder_number - 1

    if next_folder_number > total_folders and previous_folder_number < 1:
        return
    else:
        if next_folder_number > total_folders:
            next_folder_number = total_folders
        if previous_folder_number < 1:
            previous_folder_number = 1

        next_folder_number = str(next_folder_number)
        previous_folder_number = str(previous_folder_number)

        # Recursive call
        take_average_where_nan(base_folder, current_folder, os.path.join(base_folder, previous_folder_number), os.path.join(base_folder, next_folder_number), total_folders, file_type)
    return

def replace_nans_recursive(base_folder,current_folder, next_folder, total_folders):
    print(current_folder, next_folder)
    file_types = ['vx', 'vy', 'ex', 'ey']

    for file_type in file_types:
        current_files = [file for file in os.listdir(current_folder) if file.startswith("ASE") and file.endswith(f"{file_type}_v05.0_nan.tif")]
        #print(current_files)
        next_files = [file for file in os.listdir(next_folder) if file.startswith("ASE") and file.endswith(f"{file_type}_v05.0_nan.tif")]
        #print(next_files)

        for current_file, next_file in zip(current_files, next_files):
            current_file_path = os.path.join(current_folder, current_file)
            next_file_path = os.path.join(next_folder, next_file)

            # Read TIF files with rasterio
            with rasterio.open(current_file_path) as current_src:
                current_data = current_src.read(1)

            with rasterio.open(next_file_path) as next_src:
                next_data = next_src.read(1)

            # Replace NaN values in current_data with values from next_data
            current_data[np.isnan(current_data)] = next_data[np.isnan(current_data)]

            # Get metadata from the original file
            metadata = current_src.meta

            # Write the modified data back to the original file
            with rasterio.open(current_file_path, 'w', **metadata) as dst:
                dst.write(current_data, 1)

    # Get the current and next folder numbers
    current_folder_number = int(current_folder.split(os.path.sep)[-1])
    next_folder_number = int(next_folder.split(os.path.sep)[-1])
    
    # Terminate recursion if the next folder becomes the current folder
    if next_folder_number == current_folder_number:
        return

    print(current_folder_number, next_folder_number)
    # Calculate the next folder number in a circular manner
    next_folder_number = int((next_folder_number + 1) % total_folders)
    if next_folder_number == 0:
        next_folder_number = total_folders
    next_folder_number = str(next_folder_number)
    #print(next_folder_number)

    # Recursive call with the next folder and the current folder
    replace_nans_recursive(base_folder, current_folder,
                           os.path.join(base_folder, str(next_folder_number)),
                           total_folders)

def clean_imported_data(name):
    """
    Clean and interpolate values in imported data files.

    :param name: Name of the dataset.
    """
    # File names
    vx_filename = name + '_vx_v05.0.tif'
    vy_filename = name + '_vy_v05.0.tif'
    stdx_filename = name + '_ex_v05.0.tif'
    stdy_filename = name + '_ey_v05.0.tif'

    # Open raster files
    vx_file = rasterio.open(vx_filename, 'r+')
    vy_file = rasterio.open(vy_filename, 'r+')
    stdx_file = rasterio.open(stdx_filename, 'r+')
    stdy_file = rasterio.open(stdy_filename, 'r+')

    # Replace values in vx_file
    vx_data = vx_file.read(1)
    replace_value = -1.9e+9
    replace_value_std = -0.99
    vx_data[vx_data < replace_value] = np.nan

    # Replace values in vy_file
    vy_data = vy_file.read(1)
    vy_data[vy_data < replace_value] = np.nan

    # Replace values in stdx_file
    stdx_data = stdx_file.read(1)
    stdx_data[stdx_data < replace_value_std] = np.nan

    # Replace values in stdy_file
    stdy_data = stdy_file.read(1)
    stdy_data[stdy_data < replace_value_std] = np.nan

    # Get metadata from the original files
    metadata = vx_file.meta
    metadata_std = stdx_file.meta

    # Write the modified data back to the files
    with rasterio.open(vx_filename.replace('.tif', '_nan.tif'), 'w', **metadata) as dst:
        dst.write(vx_data, 1)

    with rasterio.open(vy_filename.replace('.tif', '_nan.tif'), 'w', **metadata) as dst:
        dst.write(vy_data, 1)

    with rasterio.open(stdx_filename.replace('.tif', '_nan.tif'), 'w', **metadata_std) as dst:
        dst.write(stdx_data, 1)

    with rasterio.open(stdy_filename.replace('.tif', '_nan.tif'), 'w', **metadata_std) as dst:
        dst.write(stdy_data, 1)

def read_raster_file(filename):
    """
    Read raster file using rasterio.

    :param filename: Name of the file.
    :return: Raster data.
    """
    file = rasterio.open(filename, 'r')
    return file

def get_windowed_velocity_file(name, outline, δ, modified_exists=False):
    """
    Get windowed velocity data from modified raster files.

    :param name: Name of the dataset.
    :param outline: GeoJSON outline data.
    :param δ: Buffer distance.
    :param modified_exists: Whether modified files exist.
    :return: Velocity data, raster files, window, and transform.
    """
    if name is None:
        print("Reading velocity from measures database")
        velocity_filename = icepack.datasets.fetch_measures_antarctica()
        vx_file = rasterio.open(f"netcdf:{velocity_filename}:VX", "r")
        vy_file = rasterio.open(f"netcdf:{velocity_filename}:VY", "r")
        stdx_file = rasterio.open(f"netcdf:{velocity_filename}:ERRX", "r")
        stdy_file = rasterio.open(f"netcdf:{velocity_filename}:ERRY", "r")
    else:
        print("Reading velocity from specified file")
        if not modified_exists:
            clean_imported_data(name)
    
        # Modified file names
        modified_vx_filename = name + '_vx_v05.0' + '_nan.tif'
        modified_vy_filename = name + '_vy_v05.0' + '_nan.tif'
        modified_stdx_filename = name + '_ex_v05.0' + '_nan.tif'
        modified_stdy_filename = name + '_ey_v05.0' + '_nan.tif'
    
        # Open modified raster files
        vx_file = rasterio.open(modified_vx_filename, 'r')
        vy_file = rasterio.open(modified_vy_filename, 'r')
        stdx_file = rasterio.open(modified_stdx_filename, 'r')
        stdy_file = rasterio.open(modified_stdy_filename, 'r')

    xmin, xmax, ymin, ymax = get_min_max_coords(outline, δ)
    window = rasterio.windows.from_bounds(
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax,
        transform=vx_file.transform,
    ).round_lengths().round_offsets()
    transform = vx_file.window_transform(window)

    vx = vx_file.read(indexes=1, window=window)
    vy = vy_file.read(indexes=1, window=window)
    stdx = stdx_file.read(indexes=1, window=window)
    stdy = stdy_file.read(indexes=1, window=window)

    # Close the raster files
    #vx_file.close()
    #vy_file.close()
    #stdx_file.close()
    #stdy_file.close()

    return vx, vx_file, vy, vy_file, stdx, stdx_file, stdy, stdy_file, window, transform


def create_vertex_only_mesh_for_sparse_data(mesh, window, transform, stdx):
    """
    Create a vertex-only mesh for sparse data.

    :param mesh: Firedrake mesh.
    :param window: Rasterio window.
    :param transform: Rasterio transform.
    :param stdx: Standard deviation data.
    :return: Vertex-only mesh and indices.
    """
    indices = np.array(
        [
            (i, j)
            for i in range(window.width)
            for j in range(window.height)
            if (
                mesh.locate_cell(transform * (i, j)) and
                stdx[j, i] > 0.0
            )
        ]
    )
    xs = np.array([transform * idx for idx in indices])
    point_set = firedrake.VertexOnlyMesh(
        mesh, xs, missing_points_behaviour="error"
    )
    Δ = firedrake.FunctionSpace(point_set, "DG", 0)
    return Δ, indices


def interpolate_data_onto_vertex_only_mesh(Δ, variable_value, indices):
    """
    Interpolate data onto a vertex-only mesh.

    :param Δ: Firedrake function space.
    :param variable_value: Variable values.
    :param indices: Indices of valid points.
    :return: Interpolated variable.
    """
    variable = firedrake.Function(Δ)
    variable.dat.data[:] = variable_value[indices[:, 1], indices[:, 0]]
    return variable
    
