import rasterio
import numpy as np
import firedrake
import os
import icepack
import xarray as xr
import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path
from pyproj import CRS
from scipy.spatial import cKDTree
from src.helper_functions import interpolate_2d_array, get_min_max_coords


def stable_xy_row_ids(x, y):
    """Create order-independent versioned identifiers from float64 x/y pairs."""
    x_values = np.asarray(x, dtype="float64")
    y_values = np.asarray(y, dtype="float64")
    if x_values.shape != y_values.shape:
        raise ValueError("x and y coordinate arrays must have matching shapes.")
    if not (np.isfinite(x_values).all() and np.isfinite(y_values).all()):
        raise ValueError("Stable row IDs require finite projected coordinates.")
    pack = struct.Struct(">dd").pack
    return np.asarray(
        [
            "xyh1-" + hashlib.sha256(pack(float(x_value), float(y_value))).hexdigest()[:32]
            for x_value, y_value in zip(x_values, y_values)
        ],
        dtype=object,
    )


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

def _netcdf_spatial_coordinates(dataset):
    """Return the verified projected horizontal coordinate names."""
    spatial = {}
    for name, coordinate in dataset.coords.items():
        standard_name = coordinate.attrs.get("standard_name", "").lower()
        axis = coordinate.attrs.get("axis", "").upper()
        if standard_name == "projection_x_coordinate" or axis == "X":
            spatial["x"] = name
        elif standard_name == "projection_y_coordinate" or axis == "Y":
            spatial["y"] = name

    # Projected NetCDF products commonly use literal x/y names without an
    # axis attribute. Accept those names only when their units are metres.
    for axis_name in ("x", "y"):
        if axis_name not in spatial and axis_name in dataset.coords:
            units = str(dataset.coords[axis_name].attrs.get("units", "")).lower()
            if units in {"m", "metre", "metres", "meter", "meters"}:
                spatial[axis_name] = axis_name

    if set(spatial) != {"x", "y"}:
        raise ValueError(
            "NetCDF raster must provide projected x and y coordinates with "
            "verifiable coordinate metadata and metre units."
        )

    for axis_name, coordinate_name in spatial.items():
        coordinate = dataset.coords[coordinate_name]
        units = str(coordinate.attrs.get("units", "")).lower()
        if coordinate.ndim != 1 or units not in {
            "m", "metre", "metres", "meter", "meters"
        }:
            raise ValueError(
                f"NetCDF {axis_name}-coordinate {coordinate_name!r} must be "
                "one-dimensional and expressed in metres."
            )
        values = np.asarray(coordinate.values)
        differences = np.diff(values)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(
                f"NetCDF coordinate {coordinate_name!r} is not strictly "
                "monotonic."
            )
    return spatial


def _netcdf_raster_candidates(dataset, spatial):
    """List two-dimensional data variables defined on both spatial axes."""
    x_name = spatial["x"]
    y_name = spatial["y"]
    return [
        name
        for name, variable in dataset.data_vars.items()
        if variable.ndim == 2
        and x_name in variable.dims
        and y_name in variable.dims
    ]


def _netcdf_crs(dataset, variable):
    """Read the variable's declared grid mapping without inventing a CRS."""
    grid_mapping = variable.attrs.get("grid_mapping")
    if not grid_mapping or grid_mapping not in dataset.variables:
        raise ValueError(
            f"NetCDF variable {variable.name!r} has no valid grid_mapping "
            "reference."
        )
    attributes = dict(dataset[grid_mapping].attrs)
    try:
        return CRS.from_cf(attributes)
    except Exception as error:
        for key in ("crs_wkt", "spatial_ref"):
            if attributes.get(key):
                try:
                    return CRS.from_wkt(attributes[key])
                except Exception:
                    pass
        raise ValueError(
            f"Could not parse CRS metadata for NetCDF variable "
            f"{variable.name!r}."
        ) from error


def _read_netcdf_raster(filename, variable=None, expected_crs=3031):
    """Load a validated NetCDF raster as a detached xarray DataArray."""
    with xr.open_dataset(filename, decode_cf=True, mask_and_scale=True) as dataset:
        spatial = _netcdf_spatial_coordinates(dataset)
        candidates = _netcdf_raster_candidates(dataset, spatial)
        if variable is None:
            if len(candidates) != 1:
                raise ValueError(
                    "NetCDF variable selection is ambiguous. Raster "
                    f"candidates: {candidates}. Pass an explicit variable."
                )
            variable = candidates[0]
        elif variable not in candidates:
            raise ValueError(
                f"NetCDF variable {variable!r} is not a two-dimensional "
                f"projected raster. Raster candidates: {candidates}."
            )

        data = dataset[variable]
        crs = _netcdf_crs(dataset, data)
        if expected_crs is not None:
            expected = CRS.from_user_input(expected_crs)
            if not crs.equals(expected):
                raise ValueError(
                    f"NetCDF variable {variable!r} has CRS {crs.to_string()}, "
                    f"expected {expected.to_string()}."
                )

        rename = {
            coordinate_name: axis_name
            for axis_name, coordinate_name in spatial.items()
            if coordinate_name != axis_name
        }
        if rename:
            data = data.rename(rename)

        # Icepack samples xarray.DataArray objects by named x/y coordinates.
        # Loading here detaches the small project rasters from the file so the
        # NetCDF handle is closed deterministically. Dimension order and axis
        # direction are retained exactly as declared by the coordinates.
        data.load()
        data = data.copy(deep=False)
        data.attrs = dict(data.attrs)
        data.attrs["validated_crs"] = crs.to_string()
        data.attrs["x_direction"] = (
            "ascending" if data.x.values[-1] > data.x.values[0] else "descending"
        )
        data.attrs["y_direction"] = (
            "ascending" if data.y.values[-1] > data.y.values[0] else "descending"
        )
        return data


def _read_geotiff_cell_centers(
    filename, expected_crs=3031, assumed_crs=None
):
    """Load a north-up GeoTIFF with values labeled at true pixel centers.

    ``assumed_crs`` is an explicit, provenance-visible exception for a
    hash-locked legacy raster that has no embedded CRS. It is rejected when a
    source CRS is present and must match ``expected_crs`` when both are given.
    """

    with rasterio.open(filename, "r") as dataset:
        transform = dataset.transform
        if not (np.isclose(transform.b, 0.0) and np.isclose(transform.d, 0.0)):
            raise ValueError(
                "Cell-center x/y coordinates require an unrotated GeoTIFF; "
                f"got transform {transform}."
            )
        source_crs_missing = dataset.crs is None
        if source_crs_missing:
            if assumed_crs is None:
                raise ValueError(f"GeoTIFF {filename!s} has no declared CRS.")
            actual = CRS.from_user_input(assumed_crs)
        else:
            if assumed_crs is not None:
                raise ValueError(
                    "assumed_crs is allowed only when the source GeoTIFF has "
                    "no embedded CRS."
                )
            actual = CRS.from_user_input(dataset.crs)
        if expected_crs is not None:
            expected = CRS.from_user_input(expected_crs)
            if not actual.equals(expected):
                raise ValueError(
                    f"GeoTIFF {filename!s} has CRS {actual.to_string()}, "
                    f"expected {expected.to_string()}."
                )

        values = dataset.read(1, masked=True).astype("float64").filled(np.nan)
        x = transform.c + transform.a * (np.arange(dataset.width) + 0.5)
        y = transform.f + transform.e * (np.arange(dataset.height) + 0.5)
        attributes = dict(dataset.tags(1))
        attributes.update(
            {
                "validated_crs": (
                    actual.to_string()
                ),
                "source_crs_missing": source_crs_missing,
                "assumed_crs": (
                    actual.to_string() if source_crs_missing else None
                ),
                "coordinate_mode": "cell_center",
                "source_path": os.fspath(filename),
                "x_direction": "ascending" if x[-1] > x[0] else "descending",
                "y_direction": "ascending" if y[-1] > y[0] else "descending",
            }
        )
        if dataset.nodata is not None:
            attributes["source_nodata"] = float(dataset.nodata)

    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        attrs=attributes,
        name=Path(filename).stem,
    )


def read_raster_file(
    filename, variable=None, expected_crs=3031, coordinate_mode=None,
    assumed_crs=None
):
    """Read a GeoTIFF or a projected NetCDF raster for ``icepack.interpolate``.

    A bare GeoTIFF retains historical behavior and returns a
    ``rasterio.DatasetReader``. A GeoTIFF descriptor with
    ``coordinate_mode='cell_center'`` returns an in-memory ``xarray.DataArray``
    labeled at true pixel centers. NetCDF returns an in-memory DataArray using
    its declared projected x/y coordinates after validating metadata and CRS.

    ``filename`` may be a path or a descriptor such as::

        {"path": "ADMAP_2S_epsg3031_gdal.nc", "variable": "z"}
        {"path": "AntGG2021_Gravity_disturbance_at-surface.nc",
         "variable": "grav_dist", "expected_crs": 3031}
        {"path": "GeothermalHeatFlux_5km.tif",
         "coordinate_mode": "cell_center"}

    A plain NetCDF path is accepted only when exactly one unambiguous raster
    variable exists. Existing calls such as ``read_raster_file("field.tif")``
    retain their original behavior.
    """
    if isinstance(filename, Mapping):
        descriptor = dict(filename)
        unknown = set(descriptor) - {
            "path", "variable", "expected_crs", "method", "coordinate_mode",
            "assumed_crs"
        }
        if unknown:
            raise ValueError(f"Unknown raster descriptor fields: {sorted(unknown)}")
        if "path" not in descriptor:
            raise ValueError("Raster descriptor must contain a 'path' field.")
        if variable is not None and descriptor.get("variable") not in (None, variable):
            raise ValueError("Conflicting NetCDF variable selections were supplied.")
        filename = descriptor["path"]
        variable = descriptor.get("variable", variable)
        expected_crs = descriptor.get("expected_crs", expected_crs)
        descriptor_assumed_crs = descriptor.get("assumed_crs")
        if assumed_crs is not None and descriptor_assumed_crs not in (
            None, assumed_crs
        ):
            raise ValueError("Conflicting assumed CRS values were supplied.")
        assumed_crs = descriptor_assumed_crs or assumed_crs
        descriptor_mode = descriptor.get("coordinate_mode")
        if coordinate_mode is not None and descriptor_mode not in (None, coordinate_mode):
            raise ValueError("Conflicting raster coordinate modes were supplied.")
        coordinate_mode = descriptor_mode or coordinate_mode

    filename = os.fspath(filename)
    suffix = Path(filename).suffix.lower()
    if suffix in {".nc", ".nc4", ".cdf"}:
        if coordinate_mode is not None:
            raise ValueError(
                "NetCDF inputs use their declared coordinates and do not accept "
                "a GeoTIFF coordinate_mode."
            )
        if assumed_crs is not None:
            raise ValueError("NetCDF inputs do not accept assumed_crs.")
        return _read_netcdf_raster(
            filename, variable=variable, expected_crs=expected_crs
        )
    if variable is not None:
        raise ValueError("A data-variable name is valid only for NetCDF inputs.")
    if coordinate_mode in (None, "legacy_corner"):
        if assumed_crs is not None:
            raise ValueError(
                "assumed_crs requires coordinate_mode='cell_center'."
            )
        return rasterio.open(filename, "r")
    if coordinate_mode == "cell_center":
        return _read_geotiff_cell_centers(
            filename, expected_crs=expected_crs, assumed_crs=assumed_crs
        )
    raise ValueError(
        f"Unsupported GeoTIFF coordinate_mode {coordinate_mode!r}; use "
        "'legacy_corner' or 'cell_center'."
    )

def _finite_unmasked(values):
    """Return validity from a raster read without treating numeric zero as missing."""
    array = np.ma.asarray(values)
    data = np.asarray(np.ma.getdata(array))
    return (~np.ma.getmaskarray(array)) & np.isfinite(data)


def build_velocity_observation_mask(vx, vy, errx, erry, source=None):
    """Build the common observation mask used by inversion and CSV export.

    Velocity availability is defined by valid VX/VY values and, when the
    MEaSUREs SOURCE layer is available, SOURCE > 0. ERRX/ERRY are recorded as
    diagnostics but do not define the population for an unweighted inversion.
    """
    arrays = {
        "vx": np.ma.asarray(vx),
        "vy": np.ma.asarray(vy),
        "errx": np.ma.asarray(errx),
        "erry": np.ma.asarray(erry),
    }
    shapes = {key: value.shape for key, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Velocity component/error arrays must align; got {shapes}.")

    vx_valid = _finite_unmasked(arrays["vx"])
    vy_valid = _finite_unmasked(arrays["vy"])
    velocity_valid = vx_valid & vy_valid

    source_counts = {}
    if source is not None:
        source_array = np.ma.asarray(source)
        if source_array.shape != arrays["vx"].shape:
            raise ValueError(
                "MEaSUREs SOURCE must align with VX/VY; "
                f"got {source_array.shape} and {arrays['vx'].shape}."
            )
        source_data = np.asarray(np.ma.getdata(source_array))
        source_valid = _finite_unmasked(source_array) & (source_data > 0)
        velocity_valid &= source_valid
        for code in (0, 1, 2, 3):
            source_counts[str(code)] = int(
                np.count_nonzero(_finite_unmasked(source_array) & (source_data == code))
            )

    errx_data = np.asarray(np.ma.getdata(arrays["errx"]))
    erry_data = np.asarray(np.ma.getdata(arrays["erry"]))
    error_valid = (
        _finite_unmasked(arrays["errx"])
        & _finite_unmasked(arrays["erry"])
        & (errx_data > 0)
        & (erry_data > 0)
    )
    summary = {
        "window_pixels": int(velocity_valid.size),
        "valid_vx": int(np.count_nonzero(vx_valid)),
        "valid_vy": int(np.count_nonzero(vy_valid)),
        "valid_velocity_and_source": int(np.count_nonzero(velocity_valid)),
        "valid_error_pair": int(np.count_nonzero(error_valid)),
        "valid_observation_with_error_pair": int(
            np.count_nonzero(velocity_valid & error_valid)
        ),
        "source_available": source is not None,
        "source_counts": source_counts,
        "policy": "finite_unmasked_vx_vy_and_source_gt_zero_when_available",
    }
    return velocity_valid, error_valid, summary


def _filled_with_nan(values):
    """Convert a masked raster window to a numeric array with explicit NaNs."""
    return np.asarray(np.ma.asarray(values).astype("float64").filled(np.nan))


def _validate_velocity_raster_alignment(named_datasets):
    """Reject component rasters that do not share one pixel grid and CRS."""
    reference_name, reference = named_datasets[0]
    for name, dataset in named_datasets[1:]:
        mismatches = []
        if (dataset.width, dataset.height) != (reference.width, reference.height):
            mismatches.append("shape")
        if dataset.transform != reference.transform:
            mismatches.append("transform")
        if dataset.crs != reference.crs:
            mismatches.append("CRS")
        if mismatches:
            raise ValueError(
                f"Velocity raster {name} is not aligned with {reference_name}: "
                + ", ".join(mismatches)
            )


def get_windowed_velocity_file(
    name, outline, δ, modified_exists=False, *, return_validity=False
):
    """
    Get windowed velocity data from modified raster files.

    :param name: Name of the dataset.
    :param outline: GeoJSON outline data.
    :param δ: Buffer distance.
    :param modified_exists: Whether modified files exist.
    :param return_validity: Append SOURCE and revised validity diagnostics when
        True. The default preserves the historical ten-item return contract.
    :return: Historical velocity/error data, raster handles, window, and
        transform; optionally followed by revised validity diagnostics.
    """
    source_file = None
    if name is None:
        print("Reading velocity from measures database")
        velocity_filename = icepack.datasets.fetch_measures_antarctica() #name = "Antarctica_ice_velocity_2018_2019_1km_v01.1.nc")
        vx_file = rasterio.open(f"netcdf:{velocity_filename}:VX", "r")
        vy_file = rasterio.open(f"netcdf:{velocity_filename}:VY", "r")
        stdx_file = rasterio.open(f"netcdf:{velocity_filename}:ERRX", "r")
        stdy_file = rasterio.open(f"netcdf:{velocity_filename}:ERRY", "r")
        source_file = rasterio.open(f"netcdf:{velocity_filename}:SOURCE", "r")
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

    velocity_datasets = [
        ("VX", vx_file),
        ("VY", vy_file),
        ("ERRX", stdx_file),
        ("ERRY", stdy_file),
    ]
    if source_file is not None:
        velocity_datasets.append(("SOURCE", source_file))
    _validate_velocity_raster_alignment(velocity_datasets)

    xmin, xmax, ymin, ymax = get_min_max_coords(outline, δ)
    window = rasterio.windows.from_bounds(
        left=xmin,
        bottom=ymin,
        right=xmax,
        top=ymax,
        transform=vx_file.transform,
    ).round_lengths().round_offsets()
    transform = vx_file.window_transform(window)

    vx_masked = vx_file.read(indexes=1, window=window, masked=True)
    vy_masked = vy_file.read(indexes=1, window=window, masked=True)
    stdx_masked = stdx_file.read(indexes=1, window=window, masked=True)
    stdy_masked = stdy_file.read(indexes=1, window=window, masked=True)
    if source_file is not None:
        try:
            source_masked = source_file.read(indexes=1, window=window, masked=True)
        finally:
            source_file.close()
    else:
        source_masked = None

    valid_velocity, valid_error, validity_summary = build_velocity_observation_mask(
        vx_masked, vy_masked, stdx_masked, stdy_masked, source=source_masked
    )
    vx = _filled_with_nan(vx_masked)
    vy = _filled_with_nan(vy_masked)
    stdx = _filled_with_nan(stdx_masked)
    stdy = _filled_with_nan(stdy_masked)
    source = (
        _filled_with_nan(source_masked)
        if source_masked is not None
        else np.full(vx.shape, np.nan, dtype="float64")
    )

    # Close the raster files
    #vx_file.close()
    #vy_file.close()
    #stdx_file.close()
    #stdy_file.close()

    legacy_result = (
        vx,
        vx_file,
        vy,
        vy_file,
        stdx,
        stdx_file,
        stdy,
        stdy_file,
        window,
        transform,
    )
    if not return_validity:
        return legacy_result
    return legacy_result[:-2] + (
        source,
        valid_velocity,
        valid_error,
        validity_summary,
        window,
        transform,
    )


def _raster_pixel_coordinate(transform, i, j, coordinate_mode):
    if coordinate_mode == "legacy_corner":
        offset = 0.0
    elif coordinate_mode == "cell_center":
        offset = 0.5
    else:
        raise ValueError(
            f"Unsupported velocity coordinate_mode {coordinate_mode!r}; use "
            "'legacy_corner' or 'cell_center'."
        )
    return transform * (i + offset, j + offset)


def raster_window_dataarray(values, transform, *, name=None):
    """Label a north-up raster window at its true pixel-center coordinates.

    This is the coordinate-aware counterpart of Icepack's historical
    ``rasterio.DatasetReader`` sampler, which labels values at pixel corners.
    Rotation/shear is rejected because an x/y rectilinear DataArray cannot
    represent a rotated grid without silently changing its geometry.
    """
    array = np.asarray(values, dtype="float64")
    if array.ndim != 2:
        raise ValueError(f"Raster window must be two-dimensional; got {array.shape}.")
    if not (np.isclose(transform.b, 0.0) and np.isclose(transform.d, 0.0)):
        raise ValueError("Rotated/sheared raster windows are not supported.")
    height, width = array.shape
    x = transform.c + transform.a * (np.arange(width, dtype="float64") + 0.5)
    y = transform.f + transform.e * (np.arange(height, dtype="float64") + 0.5)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("Raster-center coordinates must be finite.")
    if width > 1 and not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0)):
        raise ValueError("Raster x coordinates must be strictly monotonic.")
    if height > 1 and not (np.all(np.diff(y) > 0) or np.all(np.diff(y) < 0)):
        raise ValueError("Raster y coordinates must be strictly monotonic.")
    return xr.DataArray(
        array,
        dims=("y", "x"),
        coords={"y": y, "x": x},
        name=name,
        attrs={"coordinate_mode": "cell_center"},
    )


def nearest_valid_velocity_samples(points, vx, vy, valid_mask, transform):
    """Return nearest genuinely observed VX/VY values for query points.

    This helper is for the finite diagnostic-solver initial/boundary field
    only. It does not modify the observation mesh or turn filled values into
    inversion, training, or evaluation observations.
    """
    queries = np.asarray(points, dtype="float64")
    vx_values = np.asarray(vx, dtype="float64")
    vy_values = np.asarray(vy, dtype="float64")
    valid = np.asarray(valid_mask, dtype=bool)
    if queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError("Velocity query points must have shape (n, 2).")
    if vx_values.shape != vy_values.shape or vx_values.shape != valid.shape:
        raise ValueError("VX, VY, and the valid mask must have one common shape.")
    if not np.isfinite(queries).all():
        raise ValueError("Velocity query coordinates must be finite.")
    rows, columns = np.nonzero(
        valid & np.isfinite(vx_values) & np.isfinite(vy_values)
    )
    if len(rows) == 0:
        raise ValueError("Cannot fill initial velocity without valid observations.")
    observed_points = np.column_stack(
        (
            transform.c + transform.a * (columns.astype("float64") + 0.5),
            transform.f + transform.e * (rows.astype("float64") + 0.5),
        )
    )
    distances, indices = cKDTree(observed_points).query(queries)
    source_rows = rows[indices]
    source_columns = columns[indices]
    values = np.column_stack(
        (
            vx_values[source_rows, source_columns],
            vy_values[source_rows, source_columns],
        )
    )
    if not np.isfinite(values).all():
        raise RuntimeError("Nearest-valid velocity lookup returned invalid values.")
    return values, np.asarray(distances, dtype="float64")


def select_velocity_observation_indices(
    mesh, window, transform, valid_velocity, *, coordinate_mode="legacy_corner"
):
    """Select valid raster indices that locate anywhere in the FE domain."""
    valid_velocity = np.asarray(valid_velocity, dtype=bool)
    expected_shape = (int(window.height), int(window.width))
    if valid_velocity.shape != expected_shape:
        raise ValueError(
            f"Velocity-validity mask has shape {valid_velocity.shape}; "
            f"expected {expected_shape}."
        )
    indices = np.asarray(
        [
            (i, j)
            for i in range(int(window.width))
            for j in range(int(window.height))
            if (
                valid_velocity[j, i]
                and mesh.locate_cell(
                    _raster_pixel_coordinate(transform, i, j, coordinate_mode)
                ) is not None
            )
        ],
        dtype="int64",
    ).reshape((-1, 2))
    if len(indices) == 0:
        raise ValueError("No valid velocity observations locate in the computational mesh.")
    return indices


def create_vertex_only_mesh_for_sparse_data(
    mesh, window, transform, stdx=None, *, valid_velocity=None,
    coordinate_mode="legacy_corner"
):
    """
    Create a vertex-only mesh for sparse data.

    :param mesh: Firedrake mesh.
    :param window: Rasterio window.
    :param transform: Rasterio transform.
    :param stdx: Legacy ERRX array. Used only when valid_velocity is omitted.
    :param valid_velocity: Explicit common mask from VX/VY/SOURCE. Preferred for
        revised runs.
    :param coordinate_mode: Historical calls use pixel corners; revised runs
        explicitly request true pixel centers.
    :return: Vertex-only mesh and indices.
    """
    if valid_velocity is None:
        if stdx is None:
            raise ValueError("Supply either legacy ERRX or a valid_velocity mask.")
        valid_velocity = np.asarray(stdx) > 0.0
    indices = select_velocity_observation_indices(
        mesh, window, transform, valid_velocity, coordinate_mode=coordinate_mode
    )
    xs = np.array(
        [
            _raster_pixel_coordinate(transform, i, j, coordinate_mode)
            for i, j in indices
        ]
    )
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
    
