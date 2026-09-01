"""Audited raster specifications for the post-review data path.

These specifications are configuration only. Importing this module does not
run an inversion or create a training dataset.
"""


REVISED_GEOPHYSICS_INPUTS = [
    {
        "path": "data/geophysics/ADMAP_2S_epsg3031_gdal.nc",
        "variable": "z",
        "expected_crs": 3031,
        "method": "linear",
    },
    # Deliberately omitted. Bouguer anomaly was not a paper predictor and is
    # neither a revised predictor nor an eligibility filter. Keep the None
    # placeholder to preserve the legacy eight-slot importer API.
    None,
    {
        "path": "data/geophysics/GeothermalHeatFlux_5km.tif",
        "method": "linear",
        "coordinate_mode": "cell_center",
    },
    {
        "path": "data/geophysics/ALBMAP_SurfaceAirTemperature_5km.tif",
        "method": "linear",
        "coordinate_mode": "cell_center",
    },
    {
        "path": "data/geophysics/AntGG2021_Gravity_disturbance_at-surface.nc",
        "variable": "grav_dist",
        "expected_crs": 3031,
        "method": "linear",
    },
    # Deliberately omitted from revised exports. The historical raster has
    # unresolved sentinel/extreme values and was not used by the paper models.
    # Keep this None placeholder to preserve the legacy eight-slot importer API.
    None,
    {
        "path": "data/geophysics/Englacial_temp_Pattyn_2013.tif",
        "method": "linear",
        "coordinate_mode": "cell_center",
    },
    {
        "path": "data/geophysics/bed_class_oct29_2025.tif",
        "method": "nearest",
        "coordinate_mode": "cell_center",
        # This exact legacy auxiliary file has no embedded CRS. Production
        # preflight hash-locks it before this explicit EPSG:3031 assumption is
        # permitted; the generic loader otherwise rejects missing CRS.
        "expected_crs": 3031,
        "assumed_crs": 3031,
    },
]


REVISED_ML_PREDICTOR_POLICY = {
    "units": {
        "h": "m",
        "mag_h": "dimensionless",
        "mag_s": "dimensionless",
        "mag_b": "dimensionless",
        "driving_stress": "MPa",
    },
    "allowed": [
        "s",
        "b",
        "h",
        "mag_h",
        "mag_s",
        "mag_b",
        "driving_stress",
        "surface_air_temp",
        "heatflux",
        "gravity_disturbance",
        "mag_anomaly",
        "cos_theta_bs",
    ],
    "excluded": [
        "boug_anomaly",
        "x_velocity",
        "y_velocity",
        "vel_mag",
        "velocity_direction",
        "snow_accumulation",
        "bed_class",
        "bedmachine_source",
        "bedmachine_errbed",
    ],
}


REVISED_CSV_EXPORT_POLICY = {
    "auxiliary": ["bed_class", "bedmachine_source", "bedmachine_errbed"],
    "excluded": ["snow_accumulation", "boug_anomaly"],
}
