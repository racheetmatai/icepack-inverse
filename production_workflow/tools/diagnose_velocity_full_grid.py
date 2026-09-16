#!/usr/bin/env python3
"""Diagnose centered MEaSUREs interpolation on the frozen FE spaces."""

import json

import firedrake
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


MESH = "/home/firedrake/icepack/icepack-inverse/amundsen.msh"
VELOCITY = "/home/firedrake/.cache/icepack/antarctic_ice_vel_phase_map_v01.nc"
DIRICHLET = [1, 3, 5, 6, 7, 8, 9, 10, 11]


def sample(data, points, method):
    x = xr.DataArray(points[:, 0], dims="z")
    y = xr.DataArray(points[:, 1], dims="z")
    return data.interp(x=x, y=y, method=method).to_numpy()


mesh = firedrake.Mesh(MESH)
Q = firedrake.FunctionSpace(mesh, "CG", 2)
V = firedrake.VectorFunctionSpace(mesh, "CG", 2)
Xq = firedrake.interpolate(mesh.coordinates, firedrake.VectorFunctionSpace(mesh, "CG", 2)).dat.data_ro[:, :2]
Xv = firedrake.interpolate(mesh.coordinates, V).dat.data_ro[:, :2]

with xr.open_dataset(VELOCITY) as dataset:
    xmin, ymin = Xq.min(axis=0) - 1000.0
    xmax, ymax = Xq.max(axis=0) + 1000.0
    x_values = dataset.x.to_numpy()
    y_values = dataset.y.to_numpy()
    x_indices = np.flatnonzero((x_values >= xmin) & (x_values <= xmax))
    y_indices = np.flatnonzero((y_values >= ymin) & (y_values <= ymax))
    subset = dataset.isel(x=x_indices, y=y_indices)
    vx_grid = subset["VX"].to_numpy()
    vy_grid = subset["VY"].to_numpy()
    source_grid = subset["SOURCE"].to_numpy()
    grid_valid = (
        np.isfinite(vx_grid) & np.isfinite(vy_grid)
        & np.isfinite(source_grid) & (source_grid > 0)
    )
    rows, columns = np.nonzero(grid_valid)
    valid_points = np.column_stack(
        (subset.x.to_numpy()[columns], subset.y.to_numpy()[rows])
    )
    tree = cKDTree(valid_points)
    output = {}
    for label, points in (("Q", Xq), ("V", Xv)):
        linear_x = sample(dataset["VX"], points, "linear")
        linear_y = sample(dataset["VY"], points, "linear")
        nearest_x = sample(dataset["VX"], points, "nearest")
        nearest_y = sample(dataset["VY"], points, "nearest")
        linear_valid = np.isfinite(linear_x) & np.isfinite(linear_y)
        nearest_valid = np.isfinite(nearest_x) & np.isfinite(nearest_y)
        output[label] = {
            "count": int(len(points)),
            "linear_valid": int(linear_valid.sum()),
            "linear_invalid": int((~linear_valid).sum()),
            "nearest_valid": int(nearest_valid.sum()),
            "nearest_invalid": int((~nearest_valid).sum()),
            "linear_invalid_nearest_valid": int((~linear_valid & nearest_valid).sum()),
        }
        distances, _ = tree.query(points[~linear_valid])
        output[label]["linear_invalid_distance_to_valid_m"] = {
            "minimum": float(distances.min()),
            "median": float(np.median(distances)),
            "maximum": float(distances.max()),
        }
        if label == "V":
            nodes = firedrake.DirichletBC(
                V, firedrake.Constant((0.0, 0.0)), DIRICHLET
            ).nodes
            output[label]["dirichlet_count"] = int(len(nodes))
            output[label]["dirichlet_linear_invalid"] = int(
                (~linear_valid[nodes]).sum()
            )
            output[label]["dirichlet_nearest_invalid"] = int(
                (~nearest_valid[nodes]).sum()
            )
            boundary_invalid = nodes[~linear_valid[nodes]]
            boundary_distances, _ = tree.query(points[boundary_invalid])
            output[label]["dirichlet_invalid_distance_to_valid_m"] = {
                "minimum": float(boundary_distances.min()),
                "median": float(np.median(boundary_distances)),
                "maximum": float(boundary_distances.max()),
            }
            output[label]["dirichlet_invalid_coordinates"] = points[
                boundary_invalid
            ].tolist()

print(json.dumps(output, indent=2, sort_keys=True))
