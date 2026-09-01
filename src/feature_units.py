"""Unit-safe definitions for derived ML predictors.

Icepack uses metres, years, and megapascals.  The constants below reproduce
``icepack.constants.ice_density * icepack.constants.gravity`` without making
the portable CUDA training code depend on Firedrake or Icepack.
"""

import numpy as np


ICE_DENSITY_KG_M3 = 917.0
GRAVITY_M_S2 = 9.81
PASCALS_PER_MEGAPASCAL = 1.0e6

DRIVING_STRESS_UNIT = "MPa"
DRIVING_STRESS_MPA_PER_M = (
    ICE_DENSITY_KG_M3 * GRAVITY_M_S2 / PASCALS_PER_MEGAPASCAL
)

SURFACE_SLOPE_FLOOR = 1.5e-4
BED_SLOPE_FLOOR = 5.4e-3


def driving_stress_mpa(ice_thickness_m, surface_slope):
    """Return ``rho_i * g * h * |grad(s)|`` in Icepack-native MPa."""

    return DRIVING_STRESS_MPA_PER_M * ice_thickness_m * surface_slope


def stabilized_gradient_alignment(
    bed_gradient_x,
    bed_gradient_y,
    surface_gradient_x,
    surface_gradient_y,
    *,
    bed_floor=BED_SLOPE_FLOOR,
    surface_floor=SURFACE_SLOPE_FLOOR,
):
    """Return the bounded, soft-normalized bed--surface gradient cosine.

    The nonzero slope floors make the feature finite and smoothly send it to
    zero when either gradient is effectively flat. Inputs may be scalars or
    array-like values.
    """

    bed_x = np.asarray(bed_gradient_x, dtype="float64")
    bed_y = np.asarray(bed_gradient_y, dtype="float64")
    surface_x = np.asarray(surface_gradient_x, dtype="float64")
    surface_y = np.asarray(surface_gradient_y, dtype="float64")
    # Evaluate as normalized components rather than forming the product of two
    # squared magnitudes. This is algebraically identical but remains stable
    # for very large finite test values.
    bed_denominator = np.hypot(np.hypot(bed_x, bed_y), float(bed_floor))
    surface_denominator = np.hypot(
        np.hypot(surface_x, surface_y), float(surface_floor)
    )
    alignment = (
        (bed_x / bed_denominator) * (surface_x / surface_denominator)
        + (bed_y / bed_denominator) * (surface_y / surface_denominator)
    )
    return np.clip(alignment, -1.0, 1.0)
