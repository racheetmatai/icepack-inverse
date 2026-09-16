"""Unit-safe definitions for derived ML predictors.

Icepack uses metres, years, and megapascals.  The constants below reproduce
``icepack.constants.ice_density * icepack.constants.gravity`` without making
the portable CUDA training code depend on Firedrake or Icepack.
"""


ICE_DENSITY_KG_M3 = 917.0
GRAVITY_M_S2 = 9.81
PASCALS_PER_MEGAPASCAL = 1.0e6

DRIVING_STRESS_UNIT = "MPa"
DRIVING_STRESS_MPA_PER_M = (
    ICE_DENSITY_KG_M3 * GRAVITY_M_S2 / PASCALS_PER_MEGAPASCAL
)


def driving_stress_mpa(ice_thickness_m, surface_slope):
    """Return ``rho_i * g * h * |grad(s)|`` in Icepack-native MPa."""

    return DRIVING_STRESS_MPA_PER_M * ice_thickness_m * surface_slope
