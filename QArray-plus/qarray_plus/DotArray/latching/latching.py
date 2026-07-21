"""
Author: b-vanstraaten
Date: 17/11/2025
"""

from qarray_plus.DotArray.ground_state.ground_state import (
    ground_state_open,
    approximate_ground_state_open,
    ground_state_open_fn,
    approximate_ground_state_open_fn,
)

from qarray_plus.DotArray.latching.latching_1d import (
    latching_1d,
    latching_1d_fn,
)


# -----------------------------------------------------------------------------
# Functional version (expects cdd_fn, cdg_fn, rate_fn(vg) callables)
# -----------------------------------------------------------------------------
def latching_2d_fn(
        cdd_fn,
        cdg_fn,
        lead_tunnel_rates_fn,
        interdot_tunnel_rates_fn,
        vg,
        t_int=1.0,
        kT=1.0,
        N_recursion=1,
        key=None,
        approximate=True,
):
    """
    2D latching simulation using functional forms of the system matrices
    and tunnel rates. The initial state is computed from the first voltage
    column, then a 1D sweep is applied along the second dimension.
    """

    vg_col0 = vg[:, 0, :]
    ground_state_fn = (
        approximate_ground_state_open_fn if approximate else ground_state_open_fn
    )

    # Initial charge configuration
    n0 = ground_state_fn(cdd_fn, cdg_fn, vg_col0)

    # Run the latching simulation over the second dimension
    return latching_1d_fn(
        n0,
        cdd_fn,
        cdg_fn,
        lead_tunnel_rates_fn,
        interdot_tunnel_rates_fn,
        vg,
        t_int,
        kT,
        N_recursion,
        key,
    )


# -----------------------------------------------------------------------------
# Array version (expects fully evaluated cdd, cdg, rates)
# -----------------------------------------------------------------------------
def latching_2d(
        cdd,
        cdg,
        lead_tunnel_rates,
        interdot_tunnel_rates,
        vg,
        t_int=1.0,
        kT=1.0,
        N_recursion=1,
        key=None,
        approximate=True,
):
    """
    2D latching simulation using *array-valued* inputs. This version bypasses
    the functional interface and directly uses array data for the capacitance
    matrix, gate couplings, and tunnel rates.
    """

    vg_col0 = vg[:, 0, :]
    ground_state_fn = (
        approximate_ground_state_open if approximate else ground_state_open
    )

    # Initial charge configuration
    n0 = ground_state_fn(cdd, cdg, vg_col0)

    # Run the latching simulation over the second dimension
    return latching_1d(
        n0,
        cdd,
        cdg,
        lead_tunnel_rates,
        interdot_tunnel_rates,
        vg,
        t_int,
        kT,
        N_recursion,
        key,
    )
