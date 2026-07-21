"""
Author: b-vanstraaten
Date: 17/09/2025

Ground-state solvers for open quantum dot arrays.

This module provides:
- Ground-state solver using explicit Cdd and Cdg matrices.
- Ground-state solver using Cdd(vg) and Cdg(vg) callable functions.
- Continuous and approximate integer ground states.
- Fully JIT-compiled and VMAP-vectorized implementations.

All functions here remain side-effect-free and JAX-optimal.
"""

import jax
import jax.numpy as jnp

from qarray_plus.DotArray.ground_state.charge_configurations import (
    _charge_configurations,
)
from qarray_plus.DotArray.ground_state.free_energy import _free_energy
from qarray_plus.DotArray.ground_state.numberical_solver import (
    _numerical_solver_open,
)


# ========================================================================
#  Ground-state solver with explicit Cdd and Cdg matrices
# ========================================================================


@jax.jit
def _ground_state_open(cdd, cdg, vg):
    """
    Compute the exact integer ground state for a single gate-voltage vector vg,
    given explicit capacitance matrices Cdd and Cdg.

    Steps:
        1. Compute continuous charge distribution via the convex QP solver.
        2. Generate nearest integer charge configurations.
        3. Evaluate free energy for each candidate.
        4. Return the configuration with minimal free energy.

    Args:
        cdd: (D, D) dot-to-dot capacitance matrix.
        cdg: (D, D) gate-to-dot capacitance matrix.
        vg:  (D,)    gate voltage vector.

    Returns:
        (D,) integer charge configuration at minimal free energy.
    """
    # Step 1: continuous solution
    n_cont = _numerical_solver_open(cdd=cdd, cdg=cdg, vg=vg)

    # Step 2: enumerate nearest integer configurations
    n_nearest = _charge_configurations(n_cont, perturbations=jnp.array([0, 1]))

    # Step 3: compute free energies of candidates
    F = _free_energy(n_nearest, cdd, cdg, vg)

    # Step 4: pick lowest-energy configuration
    idx = jnp.argmin(F)
    return n_nearest[idx, :]


# VMAP over vg (batched gate voltages)
ground_state_open = jax.vmap(_ground_state_open, in_axes=(None, None, 0))


# ========================================================================
#  Ground-state solver with callable Cdd(vg) and Cdg(vg)
# ========================================================================
def _ground_state_open_fn(cdd_fn, cdg_fn, vg):
    """
    Same as `_ground_state_open`, but where Cdd and Cdg are functions of vg.
    This version is JIT-compiled with cdd_fn and cdg_fn as static arguments.
    """
    return _ground_state_open(cdd_fn(vg), cdg_fn(vg), vg)


ground_state_open_fn = jax.vmap(_ground_state_open_fn, in_axes=(None, None, 0))


# ========================================================================
#  Approximate ground state = round(continuous solution)
# ========================================================================


@jax.jit
def _approximate_ground_state_open(cdd, cdg, vg):
    """
    Faster but approximate ground state:
    Just returns round(n_cont), without evaluating free energy or checking consistency.
    """
    n_cont = _numerical_solver_open(cdd=cdd, cdg=cdg, vg=vg)
    return jnp.round(n_cont)


approximate_ground_state_open = jax.vmap(
    _approximate_ground_state_open, in_axes=(None, None, 0)
)


# ========================================================================
#  Approximate ground state with Cdd(vg), Cdg(vg)
# ========================================================================


def _approximate_ground_state_open_fn(cdd_fn, cdg_fn, vg):
    """
    Approximate ground state version using callable capacitance matrices.
    """
    return _approximate_ground_state_open(cdd_fn(vg), cdg_fn(vg), vg)


approximate_ground_state_open_fn = jax.vmap(
    _approximate_ground_state_open_fn, in_axes=(None, None, 0)
)
