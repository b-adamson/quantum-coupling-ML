"""
Optimized Lindbladian Latching (JAX-Vectorized Version)
Authors: pranavjv & b-vanstraaten
Date: 05/02/2026

Simulates charge dynamics in quantum dot arrays using an eigenbasis jump approach.
This module leverages JAX's jit-compilation and vmap-like kernels for high-speed
stochastic simulations of tunnel-coupled systems.
"""

from __future__ import annotations

from functools import lru_cache, partial
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np

# Internal qarray imports
from qarray_plus.DotArray.ground_state.free_energy import free_energy
from qarray_plus.DotArray.markov_jumps.markov_jumps import (
    cached_transition_graph_for_full_basis,
    simulate_eigenjump_scan_jax_fast,
    psi_to_charge_expectation,
)
from qarray_plus.DotArray.tunnel_coupled.tunnel_coupled_ground_state import make_H


@lru_cache(maxsize=None)
def _cached_basis_jax(n_dots: int, max_charge: int):
    """
    Computes and caches the charge state basis on the host to avoid
    redundant generation during iterative JIT calls.
    """
    states, src, dst, idx = cached_transition_graph_for_full_basis(
        n_dot=int(n_dots), max_charge=int(max_charge)
    )
    return (jnp.asarray(states, dtype=jnp.int32),
            jnp.asarray(src, dtype=jnp.int32),
            jnp.asarray(dst, dtype=jnp.int32),
            jnp.asarray(idx, dtype=jnp.int32))


@partial(jax.jit, static_argnames=("n_substeps", "per_row_reset"))
def _fast_eigenjump_kernel(
        vg_grid: jnp.ndarray,
        charge_states: jnp.ndarray,
        cdd: jnp.ndarray,
        cdg: jnp.ndarray,
        t: jnp.ndarray,
        beta: jnp.ndarray,
        dt: float,
        n_substeps: int,
        per_row_reset: bool,
        gamma_lead_vec: jnp.ndarray,
        gamma_interdot: float,
        key: jax.Array
):
    """
    Core JIT-compiled kernel. Performs Hamiltonian construction,
    eigen-decomposition, and stochastic jump simulation in a single fusion block.
    """
    ny, nx, n_gates = vg_grid.shape
    vg_flat = vg_grid.reshape(-1, n_gates)
    n_dots = charge_states.shape[1]

    # 1. Energy Calculation & Hamiltonian Construction
    # We calculate the free energy for all basis states across the voltage grid
    energies = free_energy(charge_states, cdd, cdg, vg_flat)
    h_matrices = make_H(charge_states, energies, t, n_dots)

    # 2. Spectral Decomposition
    # eigh is preferred for Hermitian/Symmetric Hamiltonians
    evals, evecs = jnp.linalg.eigh(h_matrices.astype(jnp.float32))

    # 3. Charge Sector Mapping
    # Determine the expected charge for each eigenstate to track transitions correctly
    n_basis = jnp.sum(charge_states, axis=-1).astype(jnp.float32)
    # Project charge expectations: <psi|N|psi>
    sector_labels = jnp.rint(
        jnp.einsum("pij, i -> pj", jnp.abs(evecs) ** 2, n_basis)
    ).astype(jnp.int32)

    # 4. Stochastic Trajectory Simulation
    psi_map = simulate_eigenjump_scan_jax_fast(
        evals_grid=evals.reshape(ny, nx, -1),
        evecs_grid=evecs.reshape(ny, nx, evals.shape[-1], -1).astype(jnp.complex64),
        sector_grid=sector_labels.reshape(ny, nx, -1),
        charge_states=charge_states,
        beta=beta,
        dt=dt,
        n_substeps=n_substeps,
        per_row_reset=per_row_reset,
        gamma_lead_vec=gamma_lead_vec,
        gamma_interdot=gamma_interdot,
        key0=key,
    )

    return psi_to_charge_expectation(psi_map, charge_states)


def markov_jumps(
        model,
        vg: Union[np.ndarray, jnp.ndarray],
        gamma_lead: Union[float, jnp.ndarray] = 1.0,
        gamma_interdot: float = 10.0,
        kT: float = 0.0,
        dt: float = 1.0,
        per_row_reset: bool = True,
        n_substeps: int = 1,
        max_charge: int = 2,
        seed: int = 0
) -> jnp.ndarray:
    """
    High-level entry point for Markov Jump simulations.

    Normalizes temperature relative to the system's charging energy and
    dispatches to the JIT-compiled eigenjump kernel.
    """
    vg = jnp.asarray(vg)
    orig_shape = vg.shape

    # Uniform input handling: Promote 1D/2D sweeps to 3D grid (ny, nx, n_gates)
    if vg.ndim == 1:
        vg = vg[None, None, :]
    elif vg.ndim == 2:
        vg = vg[None, :, :]

    n_dots = int(model.n_dots)

    # --- Validate rate parameters ---
    # The eigenjump head requires FINITE rates: a finite scalar gamma_interdot and a
    # finite gamma_lead (scalar or length-n_dot vector). The DotArray defaults
    # (gamma_lead=inf, gamma_interdot of shape (2, n_dot, n_dot)) are *latching*
    # conventions and are invalid here -- they otherwise yield NaNs (inf lead rate) or
    # a cryptic broadcasting error (rank-3 gamma_interdot). Fail early with guidance.
    gi = np.asarray(gamma_interdot, dtype=float)
    if gi.ndim != 0:
        raise ValueError(
            f"markov_jumps requires a scalar gamma_interdot, got shape {gi.shape}. "
            "The (2, n_dot, n_dot) form is the `latching` default; pass a finite "
            "scalar, e.g. DotArray(..., gamma_interdot=10.0)."
        )
    gl = np.asarray(gamma_lead, dtype=float)
    if gl.ndim > 1 or (gl.ndim == 1 and gl.shape != (n_dots,)):
        raise ValueError(
            f"markov_jumps requires gamma_lead to be a scalar or a length-{n_dots} "
            f"vector, got shape {gl.shape}."
        )
    if not (np.all(np.isfinite(gi)) and np.all(np.isfinite(gl))):
        raise ValueError(
            "markov_jumps requires FINITE gamma_lead and gamma_interdot. The DotArray "
            "default gamma_lead=inf is a `latching` convention; pass finite rates, e.g. "
            "DotArray(..., gamma_lead=jnp.array([1.0, 0.5]), gamma_interdot=10.0)."
        )

    # --- Thermal Scaling ---
    # Scaling kT by the mean charging energy ensures that a kT of 1.0 has
    # a consistent physical meaning across different capacitance models.
    cdd_inv = jnp.linalg.inv(model.cdd)
    # Charging energy E_c = 1 / (2 * C_Sigma) ~ 0.5 * diag(cdd_inv); it scales as the
    # inverse capacitance, so this must multiply (not divide) by diag(cdd_inv).
    charging_energies = 0.5 * jnp.diag(cdd_inv)
    mean_e_c = jnp.mean(charging_energies)
    effective_kt = kT * mean_e_c

    # Handle zero-temperature limit safely for division
    beta = jnp.where(effective_kt > 0, 1.0 / jnp.maximum(effective_kt, 1e-12), jnp.inf)

    # --- Pre-flight: Basis and Rates ---
    charge_states, _, _, _ = _cached_basis_jax(n_dots, max_charge)

    if jnp.isscalar(gamma_lead):
        gamma_lead_vec = jnp.full((n_dots,), float(gamma_lead), dtype=jnp.float32)
    else:
        gamma_lead_vec = jnp.asarray(gamma_lead, dtype=jnp.float32)

    # --- Execution ---
    n_latched = _fast_eigenjump_kernel(
        vg_grid=vg,
        charge_states=charge_states,
        cdd=jnp.asarray(model.cdd),
        cdg=jnp.asarray(model.cdg),
        t=jnp.asarray(model.t),
        beta=beta,
        dt=dt,
        n_substeps=n_substeps,
        per_row_reset=per_row_reset,
        gamma_lead_vec=gamma_lead_vec,
        gamma_interdot=gamma_interdot,
        key=jax.random.PRNGKey(seed)
    )

    # Restore original batch/grid dimensions
    return n_latched.reshape((*orig_shape[:-1], n_dots))
