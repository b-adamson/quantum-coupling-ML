"""
Author: b-vanstraaten
Date: 21/11/2025
"""

from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def _free_energy(n, cdd, cdg, vg):
    # vd = n - Cdg @ vg
    vd = n - cdg @ vg  # shape (..., D)

    # Use Cholesky for maximum speed & stability
    L = jax.scipy.linalg.cho_factor(cdd, lower=True)
    x = jax.scipy.linalg.cho_solve(L, vd.T).T  # shape (..., D)

    # Quadratic form: vdᵀ Cdd⁻¹ vd = sum(vd * x)
    F = jnp.sum(vd * x, axis=-1)

    # Penalize negative charges
    F = jnp.where(jnp.any(n < 0, axis=-1), jnp.inf, F)
    return F


free_energy = jax.vmap(_free_energy, in_axes=(None, None, None, 0))


@partial(jax.jit, static_argnames=("cdd_fn", "cdg_fn"))
def _free_energy_fn(n, cdd_fn, cdg_fn, vg):
    cdd = cdd_fn(vg)
    cdg = cdg_fn(vg)
    return _free_energy(n, cdd, cdg, vg)


free_energy_fn = jax.vmap(_free_energy_fn, in_axes=(None, None, None, 0))
