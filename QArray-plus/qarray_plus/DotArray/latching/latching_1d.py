"""
Author: b-vanstraaten
Date: 05/12/2025
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax, random

from qarray_plus.DotArray.latching.latching_0d import (
    latching_0d_fn,
    latching_0d,
)


# =============================================================================
# Functional version (cdd_fn, cdg_fn, rate_fns)
# =============================================================================
@partial(
    jax.jit,
    static_argnames=(
            "cdd_fn",
            "cdg_fn",
            "lead_tunnel_rates_fn",
            "interdot_tunnel_rates_fn",
    ),
)
def _latching_1d_fn(
        n0,
        cdd_fn,
        cdg_fn,
        lead_tunnel_rates_fn,
        interdot_tunnel_rates_fn,
        vg,
        t_int=1.0,
        kT=1.0,
        N_recursion=1,
        key=None,
):
    """
    1D latching evolution using functional inputs.
    vg: array shaped (Ny, ...)
    Returns:
        n_avg_traj: (Ny, n_states)
        key_out: PRNGKey
    """

    _, key = random.split(key)

    # -------------------------------
    # Evolution for a single vg row
    # -------------------------------
    def N_steps(n_current, key, vg_row):
        def step_fn(_, carry):
            n, k, acc = carry
            k1, k2 = random.split(k)

            n_next, _ = latching_0d_fn(
                n,
                cdd_fn=cdd_fn,
                cdg_fn=cdg_fn,
                lead_tunnel_rates_fn=lead_tunnel_rates_fn,
                interdot_tunnel_rates_fn=interdot_tunnel_rates_fn,
                vg=vg_row,
                t_int=t_int / N_recursion,
                kT=kT,
                key=k1,
            )

            return (n_next, k2, acc + n_next)

        sum_init = jnp.zeros_like(n_current)
        n_last, key_out, acc_states = lax.fori_loop(
            0, N_recursion, step_fn, (n_current, key, sum_init)
        )

        return n_last, key_out, acc_states / N_recursion

    # -------------------------------
    # Scanning across sweep dimension
    # -------------------------------
    def scan_body(carry, vg_row):
        n_current, key = carry
        n_last, key_next, avg_state = N_steps(n_current, key, vg_row)
        return (n_last, key_next), avg_state

    (_, key_out), n_avg_traj = lax.scan(scan_body, (n0, key), vg)
    return n_avg_traj, key_out


# Vectorized across batch
latching_1d_fn = jax.vmap(
    _latching_1d_fn,
    in_axes=(0, None, None, None, None, 0, None, None, None, 0),
)


# =============================================================================
# Array version (cdd, cdg, precomputed rates)
# =============================================================================
@jax.jit
def _latching_1d(
        n0,
        cdd,
        cdg,
        lead_tunnel_rates,
        interdot_tunnel_rates,
        vg,
        t_int=1.0,
        kT=1.0,
        N_recursion=1,
        key=None,
):
    """
    1D latching evolution using array-valued inputs.
    vg: array shaped (Ny, ...)
    Returns:
        n_avg_traj: (Ny, n_states)
        key_out: PRNGKey
    """

    _, key = random.split(key)

    # -------------------------------
    # Evolution for a single vg row
    # -------------------------------
    def N_steps(n_current, key, vg_row):
        def step_fn(_, carry):
            n, k, acc = carry
            k1, k2 = random.split(k)

            n_next, _ = latching_0d(
                n,
                cdd=cdd,
                cdg=cdg,
                lead_tunnel_rates=lead_tunnel_rates,
                interdot_tunnel_rates=interdot_tunnel_rates,
                vg=vg_row,
                t_int=t_int / N_recursion,
                kT=kT,
                key=k1,
            )

            return (n_next, k2, acc + n_next)

        sum_init = jnp.zeros_like(n_current)
        n_last, key_out, acc_states = lax.fori_loop(
            0, N_recursion, step_fn, (n_current, key, sum_init)
        )

        return n_last, key_out, acc_states / N_recursion

    # -------------------------------
    # Scanning across sweep dimension
    # -------------------------------
    def scan_body(carry, vg_row):
        n_current, key = carry
        n_last, key_next, avg_state = N_steps(n_current, key, vg_row)
        return (n_last, key_next), avg_state

    (_, key_out), n_avg_traj = lax.scan(scan_body, (n0, key), vg)
    return n_avg_traj, key_out


# Vectorized across batch
latching_1d = jax.vmap(
    _latching_1d,
    in_axes=(0, None, None, None, None, 0, None, None, None, 0),
)
