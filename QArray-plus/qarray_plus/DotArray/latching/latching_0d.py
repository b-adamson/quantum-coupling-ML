"""
Author: b-vanstraaten
Date: 05/12/2025
"""

from functools import partial

import jax
import jax.numpy as jnp

from qarray_plus.DotArray.ground_state.free_energy import _free_energy
from qarray_plus.DotArray.latching.functions import (
    interdots,
    fermi_function,
    probability_of_transition,
)


@jax.jit
def latching_0d(
        n_current,
        cdd,
        cdg,
        lead_tunnel_rates,
        interdot_tunnel_rates,
        t_int: float,
        kT: float,
        vg=None,
        key=None,
):
    def interdot_tunnel_rates_fn(dots, n_inter):
        is_spin_blocked = jnp.all(n_inter % 2 == 1, axis=-1).astype(int)
        return interdot_tunnel_rates[is_spin_blocked, dots[:, 0], dots[:, 1]]

    # Normalize kT relative to the mean charging energy.
    # Here, kT = 1 corresponds to a thermal energy equal to the average dot charging energy.
    cdd_inv = jnp.linalg.inv(cdd)
    # Charging energy E_c = 1 / (2 * C_Sigma), i.e. inversely proportional to capacitance.
    # In the Maxwell convention C_Sigma,i = cdd_ii, so diag(cdd_inv) ~ 1 / C_Sigma and
    # E_c ~ 0.5 * diag(cdd_inv). (Using `/` here would scale as capacitance, not 1/C.)
    charging_energies = 0.5 * jnp.diag(cdd_inv)
    mean_charging_energy = charging_energies.mean()
    # Rescale kT so that kT=1 corresponds to mean charging energy.
    kT *= mean_charging_energy

    n_dots = n_current.shape[-1]

    # Elementary charge changes
    lead_loading = jnp.eye(n_dots, dtype=int)
    lead_unloading = -jnp.eye(n_dots, dtype=int)
    dots, inter = interdots(n_dots)

    # Free energies ----------------------------------------------------------
    F0 = _free_energy(n_current[None, :], cdd, cdg, vg)
    F_load = _free_energy(n_current + lead_loading, cdd, cdg, vg)
    F_unload = _free_energy(n_current + lead_unloading, cdd, cdg, vg)
    F_inter = _free_energy(n_current + inter, cdd, cdg, vg)

    # Computing the probability of lead transitions --------------------------
    # Fermi factors ----------------------------------------------------------
    f_load = fermi_function(F_load - F0, kT)
    f_unload = fermi_function(F_unload - F0, kT)
    f_inter = fermi_function(F_inter - F0, kT)

    # Lead tunnel rates ------------------------------------------------------
    Gamma0 = lead_tunnel_rates
    Gamma_load = Gamma0 * f_load
    Gamma_unload = Gamma0 * f_unload

    # Select the occupations of the relevant dots
    n_inter = n_current[dots]
    Gamma_inter = interdot_tunnel_rates_fn(dots, n_inter) * f_inter
    Gammas = jnp.concatenate([Gamma_load, Gamma_unload, Gamma_inter], axis=0)

    # setting Gammas which are nan to zero
    Gammas = jnp.where(jnp.isnan(Gammas), 0.0, Gammas)

    probs = probability_of_transition(Gammas, t_int)
    delta = jnp.concatenate([lead_loading, lead_unloading, inter], axis=0)

    # Randomise the order to avoid bias when multiple transitions are possible
    key, perm_key, bern_key = jax.random.split(key, 3)
    perm = jax.random.permutation(perm_key, probs.shape[0])
    probs = probs[perm]
    delta = delta[perm]

    flips = jax.random.bernoulli(bern_key, p=probs)
    # Detect if any flip occurred
    has_flip = flips.any()
    # # Index of first True in flips, argmax returns the *first* index because booleans convert to {0,1}
    first_idx = jnp.argmax(flips)
    # Select correct delta[i]
    chosen_delta = delta[first_idx]

    # Apply update only if a flip happened
    n_next = jnp.where(has_flip, n_current + chosen_delta, n_current)
    return n_next, bern_key


@partial(
    jax.jit,
    static_argnames=(
            "cdd_fn",
            "cdg_fn",
            "lead_tunnel_rates_fn",
            "interdot_tunnel_rates_fn",
    ),
)
def latching_0d_fn(
        n_current,
        cdd_fn,
        cdg_fn,
        lead_tunnel_rates_fn,
        interdot_tunnel_rates_fn,
        t_int: float,
        kT: float,
        vg=None,
        key=None,
):
    return latching_0d(
        n_current,
        cdd_fn(vg),
        cdg_fn(vg),
        lead_tunnel_rates_fn(vg),
        interdot_tunnel_rates_fn(vg),
        t_int,
        kT,
        vg,
        key,
    )
