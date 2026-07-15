"""
Author: b-vanstraaten
Date: 05/12/2025
"""

import jax
import jax.numpy as jnp


@jax.jit
def _charge_configurations(
    n_cont: jnp.ndarray, perturbations=jnp.array([0, 1])
) -> jnp.ndarray:
    """
    Generate all tunnel-accessible charge configurations around a continuous solution.

    Parameters
    ----------
    n_cont : jnp.ndarray
        Continuous charge solution of shape (..., n_dot)

    Returns
    -------
    jnp.ndarray
        Integer charge configurations of shape (4**n_dot, n_dot),
        with negative occupations shifted to zero.
    """

    N = perturbations.size

    # Number of dots
    n_dot = n_cont.shape[-1]

    # Build a repeated array so meshgrid broadcasts correctly
    args = jnp.broadcast_to(perturbations, (n_dot, N))

    # All combinations of perturbations
    configs = jnp.stack(jnp.meshgrid(*args), axis=-1).reshape(N**n_dot, n_dot)
    n = jnp.floor(configs + n_cont[jnp.newaxis, :]).astype(int)

    # Ensure no configuration contains negative charge
    n_min = n.min(axis=0)
    n = n - (n_min * (n_min < 0))
    return n
