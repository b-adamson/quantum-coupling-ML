"""
Author: b-vanstraaten
Date: 05/12/2025
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.nn import sigmoid


def fermi_function(E, kT):
    finite_temp = sigmoid(-E / kT)
    zero_temp_limit = jnp.where(E < 0, 1.0, 0.0)
    # Blend the sigmoid with the zero-temperature limit to ensure correct behavior at low temperatures.
    return jnp.where(kT > 0, finite_temp, zero_temp_limit)


@partial(jax.jit, static_argnames=["N"])
def interdots(N: int):
    # Create index grid
    ii, jj = jnp.indices((N, N))

    # Boolean mask
    mask = ii != jj

    # Convert mask → integer indices (shape is statically known)
    idx = jnp.nonzero(mask, size=N * (N - 1))

    ii = idx[0]
    jj = idx[1]

    interdots = jax.nn.one_hot(ii, N) - jax.nn.one_hot(jj, N)
    dots = jnp.stack([ii, jj], axis=-1)
    return dots, interdots.astype(int)


def probability_of_transition(Gamma, t):
    return 1 - jnp.exp(-Gamma * t)
