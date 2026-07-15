"""
Author: b-vanstraaten
Date: 20/11/2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as jnp

from qarray_plus.DotArray.utils import (
    GateVoltageDependantCallable,
    _ensure_callable_and_validate,
)


def pink_noise_axis(shape, axis):
    """
    Pink (1/f) noise correlated only along one axis.
    Other axes remain white.
    """
    # White noise first
    white = jnp.random.normal(size=jnp.prod(shape)).reshape(shape)

    # Move target axis to end so FFT is clean
    white_t = jnp.moveaxis(white, axis, -1)

    L = white_t.shape[-1]

    # FFT along the last axis
    f = jnp.fft.rfft(white_t, axis=-1)

    # Frequencies
    freqs = jnp.fft.rfftfreq(L)
    freqs = jnp.where(freqs == 0, 1, freqs)

    # Apply 1/sqrt(f) scaling
    f = f / jnp.sqrt(freqs)[None]

    # Inverse FFT
    pink_t = jnp.fft.irfft(f, n=L, axis=-1)

    # Normalize
    pink_t = pink_t / jnp.std(pink_t)

    # Move axis back
    pink = jnp.moveaxis(pink_t, -1, axis)
    return (pink - pink.mean()) / pink.std()


def lorentzian(x, x0, gamma):
    return jnp.reciprocal((((x - x0) / gamma) ** 2 + 1))


@dataclass
class ChargeSensor:
    n_dots: int
    n_gates: int
    n_sensor: int

    # the dot to dot capacitive coupling in the Maxwell format of shape (n_dot, n_dot)
    csd: Union[GateVoltageDependantCallable, jnp.ndarray]

    # the gate to dot capacitive coupling in the Maxwell format of shape (n_dot, n_gate)
    csg: Union[GateVoltageDependantCallable, jnp.ndarray]

    # Lorentzian comb parameters
    width: float = 0.1
    spacing: float = 1
    n_peaks: float = 4  # matches your loop range 1..4

    pink_noise_std: float = 0
    white_noise_std: float = 0

    def v_sensor(self, vg: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
        """Compute sensor voltages based on dot occupations."""

        cds = self.csd(vg) if callable(self.csd) else self.csd
        cgs = self.csg(vg) if callable(self.csg) else self.csg
        # V_sensor = cds · n - cgs · vg.  Maxwell convention: csg is supplied
        # pre-negated (csg = -C_sg,physical), mirroring cdg, so the minus sign makes
        # the physical gate coupling add: cds·n + |csg|·vg. All examples pass a
        # negated csg accordingly.
        v_sensor = jnp.einsum("ij,...j->...i", cds, n) - jnp.einsum(
            "ij,...j->...i", cgs, vg
        )
        return v_sensor

    def __post_init__(self):
        """Validate and wrap all parameter fields as callables."""

        # Wrap constant arrays as functions and validate shapes.
        self.csd = _ensure_callable_and_validate(
            self.csd,
            expected_shape=(self.n_sensor, self.n_dots),
            name="cds",
            n_gates=self.n_gates,
        )

        self.csg = _ensure_callable_and_validate(
            self.csg,
            expected_shape=(self.n_sensor, self.n_gates),
            name="cgs",
            n_gates=self.n_gates,
        )

    def simulate(self, vg, n):
        v_sensor = self.v_sensor(vg, n)
        # Peak positions: 0, ±n*spacing for n = 1..n_peaks
        centers = jnp.arange(-self.n_peaks, self.n_peaks + 1) * self.spacing

        # Vectorized Lorentzian comb: shape (..., n_peaks*2+1)
        z = lorentzian(v_sensor[..., None] % 1, x0=centers, gamma=self.width)
        # Sum across peaks
        z = jnp.sum(z, axis=-1)

        pink_noise = self.pink_noise_std * pink_noise_axis(z.shape, 1)
        white_noise = self.white_noise_std * jnp.random.normal(size=z.shape)
        return z + pink_noise + white_noise
