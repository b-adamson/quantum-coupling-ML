"""Quantum dot array class and related utilities."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Union, Dict, Optional

import jax
import jax.numpy as jnp

from qarray_plus.ChargeSensor.charge_sensor_decorator import charge_sensor
from qarray_plus.DotArray.ground_state.ground_state import (
    ground_state_open_fn,
    ground_state_open,
    _ground_state_open,
)
from qarray_plus.DotArray.ground_state.numberical_solver import (
    numerical_solver_open_fn,
    numerical_solver_open,
)
from qarray_plus.DotArray.latching.latching import latching_2d, latching_2d_fn
from qarray_plus.DotArray.latching.latching_1d import _latching_1d
from qarray_plus.DotArray.markov_jumps import markov_jumps
from qarray_plus.DotArray.tunnel_coupled.tunnel_coupled_ground_state import (
    tunnel_coupled_ground_state,
    tunnel_coupled_ground_state_fn,
)
from qarray_plus.DotArray.utils import (
    GateVoltageDependantCallable,
    _reshape_reshape,
    dispatch_functional
)
from qarray_plus.misc import triple_point, optimal_vg
from qarray_plus.misc.timing_decorator import timing

# Type alias for cleaner definitions
ArrayOrCallable = Union[GateVoltageDependantCallable, jnp.ndarray]


@dataclass
class DotArray:
    n_dots: int
    n_gates: int
    # Maxwell capacitance matrices
    cdd: ArrayOrCallable  # Dot-to-dot (n_dot, n_dot)
    cdg: ArrayOrCallable  # Gate-to-dot (n_dot, n_gate)

    t: Optional[Dict] = None
    gamma_lead: Optional[ArrayOrCallable] = None
    gamma_interdot: Optional[ArrayOrCallable] = None

    def __post_init__(self):
        if self.gamma_lead is None:
            self.gamma_lead = jnp.full((self.n_dots,), jnp.inf)

        if self.gamma_interdot is None:
            self.gamma_interdot = jnp.full(
                (2, self.n_dots, self.n_dots), jnp.inf
            )

    def triple_point(self, vg, n1, n2, n3, return_index=False):
        return triple_point(self, vg, n1, n2, n3, return_index)

    def optimal_vg(self, n: jnp.ndarray, charge_sensor=None):
        return optimal_vg(self, n, charge_sensor)

    def numerical_solver(self, vg: jnp.ndarray):
        """Solve nonlinear open-charge equations."""
        func = dispatch_functional(
            numerical_solver_open_fn, numerical_solver_open, self.cdd, self.cdg
        )
        return _reshape_reshape(func, vg)

    @timing
    @charge_sensor
    def markov_jumps(
            self, vg, kT: float = 0.0, dt: float = 1.0, per_row_reset: bool = True,
            n_substeps: int = 1, max_charge: int = 4, seed: int = 0
    ):
        return markov_jumps(
            self, vg=vg, gamma_lead=self.gamma_lead,
            gamma_interdot=self.gamma_interdot, kT=kT, dt=dt,
            per_row_reset=per_row_reset, n_substeps=n_substeps,
            max_charge=max_charge, seed=seed
        )

    @timing
    @charge_sensor
    def ground_state_open(self, vg: jnp.ndarray):
        """Compute the open-charge ground state."""
        func = dispatch_functional(
            ground_state_open_fn, ground_state_open, self.cdd, self.cdg
        )
        max_size = 1000 if self.n_dots >= 12 else None
        return _reshape_reshape(func, vg, max_size=max_size)

    @timing
    @charge_sensor
    def tunnel_coupled_ground_state(self, vg: jnp.ndarray, n_truncate: int = 32):
        """Ground state including inter-dot tunneling."""
        func = dispatch_functional(
            tunnel_coupled_ground_state_fn, tunnel_coupled_ground_state,
            self.cdd, self.cdg, self.t
        )
        return _reshape_reshape(partial(func, n_truncate=n_truncate), vg)

    @timing
    @charge_sensor
    def latching(
            self, vg: jnp.ndarray, dt: float = 1.0, kT: float = 0.0,
            n_substeps: int = 1, seed: Optional[int] = 0
    ):
        """Perform stochastic latching for a batch of gate voltages."""
        # Use JAX-style PRNG handling
        key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)
        keys = jax.random.split(key, vg.shape[0])

        func = dispatch_functional(
            latching_2d_fn, latching_2d, self.cdd, self.cdg,
            self.gamma_lead, self.gamma_interdot
        )
        n_traj, _ = func(vg, dt, kT, n_substeps, keys)
        return n_traj

    @timing
    @charge_sensor
    def latching_cont(
            self, vg: jnp.ndarray, t_int: float, kT: float,
            N_recursion: int = 1, seed: int = 0, n0: Optional[jnp.ndarray] = None
    ):
        """Continuous stochastic latching across a voltage sweep."""
        vg_flat = vg.reshape(-1, vg.shape[-1])

        if n0 is None:
            n0 = _ground_state_open(self.cdd, self.cdg, vg_flat[0])

        func = partial(
            _latching_1d, n0, self.cdd, self.cdg,
            self.gamma_lead, self.gamma_interdot,
            t_int=t_int, kT=kT, N_recursion=N_recursion,
            key=jax.random.PRNGKey(seed)
        )
        # Wrap to return only the trajectory, not the state
        f_wrapped = lambda v: func(v)[0]
        return _reshape_reshape(f_wrapped, vg)
