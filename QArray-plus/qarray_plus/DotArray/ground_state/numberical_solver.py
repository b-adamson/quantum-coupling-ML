"""
Author: b-vanstraaten
Date: 2025-12-05
"""

import jax
import jax.numpy as jnp
from jaxopt import BoxOSQP

# OSQP solver instance
qp = BoxOSQP(check_primal_dual_infeasability=False, verbose=False)


@jax.jit
def _numerical_solver_open(cdd: jnp.ndarray, cdg: jnp.ndarray, vg: jnp.ndarray):
    """
    Solve the physically correct QP:

        minimize_y  0.5 * yᵀ Cdd y  - yᵀ (Cdg vg)
        subject to  Cdd y >= 0

    The returned physical charge vector is x = Cdd y.
    """

    # BoxOSQP's internal lax.while_loop promotes its carry to the default float
    # dtype, which is float64 when JAX x64 is enabled. If the inputs are float32
    # (as the tunnel-coupled solver casts them) the carry types disagree and the
    # solver raises "while_loop ... carry input and carry output must have equal
    # types". Solve in JAX's default float dtype (float64 under x64, else float32),
    # then cast the result back. This is a no-op when x64 is disabled.
    in_dtype = cdd.dtype
    solve_dtype = jnp.zeros(()).dtype
    cdd = cdd.astype(solve_dtype)
    cdg = cdg.astype(solve_dtype)
    vg = vg.astype(solve_dtype)

    P = cdd  # (N, N) PD matrix
    q = -(cdg @ vg)  # (N,) linear term

    A = cdd  # Constraint: Cdd y ≥ 0
    l = jnp.zeros(cdd.shape[0], dtype=solve_dtype)
    u = jnp.full(cdd.shape[0], jnp.inf, dtype=solve_dtype)

    sol = qp.run(params_obj=(P, q), params_eq=A, params_ineq=(l, u)).params
    y = sol.primal[0]

    # Convert back to physical charge
    x = cdd @ y
    return jnp.clip(x, min=0.0).astype(in_dtype)


numerical_solver_open = jax.vmap(_numerical_solver_open, in_axes=(None, None, 0))


def _numerical_solver_open_fn(cdd_fn, cdg_fn, vg):
    """
    Vectorized wrapper that computes cdd and cdg from functions of vg.
    """
    return _numerical_solver_open(cdd_fn(vg), cdg_fn(vg), vg)


numerical_solver_open_fn = jax.vmap(_numerical_solver_open_fn, in_axes=(None, None, 0))
