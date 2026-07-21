import jax.numpy as jnp
import numpy as np

from qarray_plus.DotArray.ground_state.free_energy import _free_energy_fn

# scipy is only needed for the non-constant-capacitance refinement paths below
# (CASE 1B / CASE 2). It's a large dependency, so it's imported lazily right
# before use rather than at module load time -- callers using constant Cdd/Cdg
# (the common case, e.g. CASE 1A) never pay for it.


# ============================================================
# Utility functions
# ============================================================


def _wrap_to_fn(val):
    """If val is callable, return it; else wrap constant into fn(vg)=val."""
    if callable(val):
        return val
    return lambda vg, v=val: v


def is_constant(val):
    """
    Detect whether a capacitance matrix is constant.
    - If not callable → constant
    - If callable but signature like lambda vg, v=const: v → treat as constant
    """
    if not callable(val):
        return True

    # Detect constant-returning wrapper
    import inspect

    sig = inspect.signature(val)
    if len(sig.parameters) == 2:  # vg, v=constant
        return True

    return False


def analytic_solution(n, Cdd, Cdg, rcond=1e-12):
    """
    Analytic solution for constant capacitance matrices:
        R = chol(Cdd^{-1})^T
        M = (R Cdg)^+ R
        vg = M n
    """
    Cdd_inv = np.linalg.pinv(Cdd, rcond=rcond)
    R = np.linalg.cholesky(Cdd_inv).T
    M = np.linalg.pinv(R @ Cdg, rcond=rcond) @ R
    return M @ n


def constant_guess(model, n, cdd_fn, cdg_fn, rcond=1e-12):
    """
    Compute analytic guess using capacitances evaluated at vg = 0.
    """
    vg0 = np.zeros(model.n_gates)
    Cdd0 = np.asarray(cdd_fn(vg0))
    Cdg0 = np.asarray(cdg_fn(vg0))
    return analytic_solution(n, Cdd0, Cdg0, rcond=rcond)


def optimal_vg(model, n, charge_sensor=None, rcond=1e-12):
    """
    Compute optimal gate voltages for a given charge state n.
    Includes logic for:
      - fully constant capacitance matrices → analytic closed-form solution
      - nonlinear capacitances → analytic initial guess + refinement
      - charge sensor → combined optimization
    """
    n = np.asarray(n)

    # Wrap capacitances into functions
    cdd_raw = model.cdd
    cdg_raw = model.cdg

    cdd = _wrap_to_fn(cdd_raw)
    cdg = _wrap_to_fn(cdg_raw)

    constant_cdd = is_constant(cdd_raw)
    constant_cdg = is_constant(cdg_raw)

    # ============================================================
    # CASE 1 — No charge sensor
    # ============================================================
    if charge_sensor is None:
        # --------------------------------------------------------
        # CASE 1A: Both capacitances constant → fully analytic solution
        # --------------------------------------------------------
        if constant_cdd and constant_cdg:
            Cdd = np.asarray(cdd(None))
            Cdg = np.asarray(cdg(None))
            return analytic_solution(n, Cdd, Cdg, rcond=rcond)

        # --------------------------------------------------------
        # CASE 1B: Non-constant → analytic zero-bias guess + refinement
        # --------------------------------------------------------
        x0 = constant_guess(model, n, cdd, cdg, rcond=rcond)

        objective = lambda x: float(
            _free_energy_fn(jnp.asarray(n), cdd, cdg, jnp.asarray(x))
        )

        from scipy.optimize import minimize
        res = minimize(objective, x0, method="nelder-mead")
        return res.x

    # ============================================================
    # CASE 2 — Charge sensor present
    # ============================================================
    n_dots = charge_sensor.n_dots
    n_sensor = charge_sensor.n_sensor

    if n.size != n_dots + n_sensor:
        raise ValueError("Charge vector length does not match dots + sensors.")

    n_d = n[:n_dots]
    n_s = n[n_dots:]

    # ------------------------------------------------------------
    # Step 1: Dot-only optimum as a starting point
    # ------------------------------------------------------------
    x0_dots = optimal_vg(model, n_d, charge_sensor=None, rcond=rcond)

    # ------------------------------------------------------------
    # Step 2: Build combined objective (free-energy + sensor penalty)
    # ------------------------------------------------------------
    def free_energy(x):
        return float(_free_energy_fn(jnp.asarray(n_d), cdd, cdg, jnp.asarray(x)))

    def sensor_potential(x):
        return charge_sensor.v_sensor(x, n_d)

    def sensor_penalty(x):
        return np.abs(sensor_potential(x) - n_s - 0.5).sum()

    def objective(x):
        return free_energy(x) + sensor_penalty(x)

    res = minimize(objective, x0_dots, method="nelder-mead")
    return res.x
