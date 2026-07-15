"""
Author: pranavjv
Date: 05/02/2026

Lindbladian implementation of latching
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Transition graph (charge basis) – shared utility
# -----------------------------------------------------------------------------
def build_transition_graph(
        *,
        charge_states: np.ndarray,
        max_charge: int,
        include_interdot: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimized NumPy version for host-side preprocessing.
    This avoids the jax.errors.NonConcreteBooleanIndexError.
    """
    charge_states = np.asarray(charge_states, dtype=np.int32)
    n_states, n_dots = charge_states.shape

    # Unique hashing for lookup
    base = max_charge + 1
    weights = base ** np.arange(n_dots, dtype=np.int64)
    state_hashes = np.dot(charge_states, weights)

    # Create hash map for O(1) lookup
    hash_to_idx = {h: i for i, h in enumerate(state_hashes)}

    src, dst, dot_idx = [], [], []

    # Lead transitions
    for k in range(n_dots):
        for step in [-1, 1]:
            # Vectorized shift for all states for dot k
            next_states = charge_states.copy()
            next_states[:, k] += step

            # Boundary check
            in_bounds = (next_states[:, k] >= 0) & (next_states[:, k] <= max_charge)
            valid_indices = np.where(in_bounds)[0]

            for i in valid_indices:
                h = np.dot(next_states[i], weights)
                j = hash_to_idx.get(h)
                if j is not None:
                    src.append(i)
                    dst.append(j)
                    dot_idx.append(k)

    # Interdot transitions
    if include_interdot and n_dots >= 2:
        for a in range(n_dots):
            for b in range(n_dots):
                if a == b: continue
                # Filter states that can actually move a charge
                can_move = (charge_states[:, a] > 0) & (charge_states[:, b] < max_charge)
                valid_indices = np.where(can_move)[0]

                for i in valid_indices:
                    # Logic for move a -> b
                    h = state_hashes[i] - weights[a] + weights[b]
                    j = hash_to_idx.get(h)
                    if j is not None:
                        src.append(i)
                        dst.append(j)
                        dot_idx.append(-1)

    return (
        np.array(src, dtype=np.int32),
        np.array(dst, dtype=np.int32),
        np.array(dot_idx, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Lindblad scan (JAX)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class LindbladParams:
    beta: float
    dt: float = 1.0
    n_substeps: int = 1
    per_row_reset: bool = True

    gamma_lead: Union[float, Sequence[float]] = 1.0
    gamma_interdot: float = 0.0
    gamma_phi: float = 0.0  # charge-basis pure dephasing

    init: str = "diag_min"  # "diag_min" | "gibbs_diag" | "maximally_mixed"
    project_psd: bool = True
    project_psd_max_dim: int = 64
    unravel: str = "mcwf"  # "me" | "mcwf"
    seed: int = 0


def _gamma_lead_vec(gamma_lead: Union[float, Sequence[float]], n_dot: int) -> np.ndarray:
    if np.isscalar(gamma_lead):
        return np.full((int(n_dot),), float(gamma_lead), dtype=float)
    arr = np.asarray(list(gamma_lead), dtype=float)
    if arr.shape != (int(n_dot),):
        raise ValueError(f"gamma_lead must be scalar or length n_dot={n_dot}, got {arr.shape}")
    return arr


@lru_cache(maxsize=None)
def cached_transition_graph_for_full_basis(
        n_dot: int,
        max_charge: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cached full charge basis (0..max_charge per dot) and its transition edges.
    """
    n_dot = int(n_dot)
    max_charge = int(max_charge)
    if n_dot <= 0:
        raise ValueError("n_dot must be >= 1")
    if max_charge < 0:
        raise ValueError("max_charge must be >= 0")

    charge_states = np.asarray(list(np.ndindex(*([max_charge + 1] * n_dot))), dtype=np.int32)
    src, dst, dot_index = build_transition_graph(
        charge_states=charge_states,
        max_charge=max_charge,
        include_interdot=True,
    )
    return charge_states, src, dst, dot_index


def compute_edge_rates_over_grid_jax(
        energies_flat: Array,  # (P, S) energies in charge basis
        src: Array,  # (E,)
        dst: Array,  # (E,)
        dot_index: Array,  # (E,) >=0 lead dot index; <0 interdot
        *,
        beta: float,
        gamma_lead: Union[float, Sequence[float]],
        gamma_interdot: float,
        n_dot: int,
) -> Array:
    energies_flat = jnp.asarray(energies_flat, dtype=jnp.float32)
    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)
    dot_index = jnp.asarray(dot_index, dtype=jnp.int32)

    dE = energies_flat[:, dst] - energies_flat[:, src]  # (P, E)
    beta_f = float(beta)
    if math.isinf(beta_f) and beta_f > 0:
        fac = (dE <= 0.0).astype(energies_flat.dtype)
    elif not math.isfinite(beta_f) or beta_f < 0:
        raise ValueError(f"beta must be >=0 and finite or +inf (T=0 limit); got {beta_f!r}")
    else:
        fac = jnp.where(dE > 0.0, jnp.exp(-beta_f * dE), jnp.ones_like(dE))

    gamma_lead_vec = jnp.asarray(_gamma_lead_vec(gamma_lead, n_dot=n_dot), dtype=energies_flat.dtype)
    is_lead = dot_index >= 0
    lead_dot = jnp.clip(dot_index, 0, int(n_dot) - 1)
    base = jnp.where(
        is_lead,
        gamma_lead_vec[lead_dot],
        jnp.asarray(float(gamma_interdot), dtype=energies_flat.dtype),
    )
    return fac * base[None, :]


def _lindblad_rhs_jump_form(
        rho: Array,  # (S,S)
        H: Array,  # (S,S)
        src: Array,  # (E,)
        dst: Array,  # (E,)
        rates_e: Array,  # (E,)
        *,
        gamma_phi: float,
) -> Array:
    rho = jnp.asarray(rho)
    Hc = jnp.asarray(H, dtype=rho.dtype)

    comm = -1j * (Hc @ rho - rho @ Hc)

    S = rho.shape[0]
    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)
    rates_e = jnp.asarray(rates_e, dtype=jnp.real(rho).dtype)

    rho_diag = jnp.diag(rho)
    inflow = jnp.zeros((S,), dtype=rho.dtype).at[dst].add(rates_e * rho_diag[src])
    out_rate = jnp.zeros((S,), dtype=rates_e.dtype).at[src].add(rates_e)
    D = -0.5 * (out_rate[:, None] + out_rate[None, :]) * rho
    D = D.at[jnp.arange(S), jnp.arange(S)].add(inflow)

    if float(gamma_phi) > 0.0:
        D = D - float(gamma_phi) * (rho - jnp.diag(rho_diag))

    return comm + D


def _rk4_step(rhs, rho: Array, dt: float) -> Array:
    k1 = rhs(rho)
    k2 = rhs(rho + 0.5 * dt * k1)
    k3 = rhs(rho + 0.5 * dt * k2)
    k4 = rhs(rho + dt * k3)
    return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _init_rho_from_energies(energies: Array, *, beta: float, init: str) -> Array:
    init = str(init).lower().strip()
    S = int(energies.shape[0])
    if init == "maximally_mixed":
        return jnp.eye(S, dtype=jnp.complex64) / jnp.asarray(float(S), dtype=jnp.complex64)
    if init == "gibbs_diag":
        beta_f = float(beta)
        e0 = energies - jnp.min(energies)
        if math.isinf(beta_f) and beta_f > 0:
            m = e0 <= jnp.asarray(1e-12, dtype=e0.dtype)
            w = m.astype(jnp.float32)
        elif not math.isfinite(beta_f) or beta_f < 0:
            raise ValueError(f"beta must be >=0 and finite or +inf; got {beta_f!r}")
        else:
            w = jnp.exp(-beta_f * e0)
        w = w / jnp.sum(w)
        return jnp.diag(w.astype(jnp.complex64))
    # default: pure state in min-energy basis state
    i0 = jnp.argmin(energies).astype(jnp.int32)
    v = jnp.eye(S, dtype=jnp.complex64)[i0]
    return jnp.outer(v, jnp.conj(v))


@partial(jax.jit, static_argnames=("n_substeps", "per_row_reset"))
def simulate_lindblad_scan_jax(
        evals_grid: jnp.ndarray,  # (ny, nx, S)
        evecs_grid: jnp.ndarray,  # (ny, nx, S, S)
        sector_grid: jnp.ndarray,  # (ny, nx, S)
        charge_states: jnp.ndarray,  # (S, n_dot)
        *,
        beta: float,
        dt: float,
        n_substeps: int,
        per_row_reset: bool,
        gamma_lead_vec: jnp.ndarray,
        gamma_interdot: float,
        key0: jax.random.PRNGKey,
) -> jnp.ndarray:
    """
    Ultra-optimized Lindblad/Eigenjump solver.
    Uses horizontal scans for latching and vertical vmap for parallelism.
    """
    ny, nx, S = evals_grid.shape
    dt_sub = dt / n_substeps

    # --- 1. PRE-COMPUTE RATES (Grid-Wide Vectorization) ---
    # Moving this out of the scan prevents O(N) redundant matrix ops.
    def compute_pixel_rates(eps, U, sec):
        # Detailed balance: exp(-beta * dE) for dE > 0
        dE = eps[None, :] - eps[:, None]
        fac = jnp.exp(-beta * jnp.maximum(dE, 0.0))

        # Sector logic
        same = sec[:, None] == sec[None, :]
        adj = jnp.abs(sec[:, None] - sec[None, :]) == 1

        # Dot occupation weights
        n_expect = (jnp.abs(U) ** 2).T @ charge_states
        dn = n_expect[None, :, :] - n_expect[:, None, :]
        abs_dn = jnp.abs(dn)
        sum_abs = jnp.sum(abs_dn, axis=-1, keepdims=True)
        w = jnp.where(sum_abs > 0, abs_dn / sum_abs, 0.0)

        # Coupling strengths
        lead_strength = jnp.sum(w * gamma_lead_vec, axis=-1)
        R = fac * (jnp.where(same, gamma_interdot, 0.0) + jnp.where(adj, lead_strength, 0.0))
        R = R * (1.0 - jnp.eye(S))  # Mask self-transitions
        return R, jnp.sum(R, axis=1)

    # Parallelize rate computation across the entire grid
    R_grid, lam_grid = jax.vmap(jax.vmap(compute_pixel_rates))(evals_grid, evecs_grid, sector_grid)

    # --- 2. BRANCHLESS EVOLUTION KERNEL ---
    def evolve_pixel(psi, eps, U, R, lam, pixel_key):
        def substep_fn(i, carry):
            p, k = carry
            # Use fold_in instead of split for much faster PRNG generation
            k_step = jax.random.fold_in(k, i)
            ksrc, ktgt, kunif = jax.random.split(k_step, 3)

            # Non-Hermitian propagation in eigenbasis
            c = jnp.conj(U).T @ p
            decay = jnp.exp((-1j * eps - 0.5 * lam) * dt_sub)
            c_prop = c * decay

            norm_sq = jnp.real(jnp.vdot(c_prop, c_prop))
            p_jump = jnp.clip(1.0 - norm_sq, 0.0, 1.0)

            # categorical sampling is heavily optimized in XLA
            j = jax.random.categorical(ksrc, jnp.log(jnp.maximum(jnp.abs(c) ** 2 * lam, 1e-30)))
            i_target = jax.random.categorical(ktgt, jnp.log(jnp.maximum(R[j], 1e-30)))

            # Selection logic (Branchless via jax.lax.select)
            psi_jump = U[:, i_target]
            psi_no_jump = c_prop / jnp.sqrt(jnp.maximum(norm_sq, 1e-20))

            p_next = jax.lax.select(jax.random.uniform(kunif) < p_jump, psi_jump, psi_no_jump)
            return (p_next, k)

        # fori_loop is often faster than scan for small, fixed iteration counts
        final_state, _ = jax.lax.fori_loop(0, n_substeps, substep_fn, (psi, pixel_key))
        return final_state

    # --- 3. ROW DISPATCH ---
    def scan_row(psi_init, ev_row, ec_row, R_row, lam_row, row_key):
        def step_fn(psi, x_idx):
            # Deterministic key for each pixel
            pixel_key = jax.random.fold_in(row_key, x_idx)
            next_psi = evolve_pixel(
                psi, ev_row[x_idx], ec_row[x_idx],
                R_row[x_idx], lam_row[x_idx], pixel_key
            )
            return next_psi, next_psi

        _, row_results = jax.lax.scan(step_fn, psi_init, jnp.arange(nx))
        return row_results

    if per_row_reset:
        @jax.vmap
        def do_parallel_rows(y_idx):
            # Parallel ground state initialization
            i0 = jnp.argmin(evals_grid[y_idx, 0])
            psi0 = evecs_grid[y_idx, 0, :, i0].astype(jnp.complex64)
            return scan_row(
                psi0, evals_grid[y_idx], evecs_grid[y_idx],
                R_grid[y_idx], lam_grid[y_idx], jax.random.fold_in(key0, y_idx)
            )

        return do_parallel_rows(jnp.arange(ny))

    else:
        # Sequential Rows (Latching across Y)
        def step_y(psi_prev, y_idx):
            row_key = jax.random.fold_in(key0, y_idx)
            psi_start = jax.lax.select(y_idx == 0,
                                       evecs_grid[0, 0, :, jnp.argmin(evals_grid[0, 0])],
                                       psi_prev)
            psi_row = scan_row(psi_start, evals_grid[y_idx], evecs_grid[y_idx],
                               R_grid[y_idx], lam_grid[y_idx], row_key)
            return psi_row[-1], psi_row

        _, full_map = jax.lax.scan(step_y, jnp.zeros(S, dtype=jnp.complex64), jnp.arange(ny))
        return full_map


def rho_to_charge_expectation(rho_map: Array, charge_states: Array) -> Array:
    rho_map = jnp.asarray(rho_map)
    cs = jnp.asarray(charge_states, dtype=jnp.float32)
    pops = jnp.real(jnp.diagonal(rho_map, axis1=-2, axis2=-1))
    return pops @ cs


def psi_to_charge_expectation(psi_map: Array, charge_states: Array) -> Array:
    psi_map = jnp.asarray(psi_map)
    cs = jnp.asarray(charge_states, dtype=jnp.float32)
    pops = jnp.real(psi_map * jnp.conj(psi_map))
    pops = pops / jnp.sum(pops, axis=-1, keepdims=True)
    return pops @ cs


def _init_psi_from_energies(energies: Array, *, beta: float, init: str, key: Array) -> Array:
    init = str(init).lower().strip()
    S = int(energies.shape[0])
    if init == "maximally_mixed":
        i0 = jax.random.randint(key, shape=(), minval=0, maxval=int(S), dtype=jnp.int32)
        return jnp.eye(S, dtype=jnp.complex64)[i0]
    if init == "gibbs_diag":
        beta_f = float(beta)
        e0 = energies - jnp.min(energies)
        if math.isinf(beta_f) and beta_f > 0:
            m = e0 <= jnp.asarray(1e-12, dtype=e0.dtype)
            w = m.astype(jnp.float32)
        elif not math.isfinite(beta_f) or beta_f < 0:
            raise ValueError(f"beta must be >=0 and finite or +inf; got {beta_f!r}")
        else:
            w = jnp.exp(-beta_f * e0)
        w = w / jnp.sum(w)
        i0 = jax.random.categorical(key, jnp.log(w), axis=-1).astype(jnp.int32)
        return jnp.eye(S, dtype=jnp.complex64)[i0]
    i0 = jnp.argmin(energies).astype(jnp.int32)
    return jnp.eye(S, dtype=jnp.complex64)[i0]


def simulate_mcwf_scan_jax(
        H_grid: Array,  # (ny,nx,S,S) complex
        energies_grid: Array,  # (ny,nx,S) real
        src: Array,  # (E,)
        dst: Array,  # (E,)
        rates_grid: Array,  # (ny,nx,E)
        params: LindbladParams,
) -> Array:
    """
    Monte-Carlo wavefunction (quantum jump) unravelling with charge-basis jumps:
      L_e = sqrt(rate_e) |dst><src|
    """
    H_grid = jnp.asarray(H_grid, dtype=jnp.complex64)
    energies_grid = jnp.asarray(energies_grid, dtype=jnp.float32)
    rates_grid = jnp.asarray(rates_grid, dtype=jnp.float32)

    ny, nx, S, S2 = H_grid.shape
    if S != S2:
        raise ValueError("H_grid must be (...,S,S)")

    src = jnp.asarray(src, dtype=jnp.int32)
    dst = jnp.asarray(dst, dtype=jnp.int32)

    dt = float(params.dt)
    n_sub = int(max(1, params.n_substeps))
    dt_sub = dt / float(n_sub)

    def evolve_pixel(psi0: Array, Hx: Array, rates_x: Array, key: Array) -> Tuple[Array, Array]:
        out_rate = jnp.zeros((S,), dtype=rates_x.dtype).at[src].add(rates_x)
        Heff = Hx - 0.5j * jnp.diag(out_rate.astype(jnp.complex64))

        def one_substep(carry, _t):
            psi, k = carry
            k, k1, k2 = jax.random.split(k, 3)

            # RK2 midpoint for non-Hermitian evolution
            k_psi = -1j * (Heff @ psi)
            psi_mid = psi + 0.5 * dt_sub * k_psi
            k_psi2 = -1j * (Heff @ psi_mid)
            psi_prop = psi + dt_sub * k_psi2

            n2 = jnp.real(jnp.vdot(psi_prop, psi_prop))
            p_jump = jnp.clip(1.0 - n2, 0.0, 1.0)
            do_jump = jax.random.uniform(k1, shape=()) < p_jump

            def no_jump():
                return psi_prop / jnp.sqrt(jnp.maximum(n2, 1e-20))

            def yes_jump():
                pops_src = jnp.real(psi * jnp.conj(psi))
                w = rates_x * pops_src[src]
                wsum = jnp.sum(w)
                w = jnp.where(wsum > 0, w / wsum, jnp.full_like(w, 1.0 / w.shape[0]))
                cdf = jnp.cumsum(w)
                u = jax.random.uniform(k2, shape=())
                e_idx = jnp.searchsorted(cdf, u, side="right")
                e_idx = jnp.minimum(e_idx, jnp.int32(w.shape[0] - 1))
                j = dst[e_idx]
                return jnp.eye(S, dtype=jnp.complex64)[j]

            psi_next = jax.lax.cond(do_jump, yes_jump, no_jump)
            return (psi_next.astype(jnp.complex64), k), None

        (psiT, key_out), _ = jax.lax.scan(one_substep, (psi0, key), jnp.arange(n_sub))
        return psiT, key_out

    def scan_row(psi0: Array, H_row: Array, rates_row: Array, key_row: Array) -> Tuple[Array, Tuple[Array, Array]]:
        def step(carry, inputs):
            psi, k = carry
            Hx, rx = inputs
            k, kpix = jax.random.split(k, 2)
            psi2, k2 = evolve_pixel(psi, Hx, rx, kpix)
            return (psi2, k2), psi2

        (carry_out, psi_row) = jax.lax.scan(step, (psi0, key_row), (H_row, rates_row))
        psi_end, key_end = carry_out
        return psi_row, (psi_end, key_end)

    key0 = jax.random.PRNGKey(int(params.seed))
    row_keys = jax.random.split(key0, ny)

    if params.per_row_reset:
        def do_row(y, keyy):
            e0 = energies_grid[y, 0, :]
            psi0 = _init_psi_from_energies(e0, beta=float(params.beta), init=params.init, key=keyy)
            psi_row, _ = scan_row(psi0, H_grid[y], rates_grid[y], keyy)
            return psi_row

        ys = jnp.arange(ny, dtype=jnp.int32)
        return jax.vmap(do_row)(ys, row_keys)

    def step_y(carry, y):
        psi_prev_end, kprev = carry
        kprev, keyy = jax.random.split(kprev, 2)
        e0 = energies_grid[y, 0, :]
        psi0 = jax.lax.cond(
            y == 0,
            lambda _: _init_psi_from_energies(e0, beta=float(params.beta), init=params.init, key=keyy),
            lambda _: psi_prev_end,
            operand=0,
        )
        psi_row, (psi_end, key_end) = scan_row(psi0, H_grid[y], rates_grid[y], keyy)
        return (psi_end, key_end), psi_row

    psi_dummy = _init_psi_from_energies(energies_grid[0, 0, :], beta=float(params.beta), init="diag_min", key=key0)
    _, psi_map = jax.lax.scan(step_y, (psi_dummy, key0), jnp.arange(ny, dtype=jnp.int32))
    return psi_map


# -----------------------------------------------------------------------------
# Eigenstate jump unravelling (preserves tunnel hybridization)
# -----------------------------------------------------------------------------

def _sample_categorical_from_probs(key: Array, probs: Array) -> Array:
    """
    probs: (K,) nonnegative, not necessarily normalized.
    Returns: int32 index in [0, K)
    """
    probs = jnp.asarray(probs, dtype=jnp.float32)
    s = jnp.sum(probs)
    probs = jnp.where(s > 0, probs / s, jnp.full_like(probs, 1.0 / probs.shape[0]))
    cdf = jnp.cumsum(probs)
    u = jax.random.uniform(key, shape=())
    idx = jnp.searchsorted(cdf, u, side="right")
    return jnp.minimum(idx, jnp.int32(probs.shape[0] - 1)).astype(jnp.int32)


def simulate_eigenjump_scan_jax(
        evals_grid: Array,  # (ny,nx,S) real
        evecs_grid: Array,  # (ny,nx,S,S) complex, columns are eigenvectors in charge basis
        sector_grid: Array,  # (ny,nx,S) int32, rounded total charge per eigenstate
        charge_states: Array,  # (S, n_dot) int charge basis configurations
        params: LindbladParams,
) -> Array:
    """
    Quantum-jump evolution where jumps occur between *instantaneous energy eigenstates*.

    Allowed transitions:
      - interdot-like: within same sector (ΔN=0), rate ~ gamma_interdot * exp(-β max(ΔE,0))
      - lead-like: between adjacent sectors |ΔN|=1, rate ~ gamma_lead * exp(-β max(ΔE,0))

    If `params.gamma_lead` is a sequence (length n_dot), lead-like transitions use a
    dot-weighted mixture of lead rates, where weights are inferred from the change
    in per-dot occupation expectation between eigenstates.

    The state is carried in the charge basis, so diabatic mixing across pixels
    happens naturally through overlaps as eigenvectors vary with vg.

    Returns:
      psi_map: (ny,nx,S) complex64 in the charge basis
    """
    evals_grid = jnp.asarray(evals_grid, dtype=jnp.float32)
    evecs_grid = jnp.asarray(evecs_grid, dtype=jnp.complex64)
    sector_grid = jnp.asarray(sector_grid, dtype=jnp.int32)
    charge_states = jnp.asarray(charge_states, dtype=jnp.float32)

    ny, nx, S = evals_grid.shape
    n_dot = int(charge_states.shape[1])
    dt = float(params.dt)
    n_sub = int(max(1, params.n_substeps))
    dt_sub = dt / float(n_sub)
    beta = float(params.beta)

    gamma_lead_in = params.gamma_lead
    gamma_interdot = float(params.gamma_interdot)

    # gamma_lead can be scalar or per-dot.
    if np.isscalar(gamma_lead_in):
        gamma_lead_scalar = float(gamma_lead_in)
        gamma_lead_vec = None
    else:
        gamma_lead_vec_np = np.asarray(list(gamma_lead_in), dtype=float)
        if gamma_lead_vec_np.shape != (n_dot,):
            raise ValueError(
                f"gamma_lead must be scalar or length n_dot={n_dot}, got {gamma_lead_vec_np.shape}"
            )
        gamma_lead_scalar = None
        gamma_lead_vec = jnp.asarray(gamma_lead_vec_np, dtype=jnp.float32)

    key0 = jax.random.PRNGKey(int(params.seed))
    row_keys = jax.random.split(key0, ny)

    def evolve_pixel(psi0: Array, eps: Array, U: Array, sec: Array, key: Array) -> Tuple[Array, Array]:
        # Precompute rate matrix R[j,i] (from source j -> target i) in eigenbasis.
        dE = eps[None, :] - eps[:, None]  # (S,S): i - j
        if math.isinf(beta) and beta > 0:
            fac = (dE <= 0.0).astype(eps.dtype)
        elif not math.isfinite(beta) or beta < 0:
            raise ValueError(f"beta must be >=0 and finite or +inf (T=0 limit); got {beta!r}")
        else:
            fac = jnp.where(dE > 0.0, jnp.exp(-beta * dE), jnp.ones_like(dE))

        same = sec[:, None] == sec[None, :]
        adj = jnp.abs(sec[:, None] - sec[None, :]) == 1

        if gamma_lead_vec is None:
            lead_strength = jnp.asarray(float(gamma_lead_scalar), dtype=eps.dtype)
            base = jnp.where(same, gamma_interdot, 0.0) + jnp.where(adj, lead_strength, 0.0)
        else:
            # Per-eigenstate dot occupation expectation: n_expect[j, d]
            P = jnp.abs(U) ** 2  # (S_basis, S_eig)
            n_expect = (P.T @ charge_states).astype(jnp.float32)  # (S, n_dot)

            # Per transition (j->i) absolute dot-occupation change, used as weights.
            dn = n_expect[None, :, :] - n_expect[:, None, :]  # (S,S,n_dot), row=j col=i
            abs_dn = jnp.abs(dn)
            sum_abs = jnp.sum(abs_dn, axis=-1, keepdims=True)  # (S,S,1)
            w = jnp.where(sum_abs > 0, abs_dn / sum_abs, 0.0)

            lead_strength = jnp.sum(w * gamma_lead_vec[None, None, :], axis=-1).astype(eps.dtype)  # (S,S)
            base = jnp.where(same, gamma_interdot, 0.0) + jnp.where(adj, lead_strength, 0.0)

        base = base * (1.0 - jnp.eye(S, dtype=base.dtype))  # no self-jumps
        R = fac * base  # (S,S)

        lam = jnp.sum(R, axis=1)  # (S,) outgoing rate for each source eigenstate j

        def one_substep(carry, _t):
            psi, key_in = carry
            key_in, kjump, ksrc, ktgt = jax.random.split(key_in, 4)

            # Transform to eigenbasis and propagate with non-Hermitian decay
            c = jnp.conj(U).T @ psi  # (S,)
            decay = jnp.exp((-1j * eps - 0.5 * lam) * dt_sub).astype(jnp.complex64)
            c_prop = c * decay
            n2 = jnp.real(jnp.vdot(c_prop, c_prop))
            p_jump = jnp.clip(1.0 - n2, 0.0, 1.0)
            do_jump = jax.random.uniform(kjump, shape=()) < p_jump

            def no_jump():
                c_n = c_prop / jnp.sqrt(jnp.maximum(n2, 1e-20))
                return (U @ c_n).astype(jnp.complex64)

            def yes_jump():
                pop = jnp.real(c * jnp.conj(c))
                pop = pop / jnp.sum(pop)
                w_src = pop * lam
                j = _sample_categorical_from_probs(ksrc, w_src)
                i = _sample_categorical_from_probs(ktgt, R[j])
                return U[:, i].astype(jnp.complex64)

            psi_next = jax.lax.cond(do_jump, yes_jump, no_jump)
            return (psi_next, key_in), None

        (psiT, key_out), _ = jax.lax.scan(one_substep, (psi0, key), jnp.arange(n_sub))
        psiT = psiT / jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(psiT, psiT)), 1e-20))
        return psiT.astype(jnp.complex64), key_out

    def scan_row(psi0: Array, evals_row: Array, evecs_row: Array, sec_row: Array, key_row: Array):
        def step(carry, inputs):
            psi, key_in = carry
            eps, U, sec = inputs
            key_in, kpix = jax.random.split(key_in, 2)
            psi2, key2 = evolve_pixel(psi, eps, U, sec, kpix)
            return (psi2, key2), psi2

        (carry_out, psi_row) = jax.lax.scan(step, (psi0, key_row), (evals_row, evecs_row, sec_row))
        psi_end, key_end = carry_out
        return psi_row, (psi_end, key_end)

    if params.per_row_reset:
        def do_row(y, keyy):
            eps0 = evals_grid[y, 0, :]
            U0 = evecs_grid[y, 0, :, :]
            i0 = jnp.argmin(eps0).astype(jnp.int32)
            psi0 = U0[:, i0].astype(jnp.complex64)
            psi_row, _ = scan_row(psi0, evals_grid[y], evecs_grid[y], sector_grid[y], keyy)
            return psi_row

        ys = jnp.arange(ny, dtype=jnp.int32)
        return jax.vmap(do_row)(ys, row_keys)

    # sequential rows: carry end-of-row state
    def step_y(carry, y):
        psi_prev, key_prev = carry
        key_prev, keyy = jax.random.split(key_prev, 2)
        eps0 = evals_grid[y, 0, :]
        U0 = evecs_grid[y, 0, :, :]
        i0 = jnp.argmin(eps0).astype(jnp.int32)
        psi_init = jax.lax.cond(y == 0, lambda _: U0[:, i0].astype(jnp.complex64), lambda _: psi_prev, operand=0)
        psi_row, (psi_end, key_end) = scan_row(psi_init, evals_grid[y], evecs_grid[y], sector_grid[y], keyy)
        return (psi_end, key_end), psi_row

    psi_dummy = evecs_grid[0, 0, :, jnp.argmin(evals_grid[0, 0, :]).astype(jnp.int32)]
    _, psi_map = jax.lax.scan(step_y, (psi_dummy, key0), jnp.arange(ny, dtype=jnp.int32))
    return psi_map


# -----------------------------------------------------------------------------
# JIT-friendly eigenstate jump unravelling (fully JAX runtime)
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_substeps", "per_row_reset"))
def simulate_eigenjump_scan_jax_fast(
        evals_grid: Array,  # (ny,nx,S) real
        evecs_grid: Array,  # (ny,nx,S,S) complex, columns are eigenvectors in charge basis
        sector_grid: Array,  # (ny,nx,S) int32, rounded total charge per eigenstate
        charge_states: Array,  # (S, n_dot) int charge basis configurations
        *,
        beta: float,
        dt: float,
        n_substeps: int,
        per_row_reset: bool,
        gamma_lead_vec: Array,  # (n_dot,) >=0
        gamma_interdot: float,
        key0: Array,  # PRNGKey
) -> Array:
    """
    Same physics as `simulate_eigenjump_scan_jax`, but avoids Python/NumPy control
    flow so it can be called from larger `jit`-compiled pipelines.

    Notes:
    - `beta` can be finite or +inf (T=0). The detailed-balance factor uses
      exp(-beta * max(ΔE, 0)), which yields a hard cutoff at T=0.
    - `gamma_lead_vec` is per-dot lead coupling (length n_dot). Transitions
      between adjacent total-charge sectors use a dot-weighted mixture inferred
      from per-dot occupation changes between eigenstates.
    """
    evals_grid = jnp.asarray(evals_grid, dtype=jnp.float32)
    evecs_grid = jnp.asarray(evecs_grid, dtype=jnp.complex64)
    sector_grid = jnp.asarray(sector_grid, dtype=jnp.int32)
    charge_states = jnp.asarray(charge_states, dtype=jnp.float32)

    beta = jnp.asarray(beta, dtype=jnp.float32)
    dt = jnp.asarray(dt, dtype=jnp.float32)
    gamma_lead_vec = jnp.asarray(gamma_lead_vec, dtype=jnp.float32)
    gamma_interdot = jnp.asarray(gamma_interdot, dtype=jnp.float32)
    key0 = jnp.asarray(key0, dtype=jnp.uint32)

    ny, nx, S = evals_grid.shape
    n_dot = int(charge_states.shape[1])

    # Basic shape checks (static at compile-time).
    _ = n_dot  # silence lint

    dt_sub = dt / jnp.asarray(float(n_substeps), dtype=dt.dtype)
    row_keys = jax.random.split(key0, ny)

    def evolve_pixel(psi0: Array, eps: Array, U: Array, sec: Array, key: Array) -> Tuple[Array, Array]:
        # Rate matrix R[j,i] (from source j -> target i) in eigenbasis.
        dE = eps[None, :] - eps[:, None]  # (S,S): i - j
        fac = jnp.where(dE > 0.0, jnp.exp(-beta * dE), jnp.ones_like(dE))

        same = sec[:, None] == sec[None, :]
        adj = jnp.abs(sec[:, None] - sec[None, :]) == 1

        # Per-eigenstate dot occupation expectation: n_expect[j, d]
        P = jnp.abs(U) ** 2  # (S_basis, S_eig)
        n_expect = (P.T @ charge_states).astype(jnp.float32)  # (S, n_dot)

        # Per transition (j->i) absolute dot-occupation change, used as weights.
        dn = n_expect[None, :, :] - n_expect[:, None, :]  # (S,S,n_dot), row=j col=i (up to transpose)
        abs_dn = jnp.abs(dn)
        sum_abs = jnp.sum(abs_dn, axis=-1, keepdims=True)  # (S,S,1)
        w = jnp.where(sum_abs > 0, abs_dn / sum_abs, 0.0)

        lead_strength = jnp.sum(w * gamma_lead_vec[None, None, :], axis=-1).astype(eps.dtype)  # (S,S)
        base = jnp.where(same, gamma_interdot, 0.0) + jnp.where(adj, lead_strength, 0.0)

        base = base * (1.0 - jnp.eye(S, dtype=base.dtype))  # no self-jumps
        R = fac * base  # (S,S)
        lam = jnp.sum(R, axis=1)  # (S,) outgoing rate for each source eigenstate j

        def one_substep(carry, _t):
            psi, key_in = carry
            key_in, kjump, ksrc, ktgt = jax.random.split(key_in, 4)

            # Transform to eigenbasis and propagate with non-Hermitian decay
            c = jnp.conj(U).T @ psi  # (S,)
            decay = jnp.exp((-1j * eps - 0.5 * lam) * dt_sub).astype(jnp.complex64)
            c_prop = c * decay
            n2 = jnp.real(jnp.vdot(c_prop, c_prop))
            p_jump = jnp.clip(1.0 - n2, 0.0, 1.0)
            do_jump = jax.random.uniform(kjump, shape=()) < p_jump

            def no_jump():
                c_n = c_prop / jnp.sqrt(jnp.maximum(n2, 1e-20))
                return (U @ c_n).astype(jnp.complex64)

            def yes_jump():
                pop = jnp.real(c * jnp.conj(c))
                pop = pop / jnp.sum(pop)
                w_src = pop * lam
                j = _sample_categorical_from_probs(ksrc, w_src)
                i = _sample_categorical_from_probs(ktgt, R[j])
                return U[:, i].astype(jnp.complex64)

            psi_next = jax.lax.cond(do_jump, yes_jump, no_jump)
            return (psi_next, key_in), None

        (psiT, key_out), _ = jax.lax.scan(one_substep, (psi0, key), jnp.arange(n_substeps))
        psiT = psiT / jnp.sqrt(jnp.maximum(jnp.real(jnp.vdot(psiT, psiT)), 1e-20))
        return psiT.astype(jnp.complex64), key_out

    def scan_row(psi0: Array, evals_row: Array, evecs_row: Array, sec_row: Array, key_row: Array):
        def step(carry, inputs):
            psi, key_in = carry
            eps, U, sec = inputs
            key_in, kpix = jax.random.split(key_in, 2)
            psi2, key2 = evolve_pixel(psi, eps, U, sec, kpix)
            return (psi2, key2), psi2

        (carry_out, psi_row) = jax.lax.scan(step, (psi0, key_row), (evals_row, evecs_row, sec_row))
        psi_end, key_end = carry_out
        return psi_row, (psi_end, key_end)

    if per_row_reset:
        def do_row(y, keyy):
            eps0 = evals_grid[y, 0, :]
            U0 = evecs_grid[y, 0, :, :]
            i0 = jnp.argmin(eps0).astype(jnp.int32)
            psi0 = U0[:, i0].astype(jnp.complex64)
            psi_row, _ = scan_row(psi0, evals_grid[y], evecs_grid[y], sector_grid[y], keyy)
            return psi_row

        ys = jnp.arange(ny, dtype=jnp.int32)
        return jax.vmap(do_row)(ys, row_keys)

    # sequential rows: carry end-of-row state
    def step_y(carry, y):
        psi_prev, key_prev = carry
        key_prev, keyy = jax.random.split(key_prev, 2)
        eps0 = evals_grid[y, 0, :]
        U0 = evecs_grid[y, 0, :, :]
        i0 = jnp.argmin(eps0).astype(jnp.int32)
        psi_init = jax.lax.cond(y == 0, lambda _: U0[:, i0].astype(jnp.complex64), lambda _: psi_prev, operand=0)
        psi_row, (psi_end, key_end) = scan_row(psi_init, evals_grid[y], evecs_grid[y], sector_grid[y], keyy)
        return (psi_end, key_end), psi_row

    psi_dummy = evecs_grid[0, 0, :, jnp.argmin(evals_grid[0, 0, :]).astype(jnp.int32)]
    _, psi_map = jax.lax.scan(step_y, (psi_dummy, key0), jnp.arange(ny, dtype=jnp.int32))
    return psi_map
