"""
Author: pranavjv
Date: 05/02/2026

Tunnel-coupled ground state.

This replaces the original dense-eigensolve implementation with:
- truncation that avoids materializing an exponentially large local basis
- optional multi-device execution (multi-GPU / multi-host via `pmap`)
- an iterative Lanczos solver for larger truncated bases

Public API preserved:
  - `make_H(basis, F, t, n_dot)` (vmapped over F)
  - `tunnel_coupled_ground_state(cdd, cdg, t, vg, n_truncate=...)`
  - `tunnel_coupled_ground_state_fn(cdd_fn, cdg_fn, t_fn, vg, n_truncate=...)`
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import sparse as jsparse

from qarray_plus.DotArray.ground_state.free_energy import _free_energy
from qarray_plus.DotArray.ground_state.numberical_solver import _numerical_solver_open


# -----------------------------------------------------------------------------
# Dense Hamiltonian builder (kept for examples / open-system latching)
# -----------------------------------------------------------------------------

def _tunneling_dense(t: jnp.ndarray, basis: jnp.ndarray) -> jnp.ndarray:
    """
    Dense tunneling Hamiltonian in the occupation-number basis.

    Matrix elements implement directed hops (dot_from -> dot_to) with amplitude:
      -t[from,to] * sqrt(n_from * (n_to + 1))
    """
    basis = jnp.asarray(basis)
    t = jnp.asarray(t)
    n_states, n_dot = basis.shape

    # State pairs for vectorized comparisons
    state_i = basis[:, None, :]  # (S, 1, n_dot) source
    state_j = basis[None, :, :]  # (1, S, n_dot) destination
    diff = state_j - state_i  # (S, S, n_dot) = dest - src

    H = jnp.zeros((n_states, n_states), dtype=jnp.float32)

    for dot_from in range(n_dot):
        for dot_to in range(n_dot):
            if dot_from == dot_to:
                continue
            t_val = t[dot_from, dot_to]

            expected = jnp.zeros((n_dot,), dtype=diff.dtype)
            expected = expected.at[dot_from].set(-1)
            expected = expected.at[dot_to].set(+1)

            is_valid = jnp.all(diff == expected[None, None, :], axis=-1)  # (S,S)

            n_from = state_i[:, :, dot_from]
            n_to = state_i[:, :, dot_to]
            elem = -t_val * jnp.sqrt(n_from * (n_to + 1))
            H = H + is_valid * elem

    # Hermitian symmetrization for numerical hygiene.
    return 0.5 * (H + H.T)


def _make_H(basis: jnp.ndarray, F: jnp.ndarray, t: jnp.ndarray, n_dot: int) -> jnp.ndarray:
    # Diagonal (electrostatic) part
    H_F = jnp.diag(jnp.asarray(F, dtype=jnp.float32))
    # Tunneling part
    H_t = _tunneling_dense(t=t, basis=basis)
    H = H_F + H_t
    return 0.5 * (H + H.T)


make_H = jax.vmap(_make_H, in_axes=(None, 0, None, None))


# -----------------------------------------------------------------------------
# Truncation (charge-state candidate selection)
# -----------------------------------------------------------------------------

def _build_truncation_meta(
        *,
        cdg: np.ndarray,
        n_truncate: int,
        max_total_combinations: int = 1_000_000,
        strong_deltas: Tuple[int, ...] = (-1, 0, 1, 2),
        weak_deltas: Tuple[int, ...] = (0, 1),
        strong_default: int = 6,
        chunk_size_hint: int = 8192,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int, int]:
    """
    Build per-dot delta options and chunking metadata for truncation.

    Returns:
      n_deltas_per_dot : (n_dot,) int32
      delta_array_2d   : (n_dot, n_strong) int32, weak/fixed rows are padded
      fixed_mask       : (n_dot,) int32 (1 for fixed dots, 0 otherwise)
      total_combinations : python int
      chunk_size         : python int
      n_chunks           : python int
    """
    cdg = np.asarray(cdg, dtype=float)
    n_dot = int(cdg.shape[0])
    if n_dot <= 0:
        raise ValueError("cdg must have shape (n_dot, n_gates) with n_dot >= 1")

    # Rank dots by gate coupling strength (static across the scan for constant cdg).
    strength = np.sum(np.abs(cdg), axis=1)
    order = np.argsort(strength)[::-1]

    # Use a hard cap on the number of combinations (mixed-radix enumeration).
    # We allow:
    # - "fixed" dots: 1 option (delta=0) but their baseline is rounded per-pixel
    # - "weak" dots:  2 options (delta in {0,1})
    # - "strong" dots:4 options (delta in {-1,0,1,2})
    #
    # With weak-only enumeration, total_combinations ~= 2**n_var.
    # Each strong dot adds an extra factor 2 relative to weak (4 vs 2).
    L = int(max(0, math.floor(math.log2(max(1, int(max_total_combinations))))))
    n_var = min(n_dot, L)  # weak-variable dots
    slack = max(0, L - n_var)
    n_strong = int(min(strong_default, n_var, slack))

    var_dots = set(order[:n_var].tolist())
    strong_dots = set(order[:n_strong].tolist())

    n_strong_opts = len(strong_deltas)
    n_weak_opts = len(weak_deltas)

    n_deltas_per_dot = np.ones((n_dot,), dtype=np.int32)  # fixed by default
    fixed_mask = np.ones((n_dot,), dtype=np.int32)
    delta_array_2d = np.zeros((n_dot, n_strong_opts), dtype=np.int32)

    for d in range(n_dot):
        if d in var_dots:
            fixed_mask[d] = 0
            if d in strong_dots:
                n_deltas_per_dot[d] = n_strong_opts
                delta_array_2d[d, :] = np.asarray(strong_deltas, dtype=np.int32)
            else:
                n_deltas_per_dot[d] = n_weak_opts
                delta_array_2d[d, :n_weak_opts] = np.asarray(weak_deltas, dtype=np.int32)
        else:
            # fixed: delta=0 only; baseline will be rounded per pixel
            n_deltas_per_dot[d] = 1
            delta_array_2d[d, 0] = 0

    total_combinations = int(1)
    for b in n_deltas_per_dot.tolist():
        total_combinations *= int(b)

    # We always need chunk_size >= n_truncate for the "take best K" logic.
    chunk_size = int(max(int(n_truncate), int(chunk_size_hint)))
    chunk_size = int(min(chunk_size, max(1, total_combinations)))
    n_chunks = int((total_combinations + chunk_size - 1) // chunk_size)

    return (
        jnp.asarray(n_deltas_per_dot, dtype=jnp.int32),
        jnp.asarray(delta_array_2d, dtype=jnp.int32),
        jnp.asarray(fixed_mask, dtype=jnp.int32),
        total_combinations,
        chunk_size,
        n_chunks,
    )


@partial(
    jax.jit,
    static_argnames=("num_states", "chunk_size", "total_combinations", "n_chunks"),
)
def _extract_truncated_charge_states(
        vg: jnp.ndarray,
        cdd: jnp.ndarray,
        cdg: jnp.ndarray,
        n_deltas_per_dot: jnp.ndarray,
        delta_array_2d: jnp.ndarray,
        fixed_mask: jnp.ndarray,
        *,
        num_states: int,
        chunk_size: int,
        total_combinations: int,
        n_chunks: int,
) -> jnp.ndarray:
    """
    Select `num_states` low-energy integer charge configurations around the
    continuous solution without materializing the full mixed-radix grid.
    """
    n_dot = int(cdd.shape[0])
    # Pin float inputs to float32. Under JAX x64 (required by the sparse path for its
    # int64 state encoding) vg can arrive as float64, which would make `energies`
    # float64 while the scan carry `best_energies` is float32 -> a carry-dtype
    # mismatch in jax.lax.scan. The truncation/eigensolve are float32 throughout.
    vg = jnp.asarray(vg, dtype=jnp.float32)
    cdd = jnp.asarray(cdd, dtype=jnp.float32)
    cdg = jnp.asarray(cdg, dtype=jnp.float32)
    n_cont = _numerical_solver_open(cdd=cdd, cdg=cdg, vg=vg)  # (n_dot,)
    base = jnp.floor(n_cont).astype(jnp.int32)

    # For fixed dots: deterministically round (base -> base or base+1).
    frac = n_cont - jnp.floor(n_cont)
    base = base + fixed_mask * (frac > 0.5).astype(jnp.int32)

    # Mixed-radix powers for index -> per-dot delta option.
    # powers[d] multiplies the digit for dot d.
    # These products are bounded by max_total_combinations (<= 1e6 by default),
    # so int32 arithmetic is sufficient even when x64 is disabled.
    powers = jnp.cumprod(
        jnp.concatenate([jnp.array([1], dtype=jnp.int32), n_deltas_per_dot[::-1].astype(jnp.int32)[:-1]])
    )[::-1].astype(jnp.int32)  # (n_dot,)

    # Precompute for energy evaluation: v_dash = Cdg @ vg, and Cholesky factor.
    v_dash = cdg @ vg  # (n_dot,)
    chol = jax.scipy.linalg.cho_factor(cdd, lower=True)

    best_energies = jnp.full((num_states,), jnp.inf, dtype=jnp.float32)
    best_states = jnp.zeros((num_states, n_dot), dtype=jnp.int32)

    def process_chunk(carry, chunk_idx):
        best_e, best_s = carry
        start_idx = chunk_idx * jnp.asarray(chunk_size, dtype=jnp.int32)
        idxs = jnp.arange(chunk_size, dtype=jnp.int32) + start_idx  # (chunk_size,)
        within = idxs < int(total_combinations)
        safe = jnp.mod(idxs, int(total_combinations)).astype(jnp.int32)

        delta_idx = (safe[:, None] // powers[None, :]) % n_deltas_per_dot[None, :]  # (chunk_size, n_dot)

        deltas = delta_array_2d[jnp.arange(n_dot, dtype=jnp.int32)[:, None], delta_idx.T].T  # (chunk, n_dot)
        configs = base[None, :] + deltas

        valid = within & jnp.all(configs >= 0, axis=-1)

        vd = configs.astype(jnp.float32) - v_dash[None, :]
        x = jax.scipy.linalg.cho_solve(chol, vd.T).T
        energies = jnp.sum(vd * x, axis=-1)
        energies = jnp.where(valid, energies, jnp.inf)

        # Best K from this chunk
        k = int(num_states)
        idx_best = jnp.argsort(energies)[:k]
        e_best = energies[idx_best]
        s_best = configs[idx_best].astype(jnp.int32)

        # Merge with running best
        e_comb = jnp.concatenate([best_e, e_best], axis=0)
        s_comb = jnp.concatenate([best_s, s_best], axis=0)
        keep = jnp.argsort(e_comb)[:k]
        return (e_comb[keep], s_comb[keep]), None

    (best_energies, best_states), _ = jax.lax.scan(
        process_chunk, (best_energies, best_states), jnp.arange(int(n_chunks), dtype=jnp.int32)
    )
    return best_states


# -----------------------------------------------------------------------------
# Sparse tunneling matrix + Lanczos ground state solver
# -----------------------------------------------------------------------------

def _encode_charge_states(states: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    # NOTE: Requires x64 for collision-free encoding at larger n_dot.
    states = jnp.asarray(states, dtype=jnp.int64)
    base = jnp.asarray(base, dtype=jnp.int64)
    n_dot = states.shape[1]
    powers = base ** jnp.arange(n_dot - 1, -1, -1, dtype=jnp.int64)
    return jnp.sum(states * powers[None, :], axis=1)


def _tunneling_bcoo_from_basis(basis: jnp.ndarray, t: jnp.ndarray) -> jsparse.BCOO:
    """
    Build sparse tunneling Hamiltonian H_t in the provided truncated basis.

    The sparse pattern is constructed by encoding charge states to integer codes
    and using `searchsorted` to find neighbors under single-electron hops.
    """
    basis = jnp.asarray(basis, dtype=jnp.int32)
    t = jnp.asarray(t, dtype=jnp.float32)
    # Symmetrize t so the directed sparse elements form a Hermitian H, matching the
    # dense path (which applies 0.5*(H + H.T)). For already-symmetric t this is a
    # no-op; for asymmetric t it averages the forward/reverse hop amplitudes, which
    # is exactly the dense-symmetrized result (the occupation sqrt-factors agree).
    t = 0.5 * (t + t.T)
    n_states, n_dot = basis.shape

    # Choose a mixed-base encoding base big enough to be injective on `basis`.
    # This relies on x64 being enabled (otherwise int64 is truncated to int32).
    base = jnp.asarray(jnp.max(basis) + 2, dtype=jnp.int64)
    codes = _encode_charge_states(basis, base=base)  # (S,)
    perm = jnp.argsort(codes)
    codes_sorted = codes[perm]
    idx_sorted = perm.astype(jnp.int32)

    dots = jnp.arange(n_dot, dtype=jnp.int32)
    from_idx = jnp.repeat(dots, n_dot)  # (n_dot^2,)
    to_idx = jnp.tile(dots, n_dot)  # (n_dot^2,)
    is_hop = from_idx != to_idx

    # Neighbor codes for all (state, hop) pairs:
    # code' = code - pow[from] + pow[to]
    powers = base ** jnp.arange(n_dot - 1, -1, -1, dtype=jnp.int64)  # (n_dot,)
    neigh_codes = (
            codes[:, None]
            - powers[from_idx][None, :]
            + powers[to_idx][None, :]
    )  # (S, n_dot^2)

    # Find neighbor indices by binary search in the sorted code list.
    pos = jnp.searchsorted(codes_sorted, neigh_codes, side="left").astype(jnp.int32)
    pos_safe = jnp.minimum(pos, jnp.int32(n_states - 1))
    hit_code = codes_sorted[pos_safe]
    hit = (pos < n_states) & (hit_code == neigh_codes)

    cols = idx_sorted[pos_safe]  # (S, n_dot^2) (arbitrary where hit=False)
    rows = jnp.arange(n_states, dtype=jnp.int32)[:, None]
    rows = jnp.broadcast_to(rows, cols.shape)

    # Matrix elements (directed hops)
    n_from = basis[:, from_idx]  # (S, n_dot^2)
    n_to = basis[:, to_idx]  # (S, n_dot^2)
    t_vals = t[from_idx, to_idx]  # (n_dot^2,)

    valid = hit & is_hop[None, :] & (n_from > 0) & (t_vals[None, :] != 0.0)
    vals = -t_vals[None, :] * jnp.sqrt(n_from.astype(jnp.float32) * (n_to.astype(jnp.float32) + 1.0))
    vals = jnp.where(valid, vals, 0.0).astype(jnp.float32)

    # Flatten into BCOO storage (includes explicit zeros, but keeps shapes static for JIT).
    rows_f = rows.reshape(-1)
    cols_f = cols.reshape(-1)
    vals_f = vals.reshape(-1)
    indices = jnp.stack([rows_f, cols_f], axis=1)
    return jsparse.BCOO((vals_f, indices), shape=(n_states, n_states))


@partial(jax.jit, static_argnames=("n_iterations",))
def _lanczos_lowest_eigvec(
        H_t: jsparse.BCOO,
        diag: jnp.ndarray,
        *,
        n_iterations: int = 50,
) -> jnp.ndarray:
    """
    Lanczos iteration to approximate the lowest-eigenvalue eigenvector of:
      H = diag(diag) + H_t
    """
    n = int(diag.shape[0])
    diag = jnp.asarray(diag, dtype=jnp.float32)

    def matvec(v):
        return (H_t @ v) + diag * v

    k = int(min(max(2, n_iterations), n))
    v0 = jnp.ones((n,), dtype=jnp.float32) / jnp.sqrt(jnp.asarray(float(n), dtype=jnp.float32))

    alpha = jnp.zeros((k,), dtype=jnp.float32)
    beta = jnp.zeros((k - 1,), dtype=jnp.float32)
    V = jnp.zeros((n, k), dtype=jnp.float32).at[:, 0].set(v0)

    w0 = matvec(v0)
    alpha = alpha.at[0].set(jnp.dot(v0, w0))
    w0 = w0 - alpha[0] * v0

    def step(carry, j):
        v_prev, w, alpha, beta, V = carry
        b = jnp.linalg.norm(w)
        b_safe = jnp.maximum(b, 1e-12)
        beta = beta.at[j - 1].set(b)
        v = w / b_safe
        V = V.at[:, j].set(v)
        w2 = matvec(v)
        a = jnp.dot(v, w2)
        alpha = alpha.at[j].set(a)
        w2 = w2 - a * v - b * v_prev
        return (v, w2, alpha, beta, V), None

    (v_last, w_last, alpha, beta, V), _ = jax.lax.scan(
        step, (v0, w0, alpha, beta, V), jnp.arange(1, k, dtype=jnp.int32)
    )

    T = jnp.diag(alpha) + jnp.diag(beta, 1) + jnp.diag(beta, -1)
    evals, evecs = jnp.linalg.eigh(T)
    y0 = evecs[:, 0]  # lowest eigenvalue
    psi = V @ y0
    psi = psi / jnp.linalg.norm(psi)
    return psi.astype(jnp.float32)


@partial(
    jax.jit,
    static_argnames=("num_states", "chunk_size", "total_combinations", "n_chunks", "dense_threshold", "lanczos_iters"),
)
def _tunnel_coupled_ground_state_single(
        vg: jnp.ndarray,
        cdd: jnp.ndarray,
        cdg: jnp.ndarray,
        t: jnp.ndarray,
        n_deltas_per_dot: jnp.ndarray,
        delta_array_2d: jnp.ndarray,
        fixed_mask: jnp.ndarray,
        *,
        num_states: int,
        chunk_size: int,
        total_combinations: int,
        n_chunks: int,
        dense_threshold: int,
        lanczos_iters: int,
) -> jnp.ndarray:
    # Select truncated basis
    basis = _extract_truncated_charge_states(
        vg,
        cdd,
        cdg,
        n_deltas_per_dot,
        delta_array_2d,
        fixed_mask,
        num_states=int(num_states),
        chunk_size=int(chunk_size),
        total_combinations=int(total_combinations),
        n_chunks=int(n_chunks),
    )  # (S, n_dot)

    # Diagonal energies in the truncated basis
    F = _free_energy(basis, cdd, cdg, vg).astype(jnp.float32)  # (S,)

    S = int(num_states)
    if S <= int(dense_threshold):
        H = jnp.diag(F) + _tunneling_dense(t=t, basis=basis)
        _evals, evecs = jnp.linalg.eigh(H)
        psi0 = evecs[:, 0].astype(jnp.float32)
    else:
        # Tunneling Hamiltonian in truncated basis (sparse path)
        H_t = _tunneling_bcoo_from_basis(basis=basis, t=t)
        psi0 = _lanczos_lowest_eigvec(H_t, F, n_iterations=int(lanczos_iters))

    probs = jnp.square(jnp.abs(psi0))
    n_expect = probs @ basis.astype(jnp.float32)
    return n_expect


def _maybe_multi_device_map(fn, vg_batch: jnp.ndarray) -> jnp.ndarray:
    """
    Apply `fn` to vg_batch (P, n_gates) with optional sharding across devices.
    `fn` must accept vg_batch on axis 0 (i.e. vmappable).
    """
    vg_batch = jnp.asarray(vg_batch)
    P = int(vg_batch.shape[0])
    # IMPORTANT: `pmap` maps over *local* devices for this process.
    # In multi-host setups, `jax.devices()` may include non-addressable devices,
    # so always use `local_device_count()` here.
    n_devices = int(jax.local_device_count())
    if n_devices <= 1:
        return fn(vg_batch)

    # Simple heuristic: only shard if there's enough work.
    if P < 8 * n_devices:
        return fn(vg_batch)

    pad = (-P) % n_devices
    if pad:
        vg_pad = jnp.repeat(vg_batch[-1:, :], pad, axis=0)
        vg_batch = jnp.concatenate([vg_batch, vg_pad], axis=0)
    P2 = int(vg_batch.shape[0])
    per = P2 // n_devices
    vg_sharded = vg_batch.reshape(n_devices, per, vg_batch.shape[1])

    @partial(jax.pmap, in_axes=0)
    def per_device(vg_local):
        return fn(vg_local)

    out = per_device(vg_sharded).reshape(P2, -1)
    if pad:
        out = out[:P, :]
    return out


def tunnel_coupled_ground_state(
        cdd: jnp.ndarray,
        cdg: jnp.ndarray,
        t: jnp.ndarray,
        vg: jnp.ndarray,
        n_truncate: Optional[int] = 32,
) -> jnp.ndarray:
    """
    Ground state including inter-dot tunneling, evaluated over a batch of gate voltages.

    Parameters
    ----------
    cdd : (n_dot, n_dot)
    cdg : (n_dot, n_gates)
    t   : (n_dot, n_dot)
    vg  : (P, n_gates)
    n_truncate : int
        Number of truncated basis states per pixel (default 32).
    """
    vg = jnp.asarray(vg)
    if vg.ndim != 2:
        vg = vg.reshape(-1, vg.shape[-1])

    # Truncation metadata is built on host from constant cdg (cheap).
    cdg_np = np.asarray(jax.device_get(cdg))
    n_trunc = 32 if n_truncate is None else int(n_truncate)
    n_deltas_per_dot, delta_array_2d, fixed_mask, total, chunk, n_chunks = _build_truncation_meta(
        cdg=cdg_np,
        n_truncate=n_trunc,
    )
    # Effective state count cannot exceed number of enumerated combinations.
    num_states = int(min(max(1, n_trunc), max(1, total)))

    dense_threshold = 256
    lanczos_iters = 60

    # The sparse neighbor lookup uses int64 hash-encoding; require x64 when needed.
    if num_states > dense_threshold and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "Sparse tunnel-coupled solver requires x64-enabled JAX when "
            f"num_states={num_states} > dense_threshold={dense_threshold}. "
            "Set `JAX_ENABLE_X64=1` (or `jax.config.update('jax_enable_x64', True)` "
            "before importing JAX) or reduce `n_truncate`."
        )

    single = partial(
        _tunnel_coupled_ground_state_single,
        cdd=jnp.asarray(cdd, dtype=jnp.float32),
        cdg=jnp.asarray(cdg, dtype=jnp.float32),
        t=jnp.asarray(t, dtype=jnp.float32),
        n_deltas_per_dot=n_deltas_per_dot,
        delta_array_2d=delta_array_2d,
        fixed_mask=fixed_mask,
        num_states=num_states,
        chunk_size=int(chunk),
        total_combinations=int(total),
        n_chunks=int(n_chunks),
        dense_threshold=int(dense_threshold),
        lanczos_iters=int(lanczos_iters),
    )
    batched = jax.vmap(single, in_axes=0, out_axes=0)
    return _maybe_multi_device_map(batched, vg)


def tunnel_coupled_ground_state_fn(
        cdd_fn,
        cdg_fn,
        t_fn,
        vg: jnp.ndarray,
        n_truncate: Optional[int] = 32,
) -> jnp.ndarray:
    """
    Callable-parameter version of tunnel_coupled_ground_state.

    Note: truncation metadata is built once from cdg_fn(vg0), so this is best
    used when `cdg_fn` is approximately constant over the scan.
    """
    vg = jnp.asarray(vg)
    if vg.ndim != 2:
        vg = vg.reshape(-1, vg.shape[-1])
    vg0 = vg[0]
    cdd0 = jnp.asarray(cdd_fn(vg0), dtype=jnp.float32)
    cdg0 = jnp.asarray(cdg_fn(vg0), dtype=jnp.float32)
    t0 = jnp.asarray(t_fn(vg0), dtype=jnp.float32)

    cdg_np = np.asarray(jax.device_get(cdg0))
    n_trunc = 32 if n_truncate is None else int(n_truncate)
    n_deltas_per_dot, delta_array_2d, fixed_mask, total, chunk, n_chunks = _build_truncation_meta(
        cdg=cdg_np,
        n_truncate=n_trunc,
    )
    num_states = int(min(max(1, n_trunc), max(1, total)))

    dense_threshold = 256
    lanczos_iters = 60

    if num_states > dense_threshold and not bool(jax.config.read("jax_enable_x64")):
        raise ValueError(
            "Sparse tunnel-coupled solver requires x64-enabled JAX when "
            f"num_states={num_states} > dense_threshold={dense_threshold}. "
            "Set `JAX_ENABLE_X64=1` (or `jax.config.update('jax_enable_x64', True)` "
            "before importing JAX) or reduce `n_truncate`."
        )

    # NOTE: `single` takes only the traced `vg_i`; the truncation metadata and the
    # ints below are closed-over constants baked into the trace (and forwarded as the
    # already-static args of `_tunnel_coupled_ground_state_single`). Do NOT list them in
    # static_argnames here -- they are not parameters of `single`, which raises
    # "Jitted function has invalid argnames ... in static_argnames".
    @jax.jit
    def single(vg_i):
        return _tunnel_coupled_ground_state_single(
            vg_i,
            jnp.asarray(cdd_fn(vg_i), dtype=jnp.float32),
            jnp.asarray(cdg_fn(vg_i), dtype=jnp.float32),
            jnp.asarray(t_fn(vg_i), dtype=jnp.float32),
            n_deltas_per_dot,
            delta_array_2d,
            fixed_mask,
            num_states=int(num_states),
            chunk_size=int(chunk),
            total_combinations=int(total),
            n_chunks=int(n_chunks),
            dense_threshold=int(dense_threshold),
            lanczos_iters=int(lanczos_iters),
        )

    batched = jax.vmap(single, in_axes=0, out_axes=0)
    return _maybe_multi_device_map(batched, vg)
