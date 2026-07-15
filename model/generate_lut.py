"""
Precompute a lookup table of clean (noise-free) sensor signals over a 4D grid of
device parameters, so api/simulate can interpolate at request time with plain
numpy instead of needing jax/jaxlib/scipy/qarray_plus deployed on Vercel.

Grid covers the same BOUNDS as api/simulate/index.py. noise_std is intentionally
not gridded -- it's applied at request time (see api/simulate/index.py), since
ChargeSensor's noise model is pure numpy already and cheap to reapply live.

Output: ../data/simulate_lut.npz
    t_grid, cdd_off_grid, cdg_cross0_grid, cdg_cross1_grid : 1D float32 arrays
    signals : float32 array of shape (nt, ncdd, ncdg0, ncdg1, grid_size)

Usage:
    python generate_lut.py --n_points 20 --grid_size 64
"""

import argparse
import itertools
import os
import time

import jax.numpy as jnp
import numpy as np

from qarray_plus import DotArray, ChargeSensor

BOUNDS = {
    "t": (0.0, 0.15),
    "cdd_off": (0.01, 0.15),
    "cdg_cross0": (0.05, 0.2),
    "cdg_cross1": (0.05, 0.2),
}


def simulate_clean(t, cdd_off, cdg_cross0, cdg_cross1, grid_size):
    """Same physics as api/simulate/index.py's simulate_signal, with noise_std=0."""
    cdd = jnp.array([[1.0, -cdd_off], [-cdd_off, 1.0]])
    cdg = -jnp.array([[1.0, cdg_cross0], [cdg_cross1, 1.0]])
    t_mat = jnp.array([[0.0, t], [t, 0.0]])

    charge_sensor = ChargeSensor(
        n_dots=2,
        n_gates=2,
        n_sensor=1,
        csd=jnp.array([0.02, 0.1]),
        csg=-jnp.array([0.3, 0.3]),
        pink_noise_std=0.0,
        white_noise_std=0.0,
    )

    model = DotArray(n_dots=2, n_gates=2, cdd=cdd, cdg=cdg, t=t_mat)

    v0 = model.optimal_vg([0.5, 0.5])
    eps = jnp.linspace(-0.5, 0.5, grid_size)
    vg = v0[None, :] + jnp.stack([eps / 2, -eps / 2], axis=-1)

    result = model.tunnel_coupled_ground_state(vg, charge_sensor=charge_sensor)
    return np.array(result.sensor).squeeze().astype(np.float32)


def generate(n_points: int, grid_size: int, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    t_grid = np.linspace(*BOUNDS["t"], n_points, dtype=np.float32)
    cdd_off_grid = np.linspace(*BOUNDS["cdd_off"], n_points, dtype=np.float32)
    cdg_cross0_grid = np.linspace(*BOUNDS["cdg_cross0"], n_points, dtype=np.float32)
    cdg_cross1_grid = np.linspace(*BOUNDS["cdg_cross1"], n_points, dtype=np.float32)

    signals = np.zeros((n_points, n_points, n_points, n_points, grid_size), dtype=np.float32)

    total = n_points ** 4
    start = time.time()
    count = 0
    for i, t in enumerate(t_grid):
        for j, cdd_off in enumerate(cdd_off_grid):
            for k, cdg_cross0 in enumerate(cdg_cross0_grid):
                for l, cdg_cross1 in enumerate(cdg_cross1_grid):
                    signals[i, j, k, l] = simulate_clean(
                        float(t), float(cdd_off), float(cdg_cross0), float(cdg_cross1), grid_size
                    )
                    count += 1
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0
            print(f"  {count}/{total} ({rate:.0f}/s, {elapsed:.0f}s elapsed)")

    np.savez_compressed(
        out_path,
        t_grid=t_grid,
        cdd_off_grid=cdd_off_grid,
        cdg_cross0_grid=cdg_cross0_grid,
        cdg_cross1_grid=cdg_cross1_grid,
        signals=signals,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved {total} grid points to {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_points", type=int, default=20, help="grid points per parameter axis")
    parser.add_argument("--grid_size", type=int, default=64, help="signal length per grid point")
    parser.add_argument("--out", type=str, default="../data/simulate_lut.npz")
    args = parser.parse_args()

    generate(args.n_points, args.grid_size, args.out)
