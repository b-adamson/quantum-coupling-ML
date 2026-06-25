"""
Generate training data: random double-dot parameters → (N×N sensor signal, t).
Output saved to ../data/dataset.h5

Usage:
    python generate.py --n_samples 5000 --grid_size 64
"""

import argparse
import os

import h5py
import jax.numpy as jnp
import numpy as np
from jax import random

from qarray_plus import DotArray, ChargeSensor


def random_cdd(key):
    """Maxwell capacitance matrix for 2 dots: diagonal >> off-diagonal."""
    key, subkey = random.split(key)
    off = random.uniform(subkey, shape=(), minval=0.01, maxval=0.15)
    cdd = jnp.array([[1.0, -off], [-off, 1.0]])
    return cdd, key


def random_cdg(key):
    """Gate-to-dot capacitance: strong diagonal, weak cross-coupling."""
    key, subkey = random.split(key)
    cross = random.uniform(subkey, shape=(2,), minval=0.05, maxval=0.2)
    cdg = -jnp.array([[1.0, cross[0]], [cross[1], 1.0]])
    return cdg, key


def random_t(key, t_min=0.0, t_max=0.15):
    """Scalar interdot tunnel coupling."""
    key, subkey = random.split(key)
    t_val = random.uniform(subkey, shape=(), minval=t_min, maxval=t_max)
    t_mat = jnp.array([[0.0, t_val], [t_val, 0.0]])
    return t_mat, float(t_val), key


def make_voltage_grid(grid_size: int):
    vg = jnp.stack(
        jnp.meshgrid(
            jnp.linspace(-3, 0, grid_size),
            jnp.linspace(-3, 0, grid_size),
        ),
        axis=-1,
    )
    return vg


def simulate_one(cdd, cdg, t_mat, vg):
    model = DotArray(n_dots=2, n_gates=2, cdd=cdd, cdg=cdg, t=t_mat)

    charge_sensor = ChargeSensor(
        n_dots=2,
        n_gates=2,
        n_sensor=1,
        csd=jnp.array([0.1, 0.02]),
        csg=-jnp.array([0.5, 0.15]),
        pink_noise_std=0.01,
        white_noise_std=0.005,
    )

    result = model.tunnel_coupled_ground_state(vg, charge_sensor=charge_sensor)
    return np.array(result.sensor)  # shape (grid_size, grid_size)


def generate(n_samples: int, grid_size: int, out_path: str, seed: int = 42):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    vg = make_voltage_grid(grid_size)
    key = random.PRNGKey(seed)

    signals = np.zeros((n_samples, grid_size, grid_size), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        cdd, key = random_cdd(key)
        cdg, key = random_cdg(key)
        t_mat, t_val, key = random_t(key)

        signal = simulate_one(cdd, cdg, t_mat, vg)
        signals[i] = signal.squeeze()
        labels[i] = t_val

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_samples}")

    with h5py.File(out_path, "w") as f:
        f.create_dataset("signals", data=signals)
        f.create_dataset("labels", data=labels)
        f.attrs["grid_size"] = grid_size
        f.attrs["n_samples"] = n_samples

    print(f"Saved {n_samples} samples to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--grid_size", type=int, default=64)
    parser.add_argument("--out", type=str, default="../data/dataset.h5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate(args.n_samples, args.grid_size, args.out, args.seed)
