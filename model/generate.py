import argparse
import os

import h5py
import jax.numpy as jnp
import numpy as np
from jax import random

from qarray_plus import DotArray, ChargeSensor


def random_cdd(key):
    key, subkey = random.split(key)
    off = random.uniform(subkey, shape=(), minval=0.01, maxval=0.15)
    cdd = jnp.array([[1.0, -off], [-off, 1.0]])
    return cdd, key

def random_cdg(key):
    key, subkey = random.split(key)
    cross = random.uniform(subkey, shape=(2,), minval=0.05, maxval=0.2)
    cdg = -jnp.array([[1.0, cross[0]], [cross[1], 1.0]])
    return cdg, key

def random_t(key, t_min=0.0, t_max=0.15):
    key, subkey = random.split(key)
    t_val = random.uniform(subkey, shape=(), minval=t_min, maxval=t_max)
    t_mat = jnp.array([[0.0, t_val], [t_val, 0.0]])
    return t_mat, float(t_val), key

def random_noise_std(key, noise_min=0.0, noise_max=0.1):
    key, subkey = random.split(key)
    noise_std = random.uniform(subkey, shape=(), minval=noise_min, maxval=noise_max)
    return float(noise_std), key

def simulate_one(cdd, cdg, t_mat, grid_size, noise_std):
    charge_sensor = ChargeSensor(
        n_dots=2,
        n_gates=2,
        n_sensor=1,
        csd=jnp.array([0.02, 0.1]),
        csg=-jnp.array([0.3, 0.3]),
        pink_noise_std=noise_std,
        white_noise_std=noise_std,
    )

    model = DotArray(n_dots=2, n_gates=2, cdd=cdd, cdg=cdg, t=t_mat)

    v0 = model.optimal_vg([0.5, 0.5])
    v0_gate1, v0_gate2 = v0[0], v0[1]

    eps = jnp.linspace(-0.5, 0.5, grid_size)
    gate1_voltages = v0_gate1 + eps / 2 
    gate2_voltages = v0_gate2 - eps / 2  

    vg = jnp.stack([gate1_voltages, gate2_voltages], axis=-1) 

    result = model.tunnel_coupled_ground_state(vg, charge_sensor=charge_sensor)
    return np.array(result.sensor).squeeze().astype(np.float32)

def generate(n_samples: int, grid_size: int, out_path: str, seed: int = 42):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    key = random.PRNGKey(seed)

    signals = np.zeros((n_samples, grid_size), dtype=np.float32)
    labels = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        cdd, key = random_cdd(key)
        cdg, key = random_cdg(key)
        t_mat, t_val, key = random_t(key)
        noise_std, key = random_noise_std(key)

        signal = simulate_one(cdd, cdg, t_mat, grid_size, noise_std)
        signals[i] = signal
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
