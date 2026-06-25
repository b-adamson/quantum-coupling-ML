"""
Visualise simulated CSDs for different tunnel couplings.

Usage:
    python visualise.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from qarray_plus import DotArray, ChargeSensor

GRID_SIZE = 100
T_VALUES = [0.0, 0.02, 0.05, 0.12]

CDD = jnp.array([[1.0, -0.05], [-0.05, 1.0]])
CDG = -jnp.array([[1.0, 0.1], [0.1, 1.0]])

CHARGE_SENSOR = ChargeSensor(
    n_dots=2,
    n_gates=2,
    n_sensor=1,
    csd=jnp.array([0.1, 0.02]),
    csg=-jnp.array([0.5, 0.15]),
    pink_noise_std=0.01,
    white_noise_std=0.005,
)

vg = jnp.stack(
    jnp.meshgrid(
        jnp.linspace(-3, 0, GRID_SIZE),
        jnp.linspace(-3, 0, GRID_SIZE),
    ),
    axis=-1,
)
extent = (float(vg[..., 0].min()), float(vg[..., 0].max()),
          float(vg[..., 1].min()), float(vg[..., 1].max()))

fig, axes = plt.subplots(1, len(T_VALUES), figsize=(4 * len(T_VALUES), 4))

for ax, t_val in zip(axes, T_VALUES):
    t_mat = jnp.array([[0.0, t_val], [t_val, 0.0]])
    model = DotArray(n_dots=2, n_gates=2, cdd=CDD, cdg=CDG, t=t_mat)
    result = model.tunnel_coupled_ground_state(vg, charge_sensor=CHARGE_SENSOR)
    signal = np.array(result.sensor).squeeze()

    ax.imshow(signal, origin="lower", extent=extent, aspect="equal", cmap="inferno")
    ax.set_title(f"t = {t_val:.2f}")
    ax.set_xlabel("$V_1$")
    ax.set_ylabel("$V_2$")

plt.suptitle("CSD for increasing tunnel coupling — note avoided crossings growing", fontsize=10)
plt.tight_layout()
plt.show()
