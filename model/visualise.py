"""
Visualise 1D detuning sweeps for different tunnel couplings.

Usage:
    python visualise.py
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from qarray_plus import DotArray, ChargeSensor

GRID_SIZE = 200
T_VALUES = [0.0, 0.02, 0.05, 0.08, 0.12, 0.15]

CDD = jnp.array([[1.0, -0.05], [-0.05, 1.0]])
CDG = -jnp.array([[1.0, 0.1], [0.1, 1.0]])

CHARGE_SENSOR = ChargeSensor(
    n_dots=2,
    n_gates=2,
    n_sensor=1,
    csd=jnp.array([0.02, 0.1]),
    csg=-jnp.array([0.3, 0.3]),
    pink_noise_std=0.0,
    white_noise_std=0.0,
)

eps = np.linspace(-0.5, 0.5, GRID_SIZE)

traces = []
for t_val in T_VALUES:
    t_mat = jnp.array([[0.0, t_val], [t_val, 0.0]])
    model = DotArray(n_dots=2, n_gates=2, cdd=CDD, cdg=CDG, t=t_mat)

    # Centre on the (1,0)↔(0,1) interdot transition
    v0 = model.optimal_vg([0.5, 0.5])
    vg = v0[None, :] + jnp.stack([jnp.array(eps / 2), jnp.array(-eps / 2)], axis=-1)

    result = model.tunnel_coupled_ground_state(vg, charge_sensor=CHARGE_SENSOR)
    signal = np.array(result.sensor).squeeze()
    traces.append(signal)

# Slope at ε=0: central difference around midpoint
mid = GRID_SIZE // 2
deps = eps[mid + 1] - eps[mid - 1]
slopes = [abs(tr[mid + 1] - tr[mid - 1]) / deps for tr in traces]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Plot 1: overlaid I(ε) traces ---
cmap = cm.plasma
colours = [cmap(i / max(len(T_VALUES) - 1, 1)) for i in range(len(T_VALUES))]

ax = axes[0]
for t_val, trace, colour in zip(T_VALUES, traces, colours):
    ax.plot(eps, trace, label=f"t = {t_val:.2f}", color=colour, linewidth=1.8)

ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
ax.set_xlabel("Detuning  ε  (arb. units)")
ax.set_ylabel("Sensor signal  I(ε)")
ax.set_title("Detuning sweep through interdot transition")
ax.legend(fontsize=9)

# --- Plot 2: slope at ε=0 vs t ---
ax = axes[1]
ax.plot(T_VALUES, slopes, "o-", color="steelblue", linewidth=2, markersize=7)
ax.set_xlabel("Tunnel coupling  t")
ax.set_ylabel("|dI/dε| at ε = 0")
ax.set_title("Slope at transition centre vs tunnel coupling")

plt.tight_layout()
import os
os.makedirs("../data", exist_ok=True)
plt.savefig("../data/detuning_traces.png", dpi=150)
plt.show()
print("Saved figure to ../data/detuning_traces.png")
