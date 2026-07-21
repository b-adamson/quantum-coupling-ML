
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from qarray_plus import DotArray, ChargeSensor

# qarray_plus's ε and t are dimensionless (a fraction of the charging energy
# E_C -- dot self-capacitance is normalised to 1.0). There's no real E_C in
# the simulator itself, so to show them in meV we have to assume one. 1 meV
# is a typical charging energy for a GaAs double quantum dot -- change this
# if you're modelling a specific device with a known/measured E_C.
E_C_MEV = 1.0

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

# --- Noisy realisations, matching generate.py's per-sample randomisation ---
# (sensor noise_std in [0, 0.1], background tilt slope in [-0.3, 0.3])
NOISY_T = 0.08
N_REALISATIONS = 5
rng = np.random.default_rng(0)

t_mat = jnp.array([[0.0, NOISY_T], [NOISY_T, 0.0]])
clean_model = DotArray(n_dots=2, n_gates=2, cdd=CDD, cdg=CDG, t=t_mat)
v0 = clean_model.optimal_vg([0.5, 0.5])
vg = v0[None, :] + jnp.stack([jnp.array(eps / 2), jnp.array(-eps / 2)], axis=-1)
clean_result = clean_model.tunnel_coupled_ground_state(vg, charge_sensor=CHARGE_SENSOR)
clean_trace = np.array(clean_result.sensor).squeeze()

noisy_traces = []
for _ in range(N_REALISATIONS):
    noise_std = float(rng.uniform(0.0, 0.01))
    slope = float(rng.uniform(-0.05, 0.05))

    noisy_sensor = ChargeSensor(
        n_dots=2, n_gates=2, n_sensor=1,
        csd=jnp.array([0.02, 0.1]), csg=-jnp.array([0.3, 0.3]),
        pink_noise_std=noise_std, white_noise_std=noise_std,
    )
    result = clean_model.tunnel_coupled_ground_state(vg, charge_sensor=noisy_sensor)
    signal = np.array(result.sensor).squeeze() + slope * eps
    noisy_traces.append((signal, noise_std, slope))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

# Plot 1: overlaid I(ε) traces ---
cmap = cm.plasma
colours = [cmap(i / max(len(T_VALUES) - 1, 1)) for i in range(len(T_VALUES))]

eps_mev = eps * E_C_MEV

ax = axes[0]
for t_val, trace, colour in zip(T_VALUES, traces, colours):
    ax.plot(eps_mev, trace, label=f"t = {t_val:.2f}", color=colour, linewidth=1.8)

ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
ax.set_xlabel(f"Detuning  ε  (meV, assuming E_C = {E_C_MEV:.1f} meV)")
ax.set_ylabel("Sensor signal  I(ε)")
ax.set_title("Detuning sweep through interdot transition")
ax.legend(fontsize=9)

# Plot 2: slope at ε=0 vs t
ax = axes[1]
t_mev = np.array(T_VALUES) * E_C_MEV
ax.plot(t_mev, slopes, "o-", color="steelblue", linewidth=2, markersize=7)
ax.set_xlabel(f"Tunnel coupling  t  (meV, assuming E_C = {E_C_MEV:.1f} meV)")
ax.set_ylabel("|dI/dε| at ε = 0")
ax.set_title("Slope at transition centre vs tunnel coupling")

# Plot 3: clean vs. noisy/tilted realisations at a fixed t
ax = axes[2]
ax.plot(eps_mev, clean_trace, color="black", linewidth=2.2, label="clean", zorder=5)
noisy_cmap = cm.viridis
for i, (signal, noise_std, slope) in enumerate(noisy_traces):
    colour = noisy_cmap(i / max(N_REALISATIONS - 1, 1))
    ax.plot(eps_mev, signal, color=colour, linewidth=1.1, alpha=0.8,
            label=f"noise={noise_std:.3f}, slope={slope:+.2f}")

ax.axvline(0, color="grey", linewidth=0.7, linestyle="--")
ax.set_xlabel(f"Detuning  ε  (meV, assuming E_C = {E_C_MEV:.1f} meV)")
ax.set_ylabel("Sensor signal  I(ε)")
ax.set_title(f"Noisy/tilted realisations (t = {NOISY_T:.2f})")
ax.legend(fontsize=7)

plt.tight_layout()
import os
os.makedirs("../data", exist_ok=True)
plt.savefig("../data/detuning_traces.png", dpi=150)
plt.show()
print("Saved figure to ../data/detuning_traces.png")
