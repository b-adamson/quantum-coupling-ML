# QArray+

**A physics-informed, GPU-accelerated simulator for semiconductor quantum dot arrays.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/accelerator-JAX%20%7C%20GPU-green)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

`QArray+` extends the original [`QArray`](https://github.com/b-van-straaten/qarray) framework to capture phenomena that
constant-capacitance equilibrium simulators cannot: **non-equilibrium charge latching**, **coherent interdot
hybridization**, and **open-system dissipative dynamics**. It is designed from the ground up for scalability, running
efficiently on a single CPU, a single GPU, or a multi-GPU / multi-node cluster via [JAX](https://github.com/google/jax).

---

## Why QArray+?

Modern semiconductor quantum dot experiments increasingly use fast RF reflectometry, where the measurement rate can
exceed charge tunneling rates. In this regime, standard equilibrium simulators that minimize electrostatic energy at
each gate-voltage point produce incorrect charge-stability diagrams (CSDs): the system cannot relax between
measurements, and the resulting **latching** and **hysteresis** are missed entirely.

Simultaneously, finite interdot tunnel coupling hybridizes near-degenerate charge configurations, producing **avoided
crossings** and fractional charge expectation values — effects absent from any purely classical capacitance model.

`QArray+` addresses both limitations through three complementary simulation heads:

| Model                          | Latching | Hybridization | Complexity                |
|--------------------------------|----------|---------------|---------------------------|
| Stochastic Capacitance         | ✓        | ✗             | O(N² N_r n_dot⁴)          |
| Spinless Hubbard               | ✗        | ✓             | O(N² · truncated basis)   |
| Open Quantum System (Lindblad) | ✓        | ✓             | O(N² · Hilbert space dim) |

---

## Features

- **Three simulation heads** — stochastic Markov-jump, spinless Hubbard ground state, and Lindblad master equation —
  selectable through a unified interface.
- **Gate-voltage-dependent capacitances and tunnel couplings** — reproduce non-linear electrostatics across a wide range
  of operating conditions.
- **Pauli spin blockade** — suppresses specified interdot transitions when both dots carry an odd occupation.
- **Isolated dot support** — model dots fully decoupled from reservoirs with user-specified initial charge states.
- **Realistic sensor readout** — generate charge-sensor signals with white and 1/f noise overlaid on simulated CSDs.
- **Finite-temperature effects** — all transition rates are modulated by Fermi–Dirac statistics.
- **JAX backend** — just-in-time compilation, automatic differentiation, and first-class GPU/TPU support.
- **Multi-GPU / multi-node scaling** — shard large scans across accelerators; a 64-dot 100×100 CSD completes in ~0.17 s
  on 8× H100.
- **Backwards compatible** — the original `QArray` equilibrium solver is preserved as an option.

---

## Installation

```bash
pip install qarray-plus
```

For GPU support, install JAX with CUDA before installing `QArray+`:

```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install qarray-plus
```

---

## Quick Start

### Stochastic latching (non-equilibrium CSD)

```python
import qarray_plus as qap
import jax.numpy as jnp

# Define a double-dot capacitance model
model = qap.StochasticModel(
    C_dd=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    C_dg=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    tunnel_rates=jnp.array([1.0, 0.5]),  # Γ_i τ
    temperature=0.01,  # k_B T in units of charging energy
)

# Sweep gate voltages and compute a 200×200 CSD
vg1 = jnp.linspace(-1, 1, 200)
vg2 = jnp.linspace(-1, 1, 200)
csd = model.compute_csd(vg1, vg2)
```

### Spinless Hubbard (coherent hybridization)

```python
model = qap.HubbardModel(
    C_dd=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    C_dg=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    tunnel_matrix=jnp.array([[0.0, 0.05], [0.05, 0.0]]),  # t_ij
    n_truncate=16,
)
csd = model.compute_csd(vg1, vg2)
```

### Open quantum system (Lindblad — latching + hybridization)

```python
model = qap.LindbladModel(
    C_dd=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    C_dg=jnp.array([[1.0, 0.1], [0.1, 1.0]]),
    tunnel_matrix=jnp.array([[0.0, 0.05], [0.05, 0.0]]),
    lead_rates=jnp.array([1.0, 0.0]),  # Γ_lead,i
    temperature=0.01,
    n_max=2,
)
csd = model.compute_csd(vg1, vg2, dt=1.0, row_reset=True)
```

---

## Models in Detail

### Stochastic Capacitance Model

Charge dynamics are modeled as a classical Markov jump process. At each pixel (integration window of duration τ), the
algorithm enumerates all elementary transitions — lead loading/unloading and interdot hops — computes their rates via
Fermi–Dirac statistics, and performs a single stochastic Bernoulli trial. The state is then propagated to the next
pixel. This captures latching in the slow-tunneling regime (`Γτ ≪ 1`) at a cost that scales polynomially as **O(n_dot⁴)
** per pixel, a dramatic improvement over equilibrium solvers that scale exponentially.

Pauli spin blockade (PSB) is supported via a rank-3 rate tensor that suppresses interdot transitions between states with
odd occupation on both dots.

### Spinless Hubbard Model

The electrostatic free energy is promoted to a diagonal Hamiltonian in the charge (occupation-number) basis, augmented
by coherent hopping terms. For each gate-voltage point, the ground state is found by diagonalizing this Hamiltonian
within a **truncated charge basis** of fixed size `n_truncate`, selected by solving a relaxed (continuous) version of
the electrostatic problem and keeping the lowest-free-energy integer configurations. Sparse Lanczos iteration is used
for large truncated dimensions, keeping the per-point cost controlled rather than exponentially scaling with dot number.

### Open Quantum System (Lindblad)

The system density matrix ρ evolves under the Lindblad master equation:

```
dρ/dt = -i[H, ρ] + Σ_e ( L_e ρ L_e† - ½{L_e†L_e, ρ} ) - γ_φ (ρ - Σ_n |n⟩⟨n|ρ|n⟩⟨n|)
```

Each incoherent charge-transfer edge defines a jump operator `L_e = √κ_e |d_e⟩⟨s_e|`, with rates satisfying detailed
balance. The density matrix is propagated pixel-by-pixel along the scan direction using RK4 integration. Row-reset modes
allow initialization to a thermal (Gibbs) state, the minimum-energy charge state, or a maximally mixed state. Pure
dephasing `γ_φ` can be added to suppress off-diagonal coherences and interpolate between the quantum and classical
limits.

---

## Benchmarks

All benchmarks on 8× NVIDIA H100 SXM5 80 GB GPUs, 100×100 CSD resolution.

**Stochastic latching vs. dot count:**

| n_dot | CPU    | Single H100 | 8× H100  |
|-------|--------|-------------|----------|
| 16    | < 1 s  | < 0.1 s     | < 0.05 s |
| 64    | ~2.8 s | ~0.4 s      | ~0.17 s  |

The original `QArray` baseline becomes impractical beyond ~16 dots; `QArray+` latching scales polynomially to 64+ dots.

**Hubbard ground state vs. QDarts:**

`QDarts` exceeds 200 s at 5 dots. `QArray+` (multi-GPU) remains tractable to 20+ dots using the truncated basis and
sparse eigensolvers.

**Lindblad open system:**

| n_dot | CPU     | Single H100 | 8× H100 |
|-------|---------|-------------|---------|
| 10    | ~1300 s | ~110 s      | ~30 s   |

---

## Comparison with Related Tools

| Feature                  | QArray | QArray+ | QDSim | QDFlow | QDarts | RF-Squad |
|--------------------------|--------|---------|-------|--------|--------|----------|
| Equilibrium CSD          | ✓      | ✓       | ✓     | ✓      | ✓      | ✓        |
| Physical latching        | ✗      | ✓       | ✗     | ✗      | ✗      | ✗        |
| Coherent hybridization   | ✗      | ✓       | ✗     | ✗      | ✓      | ✓        |
| Open quantum system      | ✗      | ✓       | ✗     | ✗      | ✗      | ✗        |
| Gate-voltage-dependent C | ✗      | ✓       | ✗     | ✗      | ✗      | ✗        |
| GPU acceleration         | ✗      | ✓       | ✗     | ✗      | ✗      | ✗        |
| Multi-GPU / multi-node   | ✗      | ✓       | ✗     | ✗      | ✗      | ✗        |

---

## Citation

If you use `QArray+` in your research, please cite:


---


---

## License

MIT License. See [LICENSE](LICENSE) for details.