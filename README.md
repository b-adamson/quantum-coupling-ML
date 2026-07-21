# Quantum Dot Coupling Prediction Model

Predicts interdot tunnel coupling `t` from a 1D charge sensor detuning sweep through the interdot transition.

## How it works

A double quantum dot device is characterised by two capacitance matrices:

- **`cdd` (2×2)** — dot-to-dot capacitance. The diagonal (self-capacitance) is fixed at 1; the off-diagonal is the mutual capacitance between the two dots, controlling how strongly they electrostatically repel each other.
- **`cdg` (2×2)** — gate-to-dot capacitance. The diagonal is the lever arm of each gate on its own dot; the off-diagonals are cross-coupling terms (how much gate 1 bleeds onto dot 2 and vice versa).

For each training sample, `cdd`, `cdg`, and `t` are drawn randomly to represent a different physical device geometry. These are handed to **QArray-plus**, which solves for the quantum ground state of the system as gate voltages are swept anti-diagonally through the (1,0)↔(0,1) interdot charge transition. The charge sensor signal comes out as an erf-shaped step — flat low plateau, smooth ramp, flat high plateau. The width of the ramp encodes `t`: large `t` → broad ramp, small `t` → sharp step.

5000 such (signal, t) pairs are generated and a 1D CNN is trained to map the signal shape back to `t`. By randomising `cdd` and `cdg` across samples, the model learns to extract `t` regardless of device-to-device variation in capacitances.

### `simulate_one`

The core simulation function (in `generate.py`, and reused by `demo.py`). Given one set of device parameters it:

1. Builds a `ChargeSensor` with fixed coupling params — equal `csg` values cancel the linear background so the output is a clean erf; `csd=[0.02, 0.1]` makes the signal run low→high across the transition
2. Calls `optimal_vg([0.5, 0.5])` to find the gate voltages that sit exactly on the (1,0)↔(0,1) interdot transition
3. Constructs a 64-point anti-diagonal sweep (V₁ = +ε/2, V₂ = −ε/2) around that centre
4. Calls `tunnel_coupled_ground_state` — QArray solves the quantum Hamiltonian at each point and returns the charge sensor reading
5. Returns the 64-point signal as a numpy array

## Usage

Everything lives in `model/`, run locally via `uv` — no server, no frontend, no ONNX.

```bash
cd model

# 1. Generate training data (~5000 simulated detuning sweeps)
uv run python3 generate.py --n_samples 5000 --grid_size 64

# 2. Sanity-check the physics (overlaid traces + slope vs t plot)
uv run python3 visualise.py

# 3. Train the 1D CNN
uv run python3 train.py --data ../data/dataset.h5 --epochs 80
```

Output files saved to `data/`: `dataset.h5`, `model.pt`.

## Trying the model out — `demo.py`

`demo.py` simulates one device, runs the trained model on it, and plots the trace with true `t` vs. predicted `t`. Parameters are set at the top of the file — edit them, then run:

```bash
uv run python3 demo.py
```

Two modes, set via `MODE` at the top of the file:

- `MODE = "build"` — simulates a device from the params below it (`T`, `CDD_OFF`, `CDG_CROSS0`, `CDG_CROSS1`, `NOISE_STD`) using the same physics as `generate.py`, live, every run.
- `MODE = "sample"` — instead pulls a random real `(signal, t)` pair out of `dataset.h5`.

`NOISE_STD` is worth trying above `0` deliberately: training data was always noiseless, so this is an out-of-distribution stress test of the model, not a normal input.
