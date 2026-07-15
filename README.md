# Quantum Dot Coupling Prediction Model

Predicts interdot tunnel coupling `t` from a 1D charge sensor detuning sweep through the interdot transition.

## How it works

A double quantum dot device is characterised by two capacitance matrices:

- **`cdd` (2×2)** — dot-to-dot capacitance. The diagonal (self-capacitance) is fixed at 1; the off-diagonal is the mutual capacitance between the two dots, controlling how strongly they electrostatically repel each other.
- **`cdg` (2×2)** — gate-to-dot capacitance. The diagonal is the lever arm of each gate on its own dot; the off-diagonals are cross-coupling terms (how much gate 1 bleeds onto dot 2 and vice versa).

For each training sample, `cdd`, `cdg`, and `t` are drawn randomly to represent a different physical device geometry. These are handed to **QArray-plus**, which solves for the quantum ground state of the system as gate voltages are swept anti-diagonally through the (1,0)↔(0,1) interdot charge transition. The charge sensor signal comes out as an erf-shaped step — flat low plateau, smooth ramp, flat high plateau. The width of the ramp encodes `t`: large `t` → broad ramp, small `t` → sharp step.

5000 such (signal, t) pairs are generated and a 1D CNN is trained to map the signal shape back to `t`. By randomising `cdd` and `cdg` across samples, the model learns to extract `t` regardless of device-to-device variation in capacitances.

### `simulate_one`

The core simulation function. Given one set of device parameters it:

1. Builds a `ChargeSensor` with fixed coupling params — equal `csg` values cancel the linear background so the output is a clean erf; `csd=[0.02, 0.1]` makes the signal run low→high across the transition
2. Calls `optimal_vg([0.5, 0.5])` to find the gate voltages that sit exactly on the (1,0)↔(0,1) interdot transition
3. Constructs a 64-point anti-diagonal sweep (V₁ = +ε/2, V₂ = −ε/2) around that centre
4. Calls `tunnel_coupled_ground_state` — QArray solves the quantum Hamiltonian at each point and returns the charge sensor reading
5. Returns the 64-point signal as a numpy array

## Input format

The API and frontend both expect the sensor signal as a flat, 1D JSON array of floats — not `(x, y)` pairs:

```json
{ "signal": [0.1805, 0.1807, 0.1809, ..., 0.2820] }
```

- Length must match the model's `grid_size` (64 by default).
- Each value is the sensor reading at one step of the detuning sweep; the detuning value itself (`ε`, from `linspace(-0.5, 0.5, grid_size)`) is never included — it's implicit in the array index, since every sample uses the same fixed set of sweep steps.
- Values should be in the same units/scale as the training data (raw sensor units); the server normalises the array internally before running the model.

### `POST /api/simulate` — build-your-own device

Instead of pasting a pre-baked signal, you can drive the physics directly. This runs the same `simulate_one` pipeline as `generate.py`, but with explicit rather than randomly-drawn parameters:

```json
// request
{ "t": 0.075, "cdd_off": 0.08, "cdg_cross0": 0.1, "cdg_cross1": 0.1, "noise_std": 0.0 }

// response
{ "signal": [...64 floats...], "true_t": 0.075 }
```

- `t`, `cdd_off`, `cdg_cross0`, `cdg_cross1` are clamped server-side to the same ranges `generate.py` trains on (keeps `Cdd` positive-definite and keeps comparisons against the model meaningful, since its output head is capped at `t ≤ 0.15`).
- `noise_std` (optional, default `0`) feeds `pink_noise_std`/`white_noise_std` on the `ChargeSensor` — training data was always noiseless, so pushing this above `0` is a deliberate out-of-distribution stress test, not a normal input.
- The frontend's **Build Your Own** tab wraps this endpoint with sliders, then automatically chains the result into `/api/predict` so you can compare the `t` you dialed in against what the model recovers — including while varying `cdd_off`/`cdg_cross*`, the confounders the model is supposed to be robust to.

## 1. Train the model

From the `model/` directory:

```bash
cd model

# Generate training data (~5000 simulated detuning sweeps)
uv run python3 generate.py --n_samples 5000 --grid_size 64

# Sanity-check the physics (overlaid traces + slope vs t plot)
uv run python3 visualise.py

# Train the 1D CNN
uv run python3 train.py --data ../data/dataset.h5 --epochs 80

# Export trained model to ONNX
uv run python3 export.py

# Precompute the lookup table /api/simulate interpolates at request time
# (grid resolution/time tradeoff: see model/generate_lut.py --help)
uv run python3 generate_lut.py --n_points 16 --grid_size 64 --out ../data/simulate_lut.npz
```

Output files saved to `data/`: `dataset.h5`, `model.pt`, `model.onnx`, `simulate_lut.npz`.

## 2. Test locally

`/api/predict` + `/api/sample` and `/api/simulate` are deployed as **separate Vercel functions** with separate dependencies. Run each locally in its own terminal:

```bash
# Copy ONNX model into api/ so the local server can find it
cp data/model.onnx api/model.onnx

# Copy the precomputed lookup table into api/simulate/ so the local server can find it
cp data/simulate_lut.npz api/simulate/lut.npz

# Terminal 1 — /api/predict + /api/sample on port 5000
cd api
uv run python3 -c "from http.server import HTTPServer; from predict import handler; HTTPServer(('', 5000), handler).serve_forever()"

# Terminal 2 — /api/simulate on port 5001
cd api/simulate
uv run python3 -c "from http.server import HTTPServer; from index import handler; HTTPServer(('', 5001), handler).serve_forever()"
```

Then open `frontend/index.html` with **VS Code Live Server** (right-click → Open with Live Server).

The frontend detects port 5500 (Live Server's default) and routes `/api/predict`/`/api/sample` calls to `localhost:5000` and `/api/simulate` calls to `localhost:5001`. Click **Load Example** to pull a real sample from the dataset, then **Predict** to run the model — or use the **Build Your Own** tab to drive `/api/simulate` directly.

### Why is `/api/simulate` a lookup table instead of live physics?

`/api/simulate` originally ran the full `qarray_plus`/JAX physics stack live, same as `generate.py`. `jax`/`jaxlib`/`scipy` alone are large enough to blow past Vercel's 500 MB Python function size limit, even in their own isolated function. Instead, `model/generate_lut.py` precomputes the clean (noise-free) signal on a 4D grid of `(t, cdd_off, cdg_cross0, cdg_cross1)` offline using the real physics, and the deployed function does 4D multilinear interpolation over that grid with plain `numpy` — no `jax` needed at request time. `noise_std` isn't gridded; it's reapplied live using the same (already-numpy) noise model from `ChargeSensor.py`, so noisy requests still get fresh random noise per call, not a cached noisy sample.
