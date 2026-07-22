# Quantum Dot Coupling Prediction Model

Predicts interdot tunnel coupling `t` from a 1D charge sensor detuning sweep through the interdot transition.

## How it works

A double quantum dot is simulated in **QArray-plus** with randomised capacitance matrices (`cdd`, `cdg`) and tunnel coupling `t`, sweeping gate voltages anti-diagonally through the (1,0)↔(0,1) charge transition. The sensor signal comes out as an erf-shaped step whose ramp width encodes `t`. 5000 (signal, t) pairs are generated this way and a 1D CNN is trained to recover `t` from the signal shape, robust to device-to-device capacitance variation.

`simulate_one` (in `generate.py`, reused by `demo.py`) builds the sensor, finds the transition center via `optimal_vg`, sweeps a 64-point anti-diagonal detuning through it, and returns the resulting signal.

## Usage

Everything lives in `model/`, run locally via `uv`.

```bash
cd model

uv run python3 generate.py --n_samples 5000 --grid_size 64   # generate training data
uv run python3 visualise.py                                   # sanity-check the physics
uv run python3 train.py --data ../data/dataset.h5 --epochs 80 # train the CNN
```

Outputs go to `data/`: `dataset.h5`, `model.pt`.

## Demo

`demo.py` simulates one device, runs the trained model, and plots true vs. predicted `t`. Edit params at the top of the file, then:

```bash
uv run python3 demo.py
```

- `MODE = "build"` — simulates a fresh device live from the params below it.
- `MODE = "sample"` — pulls a random real pair from `dataset.h5` instead.

`NOISE_STD > 0` is an out-of-distribution stress test — training data is always noiseless.
