# Quantum Dot Coupling Prediction Model

Predicts interdot tunnel coupling `t` from a 1D charge sensor detuning sweep through the interdot transition.

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
```

Output files saved to `data/`: `dataset.h5`, `model.pt`, `model.onnx`.

## 2. Test locally

```bash
# Copy ONNX model into api/ so the local server can find it
cp data/model.onnx api/model.onnx

# Start the local API server on port 5000
cd api
uv run python3 -c "from http.server import HTTPServer; from predict import handler; HTTPServer(('', 5000), handler).serve_forever()"
```

Then open `frontend/index.html` with **VS Code Live Server** (right-click → Open with Live Server).

The frontend detects port 5500 (Live Server's default) and routes API calls to `localhost:5000`. Click **Load Example** to pull a real sample from the dataset, then **Predict** to run the model.

## 3. Deploy to Vercel

```bash
cp data/model.onnx api/model.onnx
git add api/model.onnx
git push
```

Vercel picks up `api/predict.py` as a serverless function and serves `frontend/index.html` at `/` via `vercel.json`.
