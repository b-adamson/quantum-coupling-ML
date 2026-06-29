"""
FastAPI backend — POST /predict returns tunnel coupling t.

Expected input:
    { "signal": [...] }   # 1D array, shape (N,)

Returns:
    { "t": 0.043 }
"""

import os

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "../data/model.onnx")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


class PredictRequest(BaseModel):
    signal: list[float]


class PredictResponse(BaseModel):
    t: float


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    arr = np.array(req.signal, dtype=np.float32)

    if arr.ndim != 1:
        raise HTTPException(status_code=422, detail="signal must be a 1D array")

    # Normalise
    mean, std = arr.mean(), arr.std()
    arr = (arr - mean) / max(std, 1e-6)

    # Shape: (1, 1, L)
    x = arr[np.newaxis, np.newaxis, :]

    session = get_session()
    t_val = float(session.run(["coupling"], {"signal": x})[0][0])

    return PredictResponse(t=t_val)


SAMPLE_PATH = os.environ.get("SAMPLE_PATH", "../data/dataset.h5")

class SampleResponse(BaseModel):
    signal: list[float]
    true_t: float


@app.get("/sample", response_model=SampleResponse)
def sample():
    try:
        import h5py, random
        with h5py.File(SAMPLE_PATH, "r") as f:
            idx = random.randint(0, len(f["labels"]) - 1)
            signal = f["signals"][idx].tolist()
            true_t = float(f["labels"][idx])
        return SampleResponse(signal=signal, true_t=true_t)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load sample: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
