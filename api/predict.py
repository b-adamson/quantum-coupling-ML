import json, os
import numpy as np
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
app = Flask(__name__)
CORS(app)

_session = None

def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()
        arr = np.array(body["signal"], dtype=np.float32)
        mean, std = arr.mean(), arr.std()
        arr = (arr - mean) / max(float(std), 1e-6)
        x = arr[np.newaxis, np.newaxis, :, :]
        t = float(get_session().run(["coupling"], {"signal": x})[0][0])
        return jsonify({"t": t})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
