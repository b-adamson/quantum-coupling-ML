from http.server import BaseHTTPRequestHandler
import json, os
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
_session = None


def get_session():
    global _session
    if _session is None:
        _session = ort.InferenceSession(MODEL_PATH)
    return _session


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            arr = np.array(body["signal"], dtype=np.float32)
            mean, std = arr.mean(), arr.std()
            arr = (arr - mean) / max(float(std), 1e-6)
            x = arr[np.newaxis, np.newaxis, :, :]
            t = float(get_session().run(["coupling"], {"signal": x})[0][0])
            self._respond(200, {"t": t})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def _respond(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
