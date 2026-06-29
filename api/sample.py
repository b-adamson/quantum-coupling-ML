from http.server import BaseHTTPRequestHandler
import json, os, random
import h5py

DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/dataset.h5")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            with h5py.File(DATASET_PATH, "r") as f:
                idx = random.randint(0, len(f["labels"]) - 1)
                signal = f["signals"][idx].tolist()
                true_t = float(f["labels"][idx])
            self._respond(200, {"signal": signal, "true_t": true_t})
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
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
