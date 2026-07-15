from http.server import BaseHTTPRequestHandler
import json
import os

import numpy as np

# Precomputed on a 4D grid of (t, cdd_off, cdg_cross0, cdg_cross1) via
# model/generate_lut.py, which runs the real qarray_plus/jax physics offline.
# Deploying this lookup table instead of jax/jaxlib/scipy/qarray_plus keeps this
# function's bundle small enough for Vercel's standard Python size limit.
LUT_PATH = os.path.join(os.path.dirname(__file__), "lut.npz")
_lut = np.load(LUT_PATH)
T_GRID = _lut["t_grid"]
CDD_OFF_GRID = _lut["cdd_off_grid"]
CDG_CROSS0_GRID = _lut["cdg_cross0_grid"]
CDG_CROSS1_GRID = _lut["cdg_cross1_grid"]
SIGNALS = _lut["signals"]  # shape (nt, ncdd, ncdg0, ncdg1, grid_size)
GRID_SIZE = SIGNALS.shape[-1]

# Keep parameter ranges matched to what generate.py trained on, so the physics stays
# numerically stable (Cdd must stay positive-definite) and comparisons against the
# model's prediction stay meaningful (the model's output head is clamped to [0, 0.15]).
BOUNDS = {
    "t": (0.0, 0.15),
    "cdd_off": (0.01, 0.15),
    "cdg_cross0": (0.05, 0.2),
    "cdg_cross1": (0.05, 0.2),
    "noise_std": (0.0, 0.05),
}


def _clamp(value, key):
    lo, hi = BOUNDS[key]
    return min(max(float(value), lo), hi)


def _axis_weights(value, grid):
    """Returns (lower_index, upper_index, fraction) for linear interpolation."""
    idx = int(np.searchsorted(grid, value)) - 1
    idx = min(max(idx, 0), len(grid) - 2)
    x0, x1 = grid[idx], grid[idx + 1]
    frac = 0.0 if x1 == x0 else (value - x0) / (x1 - x0)
    return idx, idx + 1, frac


def _lookup_clean_signal(t, cdd_off, cdg_cross0, cdg_cross1):
    """4D multilinear interpolation over the precomputed grid."""
    i0, i1, ft = _axis_weights(t, T_GRID)
    j0, j1, fc = _axis_weights(cdd_off, CDD_OFF_GRID)
    k0, k1, fg0 = _axis_weights(cdg_cross0, CDG_CROSS0_GRID)
    l0, l1, fg1 = _axis_weights(cdg_cross1, CDG_CROSS1_GRID)

    out = np.zeros(GRID_SIZE, dtype=np.float64)
    for ii, wi in ((i0, 1 - ft), (i1, ft)):
        for jj, wj in ((j0, 1 - fc), (j1, fc)):
            for kk, wk in ((k0, 1 - fg0), (k1, fg0)):
                for ll, wl in ((l0, 1 - fg1), (l1, fg1)):
                    w = wi * wj * wk * wl
                    if w != 0:
                        out += w * SIGNALS[ii, jj, kk, ll]
    return out


# --- Noise model, copied verbatim (it's plain numpy already, see the "import
# numpy as jnp" alias) from QArray-plus/qarray_plus/ChargeSensor/ChargeSensor.py's
# pink_noise_axis, so noise behaves identically to a live simulation. ---
def _pink_noise_axis(shape, axis):
    white = np.random.normal(size=np.prod(shape)).reshape(shape)
    white_t = np.moveaxis(white, axis, -1)
    L = white_t.shape[-1]
    f = np.fft.rfft(white_t, axis=-1)
    freqs = np.fft.rfftfreq(L)
    freqs = np.where(freqs == 0, 1, freqs)
    f = f / np.sqrt(freqs)[None]
    pink_t = np.fft.irfft(f, n=L, axis=-1)
    pink_t = pink_t / np.std(pink_t)
    pink = np.moveaxis(pink_t, -1, axis)
    return (pink - pink.mean()) / pink.std()


def simulate_signal(t, cdd_off, cdg_cross0, cdg_cross1, noise_std=0.0):
    t = _clamp(t, "t")
    cdd_off = _clamp(cdd_off, "cdd_off")
    cdg_cross0 = _clamp(cdg_cross0, "cdg_cross0")
    cdg_cross1 = _clamp(cdg_cross1, "cdg_cross1")
    noise_std = _clamp(noise_std, "noise_std")

    signal = _lookup_clean_signal(t, cdd_off, cdg_cross0, cdg_cross1)

    if noise_std > 0:
        z = signal[:, None]  # (grid_size, 1), matches ChargeSensor's z.shape
        pink_noise = noise_std * _pink_noise_axis(z.shape, 1)
        white_noise = noise_std * np.random.normal(size=z.shape)
        signal = (z + pink_noise + white_noise).squeeze(-1)

    return signal.astype(np.float32).tolist(), t


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            signal, true_t = simulate_signal(
                t=body["t"],
                cdd_off=body["cdd_off"],
                cdg_cross0=body["cdg_cross0"],
                cdg_cross1=body["cdg_cross1"],
                noise_std=body.get("noise_std", 0.0),
            )
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
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
