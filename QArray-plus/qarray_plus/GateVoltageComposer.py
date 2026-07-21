import re
from dataclasses import dataclass, field
from typing import Tuple, Optional, Union

import numpy as np

# --------------------
# Regex patterns
# --------------------
PATTERNS = {
    "P": re.compile(r"^P(\d+)$"),  # Plunger gate
    "vP": re.compile(r"^vP(\d+)$"),  # Virtual plunger
    "e": re.compile(r"^e(\d+)_(\d+)$"),  # Detuning (epsilon)
    "u": re.compile(r"^[uU](\d+)_(\d+)$"),  # Interaction/Total energy (U)
}


@dataclass
class GateVoltageComposer:
    """
    Composes gate voltage grids for physical and virtual gate scans.
    """
    n_gate: int
    n_dot: Optional[int] = None
    n_sensor: int = 0
    virtual_gate_matrix: Optional[np.ndarray] = None
    v0: np.ndarray = field(default_factory=lambda: None)

    def __post_init__(self):
        if self.v0 is None:
            self.v0 = np.zeros(self.n_gate)

        if self.virtual_gate_matrix is None:
            # The virtual space includes dots + sensors
            n_virtual = self.n_dot + self.n_sensor

            # Initialize an identity-like mapping
            # Shape: (n_physical_gates, n_virtual_channels)
            self.virtual_gate_matrix = np.eye(self.n_gate, n_virtual)

        # Ensure v0 is a numpy array
        self.v0 = np.asarray(self.v0, dtype=float)

    # ------------------------
    # Validation
    # ------------------------
    def _check_gate(self, gate: int):
        if not (1 <= gate <= self.n_gate):
            raise ValueError(f"Gate index {gate} out of range (1..{self.n_gate})")

    def _check_dot(self, dot: int):
        if self.n_dot is None:
            raise ValueError("n_dot must be defined to use virtual gates.")
        if not (1 <= dot <= self.n_dot + self.n_sensor):
            raise ValueError(f"Dot/Sensor index {dot} out of range.")

    # ------------------------
    # Parsing & Logic
    # ------------------------
    def _parse_label(self, label: str) -> Tuple[str, Tuple[int, ...]]:
        for key, pattern in PATTERNS.items():
            if m := pattern.match(label):
                return key, tuple(map(int, m.groups()))
        raise ValueError(f"Invalid gate label '{label}'. Use P1, vP1, e1_2, or u1_2.")

    def _get_scan_vector(self, gate: Union[int, str], min_v: float, max_v: float, res: int) -> np.ndarray:
        """Returns a 1D scan array of shape (res, n_gate)."""
        if isinstance(gate, int):
            self._check_gate(gate)
            return self._construct_1d_physical(gate, min_v, max_v, res)

        key, idx = self._parse_label(gate)
        v_lin = np.linspace(min_v, max_v, res)

        if key == "P":
            return self._construct_1d_physical(idx[0], min_v, max_v, res)

        if key == "vP":
            return self._construct_1d_virtual(idx[0], v_lin)

        if key == "e":
            # Detuning: d1 increases, d2 decreases
            v1 = self._construct_1d_virtual(idx[0], v_lin / 2)
            v2 = self._construct_1d_virtual(idx[1], -v_lin / 2)
            return v1 + v2

        if key == "u":
            # Total Energy: Both increase
            scale = 1 / np.sqrt(2)
            v1 = self._construct_1d_virtual(idx[0], v_lin * scale)
            v2 = self._construct_1d_virtual(idx[1], v_lin * scale)
            return v1 + v2

    # ------------------------
    # Core Meshgrid Methods
    # ------------------------
    def _construct_1d_physical(self, gate_idx: int, min_v: float, max_v: float, res: int) -> np.ndarray:
        scan = np.zeros((res, self.n_gate))
        scan[:, gate_idx - 1] = np.linspace(min_v, max_v, res)
        return scan

    def _construct_1d_virtual(self, dot_idx: int, voltages: np.ndarray) -> np.ndarray:
        if self.virtual_gate_matrix is None:
            raise ValueError("virtual_gate_matrix is required for virtual plunger scans.")

        # Create a vector in virtual space (dots + sensors)
        v_virtual = np.zeros((len(voltages), self.n_dot + self.n_sensor))
        v_virtual[:, dot_idx - 1] = voltages

        # Transform to physical space: V_phys = M @ V_virt
        return np.einsum("ij,kj->ki", self.virtual_gate_matrix, v_virtual)

    # ------------------------
    # Public API
    # ------------------------
    def do1d(self, gate: Union[int, str], min_v: float, max_v: float, res: int) -> np.ndarray:
        """Perform a 1D sweep. Returns shape (res, n_gate)."""
        return self._get_scan_vector(gate, min_v, max_v, res) + self.v0

    def do2d(
            self,
            x_gate, x_min, x_max, x_res,
            y_gate, y_min, y_max, y_res
    ) -> np.ndarray:
        """
        Perform a 2D sweep.
        Returns shape (y_res, x_res, n_gate).
        """
        # Get 1D scan vectors (shape: res, n_gate)
        vx = self._get_scan_vector(x_gate, x_min, x_max, x_res)
        vy = self._get_scan_vector(y_gate, y_min, y_max, y_res)

        # Broadcast addition: (1, x_res, n_gate) + (y_res, 1, n_gate)
        return vx[np.newaxis, :, :] + vy[:, np.newaxis, :] + self.v0
