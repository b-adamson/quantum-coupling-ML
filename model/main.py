# uv run python3 main.py
import h5py
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib.pyplot as plt

from qarray_plus import DotArray, ChargeSensor
from train import CouplingCNN

# ============================== parameters ==================================

MODE = "build"  # "build" = simulate a device from the params below
                # "sample" = pull an entry out of dataset.h5

# MODE = "build":
T = 0.1               # (0 to 0.15)
CDD_OFF = 0.08        # (0.01 to 0.15)
CDG_CROSS0 = 0.10     # (0.05 to 0.2)
CDG_CROSS1 = 0.10     # (0.05 to 0.2)
NOISE_STD = 0.000     # sensor noise (0 to 0.02)
SLOPE = 0.00          # linear background slope (0 to 0.03)

#          [        1  ,  -cdd_off ]           [    -1     ,  -cdg_cross0 ]           [ 0 ,  t ]
#   cdd =  [                       ]     cdg = [                          ]     t =   [        ]
#          [ -cdd_off  ,     1     ]           [ -cdg_cross1 ,    -1      ]           [ t ,  0 ]

GRID_SIZE = 64
WEIGHTS_PATH = "../data/model.pt"
DATASET_PATH = "../data/dataset.h5"

# ==============================================================================


def simulate(t, cdd_off, cdg_cross0, cdg_cross1, noise_std, slope=0.0):
    cdd = jnp.array([[1.0, -cdd_off], [-cdd_off, 1.0]])
    cdg = -jnp.array([[1.0, cdg_cross0], [cdg_cross1, 1.0]])
    t_mat = jnp.array([[0.0, t], [t, 0.0]])

    charge_sensor = ChargeSensor(
        n_dots=2,
        n_gates=2,
        n_sensor=1,
        csd=jnp.array([0.02, 0.1]),
        csg=-jnp.array([0.3, 0.3]),
        pink_noise_std=noise_std,
        white_noise_std=noise_std,
    )
    model = DotArray(n_dots=2, n_gates=2, cdd=cdd, cdg=cdg, t=t_mat)

    v0 = model.optimal_vg([0.5, 0.5])
    eps = jnp.linspace(-0.5, 0.5, GRID_SIZE)
    vg = v0[None, :] + jnp.stack([eps / 2, -eps / 2], axis=-1)

    # result.sensor is (64,1) array of sensor outputs
    result = model.tunnel_coupled_ground_state(vg, charge_sensor=charge_sensor)
    signal = np.array(result.sensor).squeeze().astype(np.float32)
    return signal + slope * np.array(eps, dtype=np.float32)


def load_random_sample():
    # Pull one real (signal, t) pair out of the training dataset.
    with h5py.File(DATASET_PATH, "r") as f:
        idx = np.random.randint(len(f["labels"]))
        return f["signals"][idx], float(f["labels"][idx])


def predict(signal):
    # Run the trained CNN on one signal and return its predicted t.
    model = CouplingCNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    arr = np.asarray(signal, dtype=np.float32)
    # arr = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    arr = (arr - arr.mean()) / max(float(arr.std()), 1e-6)  # match training normalisation
    x = torch.tensor(arr).view(1, 1, -1)

    with torch.no_grad():
        return float(model(x).item())


def main():
    if MODE == "build":
        signal = simulate(T, CDD_OFF, CDG_CROSS0, CDG_CROSS1, NOISE_STD, SLOPE)
        true_t = T
    elif MODE == "sample":
        signal, true_t = load_random_sample()
    else:
        raise ValueError(f"MODE must be 'build' or 'sample', got {MODE!r}")

    pred_t = predict(signal)
    print(f"true t = {true_t:.4f}   predicted t = {pred_t:.4f}")

    eps = np.linspace(-0.5, 0.5, len(signal))
    plt.plot(eps, signal)
    plt.xlabel("Detuning ε")
    plt.ylabel("Sensor signal")
    plt.title(f"true t = {true_t:.4f}    predicted t = {pred_t:.4f}")
    plt.show()


if __name__ == "__main__":
    main()
