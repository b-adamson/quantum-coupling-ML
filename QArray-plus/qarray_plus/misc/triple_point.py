"""
Author: b-vanstraaten
Date: 04/12/2025
"""

import numpy as np

from qarray_plus.DotArray.ground_state.free_energy import free_energy_fn


def triple_point(model, vg, n1, n2, n3, return_index=False):
    """
    Find the gate voltages at the triple point for a given set of charge states.

    Args:
        model: A DotArray model with attributes
            - cdd
            - cdg
            - n_gates
        n1, n2, n3 (array-like): Charge configurations forming the triple point.
        vg (array-like): Initial gate-voltage mesh of shape (M, n_gates).

    Returns:
        np.ndarray: Gate voltages (1D array of length n_gates) at the triple point.
    """
    n1, n2, n3 = map(np.asarray, (n1, n2, n3))

    cdd = model.cdd if callable(model.cdd) else (lambda vg, v=model.cdd: v)
    cdg = model.cdg if callable(model.cdg) else (lambda vg, v=model.cdg: v)

    vg_flat = vg.reshape(-1, model.n_gates)

    F1 = free_energy_fn(n1, cdd, cdg, vg_flat)
    F2 = free_energy_fn(n2, cdd, cdg, vg_flat)
    F3 = free_energy_fn(n3, cdd, cdg, vg_flat)

    o = np.maximum.reduce([
        np.abs(F1 - F2),
        np.abs(F2 - F3),
        np.abs(F3 - F1),
    ])

    flat_index = np.argmin(o)

    if not return_index:
        return vg_flat[np.argmin(o)]
    else:
        grid_shape = vg.shape[:-1]
        return np.unravel_index(flat_index, grid_shape)
