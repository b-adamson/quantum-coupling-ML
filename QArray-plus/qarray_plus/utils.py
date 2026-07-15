"""
Quantum Dot Array Utilities
Author: b-vanstraaten
Date: 20/11/2025
"""

import numpy as np


def charge_state_to_scalar(n: np.ndarray) -> np.ndarray:
    """
    Converts binary charge states to scalar integers using bit-shifting.
    Returns: Same shape as n but with the last axis summed.
    """
    n = np.asarray(n)
    # Using bit-shift (1 << i) is faster than power (2**i)
    powers_of_two = 1 << np.arange(n.shape[-1])
    return np.sum(powers_of_two * n, axis=-1)


def charge_state_to_color(n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Maps charge states to color values via Einstein summation.
    Returns: Weighted sum across the last axis.
    """
    return np.einsum("...k, k->...", np.asarray(n), np.asarray(c))


def unique_last_axis(arr: np.ndarray) -> np.ndarray:
    """
    Find unique vectors in the last axis of a numpy ndarray.

    Returns:
        unique_arrays (np.ndarray): The unique vectors found.
    """
    arr = np.asarray(arr)
    original_shape = arr.shape

    # Reshape to 2D (N_samples, Vector_Length)
    reshaped_arr = arr.reshape(-1, original_shape[-1])

    # axis=0 finds unique rows (the vectors from the last axis)
    # We only return the unique_arrays to match original function signature
    unique_arrays, _, _ = np.unique(
        reshaped_arr, axis=0, return_index=True, return_inverse=True
    )

    return unique_arrays
