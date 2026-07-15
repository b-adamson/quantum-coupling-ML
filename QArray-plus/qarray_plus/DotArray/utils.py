"""
Author: b-vanstraaten
Date: 17/11/2025
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Protocol
from typing import Union, Callable, Any

import jax.numpy as jnp
from colorama import Fore


def _as_callable(val: Any) -> Callable:
    """Ensures a value is a callable for consistent function dispatch."""
    if callable(val):
        return val
    return lambda _: jnp.asarray(val).astype(float)


def dispatch_functional(func_fn: Callable, func_arr: Callable, *args) -> Callable:
    """
    Dispatches to either a pure-array or functional implementation.
    Prevents tracing overhead by normalizing mixed inputs.
    """
    is_callable = [callable(a) for a in args]

    if not any(is_callable):
        # Pure array path
        float_args = [jnp.asarray(a).astype(float) for a in args]
        return partial(func_arr, *float_args)

    if not all(is_callable):
        warnings.warn(
            f"{Fore.RED}Mixed callable/array arguments detected. "
            f"Converting arrays to constant functions, which may trigger tracing overhead.{Fore.RESET}"
        )

    callable_args = [_as_callable(a) for a in args]
    return partial(func_fn, *callable_args)


def _reshape_reshape(f, vg, max_size=None):
    """
    Apply `f` to `vg` after flattening all leading dimensions into one
    batch dimension. If `max_size` is provided, the flattened input is
    padded into chunks of exactly `max_size` rows before calling `f`,
    and outputs are unpadded afterwards.

    Padding is done with zeros along the feature dimension.

    Parameters
    ----------
    f : callable
        A function that accepts an array of shape (N, D) and returns an
        array whose first dimension is N.
    vg : array_like
        Input array. The last dimension is treated as the feature dimension.
    max_size : int, optional
        If provided, chunks are padded to exactly this size before `f` is called.

    Returns
    -------
    jnp.ndarray
        Output array with shape (*vg.shape[:-1], -1).
    """
    vg = jnp.atleast_2d(vg)
    leading_shape = vg.shape[:-1]

    # Flatten all leading dims into a single batch dimension
    vg_flat = vg.reshape(-1, vg.shape[-1])
    N, D = vg_flat.shape

    # --- No chunking: behave like the original function ---
    if max_size is None:
        out = jnp.asarray(f(vg_flat))
        return out.reshape(*leading_shape, -1)

    # --- Chunking with padding ---
    outputs = []
    for start in range(0, N, max_size):
        chunk = vg_flat[start: start + max_size]
        chunk_size = chunk.shape[0]

        # Pad chunk to exactly max_size rows if needed
        if chunk_size < max_size:
            pad_len = max_size - chunk_size
            pad = jnp.zeros((pad_len, D), dtype=chunk.dtype)
            chunk = jnp.vstack([chunk, pad])

        # Call f on the padded chunk
        padded_out = jnp.asarray(f(chunk))

        # Keep only the first `chunk_size` rows (remove padding effects)
        outputs.append(padded_out[:chunk_size])

    # Concatenate all unpadded outputs
    out = jnp.concatenate(outputs, axis=0)

    # Restore leading dims
    return out.reshape(*leading_shape, -1)


class GateVoltageDependantCallable(Protocol):
    def __call__(self, vg: jnp.ndarray) -> jnp.ndarray: ...


def _validate_shape(arr: jnp.ndarray, expected_shape: tuple, name: str) -> None:
    """Raise ValueError if arr.shape != expected_shape."""
    arr = jnp.array(arr)
    if arr.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {arr.shape}")


def _ensure_callable_and_validate(
        param: Union[GateVoltageDependantCallable, jnp.ndarray, float],
        expected_shape: tuple,
        name: str,
        n_gates: int,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Ensures param is a callable. If it's an array, validates shape and wraps as a constant-returning function.
    If it's a callable, checks output shape with zeros input.
    Returns a callable.
    """
    if not callable(param):
        arr = jnp.array(param).reshape(expected_shape)
        _validate_shape(arr, expected_shape, name)

        def _const_func(vg: jnp.ndarray) -> jnp.ndarray:
            return arr

        return _const_func
    else:
        test_out = param(jnp.zeros(n_gates))
        _validate_shape(test_out, expected_shape, f"{name} (callable output)")
        return param
