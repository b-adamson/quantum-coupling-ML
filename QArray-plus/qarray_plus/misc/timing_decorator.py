import functools
import os
import time
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from qarray_plus.ChargeSensor.charge_sensor_decorator import ChargeSensorResult

# termplotlib is only needed for the opt-in benchmark-mode histogram below
# (N_TIMING_AVG env var set). It's a sizeable dependency for a debug feature,
# so it's imported lazily rather than at module load time.


def hist_horizontal(
        counts: List[int],
        bin_edges: List[float],
        max_width: int = 40,
        bar_width: int = 1,
        show_bin_edges: bool = True,
        show_counts: bool = True,
        force_ascii: bool = False,
):
    from termplotlib.barh import barh

    if show_bin_edges:
        labels = [
            f"{bin_edges[k]:.1f} - {bin_edges[k + 1]:.1f}"
            for k in range(len(bin_edges) - 1)
        ]
    else:
        labels = None

    return barh(
        counts,
        labels=labels,
        max_width=max_width,
        bar_width=bar_width,
        show_vals=show_counts,
        force_ascii=force_ascii,
    )


@dataclass
class TimingResult:
    first_call_time: float
    n_calls: int = None
    times: np.ndarray = None


@dataclass
class ChargeSensorResult:
    n: jnp.ndarray
    sensor: jnp.ndarray | None = None
    timing: TimingResult = None


def _jax_timed_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out.sensor if out.sensor is not None else out.n)
    elapsed = 1000 * (time.perf_counter() - t0)
    return out, elapsed


def timing(fn):
    @functools.wraps(fn)
    def wrapper(*args, n_timing_avg=os.getenv("N_TIMING_AVG", None), **kwargs):
        # --------------------------------------------------------------
        # Case 1: Timing disabled (just measure one execution)
        # --------------------------------------------------------------
        n_timing_avg = None if n_timing_avg is None else int(n_timing_avg)

        if n_timing_avg is None:
            out, t = _jax_timed_call(fn, *args, **kwargs)
            return ChargeSensorResult(
                **out.__dict__,
                timing=TimingResult(first_call_time=t, times=np.asarray([t])),
            )
        # --------------------------------------------------------------
        # Case 2: Benchmark mode
        # First call = compile + execute
        # Next calls = execute only (averaged)
        # --------------------------------------------------------------
        # Warmup / compile timing
        out, first_call_time = _jax_timed_call(fn, *args, **kwargs)

        # Execution-only timings
        if n_timing_avg > 0:
            times = []
            for _ in range(n_timing_avg):
                _, t = _jax_timed_call(fn, *args, **kwargs)
                times.append(t)
            times = np.asarray(times)

            import termplotlib as tpl

            print(f"\n[{fn.__name__}] {n_timing_avg} runs (ms):")
            counts, bin_edges = np.histogram(times, bins=5)
            # Convert bin edges to integers
            fig = tpl.figure()
            hist = hist_horizontal(counts, bin_edges, force_ascii=False)
            fig._content.append(hist)
            fig.show()
            print(
                f"mean     : {times.mean():8.1f}\n"
                f"std      : {times.std():8.1f}\n"
                f"min      : {times.min():8.1f}\n"
                f"max      : {times.max():8.1f}"
            )
        else:
            times = np.asarray([])

        return ChargeSensorResult(
            **out.__dict__,
            timing=TimingResult(
                first_call_time=first_call_time,
                n_calls=n_timing_avg,
                times=times,
            ),
        )

    return wrapper
