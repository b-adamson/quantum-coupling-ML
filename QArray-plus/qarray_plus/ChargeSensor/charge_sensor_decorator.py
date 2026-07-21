import functools
import inspect
from dataclasses import dataclass
from typing import Optional, Callable

import jax.numpy as jnp


@dataclass(frozen=True)
class ChargeSensorResult:
    """Container for simulation results and corresponding sensor signals."""
    n: jnp.ndarray
    sensor: Optional[jnp.ndarray] = None

    def __iter__(self):
        """Allows unpacking: n, sensor = result"""
        return iter((self.n, self.sensor))


def charge_sensor(fn: Callable) -> Callable:
    """
    Decorator to automatically simulate charge sensor signals.
    Expects the wrapped function to have a 'vg' argument.
    """
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, charge_sensor=None, **kwargs) -> ChargeSensorResult:
        # 1. Execute the simulation
        result = fn(*args, **kwargs)

        # 2. Handle nested decorators: if result is already a ChargeSensorResult, extract 'n'
        n = result.n if isinstance(result, ChargeSensorResult) else result

        # 3. Short-circuit if no sensor is provided
        if charge_sensor is None:
            return ChargeSensorResult(n=n, sensor=None)

        # 4. Resolve 'vg' from arguments
        # We bind the arguments to the signature to find where 'vg' lives
        bound = sig.bind_partial(*args, **kwargs)
        vg = bound.arguments.get("vg")

        if vg is None:
            raise ValueError(
                f"@charge_sensor: Function '{fn.__name__}' must receive a 'vg' argument "
                "to simulate sensor signals."
            )

        # 5. Simulate the signal
        # sensor_signal should be a jnp.ndarray representing the sensor current/voltage
        sensor_signal = charge_sensor.simulate(vg=vg, n=n)

        return ChargeSensorResult(n=n, sensor=sensor_signal)

    return wrapper
