"""Low-pass filtering for the published mobile-base velocity command."""

import math

import numpy as np
import numpy.typing as npt


class LowPassFilter:
    """Three independent first-order LPFs for [vx, vy, wz].

    The filter starts at zero velocity to avoid an unfiltered startup jump.
    Its sampling period follows the WBC control-loop configuration.
    """

    def __init__(self, sample_hz: float, cutoff_hz: float):
        if not math.isfinite(sample_hz) or sample_hz <= 0.0:
            raise ValueError("sample_hz must be positive and finite")
        if not math.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be positive and finite")

        # Exact discretization of a first-order continuous-time LPF.
        # alpha = 1 - exp(-2*pi*fc*dt), where dt = 1/sample_hz.
        self.alpha = -math.expm1(-2.0 * math.pi * cutoff_hz / sample_hz)
        self._state = np.zeros(3, dtype=np.float64)

    def update(self, command: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the filtered [vx, vy, wz] without changing the input."""
        target = np.asarray(command, dtype=np.float64)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("base command must be three finite values [vx, vy, wz]")
        self._state += self.alpha * (target - self._state)
        return self._state.copy()
