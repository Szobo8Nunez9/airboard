"""Fixed EMA with an optional displacement-adaptive gain.

Fixed mode uses P[t] = alpha * input[t] + (1-alpha) * P[t-1]. Adaptive
mode changes that gain heuristically using pixel displacement per update.
It is not a full One Euro filter, and no zero-lag claim is made.
"""
from collections import deque
from typing import Optional, Tuple
import math


class ExponentialSmoother:
    """Smooth pixel coordinates; alpha=1 always returns the rounded raw input.

    Fixed mode is the default for reproducible alpha comparisons. Adaptive
    mode follows the upstream speed-aware gain heuristic as an opt-in.
    """

    def __init__(self, alpha: float = 0.5, adaptive: bool = False):
        self.set_alpha(alpha)
        self.adaptive = bool(adaptive)
        self.prev_x: Optional[float] = None
        self.prev_y: Optional[float] = None
        self.history = deque(maxlen=600)

    def set_alpha(self, alpha: float) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f'Smoothing factor alpha must be in (0.0, 1.0], got {alpha}')
        self.alpha = float(alpha)

    def smooth(self, x: float, y: float) -> Tuple[int, int]:
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError('Coordinates must be finite')
        if self.prev_x is None or self.prev_y is None or self.alpha == 1.0:
            self.prev_x, self.prev_y = x, y
        else:
            effective_alpha = self.alpha
            if self.adaptive:
                displacement = math.hypot(x - self.prev_x, y - self.prev_y)
                speed_factor = min(1.0, displacement / 35.0)
                effective_alpha = max(0.18, min(0.85, self.alpha * (0.6 + 0.9 * speed_factor)))
            self.prev_x = effective_alpha * x + (1.0 - effective_alpha) * self.prev_x
            self.prev_y = effective_alpha * y + (1.0 - effective_alpha) * self.prev_y
        self.history.append((self.prev_x, self.prev_y))
        return int(round(self.prev_x)), int(round(self.prev_y))

    def reset(self) -> None:
        """Reseed the next stroke at its current input rather than the prior stroke."""
        self.prev_x = None
        self.prev_y = None

    def clear_history(self) -> None:
        self.history.clear()

    @property
    def current(self) -> Optional[Tuple[int, int]]:
        if self.prev_x is not None and self.prev_y is not None:
            return int(round(self.prev_x)), int(round(self.prev_y))
        return None
