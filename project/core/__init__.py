"""
airboard core modules: hand tracking, gesture recognition, smoothing, and canvas management.
"""

from .hand_tracking import HandTracker
from .gestures import GestureClassifier, GestureMode
from .smoothing import ExponentialSmoother
from .canvas import AirCanvas

__all__ = [
    "HandTracker",
    "GestureClassifier",
    "GestureMode",
    "ExponentialSmoother",
    "AirCanvas",
]
