"""
Gesture recognition engine and hold-timer state machine.
Maps 21 hand landmarks to modes:
- Index up = DRAW
- Index + Middle = ERASE
- Open Palm = HOVER
- Fist held 1 second = CLEAR
"""

from enum import Enum
import math
import time
from typing import List, Tuple, Optional, Dict, Any


class GestureMode(str, Enum):
    IDLE = "IDLE"
    DRAW = "DRAW"
    HOVER = "HOVER"
    ERASE = "ERASE"
    CLEAR = "CLEAR"


class GestureClassifier:
    """
    Classifies 21 MediaPipe hand landmarks into airboard control modes.
    Manages the 1-second hold timer for fist-based board clearing.
    """

    def __init__(self, clear_hold_seconds: float = 1.0):
        self.clear_hold_seconds = clear_hold_seconds
        self.fist_start_time: Optional[float] = None
        self.clear_triggered: bool = False
        self.last_mode: GestureMode = GestureMode.IDLE
        self.draw_grace_counter = 0

    def get_finger_states(
        self, landmarks_px: List[Tuple[int, int]]
    ) -> Dict[str, bool]:
        """
        Determines which fingers are extended (up/open).
        Landmarks:
          Wrist: 0
          Thumb: 1-4
          Index: 5-8
          Middle: 9-12
          Ring: 13-16
          Pinky: 17-20
        """
        if len(landmarks_px) < 21:
            return {
                "thumb": False,
                "index": False,
                "middle": False,
                "ring": False,
                "pinky": False,
            }

        # Wrist and MCP references
        wrist = landmarks_px[0]
        palm_size = math.dist(wrist, landmarks_px[9])  # Distance from wrist to middle MCP
        if palm_size < 1e-4:
            palm_size = 1.0

        # Compare radial extension with a vertical fallback; this remains a pose heuristic
        def _check_extension(tip_idx: int, pip_idx: int) -> bool:
            d_tip = math.dist(wrist, landmarks_px[tip_idx])
            d_pip = math.dist(wrist, landmarks_px[pip_idx])
            is_dist_ext = d_tip > 1.15 * d_pip
            is_vert_up = landmarks_px[tip_idx][1] < landmarks_px[pip_idx][1]
            return is_dist_ext or (is_vert_up and d_tip > 0.95 * d_pip)

        index_up = _check_extension(8, 6)
        middle_up = _check_extension(12, 10)
        ring_up = _check_extension(16, 14)
        pinky_up = _check_extension(20, 18)

        # Thumb extension check: distance between thumb tip (4) and pinky MCP (17) or index MCP (5)
        thumb_tip = landmarks_px[4]
        index_mcp = landmarks_px[5]
        thumb_dist = math.dist(thumb_tip, index_mcp)
        thumb_up = thumb_dist > 0.45 * palm_size

        return {
            "thumb": thumb_up,
            "index": index_up,
            "middle": middle_up,
            "ring": ring_up,
            "pinky": pinky_up,
        }

    def is_fist(
        self, landmarks_px: List[Tuple[int, int]], finger_states: Dict[str, bool]
    ) -> bool:
        """
        Fist detection using curled-finger and palm-distance heuristics:
        Checks that all non-thumb fingers are curled down and fingertips are near palm center.
        """
        if len(landmarks_px) < 21:
            return False

        # All 4 main fingers must not be up
        if any([finger_states["index"], finger_states["middle"], finger_states["ring"], finger_states["pinky"]]):
            return False

        wrist = landmarks_px[0]
        palm_center = landmarks_px[9]
        palm_length = math.dist(wrist, palm_center)
        if palm_length < 1e-4:
            return False

        # All tips must be close to palm base/MCP cluster
        tips = [landmarks_px[4], landmarks_px[8], landmarks_px[12], landmarks_px[16], landmarks_px[20]]
        curled_count = 0
        for tip in tips:
            dist_to_center = math.dist(tip, palm_center)
            dist_to_wrist = math.dist(tip, wrist)
            if dist_to_center < 1.1 * palm_length and dist_to_wrist < 1.2 * palm_length:
                curled_count += 1

        return curled_count >= 4

    def classify(
        self, landmarks_px: List[Tuple[int, int]], current_time: Optional[float] = None
    ) -> Tuple[GestureMode, float, Dict[str, Any]]:
        """
        Classifies current hand landmarks into a GestureMode with hysteresis debounce.
        """
        if current_time is None:
            current_time = time.monotonic()

        if len(landmarks_px) < 21:
            self._reset_fist_timer()
            self.draw_grace_counter = 0
            self.last_mode = GestureMode.IDLE
            return GestureMode.IDLE, 0.0, {"fingers": {}, "raw_fist": False}

        fingers = self.get_finger_states(landmarks_px)
        fist_detected = self.is_fist(landmarks_px, fingers)

        clear_progress = 0.0
        active_mode = GestureMode.IDLE

        # 1. CLEAR: Fist held for 1 second
        if fist_detected:
            self.draw_grace_counter = 0
            if self.fist_start_time is None:
                self.fist_start_time = current_time
                self.clear_triggered = False

            elapsed = current_time - self.fist_start_time
            clear_progress = min(1.0, max(0.0, elapsed / self.clear_hold_seconds))

            if clear_progress >= 1.0:
                if not self.clear_triggered:
                    active_mode = GestureMode.CLEAR
                    self.clear_triggered = True
                else:
                    active_mode = GestureMode.IDLE
            else:
                active_mode = GestureMode.HOVER  # Pending clear countdown
        else:
            self._reset_fist_timer()

            # 2. HOVER: Open palm (four main fingers extended).
            if fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
                self.draw_grace_counter = 0
                active_mode = GestureMode.HOVER

            # 3. ERASE: Index and middle extended; ring and pinky folded.
            elif fingers["index"] and fingers["middle"] and (not fingers["ring"]) and (not fingers["pinky"]):
                self.draw_grace_counter = 0
                active_mode = GestureMode.ERASE

            # 4. DRAW: Index up only (Middle, Ring & Pinky down)
            elif fingers["index"] and (not fingers["middle"]) and (not fingers["ring"]) and (not fingers["pinky"]):
                self.draw_grace_counter = 2  # refresh grace period
                active_mode = GestureMode.DRAW

            else:
                # Preserve DRAW across at most two ambiguous observed frames.
                # Missing-hand frames take the reset path above and never receive grace.
                if self.draw_grace_counter > 0 and self.last_mode == GestureMode.DRAW and fingers["index"]:
                    self.draw_grace_counter -= 1
                    active_mode = GestureMode.DRAW
                else:
                    self.draw_grace_counter = 0
                    active_mode = GestureMode.IDLE

        self.last_mode = active_mode
        diagnostics = {
            "fingers": fingers,
            "raw_fist": fist_detected,
            "fist_hold_elapsed": (current_time - self.fist_start_time) if self.fist_start_time is not None else 0.0,
        }
        return active_mode, clear_progress, diagnostics

    def _reset_fist_timer(self) -> None:
        self.fist_start_time = None
        self.clear_triggered = False
