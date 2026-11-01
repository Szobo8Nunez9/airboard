"""
MediaPipe Hands wrapper (0.10.x, mp.solutions API).
Detects 21 3D hand landmarks in real-time camera frames.
"""

from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np


class HandTracker:
    """
    Robust 21-landmark hand tracking using MediaPipe Solutions API.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.6,
    ):
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError('Install requirements with Python 3.11, or start with --mouse.') from error
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('This project needs MediaPipe 0.10.14 with mp.solutions. Use Python 3.11.')
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.results: Any = None

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """
        Processes BGR OpenCV frame and optionally draws landmarks and connections.
        """
        # Convert BGR to RGB for MediaPipe inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        self.results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )
        return frame

    def get_landmark_positions(
        self, frame: np.ndarray, hand_idx: int = 0
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float, float]], Optional[Tuple[int, int, int, int]]]:
        """
        Extracts 21 landmark positions in pixel coordinates and normalized coordinates.
        Returns:
            (landmarks_px, landmarks_norm, bounding_box)
            bounding_box is (x_min, y_min, width, height)
        """
        landmarks_px: List[Tuple[int, int]] = []
        landmarks_norm: List[Tuple[float, float, float]] = []

        if not self.results or not self.results.multi_hand_landmarks:
            return landmarks_px, landmarks_norm, None

        if hand_idx >= len(self.results.multi_hand_landmarks):
            return landmarks_px, landmarks_norm, None

        selected_hand = self.results.multi_hand_landmarks[hand_idx]
        h, w, _ = frame.shape

        x_coords: List[int] = []
        y_coords: List[int] = []

        for lm in selected_hand.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks_px.append((cx, cy))
            landmarks_norm.append((lm.x, lm.y, lm.z))
            x_coords.append(cx)
            y_coords.append(cy)

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

        return landmarks_px, landmarks_norm, bbox

    def close(self) -> None:
        """Releases MediaPipe hand landmarker resources."""
        if hasattr(self, "hands") and self.hands:
            self.hands.close()
