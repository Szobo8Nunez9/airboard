"""
Virtual drawing canvas with overlay blending, whiteboard mode, and PNG snapshot export.
"""

from typing import Tuple, Optional, Dict
import math
import cv2
import numpy as np


class AirCanvas:
    """
    Manages drawing strokes on a separate canvas buffer.
    Supports overlay blending on video and crisp whiteboard mode.
    """

    COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
        "Royal Blue": (230, 110, 20),      # BGR
        "Emerald Green": (40, 190, 60),    # BGR
        "Crimson Red": (30, 30, 230),      # BGR
        "Amber Orange": (20, 140, 255),    # BGR
        "Deep Purple": (190, 40, 160),     # BGR
        "Charcoal Black": (30, 30, 30),    # BGR
    }

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        # Canvas holds RGB/BGR strokes on pure black (0,0,0) background
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.current_color_name = "Royal Blue"
        self.current_color = self.COLOR_PALETTE[self.current_color_name]
        self.brush_thickness = 7
        self.eraser_radius = 45

        self.prev_point: Optional[Tuple[int, int]] = None
        self.whiteboard_mode = False

    def set_color(self, color_name: str) -> None:
        """Switch active brush color."""
        if color_name in self.COLOR_PALETTE:
            self.current_color_name = color_name
            self.current_color = self.COLOR_PALETTE[color_name]

    def set_thickness(self, thickness: int) -> None:
        self.brush_thickness = max(1, thickness)

    def draw_segment(self, start_pt: Tuple[int, int], end_pt: Tuple[int, int]) -> None:
        """
        Draws a continuous, round-capped anti-aliased stroke segment.
        Interpolates sub-points between start and end to eliminate gaps and sharp facets.
        """
        dist = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
        radius = max(1, self.brush_thickness // 2)

        # Draw primary anti-aliased line
        cv2.line(
            self.canvas,
            start_pt,
            end_pt,
            self.current_color,
            self.brush_thickness,
            cv2.LINE_AA,
        )

        # Sub-step circular discs for smooth continuous ink
        steps = max(1, int(dist / 3.0))
        for s in range(steps + 1):
            t = s / steps
            x = int(round(start_pt[0] + t * (end_pt[0] - start_pt[0])))
            y = int(round(start_pt[1] + t * (end_pt[1] - start_pt[1])))
            cv2.circle(self.canvas, (x, y), radius, self.current_color, -1, cv2.LINE_AA)

    def erase_at(self, center_pt: Tuple[int, int]) -> None:
        """Erases strokes within radius around center_pt."""
        cv2.circle(
            self.canvas,
            center_pt,
            self.eraser_radius,
            (0, 0, 0),
            thickness=-1,
        )

    def clear(self) -> None:
        """Wipes the canvas completely."""
        self.canvas[:] = 0
        self.prev_point = None

    def reset_stroke(self) -> None:
        """Ends current contiguous stroke segment."""
        self.prev_point = None

    def render_view(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Renders either the camera frame with overlaid drawing or a pure whiteboard.
        """
        if camera_frame.shape[:2] != (self.height, self.width):
            camera_frame = cv2.resize(camera_frame, (self.width, self.height))

        if self.whiteboard_mode:
            # Create crisp pure white background
            board = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
            # Create mask of drawn areas
            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            # Copy drawing onto white board
            board[mask > 0] = self.canvas[mask > 0]
            return board
        else:
            # Video overlay mode: seamless blending
            gray_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_canvas, 10, 255, cv2.THRESH_BINARY)
            inv_mask = cv2.bitwise_not(mask)

            # Darken camera slightly behind drawing to make ink pop
            frame_bg = cv2.bitwise_and(camera_frame, camera_frame, mask=inv_mask)
            canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)
            combined = cv2.add(frame_bg, canvas_fg)
            return combined

    def get_export_image(self) -> np.ndarray:
        """
        Returns high-contrast image suitable for vision AI models.
        White background with vibrant strokes.
        """
        board = np.full((self.height, self.width, 3), 255, dtype=np.uint8)
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        board[mask > 0] = self.canvas[mask > 0]
        return board

    def get_png_bytes(self) -> bytes:
        """Encodes whiteboard image as in-memory PNG bytes."""
        export_img = self.get_export_image()
        # OpenCV encoders expect BGR and write standard PNG colors.
        success, encoded_img = cv2.imencode(".png", export_img)
        if not success:
            raise RuntimeError("Failed to encode canvas as PNG image.")
        return encoded_img.tobytes()

    def save_snapshot(self, filepath: str) -> None:
        """Saves current board snapshot to disk."""
        export_img = self.get_export_image()
        if not cv2.imwrite(filepath, export_img):
            raise OSError(f"Could not save snapshot to {filepath}")

    def is_empty(self) -> bool:
        """Returns True if nothing has been drawn on the canvas."""
        return not np.any(self.canvas > 10)
