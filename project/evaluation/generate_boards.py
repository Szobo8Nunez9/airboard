"""
Generates 15 reproducible test boards for AI vision benchmark:
- 5 Equations (Linear, Quadratic, Derivative, Fraction, Definite Integral)
- 5 Sketches (Flowchart, House, RC Circuit, Stick Figure, Tree & Sun)
- 5 Messy (Scribbled handwriting, Crossouts/Corrections, Open Polygons, Noisy Fraction, Abstract Chaos)
"""

import os
from typing import Dict, List, Tuple
import cv2
import numpy as np


def create_blank_canvas(width: int = 1280, height: int = 720) -> np.ndarray:
    """Returns pure white canvas (255, 255, 255)."""
    return np.full((height, width, 3), 255, dtype=np.uint8)


def add_handwritten_stroke(
    img: np.ndarray,
    pts: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (200, 50, 30),
    thickness: int = 6,
    jitter: float = 1.2,
) -> None:
    """Simulates natural airboard hand stroke with subtle jitter."""
    if len(pts) < 2:
        return
    np.random.seed(42 + len(pts))
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        j1 = (int(p1[0] + np.random.normal(0, jitter)), int(p1[1] + np.random.normal(0, jitter)))
        j2 = (int(p2[0] + np.random.normal(0, jitter)), int(p2[1] + np.random.normal(0, jitter)))
        cv2.line(img, j1, j2, color, thickness, cv2.LINE_AA)


def generate_15_test_boards(output_dir: str = "evaluation/test_boards") -> List[Dict[str, str]]:
    os.makedirs(output_dir, exist_ok=True)
    manifest = []

    # -------------------------------------------------------------
    # 5 EQUATIONS
    # -------------------------------------------------------------
    # 1. Linear: 2x + 7 = 19
    b1 = create_blank_canvas()
    cv2.putText(b1, "2x + 7 = 19", (340, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.2, (220, 70, 20), 8, cv2.LINE_AA)
    cv2.putText(b1, "Solve for x", (420, 470), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3, cv2.LINE_AA)
    p1 = os.path.join(output_dir, "eq_01_linear.png")
    cv2.imwrite(p1, b1)
    manifest.append({
        "id": "eq_01",
        "category": "equation",
        "title": "Linear Equation (2x + 7 = 19)",
        "ground_truth": "x = 6",
        "path": p1,
    })

    # 2. Quadratic: x^2 - 5x + 6 = 0
    b2 = create_blank_canvas()
    cv2.putText(b2, "x^2 - 5x + 6 = 0", (300, 350), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.0, (20, 160, 40), 8, cv2.LINE_AA)
    cv2.putText(b2, "Find roots", (440, 460), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3, cv2.LINE_AA)
    p2 = os.path.join(output_dir, "eq_02_quadratic.png")
    cv2.imwrite(p2, b2)
    manifest.append({
        "id": "eq_02",
        "category": "equation",
        "title": "Quadratic Equation (x^2 - 5x + 6 = 0)",
        "ground_truth": "x = 2, 3",
        "path": p2,
    })

    # 3. Calculus derivative: d/dx(3x^2 + 4x)
    b3 = create_blank_canvas()
    cv2.putText(b3, "d/dx [ 3x^2 + 4x ] = ?", (220, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.7, (180, 40, 160), 8, cv2.LINE_AA)
    p3 = os.path.join(output_dir, "eq_03_derivative.png")
    cv2.imwrite(p3, b3)
    manifest.append({
        "id": "eq_03",
        "category": "equation",
        "title": "Derivative d/dx(3x^2 + 4x)",
        "ground_truth": "6x + 4",
        "path": p3,
    })

    # 4. Fraction arithmetic: 3/4 + 2/3
    b4 = create_blank_canvas()
    cv2.putText(b4, "3/4 + 2/3 = ?", (380, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.2, (20, 110, 230), 8, cv2.LINE_AA)
    p4 = os.path.join(output_dir, "eq_04_fraction.png")
    cv2.imwrite(p4, b4)
    manifest.append({
        "id": "eq_04",
        "category": "equation",
        "title": "Fraction Addition (3/4 + 2/3)",
        "ground_truth": "17/12 (approx 1.417)",
        "path": p4,
    })

    # 5. Definite integral: int_0^2 2x dx
    b5 = create_blank_canvas()
    cv2.putText(b5, "int_0^2 (2x) dx = ?", (320, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.9, (180, 80, 20), 8, cv2.LINE_AA)
    p5 = os.path.join(output_dir, "eq_05_integral.png")
    cv2.imwrite(p5, b5)
    manifest.append({
        "id": "eq_05",
        "category": "equation",
        "title": "Definite Integral (int_0^2 2x dx)",
        "ground_truth": "4",
        "path": p5,
    })

    # -------------------------------------------------------------
    # 5 SKETCHES
    # -------------------------------------------------------------
    # 6. Flowchart (User -> Server -> DB)
    b6 = create_blank_canvas()
    # Boxes
    cv2.rectangle(b6, (140, 280), (340, 420), (220, 80, 30), 6)
    cv2.putText(b6, "Client", (180, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 3)
    cv2.arrowedLine(b6, (350, 350), (520, 350), (40, 40, 40), 5, tipLength=0.15)

    cv2.rectangle(b6, (530, 280), (750, 420), (30, 160, 40), 6)
    cv2.putText(b6, "Server", (575, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 3)
    cv2.arrowedLine(b6, (760, 350), (930, 350), (40, 40, 40), 5, tipLength=0.15)

    cv2.ellipse(b6, (1040, 320), (90, 40), 0, 0, 360, (180, 40, 160), 6)
    cv2.line(b6, (950, 320), (950, 400), (180, 40, 160), 6)
    cv2.line(b6, (1130, 320), (1130, 400), (180, 40, 160), 6)
    cv2.ellipse(b6, (1040, 400), (90, 40), 0, 0, 180, (180, 40, 160), 6)
    cv2.putText(b6, "Database", (980, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    p6 = os.path.join(output_dir, "sk_01_flowchart.png")
    cv2.imwrite(p6, b6)
    manifest.append({
        "id": "sk_01",
        "category": "sketch",
        "title": "System Architecture Flowchart",
        "ground_truth": "Three-tier architecture diagram: Client connected to Server connected to Database",
        "path": p6,
    })

    # 7. House sketch
    b7 = create_blank_canvas()
    # Roof triangle
    pts_roof = np.array([[640, 160], [420, 320], [860, 320]], np.int32)
    cv2.polylines(b7, [pts_roof], isClosed=True, color=(30, 30, 220), thickness=7)
    # Walls
    cv2.rectangle(b7, (450, 320), (830, 600), (30, 30, 220), 7)
    # Door
    cv2.rectangle(b7, (590, 440), (690, 600), (160, 80, 30), 6)
    cv2.circle(b7, (675, 520), 5, (40, 40, 40), -1)
    # Windows
    cv2.rectangle(b7, (490, 360), (560, 430), (200, 140, 20), 5)
    cv2.line(b7, (525, 360), (525, 430), (200, 140, 20), 3)
    cv2.line(b7, (490, 395), (560, 395), (200, 140, 20), 3)
    p7 = os.path.join(output_dir, "sk_02_house.png")
    cv2.imwrite(p7, b7)
    manifest.append({
        "id": "sk_02",
        "category": "sketch",
        "title": "Suburban House Sketch",
        "ground_truth": "Drawing of a house with triangular gabled roof, door, and divided window pane",
        "path": p7,
    })

    # 8. RC Circuit
    b8 = create_blank_canvas()
    # Battery
    cv2.line(b8, (300, 300), (300, 420), (40, 40, 40), 5)
    cv2.line(b8, (270, 340), (330, 340), (40, 40, 40), 7)
    cv2.line(b8, (285, 380), (315, 380), (40, 40, 40), 7)
    cv2.putText(b8, "+ V -", (220, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    # Resistor zigzag
    cv2.line(b8, (300, 300), (500, 300), (40, 40, 40), 5)
    res_pts = np.array([[500, 300], [530, 270], [560, 330], [590, 270], [620, 330], [650, 300]], np.int32)
    cv2.polylines(b8, [res_pts], False, (20, 160, 40), 6)
    cv2.putText(b8, "R", (560, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 160, 40), 3)
    # Capacitor
    cv2.line(b8, (650, 300), (900, 300), (40, 40, 40), 5)
    cv2.line(b8, (900, 300), (900, 340), (40, 40, 40), 5)
    cv2.line(b8, (860, 340), (940, 340), (180, 40, 160), 6)
    cv2.line(b8, (860, 370), (940, 370), (180, 40, 160), 6)
    cv2.putText(b8, "C", (960, 365), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (180, 40, 160), 3)
    cv2.line(b8, (900, 370), (900, 420), (40, 40, 40), 5)
    cv2.line(b8, (900, 420), (300, 420), (40, 40, 40), 5)
    p8 = os.path.join(output_dir, "sk_03_circuit.png")
    cv2.imwrite(p8, b8)
    manifest.append({
        "id": "sk_03",
        "category": "sketch",
        "title": "Series RC Circuit Diagram",
        "ground_truth": "Schematic of electrical series circuit with voltage source, resistor R, and capacitor C",
        "path": p8,
    })

    # 9. Stick Figure
    b9 = create_blank_canvas()
    cv2.circle(b9, (640, 240), 50, (30, 30, 30), 6)
    cv2.line(b9, (640, 290), (640, 480), (30, 30, 30), 6)
    cv2.line(b9, (640, 340), (540, 400), (30, 30, 30), 6)
    cv2.line(b9, (640, 340), (740, 260), (30, 30, 30), 6)  # waving arm
    cv2.line(b9, (740, 260), (770, 210), (30, 30, 30), 6)
    cv2.line(b9, (640, 480), (560, 610), (30, 30, 30), 6)
    cv2.line(b9, (640, 480), (720, 610), (30, 30, 30), 6)
    p9 = os.path.join(output_dir, "sk_04_stickfigure.png")
    cv2.imwrite(p9, b9)
    manifest.append({
        "id": "sk_04",
        "category": "sketch",
        "title": "Waving Stick Figure",
        "ground_truth": "Stick figure person with right hand raised waving",
        "path": p9,
    })

    # 10. Tree & Sun
    b10 = create_blank_canvas()
    # Sun
    cv2.circle(b10, (250, 180), 55, (0, 180, 255), 7)
    for ang in range(0, 360, 45):
        rad = np.radians(ang)
        x1 = int(250 + 75 * np.cos(rad))
        y1 = int(180 + 75 * np.sin(rad))
        x2 = int(250 + 105 * np.cos(rad))
        y2 = int(180 + 105 * np.sin(rad))
        cv2.line(b10, (x1, y1), (x2, y2), (0, 180, 255), 5)
    # Tree trunk
    cv2.rectangle(b10, (690, 380), (750, 620), (20, 70, 140), -1)
    # Foliage circles
    cv2.circle(b10, (720, 320), 110, (40, 180, 50), -1)
    cv2.circle(b10, (650, 300), 80, (50, 200, 60), -1)
    cv2.circle(b10, (790, 300), 80, (30, 160, 40), -1)
    p10 = os.path.join(output_dir, "sk_05_tree_sun.png")
    cv2.imwrite(p10, b10)
    manifest.append({
        "id": "sk_05",
        "category": "sketch",
        "title": "Tree and Sun Landscape",
        "ground_truth": "Landscape sketch showing radiant sun with rays next to leafy tree with brown trunk",
        "path": p10,
    })

    # -------------------------------------------------------------
    # 5 MESSY BOARDS
    # -------------------------------------------------------------
    # 11. Scribbled handwriting: rushed y = mx + b
    b11 = create_blank_canvas()
    cv2.putText(b11, "y = mx + b?", (320, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.4, (120, 50, 20), 6, cv2.LINE_AA)
    # Add random ink blots
    cv2.circle(b11, (430, 310), 12, (120, 50, 20), -1)
    cv2.line(b11, (290, 420), (750, 410), (140, 60, 30), 4)
    p11 = os.path.join(output_dir, "ms_01_scribble_linear.png")
    cv2.imwrite(p11, b11)
    manifest.append({
        "id": "ms_01",
        "category": "messy",
        "title": "Scribbled Linear Equation with Ink Blot",
        "ground_truth": "y = mx + b with stray underlines and blotches",
        "path": p11,
    })

    # 12. Crossouts & corrections: 4 * 5 = 18 20
    b12 = create_blank_canvas()
    cv2.putText(b12, "4 x 5 = 18", (300, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.0, (30, 30, 180), 7, cv2.LINE_AA)
    # Heavy crossout over 18
    cv2.line(b12, (620, 280), (740, 390), (0, 0, 255), 9)
    cv2.line(b12, (620, 390), (740, 280), (0, 0, 255), 9)
    # Correction written above
    cv2.putText(b12, "20", (630, 240), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.6, (20, 160, 40), 7, cv2.LINE_AA)
    p12 = os.path.join(output_dir, "ms_02_crossout.png")
    cv2.imwrite(p12, b12)
    manifest.append({
        "id": "ms_02",
        "category": "messy",
        "title": "Crossout Arithmetic Correction",
        "ground_truth": "4 x 5 = 20 (with 18 crossed out in red and 20 corrected)",
        "path": p12,
    })

    # 13. Incomplete geometric sketch (broken polygons)
    b13 = create_blank_canvas()
    # Unclosed triangle & disconnected box
    cv2.line(b13, (350, 250), (550, 220), (60, 60, 60), 6)
    cv2.line(b13, (550, 220), (460, 420), (60, 60, 60), 6)
    # Gap left open
    cv2.line(b13, (650, 260), (850, 260), (40, 120, 200), 6)
    cv2.line(b13, (850, 260), (840, 440), (40, 120, 200), 6)
    cv2.line(b13, (840, 440), (660, 430), (40, 120, 200), 6)
    cv2.putText(b13, "??", (500, 340), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)
    p13 = os.path.join(output_dir, "ms_03_broken_polygons.png")
    cv2.imwrite(p13, b13)
    manifest.append({
        "id": "ms_03",
        "category": "messy",
        "title": "Incomplete Geometric Outlines",
        "ground_truth": "Fragmented incomplete polygons with question marks",
        "path": p13,
    })

    # 14. Noisy faint fraction: 1/x + 1/y = 1
    b14 = create_blank_canvas()
    cv2.putText(b14, "1/x + 1/y = 1", (320, 360), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2.5, (160, 160, 160), 4, cv2.LINE_AA)
    # Add random noise dots across canvas
    np.random.seed(99)
    noise_x = np.random.randint(200, 1000, 80)
    noise_y = np.random.randint(150, 550, 80)
    for nx, ny in zip(noise_x, noise_y):
        cv2.circle(b14, (int(nx), int(ny)), int(np.random.randint(2, 6)), (140, 140, 140), -1)
    p14 = os.path.join(output_dir, "ms_04_noisy_fraction.png")
    cv2.imwrite(p14, b14)
    manifest.append({
        "id": "ms_04",
        "category": "messy",
        "title": "Low-Contrast Faint Equation with Noise",
        "ground_truth": "1/x + 1/y = 1 obscured by optical noise specks",
        "path": p14,
    })

    # 15. Abstract chaotic scribble
    b15 = create_blank_canvas()
    np.random.seed(123)
    pts = []
    x, y = 300, 360
    for _ in range(60):
        x = np.clip(x + np.random.randint(-50, 50), 200, 1080)
        y = np.clip(y + np.random.randint(-40, 40), 150, 580)
        pts.append((int(x), int(y)))
    for i in range(len(pts) - 1):
        cv2.line(b15, pts[i], pts[i + 1], (40, 40, 180), 5, cv2.LINE_AA)
    cv2.putText(b15, "STORM", (780, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (180, 40, 30), 3)
    p15 = os.path.join(output_dir, "ms_05_chaotic_scribble.png")
    cv2.imwrite(p15, b15)
    manifest.append({
        "id": "ms_05",
        "category": "messy",
        "title": "Chaotic Abstract Scribble",
        "ground_truth": "Chaotic dense stroke scribble labeled STORM",
        "path": p15,
    })

    print(f"[generate_boards] Successfully created 15 test boards in '{output_dir}'.")
    return manifest


if __name__ == "__main__":
    generate_15_test_boards()
