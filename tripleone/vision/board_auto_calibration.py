# vision/board_auto_calibration.py
# Phase 6.1:
# - automatische Board-Orientierung
# - erkennt Kreis / Ellipse des Boards
# - schätzt Mittelpunkt und Radius
# - erzeugt daraus automatisch 4 Randpunkte + Bullzentrum

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _clamp_point(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    return (
        int(max(0, min(w - 1, round(x)))),
        int(max(0, min(h - 1, round(y)))),
    )


def _sample_point_on_circle(cx: float, cy: float, radius: float, angle_deg: float, w: int, h: int) -> Tuple[int, int]:
    angle_rad = math.radians(angle_deg)
    x = cx + math.cos(angle_rad) * radius
    y = cy - math.sin(angle_rad) * radius
    return _clamp_point(x, y, w, h)


def detect_board_geometry(frame_bgr: np.ndarray) -> Optional[Dict]:
    """
    Liefert grobe Board-Geometrie:
    - center_x
    - center_y
    - radius
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 0)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=200,
        param1=120,
        param2=30,
        minRadius=120,
        maxRadius=min(frame_bgr.shape[0], frame_bgr.shape[1]),
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)

    h, w = frame_bgr.shape[:2]
    image_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    best_circle = None
    best_score = None

    for c in circles:
        x, y, r = int(c[0]), int(c[1]), int(c[2])

        if r < 100:
            continue

        dist_to_center = np.linalg.norm(np.array([x, y], dtype=np.float32) - image_center)

        # Board liegt meist ungefähr mittig, Radius eher groß
        score = dist_to_center - (r * 0.35)

        if best_score is None or score < best_score:
            best_score = score
            best_circle = (x, y, r)

    if best_circle is None:
        return None

    x, y, r = best_circle
    return {
        "center_x": int(x),
        "center_y": int(y),
        "radius": int(r),
    }


def estimate_board_rotation_from_top(frame_bgr: np.ndarray, center_x: int, center_y: int, radius: int) -> float:
    """
    Sehr grobe Rotationsschätzung:
    Wir suchen im oberen Boardbereich nach dem hellsten Zahlen-/Top-Bereich.
    Rückgabe:
    - Winkel in Grad
    - 90° bedeutet geometrisch "oben"
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    scan_radius = radius * 1.08
    best_angle = 90.0
    best_score = None

    # Suche nur im oberen Halbkreis
    for angle_deg in np.linspace(40, 140, 101):
        x, y = _sample_point_on_circle(center_x, center_y, scan_radius, angle_deg, w, h)

        x0 = max(0, x - 18)
        x1 = min(w, x + 19)
        y0 = max(0, y - 18)
        y1 = min(h, y + 19)

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        # Zahlenbereich / helle Top-Region
        score = float(np.mean(roi))

        if best_score is None or score > best_score:
            best_score = score
            best_angle = float(angle_deg)

    return best_angle


def build_auto_calibration_points(frame_bgr: np.ndarray) -> Optional[Dict]:
    """
    Erzeugt automatisch:
    - P1 = Grenze 20|1
    - P2 = Grenze 6|10
    - P3 = Grenze 3|19
    - P4 = Grenze 11|14
    - C  = Bullzentrum

    Wichtig:
    Das ist eine grobe automatische Initialisierung.
    """
    geometry = detect_board_geometry(frame_bgr)
    if geometry is None:
        return None

    h, w = frame_bgr.shape[:2]
    cx = geometry["center_x"]
    cy = geometry["center_y"]
    r = geometry["radius"]

    top_angle = estimate_board_rotation_from_top(frame_bgr, cx, cy, r)

    # P1 ist die Grenze 20|1.
    # Wenn 20 oben liegt, dann ist diese Grenze leicht rechts von oben.
    # Das Modell nimmt 9° pro Halbfeld, also 18° pro Segment.
    # P1 = Top-Achse - 9°
    p1_angle = top_angle - 9.0
    p2_angle = p1_angle - (5 * 18.0)   # weiter im Uhrzeigersinn
    p3_angle = p2_angle - (7 * 18.0)
    p4_angle = p3_angle - (5 * 18.0)

    outer_radius = r * 0.92

    p1 = _sample_point_on_circle(cx, cy, outer_radius, p1_angle, w, h)
    p2 = _sample_point_on_circle(cx, cy, outer_radius, p2_angle, w, h)
    p3 = _sample_point_on_circle(cx, cy, outer_radius, p3_angle, w, h)
    p4 = _sample_point_on_circle(cx, cy, outer_radius, p4_angle, w, h)

    return {
        "center_x": cx,
        "center_y": cy,
        "radius": r,
        "top_angle": top_angle,
        "points": [
            {"x_px": p1[0], "y_px": p1[1]},
            {"x_px": p2[0], "y_px": p2[1]},
            {"x_px": p3[0], "y_px": p3[1]},
            {"x_px": p4[0], "y_px": p4[1]},
            {"x_px": int(cx), "y_px": int(cy)},
        ],
    }