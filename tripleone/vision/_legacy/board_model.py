# vision/board_model.py
# Diese Datei enthält die Board-Geometrie, die Homography-Berechnung
# und die Score-Ermittlung für Bildpunkte.
#
# Phase 3.4b:
# - 4 Boundary-Punkte + 1 Center-Punkt
# - echte Homography
#
# Phase 3.5:
# - Klick -> Boardpunkt
# - Boardpunkt -> Ring / Segment / Score
# - Hilfsfunktionen für Visualisierung des Trefferpunkts

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


SECTOR_ORDER = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

BOARD_RADIUS_MM = 170.0

RING_RADII = {
    "double_outer": 170.0 / BOARD_RADIUS_MM,
    "double_inner": 162.0 / BOARD_RADIUS_MM,
    "triple_outer": 107.0 / BOARD_RADIUS_MM,
    "triple_inner": 99.0 / BOARD_RADIUS_MM,
    "outer_bull": 15.9 / BOARD_RADIUS_MM,
    "inner_bull": 6.35 / BOARD_RADIUS_MM,
}

BOUNDARY_ANGLES_DEG = [-81.0, 27.0, 117.0, 207.0]


@dataclass
class BoardHitResult:
    image_x_px: int
    image_y_px: int
    board_x: float
    board_y: float
    radius: float
    angle_deg: float
    sector_value: Optional[int]
    ring_name: str
    multiplier: int
    score: int
    label: str


def build_canonical_reference_points() -> np.ndarray:
    """
    Erzeugt 5 normierte Referenzpunkte:
    4 Boundary-Punkte am äußeren Double-Ring + Center.
    """
    points: List[List[float]] = []

    for angle_deg in BOUNDARY_ANGLES_DEG:
        angle_rad = math.radians(angle_deg)
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        points.append([x, y])

    points.append([0.0, 0.0])
    return np.array(points, dtype=np.float32)


def build_canonical_overlay_reference_points(canvas_size: int) -> np.ndarray:
    """
    Dieselben 5 Referenzpunkte im Pixelraum des kanonischen Overlay-Bildes.
    """
    cx = canvas_size / 2.0
    cy = canvas_size / 2.0
    r = canvas_size * 0.40

    points: List[List[float]] = []
    for angle_deg in BOUNDARY_ANGLES_DEG:
        angle_rad = math.radians(angle_deg)
        x = cx + r * math.cos(angle_rad)
        y = cy + r * math.sin(angle_rad)
        points.append([x, y])

    points.append([cx, cy])
    return np.array(points, dtype=np.float32)


def build_image_reference_points(calibration: Dict) -> np.ndarray:
    """
    Liest die 5 Bildpunkte aus der Kalibrierung.
    """
    raw_points = calibration.get("points", [])
    points: List[List[float]] = []

    for i in range(5):
        if i < len(raw_points) and isinstance(raw_points[i], dict):
            x = float(raw_points[i].get("x_px", 0))
            y = float(raw_points[i].get("y_px", 0))
        else:
            x = 0.0
            y = 0.0
        points.append([x, y])

    return np.array(points, dtype=np.float32)


def build_image_to_board_homography(calibration: Dict) -> Optional[np.ndarray]:
    """
    Homography vom Kamerabild -> normiertes Board.
    """
    image_points = build_image_reference_points(calibration)
    board_points = build_canonical_reference_points()

    try:
        h, _ = cv2.findHomography(image_points, board_points, method=0)
    except cv2.error:
        return None

    return h


def build_board_to_image_homography(calibration: Dict) -> Optional[np.ndarray]:
    """
    Homography vom normierten Board -> Kamerabild.
    """
    image_points = build_image_reference_points(calibration)
    board_points = build_canonical_reference_points()

    try:
        h, _ = cv2.findHomography(board_points, image_points, method=0)
    except cv2.error:
        return None

    return h


def build_overlay_to_image_homography(calibration: Dict, overlay_size: int) -> Optional[np.ndarray]:
    """
    Homography vom kanonischen Overlay-Pixelraum -> Kamerabild.
    """
    overlay_points = build_canonical_overlay_reference_points(overlay_size)
    image_points = build_image_reference_points(calibration)

    try:
        h, _ = cv2.findHomography(overlay_points, image_points, method=0)
    except cv2.error:
        return None

    return h


def project_image_point_to_board(x_px: int, y_px: int, calibration: Dict) -> Optional[Tuple[float, float]]:
    """
    Rechnet einen Bildpunkt in normierte Board-Koordinaten um.
    """
    h = build_image_to_board_homography(calibration)
    if h is None:
        return None

    point = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)

    try:
        transformed = cv2.perspectiveTransform(point, h)
    except cv2.error:
        return None

    return float(transformed[0][0][0]), float(transformed[0][0][1])


def project_board_point_to_image(x_board: float, y_board: float, calibration: Dict) -> Optional[Tuple[float, float]]:
    """
    Rechnet einen normierten Board-Punkt zurück ins Kamerabild.
    """
    h = build_board_to_image_homography(calibration)
    if h is None:
        return None

    point = np.array([[[float(x_board), float(y_board)]]], dtype=np.float32)

    try:
        transformed = cv2.perspectiveTransform(point, h)
    except cv2.error:
        return None

    return float(transformed[0][0][0]), float(transformed[0][0][1])


def board_point_to_overlay_pixel(x_board: float, y_board: float, overlay_size: int) -> Tuple[int, int]:
    """
    Rechnet normierte Board-Koordinaten in Pixel auf dem kanonischen Overlay um.
    """
    cx = overlay_size / 2.0
    cy = overlay_size / 2.0
    r = overlay_size * 0.40

    x_px = int(round(cx + x_board * r))
    y_px = int(round(cy + y_board * r))
    return x_px, y_px


def _resolve_sector(angle_deg: float) -> int:
    """
    Bestimmt das Segment.
    """
    sector_index = int(((angle_deg + 90.0 + 9.0) % 360.0) // 18.0)
    return SECTOR_ORDER[sector_index]


def calculate_board_hit_from_board_point(x_board: float, y_board: float) -> BoardHitResult:
    radius = math.sqrt(x_board * x_board + y_board * y_board)
    angle_deg = math.degrees(math.atan2(y_board, x_board))

    if radius > RING_RADII["double_outer"]:
        return BoardHitResult(
            image_x_px=-1,
            image_y_px=-1,
            board_x=x_board,
            board_y=y_board,
            radius=radius,
            angle_deg=angle_deg,
            sector_value=None,
            ring_name="MISS",
            multiplier=0,
            score=0,
            label="Miss",
        )

    if radius <= RING_RADII["inner_bull"]:
        return BoardHitResult(
            image_x_px=-1,
            image_y_px=-1,
            board_x=x_board,
            board_y=y_board,
            radius=radius,
            angle_deg=angle_deg,
            sector_value=25,
            ring_name="INNER_BULL",
            multiplier=2,
            score=50,
            label="Bull",
        )

    if radius <= RING_RADII["outer_bull"]:
        return BoardHitResult(
            image_x_px=-1,
            image_y_px=-1,
            board_x=x_board,
            board_y=y_board,
            radius=radius,
            angle_deg=angle_deg,
            sector_value=25,
            ring_name="OUTER_BULL",
            multiplier=1,
            score=25,
            label="Outer Bull",
        )

    sector_value = _resolve_sector(angle_deg)

    if radius >= RING_RADII["double_inner"]:
        return BoardHitResult(
            image_x_px=-1,
            image_y_px=-1,
            board_x=x_board,
            board_y=y_board,
            radius=radius,
            angle_deg=angle_deg,
            sector_value=sector_value,
            ring_name="DOUBLE",
            multiplier=2,
            score=sector_value * 2,
            label=f"D{sector_value}",
        )

    if RING_RADII["triple_inner"] <= radius <= RING_RADII["triple_outer"]:
        return BoardHitResult(
            image_x_px=-1,
            image_y_px=-1,
            board_x=x_board,
            board_y=y_board,
            radius=radius,
            angle_deg=angle_deg,
            sector_value=sector_value,
            ring_name="TRIPLE",
            multiplier=3,
            score=sector_value * 3,
            label=f"T{sector_value}",
        )

    return BoardHitResult(
        image_x_px=-1,
        image_y_px=-1,
        board_x=x_board,
        board_y=y_board,
        radius=radius,
        angle_deg=angle_deg,
        sector_value=sector_value,
        ring_name="SINGLE",
        multiplier=1,
        score=sector_value,
        label=f"S{sector_value}",
    )


def calculate_board_hit_from_image_point(x_px: int, y_px: int, calibration: Dict) -> Optional[BoardHitResult]:
    projected = project_image_point_to_board(x_px, y_px, calibration)
    if projected is None:
        return None

    x_board, y_board = projected
    result = calculate_board_hit_from_board_point(x_board, y_board)
    result.image_x_px = x_px
    result.image_y_px = y_px
    return result