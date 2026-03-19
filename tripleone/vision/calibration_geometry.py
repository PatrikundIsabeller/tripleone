# vision/calibration_geometry.py
# Triple One - Schritt 1:
# Zentrale, EINHEITLICHE Geometrie für:
# - 4-Punkt-Kalibrierung
# - Bull-Berechnung
# - Overlay
# - Testpunkt-Scoring
# - Top-Down-Warp
#
# Diese Datei ist die einzige Wahrheit für die Board-Geometrie.
# Alle anderen Module sollen diese Datei verwenden
# statt eigene Ring-/Winkel-/Bull-Logik zu rechnen.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------
# Globale Board-Konstanten
# ------------------------------------------------------------

TOPDOWN_SIZE = 900
TOPDOWN_CENTER_X = TOPDOWN_SIZE / 2.0
TOPDOWN_CENTER_Y = TOPDOWN_SIZE / 2.0
TOPDOWN_CENTER = (TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y)

# Outer Double = Referenzradius
OUTER_DOUBLE_RADIUS_PX = TOPDOWN_SIZE * 0.36

# Echte Board-Radien relativ zu 170 mm Outer Double
INNER_DOUBLE_REL = 162.0 / 170.0
OUTER_TRIPLE_REL = 107.0 / 170.0
INNER_TRIPLE_REL = 99.0 / 170.0
OUTER_BULL_REL = 15.9 / 170.0
INNER_BULL_REL = 6.35 / 170.0

# Für Overlay-Zeichnung
RING_RADII_REL = [
    1.0,                # äußerer Double-Rand
    INNER_DOUBLE_REL,   # innerer Double-Rand
    OUTER_TRIPLE_REL,   # äußerer Triple-Rand
    INNER_TRIPLE_REL,   # innerer Triple-Rand
    OUTER_BULL_REL,     # äußerer Bull-Rand
    INNER_BULL_REL,     # innerer Bull-Rand
]

# Reihenfolge clockwise ab oben
SEGMENT_ORDER = [
    20, 1, 18, 4, 13,
    6, 10, 15, 2, 17,
    3, 19, 7, 16, 8,
    11, 14, 9, 12, 5,
]

# Feste manuelle Referenzpunkte auf dem äußeren Double-Draht
# P1 = 20|1
# P2 = 6|10
# P3 = 3|19
# P4 = 11|14
MANUAL_LABELS = ["P1(20|1)", "P2(6|10)", "P3(3|19)", "P4(11|14)"]
MANUAL_BOUNDARY_ANGLES = [81.0, 351.0, 261.0, 171.0]


# ------------------------------------------------------------
# Datenklassen
# ------------------------------------------------------------

@dataclass
class HitResult:
    image_x_px: int
    image_y_px: int
    topdown_x_px: float
    topdown_y_px: float
    board_x: float
    board_y: float
    radius: float
    angle_deg: float
    segment_value: Optional[int]
    multiplier: int
    ring_name: str
    score: int
    label: str


# ------------------------------------------------------------
# Grundlegende Helpers
# ------------------------------------------------------------

def get_manual_labels() -> List[str]:
    """
    Liefert die Anzeigenamen der 4 Kalibrierpunkte.
    """
    return list(MANUAL_LABELS)


def _normalize_manual_points(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Optional[List[Dict[str, int]]]:
    """
    Akzeptiert entweder:
    - direkt eine Liste von Punkten
    - oder ein Dict mit key="points"
    und gibt die ersten 4 Punkte normiert zurück.
    """
    points = points_like
    if isinstance(points_like, dict):
        points = points_like.get("points", [])

    if not isinstance(points, Sequence) or len(points) < 4:
        return None

    norm: List[Dict[str, int]] = []
    for item in list(points)[:4]:
        if not isinstance(item, dict):
            return None

        norm.append(
            {
                "x_px": int(item.get("x_px", 0)),
                "y_px": int(item.get("y_px", 0)),
            }
        )

    return norm


def _topdown_point_on_outer_double(boundary_angle_deg: float) -> Tuple[float, float]:
    """
    Erzeugt einen Top-Down-Referenzpunkt auf dem äußeren Double-Draht.

    Winkeldefinition:
    - 0° = rechts
    - 90° = oben
    - 180° = links
    - 270° = unten
    """
    a = math.radians(boundary_angle_deg)
    x = TOPDOWN_CENTER_X + math.cos(a) * OUTER_DOUBLE_RADIUS_PX
    y = TOPDOWN_CENTER_Y - math.sin(a) * OUTER_DOUBLE_RADIUS_PX
    return x, y


def get_destination_points_topdown() -> np.ndarray:
    """
    Zielpunkte im Top-Down-Bild für die 4 festen Referenzpunkte.
    """
    pts = [_topdown_point_on_outer_double(a) for a in MANUAL_BOUNDARY_ANGLES]
    return np.array(pts, dtype=np.float32)


# ------------------------------------------------------------
# Homography
# ------------------------------------------------------------

def compute_homography_image_to_topdown(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Optional[np.ndarray]:
    """
    Bild -> Top-Down
    """
    manual_points = _normalize_manual_points(points_like)
    if manual_points is None:
        return None

    src = np.array(
        [[float(p["x_px"]), float(p["y_px"])] for p in manual_points],
        dtype=np.float32,
    )
    dst = get_destination_points_topdown()
    return cv2.getPerspectiveTransform(src, dst)


def compute_homography_topdown_to_image(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Optional[np.ndarray]:
    """
    Top-Down -> Bild
    """
    manual_points = _normalize_manual_points(points_like)
    if manual_points is None:
        return None

    src = get_destination_points_topdown()
    dst = np.array(
        [[float(p["x_px"]), float(p["y_px"])] for p in manual_points],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def project_image_points_to_topdown(
    points_like: Sequence[Dict[str, int]] | Dict,
    image_points: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Projiziert Nx2 Bildpunkte nach Top-Down.
    """
    h = compute_homography_image_to_topdown(points_like)
    if h is None:
        return None

    pts = image_points.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, h)
    return mapped.reshape(-1, 2)


def project_topdown_points_to_image(
    points_like: Sequence[Dict[str, int]] | Dict,
    topdown_points: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Projiziert Nx2 Top-Down-Punkte zurück ins Bild.
    """
    h_inv = compute_homography_topdown_to_image(points_like)
    if h_inv is None:
        return None

    pts = topdown_points.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, h_inv)
    return mapped.reshape(-1, 2)


def image_to_topdown_point(
    points_like: Sequence[Dict[str, int]] | Dict,
    x_px: float,
    y_px: float,
) -> Optional[Tuple[float, float]]:
    mapped = project_image_points_to_topdown(
        points_like,
        np.array([[float(x_px), float(y_px)]], dtype=np.float32),
    )
    if mapped is None:
        return None
    return float(mapped[0, 0]), float(mapped[0, 1])


def topdown_to_image_point(
    points_like: Sequence[Dict[str, int]] | Dict,
    x_px: float,
    y_px: float,
) -> Optional[Tuple[float, float]]:
    mapped = project_topdown_points_to_image(
        points_like,
        np.array([[float(x_px), float(y_px)]], dtype=np.float32),
    )
    if mapped is None:
        return None
    return float(mapped[0, 0]), float(mapped[0, 1])


def warp_frame_to_topdown(
    frame_bgr: np.ndarray,
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Optional[np.ndarray]:
    """
    Entzerrt ein Bild in die Top-Down-Boardansicht.
    """
    h = compute_homography_image_to_topdown(points_like)
    if h is None:
        return None

    return cv2.warpPerspective(
        frame_bgr,
        h,
        (TOPDOWN_SIZE, TOPDOWN_SIZE),
        flags=cv2.INTER_LINEAR,
    )


# ------------------------------------------------------------
# Bull / Pipeline-Punkte
# ------------------------------------------------------------

def compute_bull_from_manual_points(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Dict[str, int]:
    """
    Berechnet den Bullpunkt aus der 4-Punkt-Homography.
    """
    bull_img = topdown_to_image_point(points_like, TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y)
    if bull_img is None:
        return {"x_px": 0, "y_px": 0}

    return {
        "x_px": int(round(bull_img[0])),
        "y_px": int(round(bull_img[1])),
    }


def build_pipeline_points(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> List[Dict[str, int]]:
    """
    Baut das kompatible 5-Punkte-Format:
    - 4 manuelle Marker
    - 1 automatisch berechneter Bull
    """
    manual_points = _normalize_manual_points(points_like)
    if manual_points is None:
        return []

    bull = compute_bull_from_manual_points(manual_points)
    return manual_points + [bull]


# ------------------------------------------------------------
# Overlay-Geometrie
# ------------------------------------------------------------

def generate_ring_polylines_image(
    points_like: Sequence[Dict[str, int]] | Dict,
    degree_step: int = 3,
) -> List[np.ndarray]:
    """
    Erzeugt Ring-Polylinien im Bildraum.
    """
    polylines: List[np.ndarray] = []

    for rel_r in RING_RADII_REL:
        r = OUTER_DOUBLE_RADIUS_PX * rel_r
        pts = []

        for deg in range(0, 361, degree_step):
            a = math.radians(deg)
            x = TOPDOWN_CENTER_X + math.cos(a) * r
            y = TOPDOWN_CENTER_Y - math.sin(a) * r
            pts.append([x, y])

        pts_np = np.array(pts, dtype=np.float32)
        mapped = project_topdown_points_to_image(points_like, pts_np)
        if mapped is not None:
            polylines.append(mapped)

    return polylines


def generate_sector_lines_image(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> List[np.ndarray]:
    """
    Erzeugt die 20 Sektorgrenzen vom Bull bis zum äußeren Double.
    """
    lines: List[np.ndarray] = []

    for i in range(20):
        angle_deg = 99.0 - i * 18.0
        a = math.radians(angle_deg)

        p0 = [TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y]
        p1 = [
            TOPDOWN_CENTER_X + math.cos(a) * OUTER_DOUBLE_RADIUS_PX,
            TOPDOWN_CENTER_Y - math.sin(a) * OUTER_DOUBLE_RADIUS_PX,
        ]

        pts = np.array([p0, p1], dtype=np.float32)
        mapped = project_topdown_points_to_image(points_like, pts)
        if mapped is not None:
            lines.append(mapped)

    return lines


def generate_number_positions_image(
    points_like: Sequence[Dict[str, int]] | Dict,
) -> List[Tuple[str, Tuple[float, float]]]:
    """
    Positionen der Segmentzahlen im Bildraum.
    """
    label_r = OUTER_DOUBLE_RADIUS_PX * 1.12
    result: List[Tuple[str, Tuple[float, float]]] = []

    for i, value in enumerate(SEGMENT_ORDER):
        center_angle_deg = 90.0 - i * 18.0
        a = math.radians(center_angle_deg)

        x = TOPDOWN_CENTER_X + math.cos(a) * label_r
        y = TOPDOWN_CENTER_Y - math.sin(a) * label_r

        mapped = project_topdown_points_to_image(
            points_like,
            np.array([[x, y]], dtype=np.float32),
        )
        if mapped is not None:
            result.append((str(value), (float(mapped[0, 0]), float(mapped[0, 1]))))

    return result


def generate_twenty_segment_polygon_image(
    points_like: Sequence[Dict[str, int]] | Dict,
    steps: int = 64,
) -> Optional[np.ndarray]:
    """
    Polygon des 20er-Segments im Bildraum.
    Grenzen:
    - 20|5 = 99°
    - 20|1 = 81°
    """
    angles = np.linspace(99.0, 81.0, steps)

    outer_pts = []
    for deg in angles:
        a = math.radians(float(deg))
        x = TOPDOWN_CENTER_X + math.cos(a) * OUTER_DOUBLE_RADIUS_PX
        y = TOPDOWN_CENTER_Y - math.sin(a) * OUTER_DOUBLE_RADIUS_PX
        outer_pts.append([x, y])

    inner_pts = [[TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y]]
    poly_top = np.array(outer_pts + inner_pts, dtype=np.float32)

    return project_topdown_points_to_image(points_like, poly_top)


# ------------------------------------------------------------
# Scoring / Hit-Berechnung
# ------------------------------------------------------------

def _angle_deg_from_topdown(x_px: float, y_px: float) -> float:
    """
    Winkel im Top-Down-Bild:
    - 0° = rechts
    - 90° = oben
    """
    return (math.degrees(math.atan2(TOPDOWN_CENTER_Y - y_px, x_px - TOPDOWN_CENTER_X)) + 360.0) % 360.0


def _segment_value_from_angle(angle_deg: float) -> int:
    """
    Bestimmt den Segmentwert anhand des Winkels.
    20 ist auf 90° zentriert.
    """
    relative = (90.0 - angle_deg) % 360.0
    sector_index = int(((relative + 9.0) % 360.0) // 18.0)
    return SEGMENT_ORDER[sector_index]


def calculate_hit_from_topdown_point(
    x_px: float,
    y_px: float,
) -> HitResult:
    """
    Berechnet den Treffer aus einem Top-Down-Punkt.
    """
    dx = x_px - TOPDOWN_CENTER_X
    dy_math = TOPDOWN_CENTER_Y - y_px  # y nach oben positiv
    radius_px = math.sqrt(dx * dx + dy_math * dy_math)
    radius = radius_px / OUTER_DOUBLE_RADIUS_PX
    angle_deg = _angle_deg_from_topdown(x_px, y_px)

    if radius > 1.02:
        return HitResult(
            image_x_px=int(round(x_px)),
            image_y_px=int(round(y_px)),
            topdown_x_px=x_px,
            topdown_y_px=y_px,
            board_x=dx / OUTER_DOUBLE_RADIUS_PX,
            board_y=dy_math / OUTER_DOUBLE_RADIUS_PX,
            radius=radius,
            angle_deg=angle_deg,
            segment_value=None,
            multiplier=0,
            ring_name="MISS",
            score=0,
            label="MISS",
        )

    if radius <= INNER_BULL_REL:
        ring_name = "DBULL"
        multiplier = 2
        score = 50
        label = "DBULL"
        segment_value = None
    elif radius <= OUTER_BULL_REL:
        ring_name = "SBULL"
        multiplier = 1
        score = 25
        label = "SBULL"
        segment_value = None
    else:
        segment_value = _segment_value_from_angle(angle_deg)

        if INNER_TRIPLE_REL <= radius <= OUTER_TRIPLE_REL:
            ring_name = "TRIPLE"
            multiplier = 3
        elif INNER_DOUBLE_REL <= radius <= 1.0:
            ring_name = "DOUBLE"
            multiplier = 2
        else:
            ring_name = "SINGLE"
            multiplier = 1

        score = int(segment_value * multiplier)
        prefix = {1: "S", 2: "D", 3: "T"}[multiplier]
        label = f"{prefix}{segment_value}"

    return HitResult(
        image_x_px=int(round(x_px)),
        image_y_px=int(round(y_px)),
        topdown_x_px=x_px,
        topdown_y_px=y_px,
        board_x=dx / OUTER_DOUBLE_RADIUS_PX,
        board_y=dy_math / OUTER_DOUBLE_RADIUS_PX,
        radius=radius,
        angle_deg=angle_deg,
        segment_value=segment_value,
        multiplier=multiplier,
        ring_name=ring_name,
        score=score,
        label=label,
    )


def calculate_hit_from_image_point(
    x_px: float,
    y_px: float,
    points_like: Sequence[Dict[str, int]] | Dict,
) -> Optional[HitResult]:
    """
    Berechnet den Treffer aus einem Bildpunkt.

    WICHTIG:
    Diese Funktion ist die einzige Wahrheit für Testpunkt-Scoring
    und sollte später auch vom Detector verwendet werden.
    """
    top_pt = image_to_topdown_point(points_like, x_px, y_px)
    if top_pt is None:
        return None

    hit = calculate_hit_from_topdown_point(top_pt[0], top_pt[1])

    return HitResult(
        image_x_px=int(round(x_px)),
        image_y_px=int(round(y_px)),
        topdown_x_px=hit.topdown_x_px,
        topdown_y_px=hit.topdown_y_px,
        board_x=hit.board_x,
        board_y=hit.board_y,
        radius=hit.radius,
        angle_deg=hit.angle_deg,
        segment_value=hit.segment_value,
        multiplier=hit.multiplier,
        ring_name=hit.ring_name,
        score=hit.score,
        label=hit.label,
    )


# ------------------------------------------------------------
# Kompatibilitäts-Wrapper für bestehenden Code
# ------------------------------------------------------------

def calculate_board_hit_from_image_point(
    x_px: float,
    y_px: float,
    calibration: Dict,
) -> Optional[HitResult]:
    """
    Kompatibilitätsname für bestehenden Code.
    """
    return calculate_hit_from_image_point(x_px, y_px, calibration)


def project_image_point_to_board(
    x_px: float,
    y_px: float,
    calibration: Dict,
) -> Optional[Tuple[float, float]]:
    """
    Kompatibilitätsfunktion:
    liefert normalisierte Board-Koordinaten zurück.
    """
    hit = calculate_hit_from_image_point(x_px, y_px, calibration)
    if hit is None:
        return None
    return hit.board_x, hit.board_y