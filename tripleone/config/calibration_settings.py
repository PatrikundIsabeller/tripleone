# config/calibration_settings.py
# Diese Datei verwaltet das Laden und Speichern der Kalibrierungsdaten
# für das Homography-Overlay pro Kamera.
#
# Phase 3.4b:
# - 4 Boundary-Punkte
# - 1 Center-Punkt
#   P1 = Grenze 20|1
#   P2 = Grenze 6|10
#   P3 = Grenze 3|19
#   P4 = Grenze 11|14
#   C  = Bull-Mittelpunkt

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CALIBRATION_FILE = CONFIG_DIR / "calibration.json"

DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

# Feste Boundary-Winkel auf dem äußeren Double-Ring
BOUNDARY_ANGLES_DEG = [-81.0, 27.0, 117.0, 207.0]


def _default_points(frame_width: int, frame_height: int) -> List[Dict[str, int]]:
    """
    Liefert 5 Standardpunkte:
    P1 = 20|1
    P2 = 6|10
    P3 = 3|19
    P4 = 11|14
    C  = Bull-Mittelpunkt
    """
    cx = frame_width // 2
    cy = frame_height // 2
    r = int(min(frame_width, frame_height) * 0.30)

    points: List[Dict[str, int]] = []
    for angle_deg in BOUNDARY_ANGLES_DEG:
        angle_rad = math.radians(angle_deg)
        x = int(round(cx + r * math.cos(angle_rad)))
        y = int(round(cy + r * math.sin(angle_rad)))
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        points.append({"x_px": x, "y_px": y})

    points.append({"x_px": cx, "y_px": cy})
    return points


DEFAULT_CALIBRATION: Dict[str, Any] = {
    "cameras": [
        {
            "name": "Kamera 1",
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "overlay_alpha": 0.90,
            "show_numbers": True,
            "show_sector_lines": True,
            "points": _default_points(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        },
        {
            "name": "Kamera 2",
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "overlay_alpha": 0.90,
            "show_numbers": True,
            "show_sector_lines": True,
            "points": _default_points(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        },
        {
            "name": "Kamera 3",
            "frame_width": DEFAULT_FRAME_WIDTH,
            "frame_height": DEFAULT_FRAME_HEIGHT,
            "overlay_alpha": 0.90,
            "show_numbers": True,
            "show_sector_lines": True,
            "points": _default_points(DEFAULT_FRAME_WIDTH, DEFAULT_FRAME_HEIGHT),
        },
    ]
}


def ensure_calibration_exists() -> None:
    """Erstellt die Kalibrierungsdatei mit Standardwerten, falls sie fehlt."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CALIBRATION_FILE.exists():
        with CALIBRATION_FILE.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_CALIBRATION, f, indent=4, ensure_ascii=False)


def _sanitize_point(
    point: Dict[str, Any],
    frame_width: int,
    frame_height: int,
    fallback: Dict[str, int],
) -> Dict[str, int]:
    x_px = int(point.get("x_px", fallback["x_px"]))
    y_px = int(point.get("y_px", fallback["y_px"]))

    x_px = max(0, min(x_px, frame_width - 1))
    y_px = max(0, min(y_px, frame_height - 1))

    return {"x_px": x_px, "y_px": y_px}


def _migrate_old_structure(camera: Dict[str, Any], frame_width: int, frame_height: int) -> List[Dict[str, int]]:
    """
    Unterstützt alte Strukturen:
    - 5 Punkte -> direkt
    - 4 Punkte -> Center ergänzen
    - 8 Punkte -> auf 4 Boundary-Punkte reduzieren + Center ergänzen
    """
    raw_points = camera.get("points", [])
    if not isinstance(raw_points, list):
        return _default_points(frame_width, frame_height)

    if len(raw_points) == 5:
        return raw_points

    if len(raw_points) == 4:
        points_4: List[Dict[str, int]] = []
        for item in raw_points:
            if isinstance(item, dict):
                points_4.append(
                    {
                        "x_px": int(item.get("x_px", frame_width // 2)),
                        "y_px": int(item.get("y_px", frame_height // 2)),
                    }
                )
        if len(points_4) == 4:
            center_x = int(round(sum(p["x_px"] for p in points_4) / 4.0))
            center_y = int(round(sum(p["y_px"] for p in points_4) / 4.0))
            return points_4 + [{"x_px": center_x, "y_px": center_y}]

    if len(raw_points) == 8:
        def get_point(idx: int) -> Dict[str, int]:
            item = raw_points[idx]
            return {
                "x_px": int(item.get("x_px", frame_width // 2)),
                "y_px": int(item.get("y_px", frame_height // 2)),
            }

        # 8-Punkt-Struktur -> mittlere Boundary-Stützpunkte
        p1 = {
            "x_px": int(round((get_point(0)["x_px"] + get_point(1)["x_px"]) / 2.0)),
            "y_px": int(round((get_point(0)["y_px"] + get_point(1)["y_px"]) / 2.0)),
        }
        p2 = {
            "x_px": int(round((get_point(2)["x_px"] + get_point(3)["x_px"]) / 2.0)),
            "y_px": int(round((get_point(2)["y_px"] + get_point(3)["y_px"]) / 2.0)),
        }
        p3 = {
            "x_px": int(round((get_point(4)["x_px"] + get_point(5)["x_px"]) / 2.0)),
            "y_px": int(round((get_point(4)["y_px"] + get_point(5)["y_px"]) / 2.0)),
        }
        p4 = {
            "x_px": int(round((get_point(6)["x_px"] + get_point(7)["x_px"]) / 2.0)),
            "y_px": int(round((get_point(6)["y_px"] + get_point(7)["y_px"]) / 2.0)),
        }
        center_x = int(round((p1["x_px"] + p2["x_px"] + p3["x_px"] + p4["x_px"]) / 4.0))
        center_y = int(round((p1["y_px"] + p2["y_px"] + p3["y_px"] + p4["y_px"]) / 4.0))
        c = {"x_px": center_x, "y_px": center_y}
        return [p1, p2, p3, p4, c]

    return _default_points(frame_width, frame_height)


def _sanitize_camera_calibration(camera: Dict[str, Any], index: int) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_CALIBRATION["cameras"][index])
    result = deepcopy(default)

    frame_width = int(camera.get("frame_width", default["frame_width"]))
    frame_height = int(camera.get("frame_height", default["frame_height"]))

    frame_width = max(320, min(frame_width, 7680))
    frame_height = max(240, min(frame_height, 4320))

    result["name"] = str(camera.get("name", default["name"]))
    result["frame_width"] = frame_width
    result["frame_height"] = frame_height
    result["overlay_alpha"] = max(0.10, min(float(camera.get("overlay_alpha", default["overlay_alpha"])), 1.00))
    result["show_numbers"] = bool(camera.get("show_numbers", default["show_numbers"]))
    result["show_sector_lines"] = bool(camera.get("show_sector_lines", default["show_sector_lines"]))

    default_points = _default_points(frame_width, frame_height)
    migrated_points = _migrate_old_structure(camera, frame_width, frame_height)

    sanitized_points: List[Dict[str, int]] = []
    for i in range(5):
        raw = migrated_points[i] if i < len(migrated_points) and isinstance(migrated_points[i], dict) else {}
        sanitized_points.append(_sanitize_point(raw, frame_width, frame_height, default_points[i]))

    result["points"] = sanitized_points
    return result


def _sanitize_calibration(data: Dict[str, Any]) -> Dict[str, Any]:
    clean = deepcopy(DEFAULT_CALIBRATION)
    cameras = data.get("cameras", [])

    if not isinstance(cameras, list):
        cameras = []

    sanitized: List[Dict[str, Any]] = []
    for i in range(3):
        if i < len(cameras) and isinstance(cameras[i], dict):
            sanitized.append(_sanitize_camera_calibration(cameras[i], i))
        else:
            sanitized.append(deepcopy(DEFAULT_CALIBRATION["cameras"][i]))

    clean["cameras"] = sanitized
    return clean


def load_calibration() -> Dict[str, Any]:
    ensure_calibration_exists()

    try:
        with CALIBRATION_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        raw = deepcopy(DEFAULT_CALIBRATION)

    return _sanitize_calibration(raw)


def save_calibration(calibration_data: Dict[str, Any]) -> None:
    ensure_calibration_exists()
    clean = _sanitize_calibration(calibration_data)

    with CALIBRATION_FILE.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=4, ensure_ascii=False)