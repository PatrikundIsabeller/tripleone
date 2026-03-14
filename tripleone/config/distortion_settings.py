# config/distortion_settings.py
# Diese Datei verwaltet das Laden und Speichern der Distortion-/Linsenkorrektur-
# Einstellungen pro Kamera.
#
# Phase 5.3:
# - pro Kamera:
#   - enabled
#   - camera_matrix
#   - dist_coeffs
#   - image_width / image_height
#   - reprojection_error
#
# Diese Datei wird von vision/distortion.py, tools/calibrate_distortion.py
# und der CalibrationPage verwendet.

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
DISTORTION_FILE = CONFIG_DIR / "distortion.json"


DEFAULT_DISTORTION: Dict[str, Any] = {
    "cameras": [
        {
            "name": "Kamera 1",
            "enabled": False,
            "camera_matrix": None,
            "dist_coeffs": None,
            "image_width": None,
            "image_height": None,
            "reprojection_error": None,
            "source_count": 0,
        },
        {
            "name": "Kamera 2",
            "enabled": False,
            "camera_matrix": None,
            "dist_coeffs": None,
            "image_width": None,
            "image_height": None,
            "reprojection_error": None,
            "source_count": 0,
        },
        {
            "name": "Kamera 3",
            "enabled": False,
            "camera_matrix": None,
            "dist_coeffs": None,
            "image_width": None,
            "image_height": None,
            "reprojection_error": None,
            "source_count": 0,
        },
    ]
}


def ensure_distortion_exists() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not DISTORTION_FILE.exists():
        with DISTORTION_FILE.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_DISTORTION, f, indent=4, ensure_ascii=False)


def _sanitize_float_matrix(value: Any, rows: int, cols: int) -> Optional[List[List[float]]]:
    if not isinstance(value, list) or len(value) != rows:
        return None

    result: List[List[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            return None
        clean_row: List[float] = []
        for cell in row:
            try:
                clean_row.append(float(cell))
            except (TypeError, ValueError):
                return None
        result.append(clean_row)
    return result


def _sanitize_float_vector(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) == 0:
        return None

    result: List[float] = []
    for cell in value:
        try:
            result.append(float(cell))
        except (TypeError, ValueError):
            return None
    return result


def _sanitize_camera_entry(entry: Dict[str, Any], index: int) -> Dict[str, Any]:
    default = deepcopy(DEFAULT_DISTORTION["cameras"][index])
    result = deepcopy(default)

    result["name"] = str(entry.get("name", default["name"]))
    result["enabled"] = bool(entry.get("enabled", default["enabled"]))

    camera_matrix = _sanitize_float_matrix(entry.get("camera_matrix"), 3, 3)
    dist_coeffs = _sanitize_float_vector(entry.get("dist_coeffs"))

    result["camera_matrix"] = camera_matrix
    result["dist_coeffs"] = dist_coeffs

    image_width = entry.get("image_width")
    image_height = entry.get("image_height")
    reprojection_error = entry.get("reprojection_error")
    source_count = entry.get("source_count", 0)

    result["image_width"] = int(image_width) if image_width is not None else None
    result["image_height"] = int(image_height) if image_height is not None else None
    result["reprojection_error"] = float(reprojection_error) if reprojection_error is not None else None
    result["source_count"] = int(source_count)

    if result["camera_matrix"] is None or result["dist_coeffs"] is None:
        result["enabled"] = False

    return result


def _sanitize_distortion(data: Dict[str, Any]) -> Dict[str, Any]:
    clean = deepcopy(DEFAULT_DISTORTION)
    cameras = data.get("cameras", [])

    if not isinstance(cameras, list):
        cameras = []

    sanitized: List[Dict[str, Any]] = []
    for i in range(3):
        if i < len(cameras) and isinstance(cameras[i], dict):
            sanitized.append(_sanitize_camera_entry(cameras[i], i))
        else:
            sanitized.append(deepcopy(DEFAULT_DISTORTION["cameras"][i]))

    clean["cameras"] = sanitized
    return clean


def load_distortion() -> Dict[str, Any]:
    ensure_distortion_exists()

    try:
        with DISTORTION_FILE.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        raw = deepcopy(DEFAULT_DISTORTION)

    return _sanitize_distortion(raw)


def save_distortion(data: Dict[str, Any]) -> None:
    ensure_distortion_exists()
    clean = _sanitize_distortion(data)

    with DISTORTION_FILE.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=4, ensure_ascii=False)
