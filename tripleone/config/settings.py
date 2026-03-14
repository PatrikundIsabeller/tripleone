# config/settings.py
# Diese Datei verwaltet das Laden, Validieren und Speichern
# der lokalen JSON-Konfiguration für die App und die 3 Kameras.

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"

VALID_ROTATIONS = [0, 90, 180, 270]


DEFAULT_CONFIG: Dict[str, Any] = {
    "app": {
        "title": "TripleOne - Phase 1",
        "theme": "dark",
        "auto_start_cameras": True,
        "max_camera_scan": 10
    },
    "cameras": [
        {
            "name": "Kamera 1",
            "device_id": -1,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "rotation": 0,
            "flip": False,
            "enabled": True
        },
        {
            "name": "Kamera 2",
            "device_id": -1,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "rotation": 0,
            "flip": False,
            "enabled": True
        },
        {
            "name": "Kamera 3",
            "device_id": -1,
            "width": 1280,
            "height": 720,
            "fps": 30,
            "rotation": 0,
            "flip": False,
            "enabled": True
        }
    ]
}


def ensure_config_exists() -> None:
    """Erstellt die Konfigurationsdatei mit Standardwerten, falls sie fehlt."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        with CONFIG_FILE.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)


def _merge_defaults(user_config: Dict[str, Any], default_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Führt Benutzerkonfiguration und Standardkonfiguration rekursiv zusammen.
    Fehlende Schlüssel werden automatisch ergänzt.
    """
    result = deepcopy(default_config)

    for key, value in user_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_defaults(value, result[key])
        else:
            result[key] = value

    return result


def _sanitize_camera(camera: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Bereinigt einen Kamera-Eintrag und stellt sicher,
    dass alle Werte gültig und im erlaubten Bereich sind.
    """
    result = deepcopy(DEFAULT_CONFIG["cameras"][index])

    result["name"] = str(camera.get("name", result["name"]))
    result["device_id"] = int(camera.get("device_id", -1))

    width = int(camera.get("width", result["width"]))
    height = int(camera.get("height", result["height"]))
    fps = int(camera.get("fps", result["fps"]))
    rotation = int(camera.get("rotation", result["rotation"]))
    flip = bool(camera.get("flip", result["flip"]))
    enabled = bool(camera.get("enabled", result["enabled"]))

    result["width"] = max(320, min(width, 3840))
    result["height"] = max(240, min(height, 2160))
    result["fps"] = max(1, min(fps, 120))
    result["rotation"] = rotation if rotation in VALID_ROTATIONS else 0
    result["flip"] = flip
    result["enabled"] = enabled

    return result


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bereinigt die Gesamtkonfiguration und stellt sicher,
    dass genau 3 Kamera-Einträge vorhanden sind.
    """
    clean = deepcopy(DEFAULT_CONFIG)
    clean = _merge_defaults(config, clean)

    app_cfg = clean.get("app", {})
    app_cfg["title"] = str(app_cfg.get("title", DEFAULT_CONFIG["app"]["title"]))
    app_cfg["theme"] = str(app_cfg.get("theme", "dark"))
    app_cfg["auto_start_cameras"] = bool(app_cfg.get("auto_start_cameras", True))
    app_cfg["max_camera_scan"] = max(1, min(int(app_cfg.get("max_camera_scan", 10)), 20))
    clean["app"] = app_cfg

    cameras = clean.get("cameras", [])
    if not isinstance(cameras, list):
        cameras = []

    sanitized_cameras: List[Dict[str, Any]] = []
    for i in range(3):
        if i < len(cameras) and isinstance(cameras[i], dict):
            sanitized_cameras.append(_sanitize_camera(cameras[i], i))
        else:
            sanitized_cameras.append(deepcopy(DEFAULT_CONFIG["cameras"][i]))

    clean["cameras"] = sanitized_cameras
    return clean


def load_config() -> Dict[str, Any]:
    """Lädt die Konfiguration aus der JSON-Datei und bereinigt sie."""
    ensure_config_exists()

    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as f:
            user_config = json.load(f)
    except (json.JSONDecodeError, OSError):
        user_config = deepcopy(DEFAULT_CONFIG)

    config = _sanitize_config(user_config)
    return config


def save_config(config_data: Dict[str, Any]) -> None:
    """Speichert die bereinigte Konfiguration dauerhaft in die JSON-Datei."""
    ensure_config_exists()
    clean = _sanitize_config(config_data)

    with CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(clean, f, indent=4, ensure_ascii=False)