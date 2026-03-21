# tools/run_single_cam_debug.py
# ------------------------------------------------------------
# Single-Cam-Debug für Triple One
#
# Ziele:
# - echtes Frame + Referenzbild laden
# - SingleCamDetector ausführen
# - finalen Impact-Punkt numerisch debuggen
# - score_geometry_overlay.png auf das reale Bild rendern
# - Segment-/Winkel-/Radius-Debug als TXT + JSON speichern
#
# Wichtige Kompatibilität:
# - unterstützt dein aktuelles Legacy-calibration.json-Format
# - unterstützt CameraCalibrationRecord mit manual_points
# - gibt rohe config.json NICHT mehr blind als dict an den Detector weiter
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np


# ------------------------------------------------------------
# Projektpfad für direkte Script-Ausführung
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Projekt-Imports
# ------------------------------------------------------------
from vision.calibration_geometry import (
    build_pipeline_points,
    calculate_hit_from_image_point,
    calculate_hit_from_topdown_point,
    project_image_points_to_topdown,
    project_topdown_points_to_image,
)
from vision.calibration_storage import CameraCalibrationRecord
from vision.score_mapper import ScoreMapper
from vision.single_cam_detector import SingleCamDetector

# Optionale Config-Klassen defensiv importieren
try:
    from vision.single_cam_detector import SingleCamDetectorConfig
except Exception:
    SingleCamDetectorConfig = None  # type: ignore

try:
    from vision.dart_candidate_detector import CandidateDetectorConfig
except Exception:
    CandidateDetectorConfig = None  # type: ignore

try:
    from vision.impact_estimator import ImpactEstimatorConfig
except Exception:
    ImpactEstimatorConfig = None  # type: ignore


# ------------------------------------------------------------
# Logging / IO
# ------------------------------------------------------------
def _log(message: str) -> None:
    print(message, flush=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_dump_safe(data), f, ensure_ascii=False, indent=2)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _load_image(path: Path, description: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"{description} konnte nicht geladen werden: {path}")
    return image


# ------------------------------------------------------------
# JSON-/Objekt-Serialisierung
# ------------------------------------------------------------
def _json_dump_safe(obj: Any) -> Any:
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (tuple, list)):
        return [_json_dump_safe(v) for v in obj]

    if isinstance(obj, dict):
        return {str(k): _json_dump_safe(v) for k, v in obj.items()}

    if is_dataclass(obj):
        return _json_dump_safe(asdict(obj))

    if hasattr(obj, "__dict__"):
        return _json_dump_safe(vars(obj))

    return repr(obj)


# ------------------------------------------------------------
# Punkt-Normalisierung
# ------------------------------------------------------------
def _to_point_tuple(value: Any) -> Optional[tuple[float, float]]:
    """
    Unterstützt u. a.:
    - [x, y]
    - (x, y)
    - {"x": ..., "y": ...}
    - {"x_px": ..., "y_px": ...}
    - Objekte mit .x/.y oder .x_px/.y_px
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except Exception:
                return None

        if "x_px" in value and "y_px" in value:
            try:
                return float(value["x_px"]), float(value["y_px"])
            except Exception:
                return None

        if "px" in value and isinstance(value["px"], dict):
            nested = value["px"]
            if "x" in nested and "y" in nested:
                try:
                    return float(nested["x"]), float(nested["y"])
                except Exception:
                    return None
            if "x_px" in nested and "y_px" in nested:
                try:
                    return float(nested["x_px"]), float(nested["y_px"])
                except Exception:
                    return None

    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return float(value.x), float(value.y)
        except Exception:
            return None

    if hasattr(value, "x_px") and hasattr(value, "y_px"):
        try:
            return float(value.x_px), float(value.y_px)
        except Exception:
            return None

    return None


def _to_point_array(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 2), dtype=np.float32)

    if isinstance(points, np.ndarray):
        arr = points.astype(np.float32).reshape(-1, 2)
        return arr

    normalized: list[tuple[float, float]] = []
    for item in points:
        point = _to_point_tuple(item)
        if point is not None:
            normalized.append(point)

    if not normalized:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(normalized, dtype=np.float32).reshape(-1, 2)


# ------------------------------------------------------------
# Zeichnen
# ------------------------------------------------------------
def _draw_cross(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    size: int = 10,
    thickness: int = 2,
) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.line(image, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(image, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _draw_circle_point(
    image: np.ndarray,
    point: tuple[float, float],
    color: tuple[int, int, int],
    radius: int = 5,
    thickness: int = -1,
) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)


def _draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
    closed: bool = False,
) -> None:
    if points.size == 0:
        return

    pts = np.round(points).astype(np.int32).reshape(-1, 1, 2)
    if len(pts) < 2:
        return

    cv2.polylines(image, [pts], closed, color, thickness, cv2.LINE_AA)


def _draw_label(
    image: np.ndarray,
    text: str,
    point: tuple[float, float],
    color: tuple[int, int, int],
    scale: float = 0.55,
    thickness: int = 1,
    with_bg: bool = True,
) -> None:
    x, y = int(round(point[0])), int(round(point[1]))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if with_bg:
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x1, y1 = x - 2, y - h - 4
        x2, y2 = x + w + 4, y + baseline + 2
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ------------------------------------------------------------
# Allgemeine Objekt-Helfer
# ------------------------------------------------------------
def _extract_attr(obj: Any, candidate_names: Iterable[str], default: Any = None) -> Any:
    for name in candidate_names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _flatten_named_geometry(container: Any) -> dict[str, Any]:
    if container is None:
        return {}

    if isinstance(container, dict):
        return dict(container)

    if is_dataclass(container):
        return asdict(container)

    if hasattr(container, "__dict__"):
        return dict(vars(container))

    return {}


# ------------------------------------------------------------
# Calibration / Record
# ------------------------------------------------------------
def _normalize_marker_list(markers: Any) -> list[tuple[float, float]]:
    """
    Normalisiert auf genau 4 Primärmarker.
    Legacy-5-Punkt-Format wird unterstützt:
    - Punkte 0..3 = Marker
    - Punkt 4 = alter Bull -> wird ignoriert
    """
    arr = _to_point_array(markers)

    if arr.shape[0] < 4:
        raise ValueError(
            "Kalibrierung enthält zu wenige Marker. Erwartet werden 4 Marker "
            "(20|1, 6|10, 3|19, 11|14) oder ein Legacy-Format mit 5 Punkten "
            "(4 Marker + Bull)."
        )

    return [(float(x), float(y)) for x, y in arr[:4]]


def _extract_record_markers(record: Any) -> list[tuple[float, float]]:
    if hasattr(record, "markers"):
        return _normalize_marker_list(getattr(record, "markers"))

    if hasattr(record, "manual_points"):
        return _normalize_marker_list(getattr(record, "manual_points"))

    if hasattr(record, "points"):
        return _normalize_marker_list(getattr(record, "points"))

    raise AttributeError(
        "CameraCalibrationRecord enthält weder 'markers' noch 'manual_points' noch 'points'."
    )


def _extract_camera_payload(raw: dict[str, Any], camera_index: int) -> dict[str, Any]:
    cameras = raw.get("cameras")
    if isinstance(cameras, list):
        if 0 <= camera_index < len(cameras):
            camera_payload = cameras[camera_index]
            if isinstance(camera_payload, dict):
                return camera_payload

    if isinstance(cameras, dict):
        for key in (str(camera_index), camera_index):
            if key in cameras and isinstance(cameras[key], dict):
                return cameras[key]

    for key in (f"camera_{camera_index}", f"cam_{camera_index}", str(camera_index)):
        value = raw.get(key)
        if isinstance(value, dict):
            return value

    return raw


def _make_calibration_record(camera_payload: dict[str, Any], camera_index: int) -> CameraCalibrationRecord:
    """
    Baut den Record exakt passend zu deinem lokalen Signature-Stand:
    (camera_index, name, enabled, device_id, width, height, fps, rotation, flip,
     overlay_alpha, show_numbers, show_sector_lines, manual_points,
     empty_board_reference_path='')
    """
    raw_points = (
        camera_payload.get("markers")
        or camera_payload.get("marker_points")
        or camera_payload.get("image_points")
        or camera_payload.get("points")
        or camera_payload.get("calibration_points")
        or []
    )

    marker_list = _normalize_marker_list(raw_points)

    manual_points = [
        {"x_px": int(round(x)), "y_px": int(round(y))}
        for x, y in marker_list
    ]

    name = str(camera_payload.get("name", f"Kamera {camera_index + 1}"))
    width = int(camera_payload.get("frame_width", camera_payload.get("width", 1280)))
    height = int(camera_payload.get("frame_height", camera_payload.get("height", 720)))

    overlay_alpha = float(camera_payload.get("overlay_alpha", 0.4))
    show_numbers = bool(camera_payload.get("show_numbers", True))
    show_sector_lines = bool(camera_payload.get("show_sector_lines", True))

    enabled = bool(camera_payload.get("enabled", True))
    device_id = int(camera_payload.get("device_id", camera_index))
    fps = int(camera_payload.get("fps", 30))
    rotation = int(camera_payload.get("rotation", 0))
    flip = bool(camera_payload.get("flip", False))
    empty_board_reference_path = str(camera_payload.get("empty_board_reference_path", ""))

    return CameraCalibrationRecord(
        camera_index=camera_index,
        name=name,
        enabled=enabled,
        device_id=device_id,
        width=width,
        height=height,
        fps=fps,
        rotation=rotation,
        flip=flip,
        overlay_alpha=overlay_alpha,
        show_numbers=show_numbers,
        show_sector_lines=show_sector_lines,
        manual_points=manual_points,
        empty_board_reference_path=empty_board_reference_path,
    )


def _load_calibration_record(calibration_path: Path, camera_index: int) -> CameraCalibrationRecord:
    raw = _read_json(calibration_path)
    camera_payload = _extract_camera_payload(raw, camera_index)
    return _make_calibration_record(camera_payload, camera_index)


# ------------------------------------------------------------
# Config-Objekte robust aus dict bauen
# ------------------------------------------------------------
def _coerce_config_object(config_cls: Any, data: Any) -> Any:
    """
    Wandelt dict -> Config-Objekt um, falls eine passende Config-Klasse existiert.
    Gibt None zurück, wenn nichts Sinnvolles gebaut werden kann.
    """
    if config_cls is None or data is None:
        return None

    if not isinstance(data, dict):
        return data

    try:
        signature = inspect.signature(config_cls)
        accepted = set(signature.parameters.keys())
        kwargs = {k: v for k, v in data.items() if k in accepted}
        return config_cls(**kwargs)
    except Exception:
        # Fallback: Default-Objekt erzeugen und bekannte Felder setzen
        try:
            obj = config_cls()
            for key, value in data.items():
                if hasattr(obj, key):
                    try:
                        setattr(obj, key, value)
                    except Exception:
                        pass
            return obj
        except Exception:
            return None


# ------------------------------------------------------------
# Pipeline / Geometrie
# ------------------------------------------------------------
def _call_build_pipeline_points(record: CameraCalibrationRecord, image_shape: tuple[int, int, int]) -> Any:
    h, w = image_shape[:2]
    sig = inspect.signature(build_pipeline_points)
    params = sig.parameters

    kwargs: dict[str, Any] = {}
    record_markers = _extract_record_markers(record)

    if "calibration_record" in params:
        kwargs["calibration_record"] = record
    if "record" in params:
        kwargs["record"] = record
    if "camera_calibration_record" in params:
        kwargs["camera_calibration_record"] = record

    if "markers" in params:
        kwargs["markers"] = record_markers
    if "image_points" in params:
        kwargs["image_points"] = record_markers
    if "manual_points" in params:
        kwargs["manual_points"] = record_markers
    if "points" in params:
        kwargs["points"] = record_markers

    if "image_size" in params:
        kwargs["image_size"] = (w, h)
    if "frame_size" in params:
        kwargs["frame_size"] = (w, h)
    if "width" in params:
        kwargs["width"] = w
    if "height" in params:
        kwargs["height"] = h

    try:
        return build_pipeline_points(**kwargs)
    except TypeError:
        try:
            return build_pipeline_points(record)
        except TypeError:
            return build_pipeline_points(record_markers)


def _call_project_topdown_points_to_image(points_topdown: Any, pipeline: Any) -> np.ndarray:
    arr = _to_point_array(points_topdown)
    if arr.size == 0:
        return arr

    try:
        result = project_topdown_points_to_image(arr, pipeline)
        return _to_point_array(result)
    except TypeError:
        result = project_topdown_points_to_image(points=arr, pipeline=pipeline)
        return _to_point_array(result)


def _call_project_image_points_to_topdown(points_image: Any, pipeline: Any) -> np.ndarray:
    arr = _to_point_array(points_image)
    if arr.size == 0:
        return arr

    try:
        result = project_image_points_to_topdown(arr, pipeline)
        return _to_point_array(result)
    except TypeError:
        result = project_image_points_to_topdown(points=arr, pipeline=pipeline)
        return _to_point_array(result)


def _call_calculate_hit_from_image_point(point_image: Any, pipeline: Any) -> Any:
    point = _to_point_tuple(point_image)
    if point is None:
        return None

    try:
        return calculate_hit_from_image_point(point, pipeline)
    except TypeError:
        try:
            return calculate_hit_from_image_point(point=point, pipeline=pipeline)
        except TypeError:
            return calculate_hit_from_image_point(point, points_like=pipeline)


def _call_calculate_hit_from_topdown_point(point_topdown: Any, pipeline: Any) -> Any:
    point = _to_point_tuple(point_topdown)
    if point is None:
        return None

    try:
        return calculate_hit_from_topdown_point(point, pipeline)
    except TypeError:
        try:
            return calculate_hit_from_topdown_point(point=point, pipeline=pipeline)
        except TypeError:
            return calculate_hit_from_topdown_point(point, points_like=pipeline)


# ------------------------------------------------------------
# Pipeline-Geometrie lesen
# ------------------------------------------------------------
def _extract_geometry_from_pipeline(pipeline: Any) -> dict[str, Any]:
    src = _flatten_named_geometry(pipeline)

    nested_geometry = _extract_attr(pipeline, ["geometry", "board_geometry", "overlay_geometry"], None)
    if nested_geometry is not None:
        nested_dict = _flatten_named_geometry(nested_geometry)
        for k, v in nested_dict.items():
            src.setdefault(k, v)

    ring_polylines = src.get("ring_polylines") or src.get("rings") or src.get("ring_lines") or {}
    segment_lines = src.get("segment_lines") or src.get("sector_lines") or src.get("radial_lines") or []
    number_positions = src.get("number_positions") or src.get("label_positions") or src.get("segment_label_positions") or {}
    bull_position = src.get("bull") or src.get("bull_center") or src.get("board_center") or None

    return {
        "ring_polylines": ring_polylines,
        "segment_lines": segment_lines,
        "number_positions": number_positions,
        "bull_position": bull_position,
    }


# ------------------------------------------------------------
# ScoreMapper / Detector
# ------------------------------------------------------------
def _create_score_mapper(pipeline: Any, record: CameraCalibrationRecord) -> ScoreMapper:
    sig = inspect.signature(ScoreMapper)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "pipeline" in params:
        kwargs["pipeline"] = pipeline
    if "points_like" in params:
        kwargs["points_like"] = pipeline
    if "calibration_record" in params:
        kwargs["calibration_record"] = record
    if "record" in params:
        kwargs["record"] = record

    try:
        return ScoreMapper(**kwargs)
    except TypeError:
        try:
            return ScoreMapper(pipeline)
        except TypeError:
            return ScoreMapper(points_like=pipeline)


def _set_impact_strategy_on_object(obj: Any, impact_strategy: str) -> None:
    if obj is None:
        return
    for attr_name in ("impact_strategy", "strategy", "default_strategy", "default_impact_strategy"):
        if hasattr(obj, attr_name):
            try:
                setattr(obj, attr_name, impact_strategy)
            except Exception:
                pass


def _set_impact_strategy_on_detector(detector: Any, impact_strategy: str) -> None:
    if detector is None:
        return

    _set_impact_strategy_on_object(detector, impact_strategy)

    for cfg_name in ("config", "_config", "impact_estimator_config", "_impact_estimator_config"):
        cfg = getattr(detector, cfg_name, None)
        _set_impact_strategy_on_object(cfg, impact_strategy)

    for est_name in ("impact_estimator", "_impact_estimator"):
        estimator = getattr(detector, est_name, None)
        _set_impact_strategy_on_object(estimator, impact_strategy)
        estimator_cfg = getattr(estimator, "config", None) or getattr(estimator, "_config", None)
        _set_impact_strategy_on_object(estimator_cfg, impact_strategy)


def _create_single_cam_detector(
    pipeline: Any,
    score_mapper: ScoreMapper,
    record: CameraCalibrationRecord,
    frame_shape: tuple[int, int, int],
    config_json: Optional[dict[str, Any]],
    impact_strategy: str,
) -> SingleCamDetector:
    h, w = frame_shape[:2]
    sig = inspect.signature(SingleCamDetector)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "score_mapper" in params:
        kwargs["score_mapper"] = score_mapper
    if "pipeline" in params:
        kwargs["pipeline"] = pipeline
    if "calibration_record" in params:
        kwargs["calibration_record"] = record
    if "image_size" in params:
        kwargs["image_size"] = (w, h)

    if "pipeline_kwargs" in params:
        kwargs["pipeline_kwargs"] = {}

    # WICHTIG:
    # config.json NICHT blind als dict an config= geben.
    # Erst versuchen, echte Config-Objekte zu bauen.
    if config_json is not None and isinstance(config_json, dict):
        detector_cfg_raw = (
            config_json.get("single_cam_detector_config")
            or config_json.get("single_cam_config")
            or config_json.get("detector_config")
            or config_json.get("config")
        )
        candidate_cfg_raw = config_json.get("candidate_detector_config")
        impact_cfg_raw = config_json.get("impact_estimator_config")

        detector_cfg_obj = _coerce_config_object(SingleCamDetectorConfig, detector_cfg_raw)
        candidate_cfg_obj = _coerce_config_object(CandidateDetectorConfig, candidate_cfg_raw)
        impact_cfg_obj = _coerce_config_object(ImpactEstimatorConfig, impact_cfg_raw)

        if "config" in params and detector_cfg_obj is not None:
            kwargs["config"] = detector_cfg_obj

        if "candidate_detector_config" in params and candidate_cfg_obj is not None:
            kwargs["candidate_detector_config"] = candidate_cfg_obj

        if "impact_estimator_config" in params and impact_cfg_obj is not None:
            kwargs["impact_estimator_config"] = impact_cfg_obj

    try:
        detector = SingleCamDetector(**kwargs)
    except TypeError:
        fallback_kwargs = {}
        if "score_mapper" in params:
            fallback_kwargs["score_mapper"] = score_mapper
        if "pipeline" in params:
            fallback_kwargs["pipeline"] = pipeline
        if "calibration_record" in params:
            fallback_kwargs["calibration_record"] = record
        detector = SingleCamDetector(**fallback_kwargs)

    _set_impact_strategy_on_detector(detector, impact_strategy)
    return detector


# ------------------------------------------------------------
# detect(...)
# ------------------------------------------------------------
def _call_detector_detect(detector: Any, frame: np.ndarray, reference: np.ndarray) -> Any:
    detect_fn = getattr(detector, "detect")
    sig = inspect.signature(detect_fn)
    params = sig.parameters

    kwargs: dict[str, Any] = {}

    if "frame" in params:
        kwargs["frame"] = frame
    elif "image" in params:
        kwargs["image"] = frame

    if "reference" in params:
        kwargs["reference"] = reference
    if "reference_frame" in params:
        kwargs["reference_frame"] = reference
    if "reference_image" in params:
        kwargs["reference_image"] = reference

    try:
        return detect_fn(**kwargs)
    except TypeError:
        try:
            return detect_fn(frame, reference)
        except TypeError:
            return detect_fn(frame)


# ------------------------------------------------------------
# Ergebnis extrahieren
# ------------------------------------------------------------
def _find_first_candidate(result: Any) -> Any:
    candidates = _extract_attr(result, ["candidates", "ranked_candidates", "candidate_list"], None)
    if candidates and len(candidates) > 0:
        return candidates[0]
    return None


def _extract_final_impact_point(result: Any) -> Optional[tuple[float, float]]:
    direct_candidates = [
        _extract_attr(result, ["final_point_image", "impact_point_image", "final_impact_point", "point_image"], None),
        _extract_attr(result, ["impact_point", "final_point"], None),
    ]

    for value in direct_candidates:
        point = _to_point_tuple(value)
        if point is not None:
            return point

    nested_objects = [
        _extract_attr(result, ["impact_result", "impact_estimation_result", "estimation_result"], None),
        _extract_attr(result, ["debug"], None),
        _find_first_candidate(result),
    ]

    for nested in nested_objects:
        if nested is None:
            continue

        for name in (
            "final_point_image",
            "impact_point_image",
            "final_impact_point",
            "final_point",
            "impact_point",
            "point_image",
            "point",
        ):
            point = _to_point_tuple(_extract_attr(nested, [name], None))
            if point is not None:
                return point

        debug_dict = _extract_attr(nested, ["debug"], None)
        if isinstance(debug_dict, dict):
            for name in ("final_point_image", "impact_point_image", "final_point", "impact_point"):
                point = _to_point_tuple(debug_dict.get(name))
                if point is not None:
                    return point

    return None


def _extract_scored_hit(result: Any) -> Any:
    for name in ("scored_hit", "best_scored_hit", "hit", "score", "final_hit"):
        value = _extract_attr(result, [name], None)
        if value is not None:
            return value

    first_candidate = _find_first_candidate(result)
    if first_candidate is not None:
        for name in ("scored_hit", "hit", "score"):
            value = _extract_attr(first_candidate, [name], None)
            if value is not None:
                return value

    return None


def _extract_hypotheses(result: Any) -> list[dict[str, Any]]:
    hypotheses_output: list[dict[str, Any]] = []

    search_spaces = [
        _extract_attr(result, ["impact_result", "impact_estimation_result", "estimation_result"], None),
        _find_first_candidate(result),
        result,
    ]

    for space in search_spaces:
        if space is None:
            continue

        hypotheses = _extract_attr(space, ["hypotheses", "point_hypotheses", "impact_hypotheses"], None)
        if not hypotheses:
            continue

        for hypothesis in hypotheses:
            name = _extract_attr(hypothesis, ["name"], "hypothesis")
            point = _extract_attr(hypothesis, ["point_image", "point", "value"], None)
            confidence = _extract_attr(hypothesis, ["confidence", "score", "weight"], None)

            pt = _to_point_tuple(point)
            if pt is None:
                continue

            hypotheses_output.append(
                {
                    "name": str(name),
                    "point_image": pt,
                    "confidence": confidence,
                }
            )

        if hypotheses_output:
            break

    return hypotheses_output


# ------------------------------------------------------------
# Numerischer Segment-Debug
# ------------------------------------------------------------
DARTBOARD_NUMBERS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]


def _compute_manual_segment_debug(topdown_point: tuple[float, float]) -> dict[str, Any]:
    cx, cy = 450.0, 450.0
    x, y = topdown_point

    dx = x - cx
    dy = cy - y
    radius = math.hypot(dx, dy)

    angle_deg = math.degrees(math.atan2(dx, dy))
    if angle_deg < 0:
        angle_deg += 360.0

    sector_index = int(((angle_deg + 9.0) % 360.0) // 18.0)
    sector_number = DARTBOARD_NUMBERS[sector_index]

    lower_boundary = (sector_index * 18.0 - 9.0) % 360.0
    upper_boundary = (sector_index * 18.0 + 9.0) % 360.0

    return {
        "topdown_point": [x, y],
        "radius_from_center": radius,
        "angle_deg_clockwise_from_top": angle_deg,
        "manual_sector_index": sector_index,
        "manual_sector_number": sector_number,
        "manual_sector_boundary_deg": {
            "lower": lower_boundary,
            "upper": upper_boundary,
        },
    }


def _scored_hit_to_debug_dict(scored_hit: Any) -> dict[str, Any]:
    if scored_hit is None:
        return {}

    if isinstance(scored_hit, dict):
        return dict(scored_hit)

    debug = {}
    for attr_name in (
        "label",
        "multiplier",
        "segment",
        "segment_number",
        "sector",
        "score",
        "value",
        "ring",
        "is_miss",
        "raw_score",
    ):
        if hasattr(scored_hit, attr_name):
            debug[attr_name] = getattr(scored_hit, attr_name)

    if not debug:
        debug = _json_dump_safe(scored_hit)

    return debug if isinstance(debug, dict) else {"value": debug}


def _hit_obj_to_dict(hit_obj: Any) -> dict[str, Any]:
    if hit_obj is None:
        return {}

    if isinstance(hit_obj, dict):
        return hit_obj

    result = {}
    for attr_name in (
        "label",
        "ring",
        "multiplier",
        "segment",
        "segment_number",
        "sector",
        "score",
        "value",
        "is_miss",
        "topdown_point",
        "image_point",
    ):
        if hasattr(hit_obj, attr_name):
            result[attr_name] = _json_dump_safe(getattr(hit_obj, attr_name))

    if not result:
        result = {"repr": repr(hit_obj)}

    return result


# ------------------------------------------------------------
# Overlay
# ------------------------------------------------------------
def _render_score_geometry_overlay(
    frame: np.ndarray,
    pipeline: Any,
    record: CameraCalibrationRecord,
    final_impact_point: Optional[tuple[float, float]],
    hypotheses: list[dict[str, Any]],
) -> np.ndarray:
    overlay = frame.copy()
    geometry = _extract_geometry_from_pipeline(pipeline)

    ring_polylines = geometry["ring_polylines"]
    segment_lines = geometry["segment_lines"]
    number_positions = geometry["number_positions"]
    bull_position = geometry["bull_position"]

    ring_color_default = (180, 180, 180)
    color_map = {
        "double_outer": (0, 0, 255),
        "double_inner": (0, 80, 255),
        "triple_outer": (0, 200, 0),
        "triple_inner": (0, 150, 0),
        "outer": (160, 160, 160),
        "inner": (160, 160, 160),
        "single_outer": (160, 160, 160),
        "single_inner": (160, 160, 160),
        "sbull": (255, 255, 0),
        "dbull": (255, 255, 0),
        "bull_outer": (255, 255, 0),
        "bull_inner": (0, 255, 255),
    }

    if isinstance(ring_polylines, dict):
        items = ring_polylines.items()
    else:
        items = enumerate(ring_polylines)

    for key, polyline_topdown in items:
        poly_td = _to_point_array(polyline_topdown)
        if poly_td.shape[0] < 2:
            continue

        poly_img = _call_project_topdown_points_to_image(poly_td, pipeline)
        color = color_map.get(str(key), ring_color_default)
        thickness = 2 if "double" in str(key) or "triple" in str(key) or "bull" in str(key) else 1
        _draw_polyline(overlay, poly_img, color=color, thickness=thickness, closed=True)

    for line_topdown in segment_lines:
        line_td = _to_point_array(line_topdown)
        if line_td.shape[0] < 2:
            continue

        line_img = _call_project_topdown_points_to_image(line_td, pipeline)
        _draw_polyline(overlay, line_img, color=(120, 120, 120), thickness=1, closed=False)

    if isinstance(number_positions, dict):
        label_items = number_positions.items()
    else:
        label_items = enumerate(number_positions)

    for key, pos_topdown in label_items:
        pos_td = _to_point_tuple(pos_topdown)
        if pos_td is None:
            continue

        pos_img_arr = _call_project_topdown_points_to_image([pos_td], pipeline)
        if pos_img_arr.shape[0] == 0:
            continue

        pos_img = tuple(float(v) for v in pos_img_arr[0])
        _draw_label(
            overlay,
            str(key),
            pos_img,
            color=(255, 255, 255),
            scale=0.45,
            thickness=1,
            with_bg=True,
        )

    bull_td = _to_point_tuple(bull_position)
    if bull_td is not None:
        bull_img_arr = _call_project_topdown_points_to_image([bull_td], pipeline)
        if bull_img_arr.shape[0] > 0:
            bull_img = tuple(float(v) for v in bull_img_arr[0])
            _draw_circle_point(overlay, bull_img, (0, 255, 255), radius=6, thickness=-1)
            _draw_cross(overlay, bull_img, (0, 255, 255), size=10, thickness=2)
            _draw_label(overlay, "Bull", bull_img, (0, 255, 255), scale=0.45)

    record_markers = _extract_record_markers(record)
    for idx, marker in enumerate(record_markers):
        pt = _to_point_tuple(marker)
        if pt is None:
            continue

        _draw_circle_point(overlay, pt, (255, 0, 255), radius=6, thickness=-1)
        _draw_label(
            overlay,
            f"P{idx + 1}",
            (pt[0] + 8, pt[1] - 8),
            (255, 0, 255),
            scale=0.45,
            thickness=1,
            with_bg=True,
        )

    hypothesis_palette = [
        (255, 120, 0),
        (0, 255, 120),
        (255, 0, 120),
        (120, 255, 0),
        (0, 120, 255),
        (120, 0, 255),
        (255, 255, 0),
    ]

    for idx, hypothesis in enumerate(hypotheses):
        point = _to_point_tuple(hypothesis.get("point_image"))
        if point is None:
            continue

        color = hypothesis_palette[idx % len(hypothesis_palette)]
        _draw_circle_point(overlay, point, color, radius=5, thickness=-1)

        name = str(hypothesis.get("name", f"h{idx}"))
        confidence = hypothesis.get("confidence")
        if confidence is None:
            text = name
        elif isinstance(confidence, (int, float)):
            text = f"{name} ({confidence:.3f})"
        else:
            text = f"{name} ({confidence})"

        _draw_label(
            overlay,
            text,
            (point[0] + 8, point[1] - 8),
            color,
            scale=0.42,
            thickness=1,
            with_bg=True,
        )

    if final_impact_point is not None:
        _draw_circle_point(overlay, final_impact_point, (0, 0, 255), radius=7, thickness=-1)
        _draw_cross(overlay, final_impact_point, (0, 0, 255), size=14, thickness=2)
        _draw_label(
            overlay,
            "FINAL IMPACT",
            (final_impact_point[0] + 10, final_impact_point[1] - 10),
            (0, 0, 255),
            scale=0.55,
            thickness=1,
            with_bg=True,
        )

    return overlay


# ------------------------------------------------------------
# Optional existierende Debugbilder aus Result speichern
# ------------------------------------------------------------
def _maybe_save_stage_images(output_dir: Path, result: Any) -> None:
    debug_spaces = [
        _extract_attr(result, ["debug"], None),
        _extract_attr(result, ["candidate_result"], None),
        _extract_attr(result, ["impact_result", "impact_estimation_result", "estimation_result"], None),
        _find_first_candidate(result),
    ]

    saved_count = 0
    for space in debug_spaces:
        if space is None:
            continue

        candidates = []
        if isinstance(space, dict):
            candidates.append(space)
        else:
            candidates.append(_flatten_named_geometry(space))
            nested_debug = _extract_attr(space, ["debug"], None)
            if isinstance(nested_debug, dict):
                candidates.append(nested_debug)

        for debug_dict in candidates:
            for key, value in debug_dict.items():
                if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                    filename = output_dir / f"{key}.png"
                    cv2.imwrite(str(filename), value)
                    saved_count += 1

    if saved_count > 0:
        _log(f"[INFO] {saved_count} Stage-/Debugbilder gespeichert.")


# ------------------------------------------------------------
# Hauptlauf
# ------------------------------------------------------------
def run_debug(args: argparse.Namespace) -> None:
    frame_path = Path(args.frame)
    reference_path = Path(args.reference)
    calibration_path = Path(args.calibration)
    output_dir = Path(args.output_dir)
    _ensure_dir(output_dir)

    config_json: Optional[dict[str, Any]] = None
    if args.config:
        config_json = _read_json(Path(args.config))

    _log("[INFO] Lade Bilder ...")
    frame = _load_image(frame_path, "Frame")
    reference = _load_image(reference_path, "Referenzbild")

    _log("[INFO] Lade Kalibrierung ...")
    record = _load_calibration_record(calibration_path, args.camera_index)

    _log("[INFO] Erzeuge zentrale Pipeline-Geometrie ...")
    pipeline = _call_build_pipeline_points(record, frame.shape)

    _log("[INFO] Erzeuge ScoreMapper ...")
    score_mapper = _create_score_mapper(pipeline, record)

    _log("[INFO] Erzeuge SingleCamDetector ...")
    detector = _create_single_cam_detector(
        pipeline=pipeline,
        score_mapper=score_mapper,
        record=record,
        frame_shape=frame.shape,
        config_json=config_json,
        impact_strategy=args.impact_strategy,
    )

    _log(f"[INFO] Starte detect(...) mit impact_strategy='{args.impact_strategy}' ...")
    result = _call_detector_detect(detector, frame, reference)

    final_impact_point = _extract_final_impact_point(result)
    scored_hit = _extract_scored_hit(result)
    hypotheses = _extract_hypotheses(result)

    topdown_point = None
    if final_impact_point is not None:
        td = _call_project_image_points_to_topdown([final_impact_point], pipeline)
        if td.shape[0] > 0:
            topdown_point = tuple(float(v) for v in td[0])

    image_hit = _call_calculate_hit_from_image_point(final_impact_point, pipeline) if final_impact_point else None
    topdown_hit = _call_calculate_hit_from_topdown_point(topdown_point, pipeline) if topdown_point else None
    manual_segment_debug = _compute_manual_segment_debug(topdown_point) if topdown_point else {}

    result_summary = {
        "input": {
            "frame": str(frame_path),
            "reference": str(reference_path),
            "calibration": str(calibration_path),
            "config": str(args.config) if args.config else None,
            "camera_index": args.camera_index,
            "impact_strategy": args.impact_strategy,
        },
        "final_impact_point_image": final_impact_point,
        "final_impact_point_topdown": topdown_point,
        "scored_hit_from_detector": _scored_hit_to_debug_dict(scored_hit),
        "recomputed_hit_from_image_point": _hit_obj_to_dict(image_hit),
        "recomputed_hit_from_topdown_point": _hit_obj_to_dict(topdown_hit),
        "manual_segment_debug": manual_segment_debug,
        "hypotheses": hypotheses,
        "raw_result": _json_dump_safe(result),
    }

    _log("[INFO] Rendere score_geometry_overlay.png ...")
    score_overlay = _render_score_geometry_overlay(
        frame=frame,
        pipeline=pipeline,
        record=record,
        final_impact_point=final_impact_point,
        hypotheses=hypotheses,
    )

    overlay_path = output_dir / "score_geometry_overlay.png"
    cv2.imwrite(str(overlay_path), score_overlay)

    impact_only = frame.copy()
    if final_impact_point is not None:
        _draw_circle_point(impact_only, final_impact_point, (0, 0, 255), radius=7, thickness=-1)
        _draw_cross(impact_only, final_impact_point, (0, 0, 255), size=14, thickness=2)
        _draw_label(
            impact_only,
            "FINAL IMPACT",
            (final_impact_point[0] + 10, final_impact_point[1] - 10),
            (0, 0, 255),
            scale=0.55,
            thickness=1,
            with_bg=True,
        )

    impact_only_path = output_dir / "impact_only_overlay.png"
    cv2.imwrite(str(impact_only_path), impact_only)

    if args.save_stage_images:
        _maybe_save_stage_images(output_dir, result)

    result_json_path = output_dir / "segment_debug.json"
    _write_json(result_json_path, result_summary)

    lines: list[str] = []
    lines.append("TRIPLE ONE - SINGLE CAM DEBUG")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Frame:             {frame_path}")
    lines.append(f"Reference:         {reference_path}")
    lines.append(f"Calibration:       {calibration_path}")
    lines.append(f"Camera index:      {args.camera_index}")
    lines.append(f"Impact strategy:   {args.impact_strategy}")
    lines.append("")
    lines.append("FINAL IMPACT")
    lines.append("-" * 60)
    lines.append(f"Image point:       {final_impact_point}")
    lines.append(f"Topdown point:     {topdown_point}")
    lines.append("")
    lines.append("DETECTOR HIT")
    lines.append("-" * 60)
    lines.append(json.dumps(_json_dump_safe(_scored_hit_to_debug_dict(scored_hit)), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("RECOMPUTED HIT FROM IMAGE POINT")
    lines.append("-" * 60)
    lines.append(json.dumps(_json_dump_safe(_hit_obj_to_dict(image_hit)), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("RECOMPUTED HIT FROM TOPDOWN POINT")
    lines.append("-" * 60)
    lines.append(json.dumps(_json_dump_safe(_hit_obj_to_dict(topdown_hit)), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("MANUAL SEGMENT DEBUG")
    lines.append("-" * 60)
    lines.append(json.dumps(_json_dump_safe(manual_segment_debug), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("HYPOTHESES")
    lines.append("-" * 60)
    lines.append(json.dumps(_json_dump_safe(hypotheses), ensure_ascii=False, indent=2))
    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 60)
    lines.append(f"score_geometry_overlay.png : {overlay_path}")
    lines.append(f"impact_only_overlay.png    : {impact_only_path}")
    lines.append(f"segment_debug.json         : {result_json_path}")

    result_txt_path = output_dir / "segment_debug.txt"
    _write_text(result_txt_path, "\n".join(lines))

    _log("")
    _log("[DONE] Debuglauf abgeschlossen.")
    _log(f"[DONE] Overlay: {overlay_path}")
    _log(f"[DONE] Text-Debug: {result_txt_path}")
    _log(f"[DONE] JSON-Debug: {result_json_path}")

    _log("")
    _log("----- Kurzdebug -----")
    _log(f"Final impact image point : {final_impact_point}")
    _log(f"Final impact topdown     : {topdown_point}")
    _log(f"Detector scored hit      : {_json_dump_safe(_scored_hit_to_debug_dict(scored_hit))}")
    _log(f"Manual segment debug     : {_json_dump_safe(manual_segment_debug)}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triple One - Single-Cam Debug mit Score-Geometrie-Overlay und numerischem Segment-Debug"
    )

    parser.add_argument("--frame", required=True, help="Pfad zum Frame mit Dart")
    parser.add_argument("--reference", required=True, help="Pfad zum Referenzbild ohne Dart")
    parser.add_argument("--calibration", required=True, help="Pfad zur calibration.json")
    parser.add_argument("--output-dir", required=True, help="Ausgabeordner für Debug-Bilder und Reports")

    parser.add_argument("--config", default=None, help="Optionaler Pfad zur config.json")
    parser.add_argument("--camera-index", type=int, default=0, help="Kameraindex in der Kalibrierung")

    parser.add_argument(
        "--impact-strategy",
        default="blend",
        choices=[
            "blend",
            "best_hypothesis",
            "candidate_default",
            "lowest_contour_point",
            "major_axis_lower_endpoint",
            "major_axis_centerward_endpoint",
            "centerward_contour_tip",
            "directional_contour_tip",
        ],
        help="Impact-Strategie für den Debuglauf",
    )

    parser.add_argument(
        "--save-stage-images",
        action="store_true",
        help="Speichert zusätzlich vorhandene Stage-/Debugbilder aus den Result-Objekten",
    )

    # Nur als Kompatibilitätsflags akzeptiert
    parser.add_argument("--auto-board-mask", action="store_true", help="Kompatibilitätsflag")
    parser.add_argument("--auto-board-mask-scale", type=float, default=0.90, help="Kompatibilitätsflag")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        run_debug(args)
        return 0
    except Exception as exc:
        print("")
        print("[ERROR] Der Debuglauf ist fehlgeschlagen.")
        print(f"[ERROR] {exc}")
        print("")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

