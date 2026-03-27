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
# WICHTIG:
# Diese Version ist bewusst auf den AKTUELLEN Repo-Stand angepasst:
# - calibration_geometry.py arbeitet aktuell mit legacy "points_like"
# - ScoreMapper kann sich aus calibration_record selbst korrekt aufbauen
# - score_mapper.pipeline ist dann das kompatible points_like-Format
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
    compute_bull_from_manual_points,
    generate_number_positions_image,
    generate_ring_polylines_image,
    generate_sector_lines_image,
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
        return points.astype(np.float32).reshape(-1, 2)

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


def _flatten_object(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return dict(vars(obj))
    return {}


# ------------------------------------------------------------
# Calibration / Record
# ------------------------------------------------------------
def _normalize_marker_list(markers: Any) -> list[tuple[float, float]]:
    arr = _to_point_array(markers)

    if arr.shape[0] < 4:
        raise ValueError(
            "Kalibrierung enthält zu wenige Marker. Erwartet werden 4 Marker "
            "(20|1, 6|10, 3|19, 11|14) oder ein Legacy-Format mit 5 Punkten "
            "(4 Marker + Bull)."
        )

    return [(float(x), float(y)) for x, y in arr[:4]]


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


def _extract_record_marker_dicts(record: Any) -> list[dict[str, int]]:
    manual_points = getattr(record, "manual_points", None)
    if not manual_points:
        raise ValueError("Calibration record enthält keine manual_points.")
    points = _normalize_marker_list(manual_points)
    return [{"x_px": int(round(x)), "y_px": int(round(y))} for x, y in points]


# ------------------------------------------------------------
# Config-Objekte robust aus dict bauen
# ------------------------------------------------------------
def _coerce_config_object(config_cls: Any, data: Any) -> Any:
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
# Geometry-Aufrufe exakt passend zum aktuellen Repo
# ------------------------------------------------------------
def _project_image_points(points_like: Any, image_points: Any) -> Optional[np.ndarray]:
    pts = _to_point_array(image_points)
    if pts.size == 0:
        return None
    return project_image_points_to_topdown(points_like, pts)


def _project_topdown_points(points_like: Any, topdown_points: Any) -> Optional[np.ndarray]:
    pts = _to_point_array(topdown_points)
    if pts.size == 0:
        return None
    return project_topdown_points_to_image(points_like, pts)


def _calculate_hit_from_image(point: Any, points_like: Any) -> Any:
    pt = _to_point_tuple(point)
    if pt is None:
        return None
    return calculate_hit_from_image_point(float(pt[0]), float(pt[1]), points_like)


def _calculate_hit_from_topdown(point: Any) -> Any:
    pt = _to_point_tuple(point)
    if pt is None:
        return None
    return calculate_hit_from_topdown_point(float(pt[0]), float(pt[1]))


# ------------------------------------------------------------
# ScoreMapper / Detector
# ------------------------------------------------------------
def _create_score_mapper(record: CameraCalibrationRecord) -> ScoreMapper:
    width = int(getattr(record, "width", 1280))
    height = int(getattr(record, "height", 720))
    return ScoreMapper(
        calibration_record=record,
        image_size=(width, height),
    )


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
    score_mapper: ScoreMapper,
    record: CameraCalibrationRecord,
    config_json: Optional[dict[str, Any]],
    impact_strategy: str,
) -> SingleCamDetector:
    sig = inspect.signature(SingleCamDetector)
    params = sig.parameters
    kwargs: dict[str, Any] = {}

    if "score_mapper" in params:
        kwargs["score_mapper"] = score_mapper
    if "calibration_record" in params:
        kwargs["calibration_record"] = record

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

    detector = SingleCamDetector(**kwargs)
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
    # Neuer Hauptpfad: scored_estimates[0].impact_estimate.impact_point
    scored_estimates = _extract_attr(result, ["scored_estimates"], None)
    if scored_estimates and len(scored_estimates) > 0:
        first = scored_estimates[0]

        impact_estimate = _extract_attr(first, ["impact_estimate"], None)
        if impact_estimate is not None:
            point = _to_point_tuple(_extract_attr(impact_estimate, ["impact_point"], None))
            if point is not None:
                return point

        point = _to_point_tuple(_extract_attr(first, ["image_point"], None))
        if point is not None:
            return point

    # Alte Fallbacks
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
    # Neuer Hauptpfad: scored_estimates[0].scored_hit
    scored_estimates = _extract_attr(result, ["scored_estimates"], None)
    if scored_estimates and len(scored_estimates) > 0:
        first = scored_estimates[0]
        scored_hit = _extract_attr(first, ["scored_hit"], None)
        if scored_hit is not None:
            return scored_hit

    # Alte Fallbacks
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


def _extract_scored_estimates(result: Any) -> list[dict[str, Any]]:
    """
    Holt die scored_estimates robust aus dem Result-Objekt und normalisiert sie
    für das Kandidaten-Overlay.

    Erwarteter Hauptpfad:
    result.scored_estimates oder result["scored_estimates"]
    """
    scored_estimates = _extract_attr(result, ["scored_estimates"], None)
    if not scored_estimates:
        return []

    normalized: list[dict[str, Any]] = []

    for item in scored_estimates:
        impact_estimate = _extract_attr(item, ["impact_estimate"], None) or {}
        scored_hit = _extract_attr(item, ["scored_hit"], None) or {}
        candidate = _extract_attr(impact_estimate, ["candidate"], None) or {}

        normalized.append(
            {
                "rank": _extract_attr(item, ["rank"], None),
                "candidate_id": _extract_attr(item, ["candidate_id"], _extract_attr(candidate, ["candidate_id"], None)),
                "image_point": _to_point_tuple(_extract_attr(item, ["image_point"], None)),
                "combined_confidence": _extract_attr(item, ["combined_confidence"], None),
                "candidate_confidence": _extract_attr(item, ["candidate_confidence"], _extract_attr(candidate, ["confidence"], None)),
                "impact_confidence": _extract_attr(item, ["impact_confidence"], _extract_attr(impact_estimate, ["confidence"], None)),
                "label": _extract_attr(scored_hit, ["label"], None),
                "score": _extract_attr(scored_hit, ["score"], None),
                "ring": _extract_attr(scored_hit, ["ring"], None),
                "bbox": _extract_attr(candidate, ["bbox"], _extract_attr(impact_estimate, ["bbox"], None)),
                "centroid": _to_point_tuple(_extract_attr(candidate, ["centroid"], _extract_attr(impact_estimate, ["centroid"], None))),
                "impact_point": _to_point_tuple(_extract_attr(impact_estimate, ["impact_point"], None)),
            }
        )

    return normalized

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
        "ring_name",
        "multiplier",
        "segment",
        "segment_value",
        "score",
        "radius",
        "angle_deg",
        "board_x",
        "board_y",
        "topdown_x_px",
        "topdown_y_px",
        "image_x_px",
        "image_y_px",
    ):
        if hasattr(hit_obj, attr_name):
            result[attr_name] = _json_dump_safe(getattr(hit_obj, attr_name))

    if not result:
        result = {"repr": repr(hit_obj)}

    return result


# ------------------------------------------------------------
# Overlay - direkt auf Basis der echten calibration_geometry-Helfer
# ------------------------------------------------------------
def _render_score_geometry_overlay(
    frame: np.ndarray,
    points_like: Any,
    record: CameraCalibrationRecord,
    final_impact_point: Optional[tuple[float, float]],
    hypotheses: list[dict[str, Any]],
) -> np.ndarray:
    overlay = frame.copy()

    ring_color_default = (180, 180, 180)

    ring_polylines = generate_ring_polylines_image(points_like)
    for idx, poly in enumerate(ring_polylines):
        poly_arr = _to_point_array(poly)
        if poly_arr.shape[0] < 2:
            continue

        # Indizes laut RING_RADII_REL:
        # 0 outer double, 1 inner double, 2 outer triple, 3 inner triple, 4 outer bull, 5 inner bull
        if idx in (0, 1):
            color = (0, 0, 255)
            thickness = 2
        elif idx in (2, 3):
            color = (0, 200, 0)
            thickness = 2
        elif idx in (4, 5):
            color = (0, 255, 255)
            thickness = 2
        else:
            color = ring_color_default
            thickness = 1

        _draw_polyline(overlay, poly_arr, color=color, thickness=thickness, closed=True)

    sector_lines = generate_sector_lines_image(points_like)
    for line in sector_lines:
        line_arr = _to_point_array(line)
        if line_arr.shape[0] >= 2:
            _draw_polyline(overlay, line_arr, color=(120, 120, 120), thickness=1, closed=False)

    number_positions = generate_number_positions_image(points_like)
    for label, pos in number_positions:
        pt = _to_point_tuple(pos)
        if pt is not None:
            _draw_label(
                overlay,
                str(label),
                pt,
                color=(255, 255, 255),
                scale=0.45,
                thickness=1,
                with_bg=True,
            )

    bull = compute_bull_from_manual_points(points_like)
    bull_pt = _to_point_tuple(bull)
    if bull_pt is not None:
        _draw_circle_point(overlay, bull_pt, (0, 255, 255), radius=6, thickness=-1)
        _draw_cross(overlay, bull_pt, (0, 255, 255), size=10, thickness=2)
        _draw_label(overlay, "Bull", bull_pt, (0, 255, 255), scale=0.45)

    marker_dicts = _extract_record_marker_dicts(record)
    for idx, marker in enumerate(marker_dicts):
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

def _render_candidate_ranking_overlay(
    frame: np.ndarray,
    scored_estimates: list[dict[str, Any]],
) -> np.ndarray:
    """
    Zeichnet alle scored_estimates direkt auf das Bild:
    - Bounding Box
    - Rank
    - candidate_id
    - Candidate-/Combined-Confidence
    - Centroid
    - Impact-Punkt
    - Label / Score

    Farbkonzept:
    - Rank 1 = rot
    - Rank 2 = orange
    - Rank 3 = gelb
    - weitere = grau
    """
    overlay = frame.copy()

    def color_for_rank(rank: Any) -> tuple[int, int, int]:
        if rank == 1:
            return (0, 0, 255)       # rot
        if rank == 2:
            return (0, 165, 255)     # orange
        if rank == 3:
            return (0, 255, 255)     # gelb
        return (180, 180, 180)       # grau

    for est in scored_estimates:
        rank = est.get("rank")
        candidate_id = est.get("candidate_id")
        bbox = est.get("bbox")
        centroid = est.get("centroid")
        impact_point = est.get("impact_point")
        label = est.get("label")
        score = est.get("score")
        ring = est.get("ring")
        candidate_conf = est.get("candidate_confidence")
        combined_conf = est.get("combined_confidence")

        color = color_for_rank(rank)

        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x, y, w, h = [int(round(float(v))) for v in bbox[:4]]
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)

                title = f"#{rank}  id={candidate_id}"
                _draw_label(
                    overlay,
                    title,
                    (x, max(18, y - 8)),
                    color,
                    scale=0.5,
                    thickness=1,
                    with_bg=True,
                )

                detail_parts = []
                if candidate_conf is not None:
                    try:
                        detail_parts.append(f"c={float(candidate_conf):.3f}")
                    except Exception:
                        detail_parts.append(f"c={candidate_conf}")
                if combined_conf is not None:
                    try:
                        detail_parts.append(f"comb={float(combined_conf):.3f}")
                    except Exception:
                        detail_parts.append(f"comb={combined_conf}")
                if label is not None:
                    detail_parts.append(str(label))
                if score is not None:
                    detail_parts.append(f"s={score}")
                if ring is not None:
                    detail_parts.append(str(ring))

                if detail_parts:
                    _draw_label(
                        overlay,
                        " | ".join(detail_parts),
                        (x, y + h + 18),
                        color,
                        scale=0.45,
                        thickness=1,
                        with_bg=True,
                    )
            except Exception:
                pass

        # ----------------------------------------------------
        # Centroid
        # ----------------------------------------------------
        if centroid is not None:
            _draw_circle_point(overlay, centroid, color, radius=4, thickness=-1)
            _draw_label(
                overlay,
                "centroid",
                (int(round(centroid[0])) + 8, int(round(centroid[1])) - 8),
                color,
                scale=0.4,
                thickness=1,
                with_bg=True,
            )

        # ----------------------------------------------------
        # Impact-Punkt
        # ----------------------------------------------------
        if impact_point is not None:
            _draw_circle_point(overlay, impact_point, (255, 255, 255), radius=5, thickness=-1)
            _draw_cross(overlay, impact_point, color, size=10, thickness=2)
            _draw_label(
                overlay,
                f"impact #{rank}",
                (int(round(impact_point[0])) + 8, int(round(impact_point[1])) - 8),
                color,
                scale=0.42,
                thickness=1,
                with_bg=True,
            )

    return overlay


# ------------------------------------------------------------
# Optional vorhandene Debugbilder aus Result speichern
# ------------------------------------------------------------
def _maybe_save_stage_images(output_dir: Path, result: Any) -> None:
    """
    Speichert alle verfügbaren Stage-/Debugbilder robust aus dem Result.

    Wichtig:
    - bevorzugt echte debug_images-Felder aus candidate_result / impact_result
    - fällt zusätzlich auf dicts / __dict__ zurück
    """
    saved_count = 0
    saved_names: set[str] = set()

    def save_debug_dict(prefix: str, debug_dict: Any) -> None:
        nonlocal saved_count

        if not isinstance(debug_dict, dict):
            return

        for key, value in debug_dict.items():
            if value is None:
                continue

            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                filename_key = f"{prefix}{key}" if prefix else key
                if filename_key in saved_names:
                    continue

                filename = output_dir / f"{filename_key}.png"
                cv2.imwrite(str(filename), value)
                saved_names.add(filename_key)
                saved_count += 1

    # ----------------------------------------------------------
    # 1) Direkt vom Hauptresult
    # ----------------------------------------------------------
    result_debug = _extract_attr(result, ["debug_images"], None)
    save_debug_dict("", result_debug)

    # ----------------------------------------------------------
    # 2) candidate_result.debug_images
    # ----------------------------------------------------------
    candidate_result = _extract_attr(result, ["candidate_result"], None)
    if candidate_result is not None:
        save_debug_dict("", _extract_attr(candidate_result, ["debug_images"], None))

    # ----------------------------------------------------------
    # 3) impact_result.debug_images
    # ----------------------------------------------------------
    impact_result = _extract_attr(result, ["impact_result", "impact_estimation_result", "estimation_result"], None)
    if impact_result is not None:
        save_debug_dict("", _extract_attr(impact_result, ["debug_images"], None))

    # ----------------------------------------------------------
    # 4) scored_estimates / nested debug_images durchsuchen
    # ----------------------------------------------------------
    scored_estimates = _extract_attr(result, ["scored_estimates"], None)
    if isinstance(scored_estimates, list):
        for idx, estimate in enumerate(scored_estimates):
            estimate_debug = _extract_attr(estimate, ["debug"], None)
            if isinstance(estimate_debug, dict):
                for key, value in estimate_debug.items():
                    if isinstance(value, dict):
                        save_debug_dict(f"estimate_{idx}_", value)

    # ----------------------------------------------------------
    # 5) Fallback: generische Suche in bekannten Spaces
    # ----------------------------------------------------------
    debug_spaces = [
        _extract_attr(result, ["debug"], None),
        candidate_result,
        impact_result,
        _find_first_candidate(result),
    ]

    for space in debug_spaces:
        if space is None:
            continue

        if isinstance(space, dict):
            save_debug_dict("", space)
            continue

        # echtes Objekt -> debug_images + debug prüfen
        save_debug_dict("", _extract_attr(space, ["debug_images"], None))
        save_debug_dict("", _extract_attr(space, ["debug"], None))

        # __dict__ fallback
        raw = _flatten_object(space)
        for key, value in raw.items():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                if key in saved_names:
                    continue
                filename = output_dir / f"{key}.png"
                cv2.imwrite(str(filename), value)
                saved_names.add(key)
                saved_count += 1

    if saved_count > 0:
        _log(f"[INFO] {saved_count} Stage-/Debugbilder gespeichert.")
    else:
        _log("[INFO] Keine zusätzlichen Stage-/Debugbilder gefunden.")


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

    _log("[INFO] Erzeuge ScoreMapper direkt aus calibration_record ...")
    score_mapper = _create_score_mapper(record)

    # DAS ist jetzt der echte kompatible points_like-Kontext
    points_like = score_mapper.pipeline

    # Optionaler früher Check
    if not points_like:
        raise ValueError("ScoreMapper.pipeline ist leer oder ungültig.")

    _log("[INFO] Erzeuge SingleCamDetector ...")
    detector = _create_single_cam_detector(
        score_mapper=score_mapper,
        record=record,
        config_json=config_json,
        impact_strategy=args.impact_strategy,
    )

    _log(f"[INFO] Starte detect(...) mit impact_strategy='{args.impact_strategy}' ...")
    result = _call_detector_detect(detector, frame, reference)

    final_impact_point = _extract_final_impact_point(result)
    scored_hit = _extract_scored_hit(result)
    hypotheses = _extract_hypotheses(result)
    scored_estimates = _extract_scored_estimates(result)

    topdown_point = None
    if final_impact_point is not None:
        td = _project_image_points(points_like, [final_impact_point])
        if td is not None and len(td) > 0:
            topdown_point = tuple(float(v) for v in td[0])

    image_hit = _calculate_hit_from_image(final_impact_point, points_like) if final_impact_point else None
    topdown_hit = _calculate_hit_from_topdown(topdown_point) if topdown_point else None
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
        "points_like_pipeline": _json_dump_safe(points_like),
        "final_impact_point_image": final_impact_point,
        "final_impact_point_topdown": topdown_point,
        "scored_hit_from_detector": _scored_hit_to_debug_dict(scored_hit),
        "scored_estimates_compact": scored_estimates,
        "recomputed_hit_from_image_point": _hit_obj_to_dict(image_hit),
        "recomputed_hit_from_topdown_point": _hit_obj_to_dict(topdown_hit),
        "manual_segment_debug": manual_segment_debug,
        "hypotheses": hypotheses,
        "raw_result": _json_dump_safe(result),
    }

    _log("[INFO] Rendere score_geometry_overlay.png ...")
    score_overlay = _render_score_geometry_overlay(
        frame=frame,
        points_like=points_like,
        record=record,
        final_impact_point=final_impact_point,
        hypotheses=hypotheses,
    )

    overlay_path = output_dir / "score_geometry_overlay.png"
    cv2.imwrite(str(overlay_path), score_overlay)

    _log("[INFO] Rendere candidate_ranking_overlay.png ...")
    candidate_overlay = _render_candidate_ranking_overlay(
        frame=frame,
        scored_estimates=scored_estimates,
    )

    candidate_overlay_path = output_dir / "candidate_ranking_overlay.png"
    cv2.imwrite(str(candidate_overlay_path), candidate_overlay)

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
    lines.append(f"score_geometry_overlay.png    : {overlay_path}")
    lines.append(f"candidate_ranking_overlay.png : {candidate_overlay_path}")
    lines.append(f"impact_only_overlay.png       : {impact_only_path}")
    lines.append(f"segment_debug.json            : {result_json_path}")

    result_txt_path = output_dir / "segment_debug.txt"
    _write_text(result_txt_path, "\n".join(lines))

    _log("")
    _log("[DONE] Debuglauf abgeschlossen.")
    _log(f"[DONE] Overlay: {overlay_path}")
    _log(f"[DONE] Candidate Overlay: {candidate_overlay_path}")
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