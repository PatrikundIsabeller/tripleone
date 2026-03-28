# tools/run_single_cam_test_series.py
# Zweck:
# Führt eine Serie von Single-Cam-Tests gegen echte Testbilder aus
# und schreibt die Ergebnisse als CSV + JSON.
#
# Dateinamen-Schema:
#   s15_01.png
#   t20_01.png
#   d8_01.png
#   sbull_01.png
#   dbull_01.png
#   miss_01.png
#
# Erwartete Labels:
#   S<number>, T<number>, D<number>, SBULL, DBULL, MISS
#
# Beispiel:
# python tools/run_single_cam_test_series.py ^
#   --images-root "D:\tripleone\tripleone\data\test_series\camera_1" ^
#   --reference "D:\tripleone\tripleone\data\test_series\camera_1\reference\empty_board.png" ^
#   --calibration "D:\tripleone\tripleone\config\calibration.json" ^
#   --config "D:\tripleone\tripleone\config\config.json" ^
#   --camera-index 0 ^
#   --output-dir "D:\tripleone\tripleone\data\test_series\results\camera_1" ^
#   --save-overlays ^
#   --auto-board-mask ^
#   --auto-board-mask-scale 0.90 ^
#   --impact-strategy blend

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Imports aus dem Projekt
# -----------------------------------------------------------------------------
from pathlib import Path

# Projekt-Root robust in sys.path einhängen:
# tools/run_single_cam_test_series.py -> Projektroot ist das Parent von "tools"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from vision.single_cam_detector import (
        SingleCamDetector,
        SingleCamDetectorConfig,
    )
    from vision.score_mapper import build_score_mapper
except ImportError as exc:
    raise RuntimeError(
        f"Projektmodule konnten nicht importiert werden. "
        f"PROJECT_ROOT={PROJECT_ROOT}"
    ) from exc


# -----------------------------------------------------------------------------
# Datenmodelle
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class ExpectedHit:
    raw_name: str
    label: str
    ring: str
    segment: Optional[int]
    multiplier: int
    is_miss: bool


@dataclass(slots=True)
class SeriesResultRow:
    image_path: str
    image_name: str
    relative_path: str
    expected_label: str
    expected_ring: str
    expected_segment: Optional[int]
    expected_multiplier: int
    expected_is_miss: bool
    predicted_label: Optional[str]
    predicted_ring: Optional[str]
    predicted_segment: Optional[int]
    predicted_multiplier: Optional[int]
    predicted_score: Optional[int]
    predicted_is_miss: Optional[bool]
    final_image_x: Optional[float]
    final_image_y: Optional[float]
    final_topdown_x: Optional[float]
    final_topdown_y: Optional[float]
    best_candidate_id: Optional[int]
    candidate_count: int
    impact_count: int
    scored_count: int
    result_ok_exact: bool
    result_ok_segment_only: bool
    result_ok_ring_only: bool
    notes: str = ""


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def _log(message: str) -> None:
    print(message, flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single-cam test series and export CSV/JSON results."
    )

    parser.add_argument(
        "--images-root",
        required=True,
        help="Root folder containing test images (recursively).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Reference image path (empty board).",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Path to calibration.json",
    )
    parser.add_argument(
        "--config",
        default="",
        help="Optional config.json path. Currently accepted for compatibility.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index inside calibration.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where CSV/JSON and overlays are written.",
    )
    parser.add_argument(
        "--impact-strategy",
        default="blend",
        help="Impact strategy, e.g. blend / best_hypothesis / directional_contour_tip",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="If set, save a rendered overlay per test image.",
    )
    parser.add_argument(
        "--auto-board-mask",
        action="store_true",
        help="If set, create a circular board mask from calibration points.",
    )
    parser.add_argument(
        "--auto-board-mask-scale",
        type=float,
        default=0.90,
        help="Scale factor for auto board mask radius.",
    )
    parser.add_argument(
        "--extensions",
        default=".png,.jpg,.jpeg,.webp,.bmp",
        help="Comma-separated image extensions to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of test images to process. 0 = no limit.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Kalibrierung laden
# -----------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_manual_points(points: Any) -> list[dict[str, int]]:
    if not isinstance(points, list):
        raise ValueError("Calibration points must be a list.")

    normalized: list[dict[str, int]] = []
    for item in points:
        if isinstance(item, dict):
            if "x_px" in item and "y_px" in item:
                normalized.append(
                    {
                        "x_px": int(round(float(item["x_px"]))),
                        "y_px": int(round(float(item["y_px"]))),
                    }
                )
            elif "x" in item and "y" in item:
                normalized.append(
                    {
                        "x_px": int(round(float(item["x"]))),
                        "y_px": int(round(float(item["y"]))),
                    }
                )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            normalized.append(
                {
                    "x_px": int(round(float(item[0]))),
                    "y_px": int(round(float(item[1]))),
                }
            )

    if len(normalized) < 4:
        raise ValueError(
            "Calibration contains too few markers. Expected at least 4 manual points."
        )

    # Falls noch altes 5-Punkt-Format vorhanden ist, die ersten 4 verwenden.
    return normalized[:4]


@dataclass(slots=True)
class CompatibleCalibrationRecord:
    camera_index: int
    name: str
    enabled: bool
    device_id: int
    width: int
    height: int
    fps: int
    rotation: int
    flip: bool
    overlay_alpha: float
    show_numbers: bool
    show_sector_lines: bool
    manual_points: list[dict[str, int]]
    empty_board_reference_path: str = ""


def _load_calibration_record(calibration_path: Path, camera_index: int) -> CompatibleCalibrationRecord:
    payload = _load_json(calibration_path)

    cameras = payload.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("calibration.json does not contain a non-empty 'cameras' list.")

    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(
            f"camera-index {camera_index} out of range. calibration.json contains {len(cameras)} cameras."
        )

    cam = cameras[camera_index]
    if not isinstance(cam, dict):
        raise ValueError("Camera entry in calibration.json is invalid.")

    raw_points = cam.get("manual_points")
    if raw_points is None:
        raw_points = cam.get("points")

    manual_points = _normalize_manual_points(raw_points)

    return CompatibleCalibrationRecord(
        camera_index=int(camera_index),
        name=str(cam.get("name", f"Kamera {camera_index + 1}")),
        enabled=bool(cam.get("enabled", True)),
        device_id=int(cam.get("device_id", camera_index)),
        width=int(cam.get("frame_width", cam.get("width", 1280))),
        height=int(cam.get("frame_height", cam.get("height", 720))),
        fps=int(cam.get("fps", 30)),
        rotation=int(cam.get("rotation", 0)),
        flip=bool(cam.get("flip", False)),
        overlay_alpha=float(cam.get("overlay_alpha", 0.4)),
        show_numbers=bool(cam.get("show_numbers", True)),
        show_sector_lines=bool(cam.get("show_sector_lines", True)),
        manual_points=manual_points,
        empty_board_reference_path=str(cam.get("empty_board_reference_path", "")),
    )


# -----------------------------------------------------------------------------
# Dateinamen -> Soll-Label
# -----------------------------------------------------------------------------
_EXPECTED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^s([1-9]|1[0-9]|20)(?:_|$)", re.IGNORECASE), "S"),
    (re.compile(r"^t([1-9]|1[0-9]|20)(?:_|$)", re.IGNORECASE), "T"),
    (re.compile(r"^d([1-9]|1[0-9]|20)(?:_|$)", re.IGNORECASE), "D"),
    (re.compile(r"^sbull(?:_|$)", re.IGNORECASE), "SBULL"),
    (re.compile(r"^dbull(?:_|$)", re.IGNORECASE), "DBULL"),
    (re.compile(r"^miss(?:_|$)", re.IGNORECASE), "MISS"),
]


def parse_expected_hit_from_filename(path: Path) -> Optional[ExpectedHit]:
    stem = path.stem.strip().lower()

    for pattern, kind in _EXPECTED_PATTERNS:
        match = pattern.match(stem)
        if not match:
            continue

        if kind == "S":
            seg = int(match.group(1))
            return ExpectedHit(
                raw_name=path.name,
                label=f"S{seg}",
                ring="S",
                segment=seg,
                multiplier=1,
                is_miss=False,
            )
        if kind == "T":
            seg = int(match.group(1))
            return ExpectedHit(
                raw_name=path.name,
                label=f"T{seg}",
                ring="T",
                segment=seg,
                multiplier=3,
                is_miss=False,
            )
        if kind == "D":
            seg = int(match.group(1))
            return ExpectedHit(
                raw_name=path.name,
                label=f"D{seg}",
                ring="D",
                segment=seg,
                multiplier=2,
                is_miss=False,
            )
        if kind == "SBULL":
            return ExpectedHit(
                raw_name=path.name,
                label="SBULL",
                ring="SBULL",
                segment=None,
                multiplier=1,
                is_miss=False,
            )
        if kind == "DBULL":
            return ExpectedHit(
                raw_name=path.name,
                label="DBULL",
                ring="DBULL",
                segment=None,
                multiplier=2,
                is_miss=False,
            )
        if kind == "MISS":
            return ExpectedHit(
                raw_name=path.name,
                label="MISS",
                ring="MISS",
                segment=None,
                multiplier=0,
                is_miss=True,
            )

    return None


# -----------------------------------------------------------------------------
# Bilder / Masken
# -----------------------------------------------------------------------------
def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _point_tuple(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _compute_center_from_points(points: list[dict[str, int]]) -> tuple[float, float]:
    xs = [float(p["x_px"]) for p in points]
    ys = [float(p["y_px"]) for p in points]
    return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))


def _compute_auto_board_mask(
    image_shape: tuple[int, int],
    manual_points: list[dict[str, int]],
    scale: float,
) -> np.ndarray:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center_x, center_y = _compute_center_from_points(manual_points)
    dists = [
        float(np.hypot(p["x_px"] - center_x, p["y_px"] - center_y))
        for p in manual_points
    ]
    radius = max(1.0, float(np.mean(dists)) * float(scale))

    cv2.circle(
        mask,
        (int(round(center_x)), int(round(center_y))),
        int(round(radius)),
        255,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    return mask


# -----------------------------------------------------------------------------
# ScoreMapper / Detector
# -----------------------------------------------------------------------------
def _build_detector_from_calibration(
    calibration_record: CompatibleCalibrationRecord,
    impact_strategy: str,
) -> SingleCamDetector:
    score_mapper = build_score_mapper(calibration_record=calibration_record)

    detector = SingleCamDetector(
        config=SingleCamDetectorConfig(
            max_estimates_to_score=3,
            score_all_estimates=True,
            min_impact_confidence=0.01,
            min_combined_confidence=0.01,
            weight_candidate_confidence=0.40,
            weight_impact_confidence=0.60,
            prune_offboard_estimates_before_scoring=True,
            max_board_radius_rel_for_scoring=1.03,
            fallback_to_unpruned_estimates_if_all_filtered=False,
            keep_debug_images=True,
            keep_stage_results=True,
            render_stage_overlays=True,
        ),
        score_mapper=score_mapper,
    )

    # Strategy in ImpactEstimator setzen, wenn vorhanden
    if hasattr(detector, "impact_estimator") and hasattr(detector.impact_estimator, "config"):
        detector.impact_estimator.config.strategy = str(impact_strategy)

    return detector


# -----------------------------------------------------------------------------
# Ergebnis-Extraktion
# -----------------------------------------------------------------------------
def _safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _extract_best_estimate(result: Any) -> Any:
    best = _safe_attr(result, "best_estimate", None)
    if best is not None:
        return best

    scored = _safe_attr(result, "scored_estimates", None)
    if isinstance(scored, list) and scored:
        return scored[0]

    return None


def _extract_scored_hit(best_estimate: Any) -> Any:
    if best_estimate is None:
        return None
    return _safe_attr(best_estimate, "scored_hit", None)


def _extract_topdown_point(scored_hit: Any) -> Optional[tuple[float, float]]:
    if scored_hit is None:
        return None

    raw = _safe_attr(scored_hit, "raw_hit", None)
    if raw is not None:
        # Objektstil
        x = _safe_attr(raw, "topdown_x_px", None)
        y = _safe_attr(raw, "topdown_y_px", None)
        if x is not None and y is not None:
            return float(x), float(y)

        # Dictstil
        if isinstance(raw, dict):
            if "topdown_x_px" in raw and "topdown_y_px" in raw:
                return float(raw["topdown_x_px"]), float(raw["topdown_y_px"])

    # Fallback
    point = _safe_attr(scored_hit, "topdown_point", None)
    if point is not None:
        return _point_tuple(point)

    return None


def _extract_image_point(best_estimate: Any, scored_hit: Any) -> Optional[tuple[float, float]]:
    if best_estimate is not None:
        point = _safe_attr(best_estimate, "image_point", None)
        if point is not None:
            return _point_tuple(point)

    if scored_hit is not None:
        point = _safe_attr(scored_hit, "image_point", None)
        if point is not None:
            return _point_tuple(point)

    return None


def _result_row_from_pipeline(
    image_path: Path,
    relative_path: str,
    expected: ExpectedHit,
    result: Any,
) -> SeriesResultRow:
    best_estimate = _extract_best_estimate(result)
    scored_hit = _extract_scored_hit(best_estimate)

    predicted_label = _safe_attr(scored_hit, "label", None)
    predicted_ring = _safe_attr(scored_hit, "ring", None)
    predicted_segment = _safe_attr(scored_hit, "segment", None)
    predicted_multiplier = _safe_attr(scored_hit, "multiplier", None)
    predicted_score = _safe_attr(scored_hit, "score", None)

    predicted_is_miss = None
    if scored_hit is not None:
        is_miss = _safe_attr(scored_hit, "is_miss", None)
        if is_miss is None and predicted_ring is not None:
            predicted_is_miss = str(predicted_ring).upper() == "MISS"
        else:
            predicted_is_miss = bool(is_miss)

    image_point = _extract_image_point(best_estimate, scored_hit)
    topdown_point = _extract_topdown_point(scored_hit)

    metadata = _safe_attr(result, "metadata", {}) or {}
    candidate_count = int(metadata.get("candidate_count", 0))
    impact_count = int(metadata.get("impact_count", 0))
    scored_count = int(metadata.get("scored_count", 0))

    best_candidate_id = _safe_attr(best_estimate, "candidate_id", None)

    result_ok_exact = bool(predicted_label == expected.label)

    result_ok_segment_only = False
    if expected.segment is not None and predicted_segment is not None:
        result_ok_segment_only = int(expected.segment) == int(predicted_segment)

    result_ok_ring_only = False
    if expected.ring and predicted_ring:
        result_ok_ring_only = str(expected.ring).upper() == str(predicted_ring).upper()

    notes = ""
    if scored_count == 0:
        notes = "no_scored_estimate"

    return SeriesResultRow(
        image_path=str(image_path),
        image_name=image_path.name,
        relative_path=relative_path,
        expected_label=expected.label,
        expected_ring=expected.ring,
        expected_segment=expected.segment,
        expected_multiplier=expected.multiplier,
        expected_is_miss=expected.is_miss,
        predicted_label=predicted_label,
        predicted_ring=predicted_ring,
        predicted_segment=predicted_segment,
        predicted_multiplier=predicted_multiplier,
        predicted_score=predicted_score,
        predicted_is_miss=predicted_is_miss,
        final_image_x=None if image_point is None else float(image_point[0]),
        final_image_y=None if image_point is None else float(image_point[1]),
        final_topdown_x=None if topdown_point is None else float(topdown_point[0]),
        final_topdown_y=None if topdown_point is None else float(topdown_point[1]),
        best_candidate_id=None if best_candidate_id is None else int(best_candidate_id),
        candidate_count=candidate_count,
        impact_count=impact_count,
        scored_count=scored_count,
        result_ok_exact=result_ok_exact,
        result_ok_segment_only=result_ok_segment_only,
        result_ok_ring_only=result_ok_ring_only,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Export
# -----------------------------------------------------------------------------
def _write_csv(rows: list[SeriesResultRow], csv_path: Path) -> None:
    _ensure_dir(csv_path.parent)

    fieldnames = list(asdict(rows[0]).keys()) if rows else list(asdict(SeriesResultRow(
        image_path="",
        image_name="",
        relative_path="",
        expected_label="",
        expected_ring="",
        expected_segment=None,
        expected_multiplier=0,
        expected_is_miss=False,
        predicted_label=None,
        predicted_ring=None,
        predicted_segment=None,
        predicted_multiplier=None,
        predicted_score=None,
        predicted_is_miss=None,
        final_image_x=None,
        final_image_y=None,
        final_topdown_x=None,
        final_topdown_y=None,
        best_candidate_id=None,
        candidate_count=0,
        impact_count=0,
        scored_count=0,
        result_ok_exact=False,
        result_ok_segment_only=False,
        result_ok_ring_only=False,
        notes="",
    )).keys())

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(payload: Any, json_path: Path) -> None:
    _ensure_dir(json_path.parent)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Zusammenfassung
# -----------------------------------------------------------------------------
def _make_summary(rows: list[SeriesResultRow]) -> dict[str, Any]:
    total = len(rows)
    exact = sum(1 for r in rows if r.result_ok_exact)
    segment_only = sum(1 for r in rows if r.result_ok_segment_only)
    ring_only = sum(1 for r in rows if r.result_ok_ring_only)
    with_scored = sum(1 for r in rows if r.scored_count > 0)

    by_expected_ring: dict[str, dict[str, int]] = {}
    confusions: dict[str, int] = {}

    for row in rows:
        ring = row.expected_ring
        if ring not in by_expected_ring:
            by_expected_ring[ring] = {
                "total": 0,
                "exact": 0,
                "segment_only": 0,
                "ring_only": 0,
            }
        by_expected_ring[ring]["total"] += 1
        by_expected_ring[ring]["exact"] += int(row.result_ok_exact)
        by_expected_ring[ring]["segment_only"] += int(row.result_ok_segment_only)
        by_expected_ring[ring]["ring_only"] += int(row.result_ok_ring_only)

        pred = row.predicted_label or "NONE"
        if pred != row.expected_label:
            key = f"{row.expected_label} -> {pred}"
            confusions[key] = confusions.get(key, 0) + 1

    confusion_sorted = sorted(confusions.items(), key=lambda kv: kv[1], reverse=True)

    return {
        "total_images": total,
        "with_scored_estimate": with_scored,
        "exact_hits": exact,
        "segment_only_hits": segment_only,
        "ring_only_hits": ring_only,
        "exact_accuracy": 0.0 if total == 0 else exact / total,
        "segment_only_accuracy": 0.0 if total == 0 else segment_only / total,
        "ring_only_accuracy": 0.0 if total == 0 else ring_only / total,
        "by_expected_ring": by_expected_ring,
        "top_confusions": [{"pair": k, "count": v} for k, v in confusion_sorted[:20]],
    }


# -----------------------------------------------------------------------------
# Bildsammlung
# -----------------------------------------------------------------------------
def _collect_test_images(images_root: Path, extensions: list[str], limit: int) -> list[Path]:
    valid_ext = {ext.lower().strip() for ext in extensions if ext.strip()}
    all_files = []

    for path in images_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_ext:
            continue
        if "reference" in {p.lower() for p in path.parts}:
            continue
        all_files.append(path)

    all_files.sort()

    if limit > 0:
        return all_files[:limit]
    return all_files


# -----------------------------------------------------------------------------
# Overlay speichern
# -----------------------------------------------------------------------------
def _save_overlay_if_requested(
    result: Any,
    frame: np.ndarray,
    output_dir: Path,
    relative_path: str,
) -> Optional[str]:
    if not hasattr(result, "render_debug_overlay"):
        return None

    safe_rel = relative_path.replace("\\", "/")
    stem = Path(safe_rel).with_suffix("")
    overlay_path = output_dir / "overlays" / f"{stem}.png"
    _ensure_dir(overlay_path.parent)

    try:
        overlay = result.render_debug_overlay(frame)
        cv2.imwrite(str(overlay_path), overlay)
        return str(overlay_path)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    args = parse_args()

    images_root = Path(args.images_root)
    reference_path = Path(args.reference)
    calibration_path = Path(args.calibration)
    output_dir = Path(args.output_dir)

    if not images_root.exists():
        raise FileNotFoundError(f"images-root not found: {images_root}")
    if not reference_path.exists():
        raise FileNotFoundError(f"reference not found: {reference_path}")
    if not calibration_path.exists():
        raise FileNotFoundError(f"calibration not found: {calibration_path}")

    _ensure_dir(output_dir)

    _log("[INFO] Lade Kalibrierung ...")
    calibration_record = _load_calibration_record(calibration_path, args.camera_index)

    _log("[INFO] Erzeuge Detector ...")
    detector = _build_detector_from_calibration(
        calibration_record=calibration_record,
        impact_strategy=args.impact_strategy,
    )

    _log("[INFO] Lade Referenzbild ...")
    reference_image = _read_image(reference_path)

    board_mask = None
    if args.auto_board_mask:
        board_mask = _compute_auto_board_mask(
            image_shape=reference_image.shape,
            manual_points=calibration_record.manual_points,
            scale=float(args.auto_board_mask_scale),
        )

    extensions = [x.strip() for x in str(args.extensions).split(",") if x.strip()]
    test_images = _collect_test_images(images_root, extensions, int(args.limit))

    _log(f"[INFO] Gefundene Testbilder: {len(test_images)}")

    rows: list[SeriesResultRow] = []
    detailed_results: list[dict[str, Any]] = []

    for index, image_path in enumerate(test_images, start=1):
        relative_path = str(image_path.relative_to(images_root))

        expected = parse_expected_hit_from_filename(image_path)
        if expected is None:
            _log(f"[WARN] Überspringe {image_path.name}: Dateiname passt nicht zum Soll-Schema.")
            continue

        _log(f"[INFO] [{index}/{len(test_images)}] Teste {relative_path} ...")

        frame = _read_image(image_path)

        if frame.shape[:2] != reference_image.shape[:2]:
            _log(
                f"[WARN] Resizing frame for {image_path.name}: "
                f"frame={frame.shape[:2]} -> reference={reference_image.shape[:2]}"
            )
            frame = cv2.resize(
                frame,
                (reference_image.shape[1], reference_image.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        try:
            result = detector.detect(
                frame=frame,
                reference_frame=reference_image,
                board_mask=board_mask,
                board_polygon=None,
            )
        except Exception as exc:
            row = SeriesResultRow(
                image_path=str(image_path),
                image_name=image_path.name,
                relative_path=relative_path,
                expected_label=expected.label,
                expected_ring=expected.ring,
                expected_segment=expected.segment,
                expected_multiplier=expected.multiplier,
                expected_is_miss=expected.is_miss,
                predicted_label=None,
                predicted_ring=None,
                predicted_segment=None,
                predicted_multiplier=None,
                predicted_score=None,
                predicted_is_miss=None,
                final_image_x=None,
                final_image_y=None,
                final_topdown_x=None,
                final_topdown_y=None,
                best_candidate_id=None,
                candidate_count=0,
                impact_count=0,
                scored_count=0,
                result_ok_exact=False,
                result_ok_segment_only=False,
                result_ok_ring_only=False,
                notes=f"exception: {exc}",
            )
            rows.append(row)
            detailed_results.append(
                {
                    "image_path": str(image_path),
                    "relative_path": relative_path,
                    "expected": asdict(expected),
                    "error": str(exc),
                }
            )
            continue

        overlay_path = None
        if args.save_overlays:
            overlay_path = _save_overlay_if_requested(
                result=result,
                frame=frame,
                output_dir=output_dir,
                relative_path=relative_path,
            )

        row = _result_row_from_pipeline(
            image_path=image_path,
            relative_path=relative_path,
            expected=expected,
            result=result,
        )
        rows.append(row)

        detailed_results.append(
            {
                "image_path": str(image_path),
                "relative_path": relative_path,
                "expected": asdict(expected),
                "row": asdict(row),
                "overlay_path": overlay_path,
                "result": result.to_dict() if hasattr(result, "to_dict") else {},
            }
        )

    summary = _make_summary(rows)

    csv_path = output_dir / "test_series_results.csv"
    json_path = output_dir / "test_series_results.json"
    summary_path = output_dir / "test_series_summary.json"

    _write_csv(rows, csv_path)
    _write_json(
        {
            "summary": summary,
            "rows": [asdict(r) for r in rows],
            "details": detailed_results,
        },
        json_path,
    )
    _write_json(summary, summary_path)

    _log("")
    _log("[DONE] Testserie abgeschlossen.")
    _log(f"[DONE] CSV   : {csv_path}")
    _log(f"[DONE] JSON  : {json_path}")
    _log(f"[DONE] Summary: {summary_path}")
    _log("")
    _log("----- Zusammenfassung -----")
    _log(f"Total images        : {summary['total_images']}")
    _log(f"With scored estimate: {summary['with_scored_estimate']}")
    _log(f"Exact hits          : {summary['exact_hits']}")
    _log(f"Exact accuracy      : {summary['exact_accuracy']:.2%}")
    _log(f"Segment accuracy    : {summary['segment_only_accuracy']:.2%}")
    _log(f"Ring accuracy       : {summary['ring_only_accuracy']:.2%}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log("\n[ABORT] Vom Benutzer abgebrochen.")
        raise SystemExit(130)