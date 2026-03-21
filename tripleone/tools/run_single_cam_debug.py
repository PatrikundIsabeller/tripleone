#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/run_single_cam_debug.py

Zweck:
- Führt die komplette Single-Cam-Debug-Pipeline für ein Einzelbild aus
- Nutzt Referenzbild + aktuelles Bild
- Nutzt calibration.json + config.json
- Speichert Debug-Overlays und optionale Stage-Bilder
- Zeichnet zusätzlich die Score-Geometrie auf das Kamerabild

Wichtige Ausgaben:
- candidate_overlay.png
- impact_overlay.png
- single_cam_overlay.png
- score_geometry_overlay.png

Typischer Aufruf:
py tools/run_single_cam_debug.py ^
  --frame "D:\tripleone\tripleone\data\test_images\dart_01.png" ^
  --reference "D:\tripleone\tripleone\data\references\camera_1_empty_board.png" ^
  --calibration "D:\tripleone\tripleone\config\calibration.json" ^
  --config "D:\tripleone\tripleone\config\config.json" ^
  --camera-index 0 ^
  --output-dir "D:\tripleone\tripleone\debug_output\single_cam_test" ^
  --save-stage-images ^
  --auto-board-mask ^
  --auto-board-mask-scale 0.90 ^
  --impact-strategy blend
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Projekt-Imports robust halten:
try:
    from vision.single_cam_detector import (
        SingleCamDetector,
        SingleCamDetectorConfig,
    )
except ImportError:  # pragma: no cover
    from tripleone.vision.single_cam_detector import (  # type: ignore
        SingleCamDetector,
        SingleCamDetectorConfig,
    )


PointF = tuple[float, float]


# ---------------------------------------------------------------------
# Allgemeine Datei-/Bild-Helfer
# ---------------------------------------------------------------------

def load_image(image_path: str | Path) -> np.ndarray:
    """
    Lädt ein Bild von der Platte.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Bild konnte nicht geladen werden: {path}")

    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """
    Speichert ein Bild auf der Platte.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(str(out_path), image)
    if not ok:
        raise RuntimeError(f"Bild konnte nicht gespeichert werden: {out_path}")


def load_json(json_path: str | Path) -> dict[str, Any]:
    """
    Lädt eine JSON-Datei robust.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON-Datei muss ein Objekt enthalten: {path}")

    return data


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """
    Stellt sicher, dass ein Bild als BGR vorliegt.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


# ---------------------------------------------------------------------
# Kalibrierung / Config lesen
# ---------------------------------------------------------------------

def _coerce_point(value: Any) -> PointF:
    """
    Wandelt verschiedene Punktformate in (x, y) um.
    Unterstützt:
    - (x, y)
    - [x, y]
    - {"x": ..., "y": ...}
    - {"x_px": ..., "y_px": ...}
    """
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
        raise ValueError(f"Point dict muss x/y oder x_px/y_px enthalten: {value}")

    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        return float(value[0]), float(value[1])

    raise ValueError(f"Ungültiges Punktformat: {value!r}")


def _extract_manual_points_from_calibration_dict(data: dict[str, Any], camera_index: int) -> list[PointF]:
    """
    Liest 4 Kalibrierpunkte aus verschiedenen calibration.json-Formaten.

    Unterstützte Muster:
    - {"manual_points": [...]}
    - {"cameras": [{"manual_points": [...]}, ...]}
    - {"camera_calibrations": [{"manual_points": [...]}, ...]}
    - {"points": [...]}
    - {"marker_points": [...]}
    """
    candidates: list[Any] = []

    # global direkt
    for key in ("manual_points", "points", "marker_points", "image_points", "markers"):
        if key in data:
            candidates.append(data[key])

    # verschachtelt über cameras
    for list_key in ("cameras", "camera_calibrations", "camera_configs", "camera_data"):
        items = data.get(list_key)
        if isinstance(items, list) and 0 <= camera_index < len(items):
            entry = items[camera_index]
            if isinstance(entry, dict):
                for key in ("manual_points", "points", "marker_points", "image_points", "markers"):
                    if key in entry:
                        candidates.append(entry[key])

    for raw in candidates:
        if isinstance(raw, list) and len(raw) == 4:
            try:
                return [_coerce_point(v) for v in raw]
            except Exception:
                continue

    raise ValueError(
        "Konnte keine 4 Kalibrierpunkte in calibration.json finden. "
        "Erwartet z. B. manual_points / points / marker_points."
    )


def _extract_image_size_from_calibration_dict(
    data: dict[str, Any],
    camera_index: int,
    *,
    fallback_image: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """
    Liest Bildgröße aus calibration.json oder fällt auf das geladene Bild zurück.
    """
    def _coerce_size(value: Any) -> Optional[tuple[int, int]]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return int(value[0]), int(value[1])
        if isinstance(value, dict):
            if "width" in value and "height" in value:
                return int(value["width"]), int(value["height"])
            if "image_width" in value and "image_height" in value:
                return int(value["image_width"]), int(value["image_height"])
        return None

    candidates: list[Any] = []

    for key in ("image_size", "size", "frame_size"):
        if key in data:
            candidates.append(data[key])

    for list_key in ("cameras", "camera_calibrations", "camera_configs", "camera_data"):
        items = data.get(list_key)
        if isinstance(items, list) and 0 <= camera_index < len(items):
            entry = items[camera_index]
            if isinstance(entry, dict):
                for key in ("image_size", "size", "frame_size"):
                    if key in entry:
                        candidates.append(entry[key])
                if "image_width" in entry and "image_height" in entry:
                    candidates.append({"image_width": entry["image_width"], "image_height": entry["image_height"]})

    if "image_width" in data and "image_height" in data:
        candidates.append({"image_width": data["image_width"], "image_height": data["image_height"]})

    for value in candidates:
        size = _coerce_size(value)
        if size is not None:
            return size

    if fallback_image is not None:
        h, w = fallback_image.shape[:2]
        return int(w), int(h)

    raise ValueError("Konnte keine image_size bestimmen.")


def _extract_single_cam_config_dict(config_data: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Extrahiert aus config.json nur den relevanten Single-Cam-Config-Block.

    Unterstützte Muster:
    - {"single_cam": {...}}
    - {"single_cam_detector": {...}}
    - {"vision": {"single_cam": {...}}}
    - Fallback: Root-Level
    """
    if not config_data:
        return {}

    for key in ("single_cam", "single_cam_detector"):
        value = config_data.get(key)
        if isinstance(value, dict):
            return dict(value)

    vision = config_data.get("vision")
    if isinstance(vision, dict):
        for key in ("single_cam", "single_cam_detector"):
            value = vision.get(key)
            if isinstance(value, dict):
                return dict(value)

    return dict(config_data)


def _build_single_cam_config(config_dict: dict[str, Any]) -> SingleCamDetectorConfig:
    """
    Baut robust ein SingleCamDetectorConfig-Objekt.
    Nur existierende Felder werden übernommen.
    """
    config = SingleCamDetectorConfig()

    allowed_fields = set(getattr(SingleCamDetectorConfig, "__dataclass_fields__", {}).keys())

    for key, value in config_dict.items():
        if key in allowed_fields:
            setattr(config, key, value)

    return config


# ---------------------------------------------------------------------
# Board-Mask / Polygon
# ---------------------------------------------------------------------

def _parse_board_polygon(value: Optional[str]) -> Optional[list[tuple[int, int]]]:
    """
    Erwartet JSON-String, z. B.:
    [[100, 100], [200, 100], [200, 200], [100, 200]]
    """
    if not value:
        return None

    data = json.loads(value)
    if not isinstance(data, list) or len(data) < 3:
        raise ValueError("--board-polygon muss mindestens 3 Punkte enthalten.")

    points: list[tuple[int, int]] = []
    for item in data:
        x, y = _coerce_point(item)
        points.append((int(round(x)), int(round(y))))

    return points


def build_auto_board_mask(
    frame: np.ndarray,
    manual_points: list[PointF],
    *,
    scale: float = 0.90,
) -> np.ndarray:
    """
    Baut aus den 4 Kalibrierpunkten eine grobe Kreisscheibe als ROI-Maske.

    Idee:
    - Mittelpunkt = Mittelwert der 4 Punkte
    - Radius = mittlere Distanz der 4 Punkte zum Mittelpunkt
    - optionaler Skalierungsfaktor
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.asarray(manual_points, dtype=np.float64)
    center = np.mean(pts, axis=0)
    dists = np.linalg.norm(pts - center[None, :], axis=1)
    radius = float(np.mean(dists)) * float(scale)

    cx = int(round(center[0]))
    cy = int(round(center[1]))
    rr = max(1, int(round(radius)))

    cv2.circle(mask, (cx, cy), rr, 255, thickness=-1)
    return mask


# ---------------------------------------------------------------------
# Score-Geometrie-Overlay
# ---------------------------------------------------------------------

def render_score_geometry_overlay(
    frame: np.ndarray,
    score_mapper: Any,
) -> np.ndarray:
    """
    Zeichnet die vom ScoreMapper verwendete Score-Geometrie auf das Kamerabild.

    Sichtbar:
    - Bull-Zentrum
    - wichtige Ringkreise
    - Sektorlinien
    - Segmentlabels

    Ziel:
    Prüfen, ob Kalibrierung/Rotation/Winkelaufteilung zur echten Dartscheibe passt.
    """
    canvas = ensure_bgr(frame)

    if score_mapper is None:
        return canvas

    pipeline = getattr(score_mapper, "pipeline", None)
    if pipeline is None:
        return canvas

    if not hasattr(score_mapper, "topdown_point_to_image"):
        return canvas

    segment_order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

    center_td = (450.0, 450.0)

    # relative Radii passend zur bisherigen TripleOne-Geometrie
    relative_radii = [0.985, 0.93, 0.57, 0.51, 0.093, 0.037]
    outer_radius_td = 450.0
    ring_radii_td = [outer_radius_td * r for r in relative_radii]

    try:
        center_img = score_mapper.topdown_point_to_image(center_td)
    except Exception:
        center_img = None

    if center_img is None:
        return canvas

    cx = int(round(center_img[0]))
    cy = int(round(center_img[1]))
    cv2.circle(canvas, (cx, cy), 6, (0, 0, 255), -1)
    cv2.circle(canvas, (cx, cy), 12, (255, 255, 255), 1)

    circle_colors = [
        (0, 255, 255),   # outer double
        (0, 200, 255),   # inner double
        (0, 255, 0),     # outer triple
        (0, 200, 0),     # inner triple
        (255, 255, 0),   # outer bull
        (0, 0, 255),     # inner bull
    ]

    for radius_td, color in zip(ring_radii_td, circle_colors):
        pts_img: list[tuple[int, int]] = []
        for deg in range(0, 360, 5):
            rad = np.deg2rad(deg)
            x_td = center_td[0] + np.cos(rad) * radius_td
            y_td = center_td[1] - np.sin(rad) * radius_td

            try:
                p_img = score_mapper.topdown_point_to_image((x_td, y_td))
            except Exception:
                p_img = None

            if p_img is not None:
                pts_img.append((int(round(p_img[0])), int(round(p_img[1]))))

        if len(pts_img) >= 3:
            poly = np.asarray(pts_img, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    # Sektorgrenzen
    for i in range(20):
        boundary_deg = 90.0 - (i * 18.0) - 9.0
        rad = np.deg2rad(boundary_deg)

        x_td = center_td[0] + np.cos(rad) * outer_radius_td
        y_td = center_td[1] - np.sin(rad) * outer_radius_td

        try:
            p_img = score_mapper.topdown_point_to_image((x_td, y_td))
        except Exception:
            p_img = None

        if p_img is not None:
            px = int(round(p_img[0]))
            py = int(round(p_img[1]))
            cv2.line(canvas, (cx, cy), (px, py), (255, 0, 0), 1, cv2.LINE_AA)

    # Segmentlabels
    label_radius_td = outer_radius_td * 1.03

    for i, segment in enumerate(segment_order):
        angle_deg = 90.0 - (i * 18.0)
        rad = np.deg2rad(angle_deg)

        x_td = center_td[0] + np.cos(rad) * label_radius_td
        y_td = center_td[1] - np.sin(rad) * label_radius_td

        try:
            p_img = score_mapper.topdown_point_to_image((x_td, y_td))
        except Exception:
            p_img = None

        if p_img is None:
            continue

        lx = int(round(p_img[0]))
        ly = int(round(p_img[1]))

        cv2.putText(
            canvas,
            str(segment),
            (lx - 10, ly + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(segment),
            (lx - 10, ly + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------
# Detector / Strategie-Overrides
# ---------------------------------------------------------------------

def _get_score_mapper(detector: Any) -> Any:
    """
    Holt den ScoreMapper robust aus dem Detector.
    """
    public_mapper = getattr(detector, "score_mapper", None)
    if public_mapper is not None:
        return public_mapper

    return getattr(detector, "_score_mapper", None)


def _apply_runtime_overrides(detector: Any, args: argparse.Namespace) -> None:
    """
    Wendet CLI-Overrides auf Config / Unterobjekte an.
    """
    # Detector-Config
    detector_config = getattr(detector, "config", None)
    if detector_config is not None:
        if args.score_all_estimates:
            setattr(detector_config, "score_all_estimates", True)
        if args.max_estimates_to_score is not None:
            setattr(detector_config, "max_estimates_to_score", int(args.max_estimates_to_score))
        if args.min_impact_confidence is not None:
            setattr(detector_config, "min_impact_confidence", float(args.min_impact_confidence))
        if args.min_combined_confidence is not None:
            setattr(detector_config, "min_combined_confidence", float(args.min_combined_confidence))

    # Candidate-Detector-Config
    candidate_detector = getattr(detector, "candidate_detector", None)
    candidate_config = getattr(candidate_detector, "config", None) if candidate_detector is not None else None
    if candidate_config is not None:
        if args.candidate_diff_threshold is not None:
            setattr(candidate_config, "diff_threshold", float(args.candidate_diff_threshold))
        if args.candidate_min_area is not None:
            setattr(candidate_config, "min_contour_area", float(args.candidate_min_area))
        if args.candidate_min_confidence is not None:
            setattr(candidate_config, "min_confidence", float(args.candidate_min_confidence))

    # Impact-Estimator-Config
    impact_estimator = getattr(detector, "impact_estimator", None)
    impact_config = getattr(impact_estimator, "config", None) if impact_estimator is not None else None
    if impact_config is not None and args.impact_strategy:
        setattr(impact_config, "strategy", str(args.impact_strategy))


# ---------------------------------------------------------------------
# Zusammenfassung drucken
# ---------------------------------------------------------------------

def print_result_summary(
    result: Any,
    *,
    calibration_source: str,
    camera_index: int,
    image_size: tuple[int, int],
) -> None:
    """
    Druckt eine kompakte Ergebniszusammenfassung.
    """
    metadata = getattr(result, "metadata", {}) or {}
    scored_estimates = getattr(result, "scored_estimates", []) or []

    best = scored_estimates[0] if scored_estimates else None

    print()
    print("===== SINGLE CAM DEBUG RESULT =====")
    print(f"Calibration Source: {calibration_source}")
    print(f"Camera Index: {camera_index}")
    print(f"Image Size: {image_size}")

    if best is None:
        print("Best Label: None")
        print("Best Score: None")
        print("Scored Estimates: 0")
        print(f"Candidate Count: {metadata.get('candidate_count')}")
        print(f"Impact Count: {metadata.get('impact_count')}")
        return

    print(f"Best Label: {getattr(best, 'label', None)}")
    print(f"Best Score: {getattr(best, 'score', None)}")
    print(f"Scored Estimates: {len(scored_estimates)}")
    print(f"Best Candidate ID: {getattr(best, 'candidate_id', None)}")
    print(f"Image Point: {getattr(best, 'image_point', None)}")
    print(f"Candidate Confidence: {getattr(best, 'candidate_confidence', None):.4f}")
    print(f"Impact Confidence: {getattr(best, 'impact_confidence', None):.4f}")
    print(f"Combined Confidence: {getattr(best, 'combined_confidence', None):.4f}")
    print(f"Impact Method: {getattr(best, 'impact_method', None)}")
    print(f"Hypothesis Count: {getattr(best, 'hypothesis_count', None)}")
    print(f"Candidate Count: {metadata.get('candidate_count')}")
    print(f"Impact Count: {metadata.get('impact_count')}")


# ---------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-Cam-Debuglauf für TripleOne.")

    parser.add_argument("--frame", required=True, help="Aktuelles Bild mit Dart.")
    parser.add_argument("--reference", help="Referenzbild ohne Dart.")
    parser.add_argument("--auto-board-mask", action="store_true", help="Grobes Board-ROI automatisch aus Kalibrierpunkten erzeugen.")
    parser.add_argument("--auto-board-mask-scale", type=float, default=0.90, help="Skalierung der automatischen Board-Kreismaske.")
    parser.add_argument("--calibration", required=True, help="Pfad zu calibration.json.")
    parser.add_argument("--config", help="Pfad zu config.json.")
    parser.add_argument("--camera-index", type=int, default=0, help="Kameraindex in calibration/config.")
    parser.add_argument("--board-polygon", help="Optionales JSON-Polygon, z. B. [[x1,y1],[x2,y2],...].")
    parser.add_argument("--output-dir", default="debug_output/single_cam_test", help="Ausgabeordner.")
    parser.add_argument("--save-stage-images", action="store_true", help="Speichert Stage-Debugbilder aus CandidateDetector/SingleCam.")
    parser.add_argument("--no-stage-overlays", action="store_true", help="Unterdrückt candidate/impact/single-cam overlays.")
    parser.add_argument("--score-all-estimates", action="store_true", help="Scort alle ImpactEstimates statt nur die besten.")
    parser.add_argument("--max-estimates-to-score", type=int, help="Begrenzt die Anzahl der weitergescorten Impacts.")
    parser.add_argument("--min-impact-confidence", type=float, help="Mindest-Impact-Konfidenz.")
    parser.add_argument("--min-combined-confidence", type=float, help="Mindest-Combined-Konfidenz.")
    parser.add_argument("--candidate-diff-threshold", type=float, help="Override Candidate diff_threshold.")
    parser.add_argument("--candidate-min-area", type=float, help="Override Candidate min_contour_area.")
    parser.add_argument("--candidate-min-confidence", type=float, help="Override Candidate min_confidence.")

    parser.add_argument(
        "--impact-strategy",
        type=str,
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
        help="Strategie des Impact Estimators.",
    )

    return parser


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_image(args.frame)
    reference_frame = load_image(args.reference) if args.reference else frame.copy()

    calibration_data = load_json(args.calibration)
    config_data = load_json(args.config) if args.config else {}

    manual_points = _extract_manual_points_from_calibration_dict(calibration_data, int(args.camera_index))
    image_size = _extract_image_size_from_calibration_dict(
        calibration_data,
        int(args.camera_index),
        fallback_image=frame,
    )

    single_cam_config_dict = _extract_single_cam_config_dict(config_data)
    single_cam_config = _build_single_cam_config(single_cam_config_dict)

    detector = SingleCamDetector(
        config=single_cam_config,
        manual_points=manual_points,
        image_size=image_size,
    )

    _apply_runtime_overrides(detector, args)

    board_polygon = _parse_board_polygon(args.board_polygon)

    board_mask = None
    if args.auto_board_mask:
        board_mask = build_auto_board_mask(
            frame,
            manual_points=manual_points,
            scale=float(args.auto_board_mask_scale),
        )
        save_image(output_dir / "board_mask.png", board_mask)

    result = detector.detect(
        frame=frame,
        reference_frame=reference_frame,
        board_mask=board_mask,
        board_polygon=board_polygon,
    )

    # -----------------------------------------------------------------
    # Stage-Bilder speichern
    # -----------------------------------------------------------------
    if args.save_stage_images:
        # Candidate stage images
        candidate_result = getattr(result, "candidate_result", None)
        candidate_debug_images = getattr(candidate_result, "debug_images", {}) if candidate_result is not None else {}
        if isinstance(candidate_debug_images, dict):
            for name, image in candidate_debug_images.items():
                if isinstance(image, np.ndarray):
                    save_image(output_dir / f"candidate_stage_{name}.png", image)

        # SingleCam debug images
        result_debug_images = getattr(result, "debug_images", {}) or {}
        if isinstance(result_debug_images, dict):
            for name, image in result_debug_images.items():
                if isinstance(image, np.ndarray):
                    save_image(output_dir / f"single_cam_stage_{name}.png", image)

    # -----------------------------------------------------------------
    # Overlays speichern
    # -----------------------------------------------------------------
    if not args.no_stage_overlays:
        candidate_result = getattr(result, "candidate_result", None)
        impact_result = getattr(result, "impact_result", None)

        if candidate_result is not None and hasattr(candidate_result, "render_debug_overlay"):
            try:
                candidate_overlay = candidate_result.render_debug_overlay(frame.copy())
                save_image(output_dir / "candidate_overlay.png", candidate_overlay)
            except Exception as exc:
                print(f"Warnung: candidate_overlay konnte nicht erzeugt werden: {exc}")

        if impact_result is not None and hasattr(impact_result, "render_debug_overlay"):
            try:
                impact_overlay = impact_result.render_debug_overlay(frame.copy())
                save_image(output_dir / "impact_overlay.png", impact_overlay)
            except Exception as exc:
                print(f"Warnung: impact_overlay konnte nicht erzeugt werden: {exc}")

        if hasattr(result, "render_debug_overlay"):
            try:
                single_cam_overlay = result.render_debug_overlay(frame.copy())
                save_image(output_dir / "single_cam_overlay.png", single_cam_overlay)
            except Exception as exc:
                print(f"Warnung: single_cam_overlay konnte nicht erzeugt werden: {exc}")

        try:
            score_mapper = _get_score_mapper(detector)
            score_geometry_overlay = render_score_geometry_overlay(frame, score_mapper)
            save_image(output_dir / "score_geometry_overlay.png", score_geometry_overlay)
        except Exception as exc:
            print(f"Warnung: score_geometry_overlay konnte nicht erzeugt werden: {exc}")

    # -----------------------------------------------------------------
    # Ergebnis JSON speichern
    # -----------------------------------------------------------------
    result_dict = result.to_dict() if hasattr(result, "to_dict") else {"result": str(result)}
    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print_result_summary(
        result,
        calibration_source="old_calibration_json",
        camera_index=int(args.camera_index),
        image_size=image_size,
    )
    print()
    print(f"Ausgabe gespeichert in: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())