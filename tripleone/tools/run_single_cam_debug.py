# tools/run_single_cam_debug.py
# Zweck:
# Dieses Script ist ein kontrollierter Realtest für die Single-Cam-Pipeline.
#
# Es lädt:
# - ein Referenzbild (leeres Board)
# - ein aktuelles Bild (mit möglichem Dart)
# - eine Kalibrierungsdatei / einen Kalibrierungs-Record
#
# Danach führt es die komplette Single-Cam-Pipeline aus:
# 1) Candidate Detection
# 2) Impact Estimation
# 3) Score Mapping
#
# Und speichert:
# - ein finales Overlay
# - optionale Stage-Debugbilder
# - eine JSON-Ergebnisdatei
#
# Wichtiger Hinweis:
# Dieses Script ist absichtlich für reproduzierbare Einzelbildtests gebaut,
# nicht für den finalen Livebetrieb.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# Projekt-Root in sys.path aufnehmen, falls das Script direkt ausgeführt wird
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.dart_candidate_detector import CandidateDetectorConfig
from vision.impact_estimator import ImpactEstimatorConfig
from vision.single_cam_detector import (
    SingleCamDetectionResult,
    SingleCamDetector,
    SingleCamDetectorConfig,
)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Baut den CLI-Parser für das Debug-Script.
    """
    parser = argparse.ArgumentParser(
        description="Run a reproducible single-camera debug test for Triple One."
    )

    parser.add_argument(
        "--frame",
        required=True,
        help="Pfad zum aktuellen Bild (z. B. Board mit Dart).",
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Pfad zum Referenzbild des leeren Boards.",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help=(
            "Pfad zu einer JSON-Datei mit Kalibrierungsdaten. "
            "Erwartet entweder ein dict mit manual_points oder einen einzelnen Record."
        ),
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help=(
            "Optionaler Kameraindex für Store-Dateien mit mehreren Kameras. "
            "Wenn gesetzt, wird versucht, den passenden Record aus einer Store-Struktur zu lesen."
        ),
    )
    parser.add_argument(
        "--board-polygon",
        default=None,
        help=(
            "Optionaler Pfad zu einer JSON-Datei mit Polygonpunkten für die ROI. "
            "Format: [[x1, y1], [x2, y2], ...]"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="debug_output/single_cam",
        help="Ausgabeordner für Overlay, Debugbilder und JSON-Ergebnis.",
    )
    parser.add_argument(
        "--save-stage-images",
        action="store_true",
        help="Speichert alle von der Pipeline gelieferten Stage-Debugbilder.",
    )
    parser.add_argument(
        "--no-stage-overlays",
        action="store_true",
        help="Deaktiviert zusätzlich erzeugte Stage-Overlay-Bilder.",
    )
    parser.add_argument(
        "--score-all-estimates",
        action="store_true",
        help="Scort mehrere Impact-Schätzungen statt nur der besten.",
    )
    parser.add_argument(
        "--max-estimates-to-score",
        type=int,
        default=3,
        help="Maximale Anzahl an Impact-Schätzungen, die gescort werden.",
    )
    parser.add_argument(
        "--min-impact-confidence",
        type=float,
        default=0.01,
        help="Mindestkonfidenz für Impact-Schätzungen.",
    )
    parser.add_argument(
        "--min-combined-confidence",
        type=float,
        default=0.01,
        help="Mindestkonfidenz für finale gescorte Ergebnisse.",
    )
    parser.add_argument(
        "--candidate-diff-threshold",
        type=int,
        default=24,
        help="Threshold für die Differenzmaske im Candidate Detector.",
    )
    parser.add_argument(
        "--candidate-min-area",
        type=float,
        default=25.0,
        help="Minimale Konturfläche für Kandidaten.",
    )
    parser.add_argument(
        "--candidate-min-confidence",
        type=float,
        default=0.18,
        help="Minimale Kandidatenkonfidenz im Candidate Detector.",
    )
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
            "directional_contour_tip",
        ],
        help="Strategie des Impact Estimators.",
    )

    return parser


# -----------------------------------------------------------------------------
# Laden / Validieren
# -----------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """
    Lädt ein Bild robust von Platte.
    """
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")

    return image


def load_json(path: str) -> Any:
    """
    Lädt JSON von Platte.
    """
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_board_polygon(path: Optional[str]) -> Optional[list[list[float]]]:
    """
    Lädt optional ein Board-Polygon aus JSON.
    """
    if not path:
        return None

    data = load_json(path)

    if not isinstance(data, list) or len(data) < 3:
        raise ValueError(
            "Board-Polygon muss eine Liste mit mindestens 3 Punkten sein."
        )

    normalized: list[list[float]] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Ungültiger Polygonpunkt: {item!r}")
        normalized.append([float(item[0]), float(item[1])])

    return normalized


def resolve_calibration_record(data: Any, camera_index: Optional[int]) -> Any:
    """
    Extrahiert robust einen Kalibrierungs-Record aus einer JSON-Struktur.

    Unterstützte grobe Formate:
    - einzelner Record mit manual_points
    - dict mit records
    - dict mit cameras
    - dict mit camera_configs
    - dict keyed by camera index / string index

    Wichtiger Punkt:
    Das Script versucht bewusst mehrere Formate robust zu lesen, damit du
    nicht gleich wieder an Dateiformaten hängen bleibst.
    """
    # Direkt ein einzelner Record
    if isinstance(data, dict) and _looks_like_calibration_record(data):
        return data

    # Ohne camera_index: wenn genau ein Record erkennbar ist, nimm ihn
    if camera_index is None:
        single = _try_extract_single_record(data)
        if single is not None:
            return single

    # Mit camera_index gezielt auflösen
    if camera_index is not None:
        resolved = _try_extract_record_by_index(data, camera_index)
        if resolved is not None:
            return resolved

    raise ValueError(
        "Kalibrierungs-JSON konnte nicht in einen einzelnen Kamera-Record aufgelöst werden. "
        "Prüfe das Format oder verwende --camera-index."
    )


def _looks_like_calibration_record(value: Any) -> bool:
    """
    Grobe Heuristik, ob ein Objekt wie ein einzelner Calibration-Record aussieht.
    """
    if not isinstance(value, dict):
        return False

    if "manual_points" in value:
        return True

    point_keys = {"p1", "p2", "p3", "p4"}
    if point_keys.issubset(set(value.keys())):
        return True

    marker_keys = {"marker_points", "markers", "image_points", "points"}
    if marker_keys.intersection(set(value.keys())):
        return True

    return False


def _try_extract_single_record(data: Any) -> Optional[Any]:
    """
    Versucht aus einer Struktur genau einen einzelnen Record zu extrahieren.
    """
    if isinstance(data, dict):
        # records: [...]
        if "records" in data and isinstance(data["records"], list) and len(data["records"]) == 1:
            candidate = data["records"][0]
            if _looks_like_calibration_record(candidate):
                return candidate

        # cameras / camera_configs als dict
        for key in ("cameras", "camera_configs"):
            if key in data and isinstance(data[key], dict) and len(data[key]) == 1:
                only_value = next(iter(data[key].values()))
                if _looks_like_calibration_record(only_value):
                    return only_value

        # cameras / camera_configs als list
        for key in ("cameras", "camera_configs"):
            if key in data and isinstance(data[key], list) and len(data[key]) == 1:
                only_value = data[key][0]
                if _looks_like_calibration_record(only_value):
                    return only_value

        # keyed dict mit genau einem record
        matching_values = [
            value for value in data.values()
            if _looks_like_calibration_record(value)
        ]
        if len(matching_values) == 1:
            return matching_values[0]

    return None


def _try_extract_record_by_index(data: Any, camera_index: int) -> Optional[Any]:
    """
    Versucht einen Record gezielt nach Kameraindex zu extrahieren.
    """
    index_str = str(camera_index)

    if isinstance(data, dict):
        # records: [...]
        if "records" in data and isinstance(data["records"], list):
            records = data["records"]
            if 0 <= camera_index < len(records):
                candidate = records[camera_index]
                if _looks_like_calibration_record(candidate):
                    return candidate

        # cameras / camera_configs als list
        for key in ("cameras", "camera_configs"):
            if key in data and isinstance(data[key], list):
                items = data[key]
                if 0 <= camera_index < len(items):
                    candidate = items[camera_index]
                    if _looks_like_calibration_record(candidate):
                        return candidate

        # cameras / camera_configs als dict
        for key in ("cameras", "camera_configs"):
            if key in data and isinstance(data[key], dict):
                container = data[key]
                if index_str in container and _looks_like_calibration_record(container[index_str]):
                    return container[index_str]
                camera_key = f"cam_{camera_index}"
                if camera_key in container and _looks_like_calibration_record(container[camera_key]):
                    return container[camera_key]

        # direkt keyed dict
        if index_str in data and _looks_like_calibration_record(data[index_str]):
            return data[index_str]

        camera_key = f"cam_{camera_index}"
        if camera_key in data and _looks_like_calibration_record(data[camera_key]):
            return data[camera_key]

    return None


# -----------------------------------------------------------------------------
# Debug / Speichern
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """
    Erstellt einen Ordner rekursiv, falls nötig.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_image(path: Path, image: np.ndarray) -> None:
    """
    Speichert ein Bild und wirft bei Fehlern eine klare Exception.
    """
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"Bild konnte nicht gespeichert werden: {path}")


def make_json_safe(value: Any) -> Any:
    """
    Wandelt komplexere Python-/NumPy-Werte in JSON-kompatible Strukturen um.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return make_json_safe(value.to_dict())

    return str(value)


def print_console_summary(result: SingleCamDetectionResult) -> None:
    """
    Gibt eine kompakte, aber brauchbare Konsolenzusammenfassung aus.
    """
    print("\n===== SINGLE CAM DEBUG RESULT =====")
    print(f"Best Label: {result.best_label}")
    print(f"Best Score: {result.best_score}")
    print(f"Scored Estimates: {len(result.scored_estimates)}")

    best = result.best_estimate
    if best is None:
        print("Kein finales Ergebnis gefunden.")
        return

    print(f"Best Candidate ID: {best.candidate_id}")
    print(f"Image Point: {best.image_point}")
    print(f"Candidate Confidence: {best.candidate_confidence:.4f}")
    print(f"Impact Confidence: {best.impact_confidence:.4f}")
    print(f"Combined Confidence: {best.combined_confidence:.4f}")
    print(f"Impact Method: {best.impact_estimate.method}")
    print(f"Hypothesis Count: {best.impact_estimate.hypothesis_count}")

    if result.candidate_result is not None:
        print(f"Candidate Count: {len(result.candidate_result.candidates)}")

    if result.impact_result is not None:
        print(f"Impact Count: {len(result.impact_result.estimates)}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    """
    CLI-Einstieg.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    frame = load_image(args.frame)
    reference = load_image(args.reference)

    calibration_data = load_json(args.calibration)
    calibration_record = resolve_calibration_record(
        calibration_data,
        camera_index=args.camera_index,
    )

    board_polygon = load_board_polygon(args.board_polygon)

    candidate_config = CandidateDetectorConfig(
        diff_threshold=int(args.candidate_diff_threshold),
        min_contour_area=float(args.candidate_min_area),
        min_confidence=float(args.candidate_min_confidence),
        keep_debug_images=bool(args.save_stage_images),
    )

    impact_config = ImpactEstimatorConfig(
        strategy=str(args.impact_strategy),
    )

    single_cam_config = SingleCamDetectorConfig(
        score_all_estimates=bool(args.score_all_estimates),
        max_estimates_to_score=int(args.max_estimates_to_score),
        min_impact_confidence=float(args.min_impact_confidence),
        min_combined_confidence=float(args.min_combined_confidence),
        keep_debug_images=bool(args.save_stage_images),
        keep_stage_results=True,
        render_stage_overlays=not bool(args.no_stage_overlays),
    )

    detector = SingleCamDetector(
        config=single_cam_config,
        candidate_detector_config=candidate_config,
        impact_estimator_config=impact_config,
        calibration_record=calibration_record,
    )

    result = detector.detect(
        frame=frame,
        reference_frame=reference,
        board_polygon=board_polygon,
    )

    print_console_summary(result)

    # Finales Overlay speichern
    final_overlay = result.render_debug_overlay(frame)
    save_image(output_dir / "single_cam_overlay.png", final_overlay)

    # Stage-Debugbilder optional speichern
    if args.save_stage_images:
        for name, image in result.debug_images.items():
            suffix = ".png"
            save_image(output_dir / f"{name}{suffix}", image)

    # Ergebnis als JSON speichern
    result_json = {
        "frame_path": str(Path(args.frame).resolve()),
        "reference_path": str(Path(args.reference).resolve()),
        "calibration_path": str(Path(args.calibration).resolve()),
        "board_polygon_path": None if not args.board_polygon else str(Path(args.board_polygon).resolve()),
        "best_label": result.best_label,
        "best_score": result.best_score,
        "result": make_json_safe(result.to_dict()),
    }

    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    print(f"\nAusgabe gespeichert in: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())