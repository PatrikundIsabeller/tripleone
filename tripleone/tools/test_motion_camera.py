from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.dart_motion_detector import (
    DartMotionDetector,
    DartMotionDetectorConfig,
)
from vision.score_mapper import ScoreMapper


WINDOW_NAME = "TripleOne Motion + Score Test"


def _resize_for_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def _label_panel(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 38),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        canvas,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas


def _open_camera(
    device_id: int,
    width: int,
    height: int,
    fps: int,
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(
        device_id,
        cv2.CAP_DSHOW,
    )

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        raise RuntimeError(
            f"Kamera {device_id} konnte nicht geöffnet werden."
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        int(width),
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        int(height),
    )

    cap.set(
        cv2.CAP_PROP_FPS,
        int(fps),
    )

    return cap


def _load_calibration_for_camera(
    camera_index: int,
) -> tuple[list[tuple[float, float]], tuple[float, float], tuple[int, int], str]:
    calibration_path = (
        PROJECT_ROOT
        / "config"
        / "calibration.json"
    )

    if not calibration_path.exists():
        raise FileNotFoundError(
            f"Kalibrierungsdatei nicht gefunden: {calibration_path}"
        )

    data = json.loads(
        calibration_path.read_text(
            encoding="utf-8",
        )
    )

    cameras = data.get(
        "cameras",
        [],
    )

    if not (
        0 <= int(camera_index) < len(cameras)
    ):
        raise IndexError(
            f"Keine Kalibrierung für Kamera-Index {camera_index} vorhanden."
        )

    camera_record = cameras[
        int(camera_index)
    ]

    points = camera_record.get(
        "points",
        [],
    )

    if len(points) < 5:
        raise ValueError(
            f"Kamera {camera_index}: mindestens 5 Kalibrierungspunkte erwartet, "
            f"gefunden: {len(points)}"
        )

    # ScoreMapper arbeitet mit der 4-Punkt-Kalibrierung.
    manual_points = [
        (
            float(point["x_px"]),
            float(point["y_px"]),
        )
        for point in points[:4]
    ]

    # 5. Punkt ist der gespeicherte Board-/Bull-Mittelpunkt.
    center_record = points[4]

    board_center = (
        float(center_record["x_px"]),
        float(center_record["y_px"]),
    )

    image_size = (
        int(camera_record.get("frame_width", 1280)),
        int(camera_record.get("frame_height", 720)),
    )

    camera_name = str(
        camera_record.get(
            "name",
            f"Kamera {camera_index + 1}",
        )
    )

    return (
        manual_points,
        board_center,
        image_size,
        camera_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TripleOne: isolierter Motion-/Geometrie-/Score-Test "
            "ohne YOLO und ohne Fusion."
        )
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Physische Kamera-ID / Kalibrierungsindex: 0, 1 oder 2.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    (
        manual_points,
        board_center,
        calibration_image_size,
        camera_name,
    ) = _load_calibration_for_camera(
        args.camera
    )

    score_mapper = ScoreMapper(
        manual_points=manual_points,
        image_size=calibration_image_size,
    )

    detector = DartMotionDetector(
        DartMotionDetectorConfig(
            analysis_scale=0.50,
            diff_threshold=18,
            blur_kernel_size=5,
            min_contour_area=22.0,
            roi_margin_px=45,
            keep_debug_images=True,
        )
    )

    cap = _open_camera(
        device_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    reference_frame: Optional[np.ndarray] = None
    last_composite: Optional[np.ndarray] = None

    print()
    print("TripleOne Motion + Score Test")
    print("--------------------------------")
    print(f"Kamera: {camera_name} / Index {args.camera}")
    print(f"Boardzentrum: {board_center}")
    print(f"Kalibrierung: {manual_points}")
    print()
    print("Tasten:")
    print("  R = aktuelles Bild als LEERBOARD speichern")
    print("  C = Referenz löschen")
    print("  S = Screenshot speichern")
    print("  Q / ESC = beenden")
    print()
    print("Kein B mehr notwendig.")
    print("Bull-Mittelpunkt und ScoreMapper werden aus config/calibration.json geladen.")
    print()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    panel_width = 640
    panel_height = 360

    try:
        while True:
            ok, frame = cap.read()

            if not ok or frame is None:
                print(
                    "[ERROR] Kameraframe konnte nicht gelesen werden."
                )
                break

            live = frame.copy()

            # Boardzentrum auch im LIVE-Panel anzeigen.
            center_x = int(
                round(board_center[0])
            )
            center_y = int(
                round(board_center[1])
            )

            cv2.drawMarker(
                live,
                (
                    center_x,
                    center_y,
                ),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                18,
                1,
            )

            score_text = "SCORE: -"
            topdown_text = "TOP: -"

            if reference_frame is None:
                empty = np.zeros_like(
                    live
                )

                live_panel = _label_panel(
                    _resize_for_panel(
                        live,
                        panel_width,
                        panel_height,
                    ),
                    "LIVE",
                )

                reference_panel = _label_panel(
                    _resize_for_panel(
                        empty,
                        panel_width,
                        panel_height,
                    ),
                    "REFERENCE - R drücken",
                )

                diff_panel = _label_panel(
                    _resize_for_panel(
                        empty,
                        panel_width,
                        panel_height,
                    ),
                    "DIFF / MASK",
                )

                overlay_panel = _label_panel(
                    _resize_for_panel(
                        live,
                        panel_width,
                        panel_height,
                    ),
                    "MOTION OVERLAY | SCORE: -",
                )

            else:
                result = detector.detect(
                    current_frame=frame,
                    reference_frame=reference_frame,
                    board_mask=None,
                    board_center_image=board_center,
                )

                diff_image = result.debug_images[
                    "motion_diff"
                ]

                mask_image = result.debug_images[
                    "motion_cleaned"
                ]

                overlay_image = result.debug_images[
                    "motion_overlay"
                ].copy()

                scored_hit = None

                if result.dart_geometry is not None:
                    impact = (
                        float(
                            result.dart_geometry.impact_point[0]
                        ),
                        float(
                            result.dart_geometry.impact_point[1]
                        ),
                    )

                    try:
                        scored_hit = score_mapper.score_image_point(
                            impact
                        )

                        score_text = (
                            f"SCORE: {scored_hit.label} "
                            f"({scored_hit.score})"
                        )

                        if scored_hit.topdown_point is not None:
                            topdown_text = (
                                "TOP: "
                                f"({scored_hit.topdown_point[0]:.1f}, "
                                f"{scored_hit.topdown_point[1]:.1f})"
                            )

                        cv2.putText(
                            overlay_image,
                            score_text,
                            (12, 88),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.78,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.putText(
                            overlay_image,
                            topdown_text,
                            (12, 116),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

                    except Exception as exc:
                        score_text = (
                            f"SCORE ERROR: "
                            f"{type(exc).__name__}"
                        )

                        cv2.putText(
                            overlay_image,
                            score_text,
                            (12, 88),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.60,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                live_panel = _label_panel(
                    _resize_for_panel(
                        live,
                        panel_width,
                        panel_height,
                    ),
                    "LIVE",
                )

                reference_panel = _label_panel(
                    _resize_for_panel(
                        reference_frame,
                        panel_width,
                        panel_height,
                    ),
                    "REFERENCE",
                )

                diff_panel = _label_panel(
                    _resize_for_panel(
                        diff_image,
                        panel_width,
                        panel_height,
                    ),
                    (
                        "DIFF / MASK "
                        f"regions={len(result.regions)} "
                        f"ratio={result.changed_pixel_ratio:.4f}"
                    ),
                )

                mask_bgr = _resize_for_panel(
                    mask_image,
                    panel_width,
                    panel_height,
                )

                diff_panel = cv2.addWeighted(
                    diff_panel,
                    0.75,
                    mask_bgr,
                    0.25,
                    0.0,
                )

                geometry_text = (
                    "NO DART GEOMETRY"
                )

                if result.dart_geometry is not None:
                    impact = (
                        result.dart_geometry.impact_point
                    )

                    geometry_text = (
                        f"IMPACT=({impact[0]:.1f},{impact[1]:.1f}) "
                        f"conf={result.dart_geometry.confidence:.2f}"
                    )

                overlay_panel = _label_panel(
                    _resize_for_panel(
                        overlay_image,
                        panel_width,
                        panel_height,
                    ),
                    (
                        f"{geometry_text} | "
                        f"{score_text}"
                    ),
                )

            top = np.hstack(
                (
                    live_panel,
                    reference_panel,
                )
            )

            bottom = np.hstack(
                (
                    diff_panel,
                    overlay_panel,
                )
            )

            composite = np.vstack(
                (
                    top,
                    bottom,
                )
            )

            last_composite = composite

            cv2.imshow(
                WINDOW_NAME,
                composite,
            )

            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key in (
                27,
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("r"),
                ord("R"),
            ):
                reference_frame = frame.copy()

                print(
                    "[REFERENCE] Leerboard gespeichert."
                )

            elif key in (
                ord("c"),
                ord("C"),
            ):
                reference_frame = None

                print(
                    "[REFERENCE] Referenz gelöscht."
                )

            elif key in (
                ord("s"),
                ord("S"),
            ):
                if last_composite is None:
                    continue

                output_dir = (
                    Path("debug_output")
                    / "motion_score_test"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                existing = sorted(
                    output_dir.glob(
                        f"cam{args.camera}_score_*.png"
                    )
                )

                next_index = (
                    len(existing) + 1
                )

                output_path = (
                    output_dir
                    / f"cam{args.camera}_score_{next_index:03d}.png"
                )

                cv2.imwrite(
                    str(output_path),
                    last_composite,
                )

                print(
                    f"[SCREENSHOT] gespeichert: {output_path}"
                )

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
