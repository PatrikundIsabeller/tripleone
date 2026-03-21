# tools/capture_single_cam_frame.py
# Zweck:
# Dieses Script öffnet eine einzelne Kamera, zeigt die Live-Vorschau
# und speichert auf Tastendruck genau ein Frame in definierter Auflösung.
#
# Standardziel:
# - Kameraindex 0
# - 1280x720
#
# Typische Nutzung:
# 1) Leeres Board ausrichten
# 2) Taste S drücken -> Referenzbild speichern
# 3) Dart stecken
# 4) Script nochmal starten oder weiterlaufen lassen
# 5) Taste S drücken -> Testbild speichern
#
# WICHTIG:
# Dieses Script ist absichtlich klein und robust.
# Es soll dir konsistente Eingabebilder für die Debug-Pipeline liefern.

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Baut den CLI-Parser.
    """
    parser = argparse.ArgumentParser(
        description="Capture a single frame from one camera in a fixed resolution."
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV-Kameraindex. Standard: 0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Zielbreite der Kamera. Standard: 1280",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Zielhöhe der Kamera. Standard: 720",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Gewünschte FPS. Standard: 30",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optionaler Ausgabepfad. "
            "Wenn nicht gesetzt, wird ein Dateiname mit Timestamp erzeugt."
        ),
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Spiegelt die Vorschau horizontal.",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Dreht Vorschau und gespeichertes Bild. Standard: 0",
    )
    parser.add_argument(
        "--title",
        default="Triple One - Camera Capture",
        help="Fenstertitel der Vorschau.",
    )

    return parser


# -----------------------------------------------------------------------------
# Kamera
# -----------------------------------------------------------------------------

def open_camera(camera_index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """
    Öffnet die Kamera robust unter Windows.
    """
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback ohne CAP_DSHOW
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Kamera {camera_index} konnte nicht geöffnet werden.")

    # Wunschwerte setzen
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    return cap


def rotate_frame(frame, rotate: int):
    """
    Dreht ein Frame in 90°-Schritten.
    """
    if rotate == 0:
        return frame
    if rotate == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unsupported rotate value: {rotate}")


def ensure_parent_dir(path: Path) -> None:
    """
    Erstellt den Zielordner, falls nötig.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def build_default_output_path(camera_index: int) -> Path:
    """
    Erzeugt einen Timestamp-Dateinamen im Projektordner data/captures/.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("data") / "captures" / f"camera_{camera_index}_{timestamp}.png"


def draw_overlay(frame, camera_index: int, width: int, height: int) -> None:
    """
    Zeichnet einfache Bedienhinweise in die Vorschau.
    """
    lines = [
        f"Camera: {camera_index}",
        f"Requested: {width}x{height}",
        "S = speichern",
        "Q / ESC = beenden",
    ]

    x = 20
    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 30


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    """
    Script-Einstieg.
    """
    parser = build_arg_parser()
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else build_default_output_path(args.camera_index)

    cap = open_camera(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )

    try:
        cv2.namedWindow(args.title, cv2.WINDOW_NORMAL)

        print("Kamera geöffnet.")
        print(f"Kameraindex: {args.camera_index}")
        print(f"Gewünschte Auflösung: {args.width}x{args.height}")
        print("Tasten:")
        print("  S = aktuelles Frame speichern")
        print("  Q / ESC = beenden")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Fehler: Frame konnte nicht gelesen werden.")
                continue

            if args.mirror:
                frame = cv2.flip(frame, 1)

            frame = rotate_frame(frame, args.rotate)

            preview = frame.copy()
            draw_overlay(preview, args.camera_index, args.width, args.height)

            actual_h, actual_w = frame.shape[:2]
            cv2.putText(
                preview,
                f"Actual: {actual_w}x{actual_h}",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(args.title, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                print("Beendet.")
                break

            if key in (ord("s"), ord("S")):
                ensure_parent_dir(output_path)
                ok_save = cv2.imwrite(str(output_path), frame)
                if not ok_save:
                    print(f"Fehler: Bild konnte nicht gespeichert werden: {output_path}")
                    continue

                print(f"Bild gespeichert: {output_path.resolve()}")
                print(f"Gespeicherte Bildgröße: {actual_w}x{actual_h}")

        return 0

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())