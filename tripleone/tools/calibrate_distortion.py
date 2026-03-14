# tools/calibrate_distortion.py
# Dieses Tool kalibriert die Linsenverzerrung für bis zu 3 Kameras
# aus Checkerboard-Bildern und speichert das Ergebnis in config/distortion.json.
#
# Beispiel:
# py tools/calibrate_distortion.py --cam1 calibration_images/cam1 --cam2 calibration_images/cam2 --cam3 calibration_images/cam3 --cols 9 --rows 6 --square-size 25

from __future__ import annotations

import argparse
from pathlib import Path

from config.distortion_settings import load_distortion, save_distortion
from vision.distortion import calibrate_from_checkerboard_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="Distortion-Kalibrierung für TripleOne")
    parser.add_argument("--cam1", type=str, default=None, help="Ordner mit Checkerboard-Bildern für Kamera 1")
    parser.add_argument("--cam2", type=str, default=None, help="Ordner mit Checkerboard-Bildern für Kamera 2")
    parser.add_argument("--cam3", type=str, default=None, help="Ordner mit Checkerboard-Bildern für Kamera 3")
    parser.add_argument("--cols", type=int, default=9, help="Anzahl innerer Ecken horizontal")
    parser.add_argument("--rows", type=int, default=6, help="Anzahl innerer Ecken vertikal")
    parser.add_argument("--square-size", type=float, default=25.0, help="Kantenlänge eines Quadrats in mm")

    args = parser.parse_args()

    distortion_data = load_distortion()

    cam_args = [args.cam1, args.cam2, args.cam3]

    for idx, folder_str in enumerate(cam_args):
        if not folder_str:
            continue

        folder = Path(folder_str)
        if not folder.exists() or not folder.is_dir():
            print(f"[WARN] Kamera {idx + 1}: Ordner nicht gefunden: {folder}")
            continue

        print(f"[INFO] Kalibriere Kamera {idx + 1} aus: {folder}")

        try:
            result = calibrate_from_checkerboard_folder(
                folder=folder,
                board_cols=args.cols,
                board_rows=args.rows,
                square_size=args.square_size,
            )
        except Exception as exc:
            print(f"[ERROR] Kamera {idx + 1}: {exc}")
            continue

        distortion_data["cameras"][idx]["enabled"] = True
        distortion_data["cameras"][idx]["camera_matrix"] = result["camera_matrix"]
        distortion_data["cameras"][idx]["dist_coeffs"] = result["dist_coeffs"]
        distortion_data["cameras"][idx]["image_width"] = result["image_width"]
        distortion_data["cameras"][idx]["image_height"] = result["image_height"]
        distortion_data["cameras"][idx]["reprojection_error"] = result["reprojection_error"]
        distortion_data["cameras"][idx]["source_count"] = result["source_count"]

        print(
            f"[OK] Kamera {idx + 1}: "
            f"Bilder={result['source_count']}, "
            f"RMS={result['reprojection_error']:.4f}, "
            f"Size={result['image_width']}x{result['image_height']}"
        )

    save_distortion(distortion_data)
    print("[DONE] Distortion-Profile gespeichert in config/distortion.json")


if __name__ == "__main__":
    main()
