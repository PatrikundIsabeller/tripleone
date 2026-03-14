# vision/distortion.py
# Diese Datei enthält die Distortion-/Linsenkorrektur-Funktionen.
#
# Phase 5.3:
# - Checkerboard-Kalibrierung aus Bildern
# - Prüfung, ob ein Profil gültig ist
# - Frame-Undistortion für Preview, Detection und Fusion

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


def has_valid_distortion(profile: Dict) -> bool:
    return bool(
        isinstance(profile, dict)
        and profile.get("camera_matrix") is not None
        and profile.get("dist_coeffs") is not None
        and profile.get("image_width") is not None
        and profile.get("image_height") is not None
    )


def profile_to_numpy(profile: Dict) -> Tuple[np.ndarray, np.ndarray]:
    camera_matrix = np.array(profile["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(profile["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def undistort_frame(frame_bgr: np.ndarray, profile: Dict) -> np.ndarray:
    """
    Entzerrt einen Frame anhand des gespeicherten Kameraprofils.
    """
    if not has_valid_distortion(profile):
        return frame_bgr

    camera_matrix, dist_coeffs = profile_to_numpy(profile)

    h, w = frame_bgr.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted


def _collect_image_paths(folder: Path) -> List[Path]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in supported])


def calibrate_from_checkerboard_folder(
    folder: Path,
    board_cols: int = 9,
    board_rows: int = 6,
    square_size: float = 25.0,
) -> Dict:
    """
    Kalibriert eine Kamera aus mehreren Checkerboard-Bildern.
    board_cols / board_rows = innere Ecken.
    """
    image_paths = _collect_image_paths(folder)
    if not image_paths:
        raise ValueError(f"Keine Bilder im Ordner gefunden: {folder}")

    pattern_size = (board_cols, board_rows)

    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= float(square_size)

    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []

    image_size: Optional[Tuple[int, int]] = None
    used_images = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not found:
            continue

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners_refined)
        used_images += 1

    if used_images < 6:
        raise ValueError(
            f"Zu wenige gültige Checkerboard-Bilder gefunden ({used_images}). "
            "Empfohlen sind mindestens 6–10 gute Bilder."
        )

    if image_size is None:
        raise ValueError("Keine gültige Bildgröße gefunden.")

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )

    return {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "reprojection_error": float(rms),
        "source_count": int(used_images),
    }
