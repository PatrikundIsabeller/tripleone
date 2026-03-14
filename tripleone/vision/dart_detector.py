# vision/dart_detector.py
# Diese Datei enthält die erste automatische Dart-Erkennung über
# Frame-Differenz, Konturerkennung und Spitzen-Schätzung.
#
# Phase 4.0:
# - Referenzbild vom leeren Board
# - neue Bewegung / Objekt erkennen
# - Dart-Kontur finden
# - wahrscheinliche Spitze bestimmen
# - Score über bestehende Board-Geometrie berechnen

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from vision.board_model import calculate_board_hit_from_image_point, project_image_point_to_board


@dataclass
class DartDetectionResult:
    x_px: int
    y_px: int
    contour_area: float
    score_label: str
    score_value: int
    ring_name: str
    board_x: float
    board_y: float
    radius: float


class DartDetector:
    """
    Erste Dart-Erkennung für ein einzelnes neues Dartobjekt.
    Workflow:
    1. set_reference_frame() mit leerem Board
    2. arm()
    3. process_frame() auf neue Frames anwenden
    4. bei stabiler Erkennung Ergebnis zurückgeben
    """

    def __init__(self) -> None:
        self.reference_gray: Optional[np.ndarray] = None
        self.is_armed: bool = False

        self.diff_threshold: int = 28
        self.min_contour_area: float = 140.0
        self.max_contour_area: float = 25000.0

        self.required_stable_frames: int = 3
        self.match_distance_px: float = 18.0

        self._last_candidate: Optional[Tuple[int, int]] = None
        self._stable_counter: int = 0
        self._locked: bool = False

    def reset(self) -> None:
        self.is_armed = False
        self._last_candidate = None
        self._stable_counter = 0
        self._locked = False

    def clear_reference(self) -> None:
        self.reference_gray = None
        self.reset()

    def set_reference_frame(self, frame_bgr: np.ndarray) -> None:
        """
        Speichert ein Referenzbild des leeren Boards.
        """
        gray = self._to_preprocessed_gray(frame_bgr)
        self.reference_gray = gray
        self.reset()

    def arm(self) -> bool:
        """
        Scharf schalten. Nur möglich, wenn ein Referenzbild vorhanden ist.
        """
        if self.reference_gray is None:
            return False

        self.is_armed = True
        self._last_candidate = None
        self._stable_counter = 0
        self._locked = False
        return True

    def _to_preprocessed_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _build_diff_mask(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if self.reference_gray is None:
            return None

        current_gray = self._to_preprocessed_gray(frame_bgr)
        diff = cv2.absdiff(current_gray, self.reference_gray)

        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_big = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

        return mask

    def _find_best_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_area = 0.0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            if area > self.max_contour_area:
                continue
            if area > best_area:
                best_area = area
                best_contour = contour

        return best_contour

    def _estimate_tip_from_contour(self, contour: np.ndarray, calibration: Dict) -> Optional[Tuple[int, int]]:
        """
        Schätzt die Dartspitze aus der Kontur.
        Idee:
        - Alle Konturpunkte ins normierte Board projizieren
        - Der Punkt mit dem kleinsten Radius ist am ehesten die Spitze,
          weil die Dartspitze am nächsten zum Board-Zentrum liegt.
        """
        pts = contour.reshape(-1, 2)

        best_pt: Optional[Tuple[int, int]] = None
        best_radius = float("inf")

        for p in pts:
            x_px = int(p[0])
            y_px = int(p[1])

            projected = project_image_point_to_board(x_px, y_px, calibration)
            if projected is None:
                continue

            x_board, y_board = projected
            radius = math.sqrt(x_board * x_board + y_board * y_board)

            # harte Begrenzung, damit wir keine komplett unplausiblen Punkte nehmen
            if radius > 1.35:
                continue

            if radius < best_radius:
                best_radius = radius
                best_pt = (x_px, y_px)

        return best_pt

    def process_frame(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[DartDetectionResult]:
        """
        Prüft einen neuen Frame.
        Gibt genau dann ein Ergebnis zurück, wenn eine stabile neue Dart-Erkennung
        gefunden wurde.
        """
        if not self.is_armed:
            return None

        if self.reference_gray is None:
            return None

        if self._locked:
            return None

        mask = self._build_diff_mask(frame_bgr)
        if mask is None:
            return None

        contour = self._find_best_contour(mask)
        if contour is None:
            self._last_candidate = None
            self._stable_counter = 0
            return None

        area = float(cv2.contourArea(contour))
        tip = self._estimate_tip_from_contour(contour, calibration)
        if tip is None:
            self._last_candidate = None
            self._stable_counter = 0
            return None

        tx, ty = tip

        if self._last_candidate is None:
            self._last_candidate = (tx, ty)
            self._stable_counter = 1
            return None

        dx = tx - self._last_candidate[0]
        dy = ty - self._last_candidate[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= self.match_distance_px:
            self._stable_counter += 1
        else:
            self._stable_counter = 1

        self._last_candidate = (tx, ty)

        if self._stable_counter < self.required_stable_frames:
            return None

        hit = calculate_board_hit_from_image_point(tx, ty, calibration)
        if hit is None:
            return None

        self._locked = True
        self.is_armed = False

        return DartDetectionResult(
            x_px=tx,
            y_px=ty,
            contour_area=area,
            score_label=hit.label,
            score_value=hit.score,
            ring_name=hit.ring_name,
            board_x=hit.board_x,
            board_y=hit.board_y,
            radius=hit.radius,
        )