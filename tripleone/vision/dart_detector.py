# vision/dart_detector.py
# Diese Datei enthält die automatische Dart-Erkennung über
# inkrementelle Frame-Differenz, Board-Maske, Konturerkennung
# und robuste Spitzen-Schätzung.
#
# Phase 4.3:
# - inkrementelle Referenz
# - Boardbereich wird maskiert
# - robuste Konturfilter
# - stabilere Spitzen-Schätzung
# - bis zu 3 Darts nacheinander erkennen

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    Robuste Einzeldart-Erkennung für bis zu 3 Darts nacheinander.
    """

    def __init__(self) -> None:
        self.reference_gray: Optional[np.ndarray] = None
        self.is_armed: bool = False

        self.max_darts_per_round: int = 3

        self.diff_threshold: int = 22

        self.min_contour_area: float = 110.0
        self.max_contour_area: float = 14000.0

        self.required_stable_frames: int = 3
        self.match_distance_px: float = 16.0

        self.arm_grace_seconds: float = 0.8
        self.post_hit_cooldown_seconds: float = 0.9

        self._armed_at: float = 0.0
        self._cooldown_until: float = 0.0

        self._last_candidate: Optional[Tuple[int, int]] = None
        self._stable_counter: int = 0

        self.detected_darts: List[DartDetectionResult] = []

    def reset_round(self) -> None:
        self.is_armed = False
        self._armed_at = 0.0
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        self.detected_darts = []

    def clear_reference(self) -> None:
        self.reference_gray = None
        self.reset_round()

    def set_reference_frame(self, frame_bgr: np.ndarray) -> None:
        self.reference_gray = self._to_preprocessed_gray(frame_bgr)
        self.reset_round()

    def arm(self) -> bool:
        if self.reference_gray is None:
            return False

        self.is_armed = True
        self._armed_at = time.monotonic()
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        return True

    def _to_preprocessed_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _build_board_mask(self, frame_bgr: np.ndarray, calibration: Dict) -> np.ndarray:
        """
        Erzeugt eine Maske, die nur den Dartboard-Bereich zulässt.
        Verwendet die 4 Boundary-Punkte und erweitert sie leicht.
        """
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        frame_points = calibration.get("points", [])
        if len(frame_points) < 5:
            return np.full((h, w), 255, dtype=np.uint8)

        pts = []
        for item in frame_points[:4]:
            if isinstance(item, dict):
                pts.append([int(item.get("x_px", 0)), int(item.get("y_px", 0))])

        if len(pts) != 4:
            return np.full((h, w), 255, dtype=np.uint8)

        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(mask, pts_np, 255, lineType=cv2.LINE_AA)

        # etwas erweitern, damit Flights / äußere Bereiche nicht abgeschnitten werden
        kernel = np.ones((31, 31), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _build_diff_mask(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[np.ndarray]:
        if self.reference_gray is None:
            return None

        current_gray = self._to_preprocessed_gray(frame_bgr)
        diff = cv2.absdiff(current_gray, self.reference_gray)

        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        board_mask = self._build_board_mask(frame_bgr, calibration)
        mask = cv2.bitwise_and(mask, board_mask)

        kernel_small = np.ones((3, 3), np.uint8)
        kernel_big = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.dilate(mask, kernel_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_big)

        return mask

    def _contour_shape_ok(self, contour: np.ndarray) -> bool:
        area = cv2.contourArea(contour)
        if area < self.min_contour_area or area > self.max_contour_area:
            return False

        x, y, w, h = cv2.boundingRect(contour)
        short_side = max(1, min(w, h))
        long_side = max(w, h)
        aspect_ratio = long_side / short_side

        if aspect_ratio < 2.0:
            return False

        rect_area = max(1, w * h)
        fill_ratio = area / rect_area

        # Sehr volle kompakte Flächen sind meist Schatten/Hand/Fehlobjekte
        if fill_ratio > 0.72:
            return False

        # Zu wenig Substanz ist oft Rauschen
        if fill_ratio < 0.08:
            return False

        return True

    def _find_candidate_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours: List[np.ndarray] = []
        for contour in contours:
            if self._contour_shape_ok(contour):
                valid_contours.append(contour)

        valid_contours.sort(key=cv2.contourArea, reverse=True)
        return valid_contours

    def _get_axis_endpoints(self, contour: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Bestimmt über PCA die Hauptachse und liefert die beiden extremen Endpunkte
        entlang dieser Achse.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return None

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        if mean is None or eigenvectors is None or len(eigenvectors) == 0:
            return None

        center = mean[0]
        direction = eigenvectors[0]

        projections = np.dot(pts - center, direction)
        min_idx = int(np.argmin(projections))
        max_idx = int(np.argmax(projections))

        p1 = pts[min_idx]
        p2 = pts[max_idx]
        return p1, p2

    def _board_radius_of_pixel(self, x_px: int, y_px: int, calibration: Dict) -> Optional[float]:
        projected = project_image_point_to_board(x_px, y_px, calibration)
        if projected is None:
            return None

        x_board, y_board = projected
        return math.sqrt(x_board * x_board + y_board * y_board)

    def _estimate_tip_from_contour(self, contour: np.ndarray, calibration: Dict) -> Optional[Tuple[int, int]]:
        """
        Robuste Spitzen-Schätzung:
        - Dart-Hauptachse per PCA
        - beide Enden der Achse bestimmen
        - Ende näher zum Boardzentrum ist Kandidat für die Spitze
        - zusätzliche Plausibilitätsprüfung
        """
        endpoints = self._get_axis_endpoints(contour)
        if endpoints is None:
            return None

        p1, p2 = endpoints

        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        x2, y2 = int(round(float(p2[0]))), int(round(float(p2[1])))

        r1 = self._board_radius_of_pixel(x1, y1, calibration)
        r2 = self._board_radius_of_pixel(x2, y2, calibration)

        if r1 is None or r2 is None:
            return None

        # Plausibilitätsfilter: beide Endpunkte dürfen nicht komplett unplausibel weit draußen sein
        if r1 > 1.35 and r2 > 1.35:
            return None

        # Das Ende näher zum Zentrum ist typischerweise die Spitze
        if r1 <= r2:
            tip = (x1, y1)
            tip_radius = r1
        else:
            tip = (x2, y2)
            tip_radius = r2

        # Die Spitze darf nicht komplett außerhalb des Boards liegen
        if tip_radius > 1.10:
            return None

        return tip

    def _candidate_is_new(self, x_px: int, y_px: int) -> bool:
        """
        Verhindert doppelte Erkennung an fast identischer Stelle.
        """
        for dart in self.detected_darts:
            dx = x_px - dart.x_px
            dy = y_px - dart.y_px
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= 22.0:
                return False
        return True

    def _update_reference_after_hit(self, frame_bgr: np.ndarray) -> None:
        self.reference_gray = self._to_preprocessed_gray(frame_bgr)

    def process_frame(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[DartDetectionResult]:
        if not self.is_armed:
            return None

        if self.reference_gray is None:
            return None

        if len(self.detected_darts) >= self.max_darts_per_round:
            return None

        now = time.monotonic()

        if (now - self._armed_at) < self.arm_grace_seconds:
            return None

        if now < self._cooldown_until:
            return None

        mask = self._build_diff_mask(frame_bgr, calibration)
        if mask is None:
            return None

        contours = self._find_candidate_contours(mask)
        if not contours:
            self._last_candidate = None
            self._stable_counter = 0
            return None

        best_tip: Optional[Tuple[int, int]] = None
        best_area: float = 0.0

        for contour in contours:
            tip = self._estimate_tip_from_contour(contour, calibration)
            if tip is None:
                continue

            tx, ty = tip

            if not self._candidate_is_new(tx, ty):
                continue

            best_tip = tip
            best_area = float(cv2.contourArea(contour))
            break

        if best_tip is None:
            self._last_candidate = None
            self._stable_counter = 0
            return None

        tx, ty = best_tip

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

        # Miss wird in dieser Phase nicht akzeptiert
        if hit.ring_name == "MISS":
            self._last_candidate = None
            self._stable_counter = 0
            return None

        result = DartDetectionResult(
            x_px=tx,
            y_px=ty,
            contour_area=best_area,
            score_label=hit.label,
            score_value=hit.score,
            ring_name=hit.ring_name,
            board_x=hit.board_x,
            board_y=hit.board_y,
            radius=hit.radius,
        )

        self.detected_darts.append(result)

        # WICHTIG: aktueller Zustand wird Referenz für den nächsten Dart
        self._update_reference_after_hit(frame_bgr)

        self._cooldown_until = time.monotonic() + self.post_hit_cooldown_seconds
        self._last_candidate = None
        self._stable_counter = 0

        if len(self.detected_darts) >= self.max_darts_per_round:
            self.is_armed = False

        return result
