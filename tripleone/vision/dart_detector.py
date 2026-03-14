# vision/dart_detector.py
# Diese Datei enthält die automatische Dart-Erkennung über
# inkrementelle Frame-Differenz, Board-Maske, Konturerkennung
# und Spitzen-Schätzung.
#
# Phase 4.1:
# - Referenzbild vom leeren Board
# - danach automatische Referenzaktualisierung nach jedem Treffer
# - bis zu 3 Darts nacheinander erkennen
# - Boardbereich wird per Homography eingeschränkt
# - Cooldown und Sperrzeiten gegen Fehltrigger

# vision/dart_detector.py

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from vision.board_model import calculate_board_hit_from_image_point


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

    def __init__(self) -> None:

        self.reference_gray = None
        self.is_armed = False

        # maximale Darts pro Runde
        self.max_darts_per_round = 3

        self.diff_threshold = 22

        self.min_contour_area = 120
        self.max_contour_area = 15000

        self.required_stable_frames = 3
        self.match_distance_px = 18

        self.arm_grace_seconds = 0.8
        self.cooldown_seconds = 1.0

        self._armed_at = 0
        self._cooldown_until = 0

        self._last_candidate = None
        self._stable_counter = 0

        self.detected_darts = []

    def reset_round(self):

        self.is_armed = False
        self.detected_darts.clear()
        self._last_candidate = None
        self._stable_counter = 0

    def set_reference_frame(self, frame):

        self.reference_gray = self._to_gray(frame)
        self.reset_round()

    def arm(self):

        if self.reference_gray is None:
            return False

        self.is_armed = True
        self._armed_at = time.monotonic()
        return True

    def _to_gray(self, frame):

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)
        return g

    def _diff_mask(self, frame):

        g = self._to_gray(frame)

        diff = cv2.absdiff(g, self.reference_gray)

        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _find_dart_contour(self, mask):

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0

        for c in contours:

            area = cv2.contourArea(c)

            if area < self.min_contour_area:
                continue

            if area > self.max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(c)

            aspect = max(w, h) / (min(w, h) + 1)

            if aspect < 2.2:
                continue

            if area > best_area:

                best_area = area
                best = c

        return best

    def _estimate_tip(self, contour, calibration):

        pts = contour.reshape(-1, 2).astype(np.float32)

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)

        direction = eigenvectors[0]

        center = mean[0]

        projections = []

        for p in pts:

            v = p - center

            proj = np.dot(v, direction)

            projections.append(proj)

        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)

        p1 = pts[min_idx]
        p2 = pts[max_idx]

        board_center = calibration["points"][4]

        cx = board_center["x_px"]
        cy = board_center["y_px"]

        d1 = math.hypot(p1[0] - cx, p1[1] - cy)
        d2 = math.hypot(p2[0] - cx, p2[1] - cy)

        if d1 < d2:
            tip = p1
        else:
            tip = p2

        return int(tip[0]), int(tip[1])

    def process_frame(self, frame, calibration):

        if not self.is_armed:
            return None

        now = time.monotonic()

        if now - self._armed_at < self.arm_grace_seconds:
            return None

        if now < self._cooldown_until:
            return None

        mask = self._diff_mask(frame)

        contour = self._find_dart_contour(mask)

        if contour is None:
            self._last_candidate = None
            self._stable_counter = 0
            return None

        tip = self._estimate_tip(contour, calibration)

        tx, ty = tip

        if self._last_candidate is None:

            self._last_candidate = tip
            self._stable_counter = 1
            return None

        dx = tx - self._last_candidate[0]
        dy = ty - self._last_candidate[1]

        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.match_distance_px:
            self._stable_counter += 1
        else:
            self._stable_counter = 1

        self._last_candidate = tip

        if self._stable_counter < self.required_stable_frames:
            return None

        hit = calculate_board_hit_from_image_point(tx, ty, calibration)

        if hit is None:
            return None

        if hit.ring_name == "MISS":
            return None

        result = DartDetectionResult(
            x_px=tx,
            y_px=ty,
            contour_area=cv2.contourArea(contour),
            score_label=hit.label,
            score_value=hit.score,
            ring_name=hit.ring_name,
            board_x=hit.board_x,
            board_y=hit.board_y,
            radius=hit.radius,
        )

        self.detected_darts.append(result)

        self.reference_gray = self._to_gray(frame)

        self._cooldown_until = time.monotonic() + self.cooldown_seconds

        self._stable_counter = 0
        self._last_candidate = None

        return result
