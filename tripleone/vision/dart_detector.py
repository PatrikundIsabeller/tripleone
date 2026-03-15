# vision/dart_detector.py
# Phase 6.0:
# - Board-Impact Point Detection
# - Dartachse bestimmen
# - entlang der Achse den wahrscheinlichen Einschlagpunkt auf der virtuellen Boardfläche suchen
# - nur dieser Impact-Point wird gewertet

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


@dataclass
class DartDebugSnapshot:
    is_armed: bool
    diff_nonzero_pixels: int
    contour_count_total: int
    contour_count_valid: int
    chosen_area: Optional[float]
    chosen_aspect_ratio: Optional[float]
    chosen_fill_ratio: Optional[float]
    chosen_tip_x: Optional[int]
    chosen_tip_y: Optional[int]
    chosen_tip_radius: Optional[float]
    reject_reason: str
    info_text: str


class DartDetector:
    def __init__(self) -> None:
        self.reference_gray: Optional[np.ndarray] = None
        self.is_armed: bool = False

        self.max_darts_per_round: int = 3

        # Basisparameter
        self.diff_threshold: int = 22
        self.min_contour_area: float = 120.0
        self.max_contour_area: float = 18000.0

        self.required_stable_frames: int = 3
        self.match_distance_px: float = 14.0

        self.arm_grace_seconds: float = 0.7
        self.post_hit_cooldown_seconds: float = 0.9

        self._armed_at: float = 0.0
        self._cooldown_until: float = 0.0

        self._last_candidate: Optional[Tuple[int, int]] = None
        self._stable_counter: int = 0

        # Aktuell noch von der UI verwendet
        self.detected_darts: List[DartDetectionResult] = []

        self._last_debug = DartDebugSnapshot(
            is_armed=False,
            diff_nonzero_pixels=0,
            contour_count_total=0,
            contour_count_valid=0,
            chosen_area=None,
            chosen_aspect_ratio=None,
            chosen_fill_ratio=None,
            chosen_tip_x=None,
            chosen_tip_y=None,
            chosen_tip_radius=None,
            reject_reason="-",
            info_text="Noch keine Analyse",
        )

    def get_debug_snapshot(self) -> DartDebugSnapshot:
        return self._last_debug

    def _set_debug(
        self,
        *,
        diff_nonzero_pixels: int = 0,
        contour_count_total: int = 0,
        contour_count_valid: int = 0,
        chosen_area: Optional[float] = None,
        chosen_aspect_ratio: Optional[float] = None,
        chosen_fill_ratio: Optional[float] = None,
        chosen_tip_x: Optional[int] = None,
        chosen_tip_y: Optional[int] = None,
        chosen_tip_radius: Optional[float] = None,
        reject_reason: str = "-",
        info_text: str = "",
    ) -> None:
        self._last_debug = DartDebugSnapshot(
            is_armed=self.is_armed,
            diff_nonzero_pixels=diff_nonzero_pixels,
            contour_count_total=contour_count_total,
            contour_count_valid=contour_count_valid,
            chosen_area=chosen_area,
            chosen_aspect_ratio=chosen_aspect_ratio,
            chosen_fill_ratio=chosen_fill_ratio,
            chosen_tip_x=chosen_tip_x,
            chosen_tip_y=chosen_tip_y,
            chosen_tip_radius=chosen_tip_radius,
            reject_reason=reject_reason,
            info_text=info_text,
        )

    def reset_round(self) -> None:
        self.is_armed = False
        self._armed_at = 0.0
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        self.detected_darts = []
        self._set_debug(info_text="Runde zurückgesetzt")

    def clear_reference(self) -> None:
        self.reference_gray = None
        self.reset_round()
        self._set_debug(info_text="Referenz gelöscht")

    def set_reference_frame(self, frame_bgr: np.ndarray) -> None:
        self.reference_gray = self._to_preprocessed_gray(frame_bgr)
        self.reset_round()
        self._set_debug(info_text="Referenzbild gespeichert")

    def arm(self) -> bool:
        if self.reference_gray is None:
            self._set_debug(reject_reason="Keine Referenz", info_text="Arm fehlgeschlagen")
            return False

        self.is_armed = True
        self._armed_at = time.monotonic()
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        self._set_debug(info_text="Detector scharf")
        return True

    def _to_preprocessed_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def _build_board_mask(self, frame_bgr: np.ndarray, calibration: Dict) -> np.ndarray:
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

        kernel = np.ones((35, 35), np.uint8)
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

    def _contour_metrics(self, contour: np.ndarray) -> Tuple[float, float, float]:
        area = float(cv2.contourArea(contour))
        x, y, w, h = cv2.boundingRect(contour)
        short_side = max(1, min(w, h))
        long_side = max(w, h)
        aspect_ratio = long_side / short_side
        rect_area = max(1, w * h)
        fill_ratio = area / rect_area
        return area, aspect_ratio, fill_ratio

    def _contour_shape_ok(self, contour: np.ndarray) -> Tuple[bool, str]:
        area, aspect_ratio, fill_ratio = self._contour_metrics(contour)

        if area < self.min_contour_area:
            return False, f"Area zu klein ({area:.1f})"

        if area > self.max_contour_area:
            return False, f"Area zu groß ({area:.1f})"

        if aspect_ratio < 2.0:
            return False, f"Aspect zu klein ({aspect_ratio:.2f})"

        if fill_ratio > 0.80:
            return False, f"Fill zu hoch ({fill_ratio:.2f})"

        if fill_ratio < 0.04:
            return False, f"Fill zu klein ({fill_ratio:.2f})"

        return True, "OK"

    def _find_candidate_contours(self, mask: np.ndarray) -> Tuple[List[np.ndarray], int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours: List[np.ndarray] = []

        for contour in contours:
            ok, _ = self._contour_shape_ok(contour)
            if ok:
                valid_contours.append(contour)

        valid_contours.sort(key=cv2.contourArea, reverse=True)
        return valid_contours, len(contours)

    def _board_radius_of_pixel(self, x_px: int, y_px: int, calibration: Dict) -> Optional[float]:
        projected = project_image_point_to_board(x_px, y_px, calibration)
        if projected is None:
            return None

        x_board, y_board = projected
        return math.sqrt(x_board * x_board + y_board * y_board)

    def _pca_axis(self, contour: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Gibt Mittelpunkt und Hauptrichtung der Kontur zurück.
        """
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return None

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        if mean is None or eigenvectors is None or len(eigenvectors) == 0:
            return None

        center = mean[0]
        direction = eigenvectors[0]
        norm = np.linalg.norm(direction)
        if norm <= 1e-6:
            return None

        direction = direction / norm
        return center, direction

    def _sample_axis_impact_point(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        contour: np.ndarray,
        calibration: Dict,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[float], str]:
        """
        Sucht entlang der Dartachse den Punkt, der die virtuelle Boardfläche
        am wahrscheinlichsten trifft.

        Idee:
        - wir nehmen beide Richtungen der Achse
        - sampeln viele Punkte entlang der Linie
        - behalten nur Punkte innerhalb / nahe der Kontur
        - projizieren sie in den Boardraum
        - der Punkt mit kleinstem Radius innerhalb sinnvoller Boardfläche ist der Impact-Point
        """
        h, w = frame_shape[:2]

        # Konturmaske für "liegt der Punkt auf dem Dartobjekt?"
        contour_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Länge des Bounding-Rechtecks als Sampling-Bereich
        x, y, bw, bh = cv2.boundingRect(contour)
        half_length = float(max(bw, bh)) * 0.9
        step = 1.5

        candidates: List[Tuple[Tuple[int, int], float]] = []

        for sign in (-1.0, 1.0):
            t = 0.0
            while t <= half_length:
                px = center[0] + sign * direction[0] * t
                py = center[1] + sign * direction[1] * t

                ix = int(round(float(px)))
                iy = int(round(float(py)))

                if ix < 0 or iy < 0 or ix >= w or iy >= h:
                    t += step
                    continue

                # Punkt muss auf dem Dartblob oder direkt an seinem Rand liegen
                if contour_mask[iy, ix] == 0:
                    t += step
                    continue

                radius = self._board_radius_of_pixel(ix, iy, calibration)
                if radius is None:
                    t += step
                    continue

                # Nur Punkte im sinnvollen Boardbereich
                if radius <= 1.08:
                    candidates.append(((ix, iy), radius))

                t += step

        if not candidates:
            return None, None, "Keine Impact-Samples gefunden"

        # kleinster Radius = am ehesten Spitze/Impact
        candidates.sort(key=lambda item: item[1])
        impact_point, impact_radius = candidates[0]

        return impact_point, impact_radius, "OK"

    def _estimate_impact_point_from_contour(
        self,
        contour: np.ndarray,
        calibration: Dict,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[Optional[Tuple[int, int]], Optional[float], str]:
        """
        Hauptlogik Phase 6.0:
        - Dartachse per PCA
        - virtuellen Impact-Point entlang der Achse suchen
        """
        axis = self._pca_axis(contour)
        if axis is None:
            return None, None, "Keine PCA-Achse"

        center, direction = axis

        impact_point, impact_radius, reason = self._sample_axis_impact_point(
            center=center,
            direction=direction,
            contour=contour,
            calibration=calibration,
            frame_shape=frame_shape,
        )
        if impact_point is None:
            return None, None, reason

        return impact_point, impact_radius, "OK"

    def _candidate_is_new(self, x_px: int, y_px: int) -> bool:
        for dart in self.detected_darts:
            dx = x_px - dart.x_px
            dy = y_px - dart.y_px
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= 24.0:
                return False
        return True

    def _update_reference_after_hit(self, frame_bgr: np.ndarray) -> None:
        self.reference_gray = self._to_preprocessed_gray(frame_bgr)

    def process_frame(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[DartDetectionResult]:
        if not self.is_armed:
            self._set_debug(info_text="Detector nicht scharf", reject_reason="Nicht scharf")
            return None

        if self.reference_gray is None:
            self._set_debug(info_text="Keine Referenz", reject_reason="Keine Referenz")
            return None

        if len(self.detected_darts) >= self.max_darts_per_round:
            self._set_debug(info_text="Maximale Darts erreicht", reject_reason="Runde voll")
            return None

        now = time.monotonic()

        if (now - self._armed_at) < self.arm_grace_seconds:
            self._set_debug(info_text="Grace-Phase nach Arm", reject_reason="Grace")
            return None

        if now < self._cooldown_until:
            self._set_debug(info_text="Cooldown aktiv", reject_reason="Cooldown")
            return None

        mask = self._build_diff_mask(frame_bgr, calibration)
        if mask is None:
            self._set_debug(info_text="Maskenerzeugung fehlgeschlagen", reject_reason="Keine Maske")
            return None

        diff_nonzero = int(np.count_nonzero(mask))
        valid_contours, total_contours = self._find_candidate_contours(mask)

        if not valid_contours:
            self._last_candidate = None
            self._stable_counter = 0
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=0,
                reject_reason="Keine gültige Kontur",
                info_text="Keine brauchbare Kontur gefunden",
            )
            return None

        best_tip: Optional[Tuple[int, int]] = None
        best_area: Optional[float] = None
        best_aspect: Optional[float] = None
        best_fill: Optional[float] = None
        best_tip_radius: Optional[float] = None
        reject_reason = "-"
        info_text = "-"

        for contour in valid_contours:
            area, aspect_ratio, fill_ratio = self._contour_metrics(contour)
            impact_point, impact_radius, reason = self._estimate_impact_point_from_contour(
                contour=contour,
                calibration=calibration,
                frame_shape=frame_bgr.shape,
            )

            if impact_point is None:
                reject_reason = reason
                info_text = "Kontur vorhanden, aber kein Board-Impact"
                continue

            tx, ty = impact_point

            if not self._candidate_is_new(tx, ty):
                reject_reason = "Zu nah an vorhandenem Dart"
                info_text = "Impact-Kandidat verworfen"
                continue

            best_tip = impact_point
            best_area = area
            best_aspect = aspect_ratio
            best_fill = fill_ratio
            best_tip_radius = impact_radius
            reject_reason = "OK"
            info_text = "Board-Impact-Point erfolgreich geschätzt"
            break

        if best_tip is None:
            self._last_candidate = None
            self._stable_counter = 0
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                reject_reason=reject_reason,
                info_text=info_text,
            )
            return None

        tx, ty = best_tip

        if self._last_candidate is None:
            self._last_candidate = (tx, ty)
            self._stable_counter = 1
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best_area,
                chosen_aspect_ratio=best_aspect,
                chosen_fill_ratio=best_fill,
                chosen_tip_x=tx,
                chosen_tip_y=ty,
                chosen_tip_radius=best_tip_radius,
                reject_reason="Warte auf Stabilisierung",
                info_text="Erster Impact-Kandidatenframe",
            )
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
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best_area,
                chosen_aspect_ratio=best_aspect,
                chosen_fill_ratio=best_fill,
                chosen_tip_x=tx,
                chosen_tip_y=ty,
                chosen_tip_radius=best_tip_radius,
                reject_reason=f"Stabilisierung {self._stable_counter}/{self.required_stable_frames}",
                info_text="Impact-Kandidat noch nicht stabil genug",
            )
            return None

        hit = calculate_board_hit_from_image_point(tx, ty, calibration)
        if hit is None:
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best_area,
                chosen_aspect_ratio=best_aspect,
                chosen_fill_ratio=best_fill,
                chosen_tip_x=tx,
                chosen_tip_y=ty,
                chosen_tip_radius=best_tip_radius,
                reject_reason="Scoring fehlgeschlagen",
                info_text="Board-Hit-Berechnung fehlgeschlagen",
            )
            return None

        if hit.ring_name == "MISS":
            self._last_candidate = None
            self._stable_counter = 0
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best_area,
                chosen_aspect_ratio=best_aspect,
                chosen_fill_ratio=best_fill,
                chosen_tip_x=tx,
                chosen_tip_y=ty,
                chosen_tip_radius=best_tip_radius,
                reject_reason="MISS verworfen",
                info_text="Impact lag außerhalb",
            )
            return None

        result = DartDetectionResult(
            x_px=tx,
            y_px=ty,
            contour_area=float(best_area),
            score_label=hit.label,
            score_value=hit.score,
            ring_name=hit.ring_name,
            board_x=hit.board_x,
            board_y=hit.board_y,
            radius=hit.radius,
        )

        self.detected_darts.append(result)
        self._update_reference_after_hit(frame_bgr)

        self._cooldown_until = time.monotonic() + self.post_hit_cooldown_seconds
        self._last_candidate = None
        self._stable_counter = 0

        self._set_debug(
            diff_nonzero_pixels=diff_nonzero,
            contour_count_total=total_contours,
            contour_count_valid=len(valid_contours),
            chosen_area=best_area,
            chosen_aspect_ratio=best_aspect,
            chosen_fill_ratio=best_fill,
            chosen_tip_x=tx,
            chosen_tip_y=ty,
            chosen_tip_radius=best_tip_radius,
            reject_reason="OK",
            info_text=f"Impact-Treffer erkannt: {result.score_label} = {result.score_value}",
        )

        return result