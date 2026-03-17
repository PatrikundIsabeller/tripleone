# vision/dart_detector.py
# Reset C.2:
# - Top-Down Dart Detection
# - feste 4 Drahtpunkte:
#   P1 = 20|1
#   P2 = 6|10
#   P3 = 3|19
#   P4 = 11|14
# - C = Bullzentrum
# - stabile Kandidatenbewertung statt "erste gültige Kontur"
# - genau ein Dart pro Scharfstellung

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

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


@dataclass
class DartDebugSnapshot:
    is_armed: bool
    has_reference: bool
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


@dataclass
class _ContourCandidate:
    contour: np.ndarray
    area: float
    aspect_ratio: float
    fill_ratio: float
    tip_topdown: Tuple[int, int]
    tip_radius: float
    radial_alignment: float
    score: float


class DartDetector:
    def __init__(self) -> None:
        self.reference_topdown_gray: Optional[np.ndarray] = None
        self.is_armed: bool = False
        self.has_detected_dart: bool = False

        self.diff_threshold: int = 20
        self.min_contour_area: float = 80.0
        self.max_contour_area: float = 30000.0

        self.required_stable_frames: int = 2
        self.match_distance_px: float = 12.0

        self.arm_grace_seconds: float = 0.5
        self.post_hit_cooldown_seconds: float = 0.8

        self._armed_at: float = 0.0
        self._cooldown_until: float = 0.0
        self._last_candidate: Optional[Tuple[int, int]] = None
        self._stable_counter: int = 0

        self.last_detection: Optional[DartDetectionResult] = None

        self.topdown_size: int = 900
        self.board_center_topdown: Tuple[float, float] = (
            self.topdown_size / 2.0,
            self.topdown_size / 2.0,
        )
        self.outer_double_radius_topdown: float = self.topdown_size * 0.36

        self._last_debug = DartDebugSnapshot(
            is_armed=False,
            has_reference=False,
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
            has_reference=self.reference_topdown_gray is not None,
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

    def reset_detection(self) -> None:
        self.is_armed = False
        self.has_detected_dart = False
        self._armed_at = 0.0
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        self.last_detection = None
        self._set_debug(info_text="Detektion zurückgesetzt")

    def clear_reference(self) -> None:
        self.reference_topdown_gray = None
        self.reset_detection()
        self._set_debug(info_text="Referenz gelöscht")

    def arm(self) -> bool:
        if self.reference_topdown_gray is None:
            self._set_debug(
                reject_reason="Keine Referenz",
                info_text="Scharfstellen fehlgeschlagen",
            )
            return False

        self.is_armed = True
        self.has_detected_dart = False
        self._armed_at = time.monotonic()
        self._cooldown_until = 0.0
        self._last_candidate = None
        self._stable_counter = 0
        self.last_detection = None
        self._set_debug(info_text="Detector scharf")
        return True

    def _extract_src_points(self, calibration: Dict) -> Optional[np.ndarray]:
        points = calibration.get("points", [])
        if len(points) < 5:
            return None

        src = []
        for item in points[:4]:
            if not isinstance(item, dict):
                return None
            src.append([float(item.get("x_px", 0)), float(item.get("y_px", 0))])

        if len(src) != 4:
            return None

        return np.array(src, dtype=np.float32)

    def _dest_point_on_outer_double(self, boundary_angle_deg: float) -> Tuple[float, float]:
        cx, cy = self.board_center_topdown
        r = self.outer_double_radius_topdown
        a = math.radians(boundary_angle_deg)
        x = cx + math.cos(a) * r
        y = cy - math.sin(a) * r
        return x, y

    def _topdown_destination_points(self) -> np.ndarray:
        """
        Feste Zielpunkte auf dem Outer-Double-Draht.

        Grenzen:
        P1 = 20|1   -> 81°
        P2 = 6|10   -> 351°
        P3 = 3|19   -> 261°
        P4 = 11|14  -> 171°
        """
        p1 = self._dest_point_on_outer_double(81.0)
        p2 = self._dest_point_on_outer_double(351.0)
        p3 = self._dest_point_on_outer_double(261.0)
        p4 = self._dest_point_on_outer_double(171.0)
        return np.array([p1, p2, p3, p4], dtype=np.float32)

    def _compute_homography(self, calibration: Dict) -> Optional[np.ndarray]:
        src = self._extract_src_points(calibration)
        if src is None:
            return None
        dst = self._topdown_destination_points()
        return cv2.getPerspectiveTransform(src, dst)

    def _warp_to_topdown(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[np.ndarray]:
        matrix = self._compute_homography(calibration)
        if matrix is None:
            return None
        return cv2.warpPerspective(
            frame_bgr,
            matrix,
            (self.topdown_size, self.topdown_size),
            flags=cv2.INTER_LINEAR,
        )

    def _to_preprocessed_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    def set_reference_frame(self, frame_bgr: np.ndarray, calibration: Dict) -> bool:
        warped = self._warp_to_topdown(frame_bgr, calibration)
        if warped is None:
            self._set_debug(
                reject_reason="Homography fehlgeschlagen",
                info_text="Leeres Board nicht gespeichert",
            )
            return False

        self.reference_topdown_gray = self._to_preprocessed_gray(warped)
        self.reset_detection()
        self._set_debug(info_text="Leeres Board im Top-Down gespeichert")
        return True

    def _build_circular_board_mask(self) -> np.ndarray:
        s = self.topdown_size
        mask = np.zeros((s, s), dtype=np.uint8)
        center = (int(self.board_center_topdown[0]), int(self.board_center_topdown[1]))
        radius = int(self.outer_double_radius_topdown * 1.02)
        cv2.circle(mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        return mask

    def _build_diff_mask(self, frame_bgr: np.ndarray, calibration: Dict) -> Optional[np.ndarray]:
        if self.reference_topdown_gray is None:
            return None

        warped = self._warp_to_topdown(frame_bgr, calibration)
        if warped is None:
            return None

        current_gray = self._to_preprocessed_gray(warped)
        diff = cv2.absdiff(current_gray, self.reference_topdown_gray)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        board_mask = self._build_circular_board_mask()
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
        if aspect_ratio < 1.35:
            return False, f"Aspect zu klein ({aspect_ratio:.2f})"
        if fill_ratio > 0.95:
            return False, f"Fill zu hoch ({fill_ratio:.2f})"
        if fill_ratio < 0.02:
            return False, f"Fill zu klein ({fill_ratio:.2f})"
        return True, "OK"

    def _find_candidate_contours(self, mask: np.ndarray) -> Tuple[List[np.ndarray], int]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours: List[np.ndarray] = []

        for contour in contours:
            ok, _ = self._contour_shape_ok(contour)
            if ok:
                valid_contours.append(contour)

        return valid_contours, len(contours)

    def _topdown_tip_from_contour(
        self,
        contour: np.ndarray,
    ) -> Tuple[Optional[Tuple[int, int]], Optional[float], str]:
        pts = contour.reshape(-1, 2)
        if len(pts) == 0:
            return None, None, "Leere Kontur"

        cx, cy = self.board_center_topdown
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dists = np.sqrt(dx * dx + dy * dy)

        tip_index = int(np.argmax(dists))
        tip = pts[tip_index]
        tip_x = int(tip[0])
        tip_y = int(tip[1])

        tip_radius = float(dists[tip_index] / self.outer_double_radius_topdown)
        if tip_radius > 1.12:
            return None, None, f"Tip zu weit außen ({tip_radius:.3f})"

        return (tip_x, tip_y), tip_radius, "OK"

    def _pca_direction(self, contour: np.ndarray) -> Optional[np.ndarray]:
        pts = contour.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return None

        mean, eigenvectors = cv2.PCACompute(pts, mean=None)
        if mean is None or eigenvectors is None or len(eigenvectors) == 0:
            return None

        direction = eigenvectors[0]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None

        return direction / norm

    def _score_candidate(
        self,
        area: float,
        aspect_ratio: float,
        fill_ratio: float,
        tip_radius: float,
        radial_alignment: float,
    ) -> float:
        # Fläche: in deinem Setup waren gute Treffer grob zwischen 150 und 2500
        if area < 120:
            area_score = max(0.0, area / 120.0)
        elif area <= 2500:
            area_score = 1.0
        else:
            area_score = max(0.0, 1.0 - ((area - 2500.0) / 8000.0))

        # Aspect: Darts sollen eher länglich sein
        aspect_score = max(0.0, min(1.0, (aspect_ratio - 1.3) / 4.0))

        # Fill: mittlere Werte besser, zu kompakt oder zu flächig schlechter
        fill_center = 0.40
        fill_score = max(0.0, 1.0 - (abs(fill_ratio - fill_center) / 0.45))

        # Tip-Radius: Soft, damit Bull auch möglich bleibt, aber ganz außen etwas bevorzugt
        if tip_radius < 0.05:
            tip_radius_score = 0.25
        elif tip_radius < 0.20:
            tip_radius_score = 0.60
        elif tip_radius <= 1.02:
            tip_radius_score = 1.0
        else:
            tip_radius_score = max(0.0, 1.0 - ((tip_radius - 1.02) / 0.10))

        # Radiale Ausrichtung: sehr wichtig
        radial_score = max(0.0, min(1.0, radial_alignment))

        total = (
            0.20 * area_score
            + 0.20 * aspect_score
            + 0.15 * fill_score
            + 0.15 * tip_radius_score
            + 0.30 * radial_score
        )
        return float(total)

    def _evaluate_contour(self, contour: np.ndarray) -> Optional[_ContourCandidate]:
        area, aspect_ratio, fill_ratio = self._contour_metrics(contour)
        tip_topdown, tip_radius, reason = self._topdown_tip_from_contour(contour)
        if tip_topdown is None or tip_radius is None:
            return None

        direction = self._pca_direction(contour)
        if direction is None:
            return None

        cx, cy = self.board_center_topdown
        radial_vec = np.array([tip_topdown[0] - cx, tip_topdown[1] - cy], dtype=np.float32)
        radial_norm = float(np.linalg.norm(radial_vec))
        if radial_norm < 1e-6:
            return None

        radial_unit = radial_vec / radial_norm
        radial_alignment = float(abs(np.dot(direction, radial_unit)))

        # sehr schwache radial ausgerichtete Konturen verwerfen
        if radial_alignment < 0.35:
            return None

        score = self._score_candidate(
            area=area,
            aspect_ratio=aspect_ratio,
            fill_ratio=fill_ratio,
            tip_radius=tip_radius,
            radial_alignment=radial_alignment,
        )

        return _ContourCandidate(
            contour=contour,
            area=area,
            aspect_ratio=aspect_ratio,
            fill_ratio=fill_ratio,
            tip_topdown=tip_topdown,
            tip_radius=tip_radius,
            radial_alignment=radial_alignment,
            score=score,
        )

    def _choose_best_candidate(self, valid_contours: List[np.ndarray]) -> Optional[_ContourCandidate]:
        candidates: List[_ContourCandidate] = []

        for contour in valid_contours:
            cand = self._evaluate_contour(contour)
            if cand is not None:
                candidates.append(cand)

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[0]

    def _inverse_map_tip_to_camera(
        self,
        tip_topdown: Tuple[int, int],
        calibration: Dict,
    ) -> Optional[Tuple[int, int]]:
        src = self._extract_src_points(calibration)
        if src is None:
            return None

        dst = self._topdown_destination_points()
        inverse_matrix = cv2.getPerspectiveTransform(dst, src)

        point = np.array(
            [[[float(tip_topdown[0]), float(tip_topdown[1])]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(point, inverse_matrix)

        x_px = int(round(float(mapped[0, 0, 0])))
        y_px = int(round(float(mapped[0, 0, 1])))
        return x_px, y_px

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        calibration: Dict,
    ) -> Optional[DartDetectionResult]:
        if not self.is_armed:
            return None

        if self.reference_topdown_gray is None:
            self._set_debug(info_text="Keine Referenz", reject_reason="Keine Referenz")
            return None

        if self.has_detected_dart:
            return None

        now = time.monotonic()

        if (now - self._armed_at) < self.arm_grace_seconds:
            self._set_debug(info_text="Grace-Phase nach Scharfstellung", reject_reason="Grace")
            return None

        if now < self._cooldown_until:
            self._set_debug(info_text="Cooldown aktiv", reject_reason="Cooldown")
            return None

        mask = self._build_diff_mask(frame_bgr, calibration)
        if mask is None:
            self._set_debug(
                info_text="Top-Down Maskenerzeugung fehlgeschlagen",
                reject_reason="Keine Maske",
            )
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
                info_text="Keine brauchbare Top-Down-Kontur gefunden",
            )
            return None

        best = self._choose_best_candidate(valid_contours)
        if best is None:
            self._last_candidate = None
            self._stable_counter = 0
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                reject_reason="Kein stabiler Kandidat",
                info_text="Gültige Konturen vorhanden, aber keine gute Kandidatenbewertung",
            )
            return None

        tx_top, ty_top = best.tip_topdown

        if self._last_candidate is None:
            self._last_candidate = (tx_top, ty_top)
            self._stable_counter = 1
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best.area,
                chosen_aspect_ratio=best.aspect_ratio,
                chosen_fill_ratio=best.fill_ratio,
                chosen_tip_x=tx_top,
                chosen_tip_y=ty_top,
                chosen_tip_radius=best.tip_radius,
                reject_reason="Warte auf Stabilisierung",
                info_text=f"Top-Down-Kandidat score={best.score:.3f} align={best.radial_alignment:.3f}",
            )
            return None

        dx = tx_top - self._last_candidate[0]
        dy = ty_top - self._last_candidate[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance <= self.match_distance_px:
            self._stable_counter += 1
        else:
            self._stable_counter = 1

        self._last_candidate = (tx_top, ty_top)

        if self._stable_counter < self.required_stable_frames:
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best.area,
                chosen_aspect_ratio=best.aspect_ratio,
                chosen_fill_ratio=best.fill_ratio,
                chosen_tip_x=tx_top,
                chosen_tip_y=ty_top,
                chosen_tip_radius=best.tip_radius,
                reject_reason=f"Stabilisierung {self._stable_counter}/{self.required_stable_frames}",
                info_text=f"Top-Down-Kandidat score={best.score:.3f} align={best.radial_alignment:.3f}",
            )
            return None

        camera_tip = self._inverse_map_tip_to_camera(best.tip_topdown, calibration)
        if camera_tip is None:
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best.area,
                chosen_aspect_ratio=best.aspect_ratio,
                chosen_fill_ratio=best.fill_ratio,
                chosen_tip_x=tx_top,
                chosen_tip_y=ty_top,
                chosen_tip_radius=best.tip_radius,
                reject_reason="Inverse Homography fehlgeschlagen",
                info_text="Top-Down-Spitze konnte nicht zurückprojiziert werden",
            )
            return None

        cam_x, cam_y = camera_tip
        hit = calculate_board_hit_from_image_point(cam_x, cam_y, calibration)

        if hit is None:
            self._set_debug(
                diff_nonzero_pixels=diff_nonzero,
                contour_count_total=total_contours,
                contour_count_valid=len(valid_contours),
                chosen_area=best.area,
                chosen_aspect_ratio=best.aspect_ratio,
                chosen_fill_ratio=best.fill_ratio,
                chosen_tip_x=tx_top,
                chosen_tip_y=ty_top,
                chosen_tip_radius=best.tip_radius,
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
                chosen_area=best.area,
                chosen_aspect_ratio=best.aspect_ratio,
                chosen_fill_ratio=best.fill_ratio,
                chosen_tip_x=tx_top,
                chosen_tip_y=ty_top,
                chosen_tip_radius=best.tip_radius,
                reject_reason="MISS verworfen",
                info_text="Top-Down-Treffer lag außerhalb",
            )
            return None

        result = DartDetectionResult(
            x_px=cam_x,
            y_px=cam_y,
            contour_area=float(best.area),
            score_label=hit.label,
            score_value=hit.score,
            ring_name=hit.ring_name,
            board_x=hit.board_x,
            board_y=hit.board_y,
            radius=hit.radius,
        )

        self.last_detection = result
        self.has_detected_dart = True
        self.is_armed = False
        self._cooldown_until = time.monotonic() + self.post_hit_cooldown_seconds
        self._last_candidate = None
        self._stable_counter = 0

        self._set_debug(
            diff_nonzero_pixels=diff_nonzero,
            contour_count_total=total_contours,
            contour_count_valid=len(valid_contours),
            chosen_area=best.area,
            chosen_aspect_ratio=best.aspect_ratio,
            chosen_fill_ratio=best.fill_ratio,
            chosen_tip_x=tx_top,
            chosen_tip_y=ty_top,
            chosen_tip_radius=best.tip_radius,
            reject_reason="OK",
            info_text=(
                f"Top-Down-Treffer erkannt: {result.score_label} = {result.score_value} "
                f"| score={best.score:.3f} align={best.radial_alignment:.3f}"
            ),
        )

        return result