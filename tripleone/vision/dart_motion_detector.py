from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Optional

import cv2
import numpy as np

PointF = tuple[float, float]
BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class DartMotionDetectorConfig:
    # Schnelle Motion-Stufe
    analysis_scale: float = 0.50
    blur_kernel_size: int = 5
    diff_threshold: int = 18

    open_kernel_size: int = 3
    close_kernel_size: int = 5
    dilate_kernel_size: int = 3
    dilate_iterations: int = 1

    min_contour_area: float = 22.0
    max_contour_area_ratio: float = 0.08
    min_bbox_width: int = 2
    min_bbox_height: int = 5
    roi_margin_px: int = 45
    max_regions: int = 6

    # ----------------------------------------------------------
    # NEU: Fragmente zu einem Dart zusammenführen
    # ----------------------------------------------------------
    max_fragments_per_dart: int = 3

    # Maximaler Abstand eines Fragment-Zentrums zur gemeinsamen Dartachse
    merge_max_perpendicular_distance_px: float = 34.0

    # Maximaler Abstand zwischen zwei Fragment-Zentren
    merge_max_centroid_distance_px: float = 260.0

    # Ein Dart soll länglich genug sein
    min_axis_span_px: float = 45.0
    min_axis_elongation: float = 2.2

    # Wie weit darf der berechnete Impact vom nächsten echten Diff-Pixel
    # nach außen geschoben werden?
    impact_endpoint_trim_px: float = 2.0

    keep_debug_images: bool = True


@dataclass(slots=True)
class MotionRegion:
    region_id: int
    bbox: BBox
    centroid: PointF
    area: float
    changed_pixel_ratio: float
    contour: Optional[np.ndarray] = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DartGeometry:
    """
    Geometrie eines aus 1..N Motion-Fragmenten rekonstruierten Darts.
    """
    region_ids: tuple[int, ...]
    axis_point: PointF
    axis_direction: PointF
    endpoint_a: PointF
    endpoint_b: PointF
    impact_point: PointF
    flight_side_point: PointF
    axis_span_px: float
    elongation: float
    perpendicular_rms_px: float
    confidence: float
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DartMotionResult:
    regions: list[MotionRegion]
    changed_pixel_ratio: float
    motion_detected: bool
    dart_geometry: Optional[DartGeometry] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def best_region(self) -> Optional[MotionRegion]:
        return self.regions[0] if self.regions else None


class DartMotionDetector:
    def __init__(self, config: Optional[DartMotionDetectorConfig] = None) -> None:
        self.config = config or DartMotionDetectorConfig()

    def detect(
        self,
        current_frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        board_mask: Optional[np.ndarray] = None,
        board_center_image: Optional[PointF] = None,
    ) -> DartMotionResult:
        self._validate_frame(current_frame, "current_frame")
        self._validate_frame(reference_frame, "reference_frame")

        if current_frame.shape[:2] != reference_frame.shape[:2]:
            raise ValueError("current_frame und reference_frame müssen dieselbe Größe haben.")

        full_h, full_w = current_frame.shape[:2]
        scale = float(self.config.analysis_scale)

        if not 0.10 <= scale <= 1.0:
            raise ValueError("analysis_scale muss zwischen 0.10 und 1.0 liegen.")

        analysis_w = max(32, int(round(full_w * scale)))
        analysis_h = max(32, int(round(full_h * scale)))

        current_small = cv2.resize(
            current_frame, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA
        )
        reference_small = cv2.resize(
            reference_frame, (analysis_w, analysis_h), interpolation=cv2.INTER_AREA
        )

        mask_small = self._prepare_mask(
            board_mask=board_mask,
            full_shape=(full_h, full_w),
            analysis_shape=(analysis_h, analysis_w),
        )

        current_gray = self._to_gray(current_small)
        reference_gray = self._to_gray(reference_small)

        blur_k = self._odd_kernel(self.config.blur_kernel_size)
        if blur_k > 1:
            current_gray = cv2.GaussianBlur(current_gray, (blur_k, blur_k), 0)
            reference_gray = cv2.GaussianBlur(reference_gray, (blur_k, blur_k), 0)

        diff = cv2.absdiff(reference_gray, current_gray)
        diff_masked = (
            cv2.bitwise_and(diff, diff, mask=mask_small)
            if mask_small is not None
            else diff
        )

        _, binary = cv2.threshold(
            diff_masked,
            int(self.config.diff_threshold),
            255,
            cv2.THRESH_BINARY,
        )

        cleaned = binary.copy()

        open_k = self._odd_kernel(self.config.open_kernel_size)
        if open_k > 1:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_OPEN,
                np.ones((open_k, open_k), dtype=np.uint8),
            )

        close_k = self._odd_kernel(self.config.close_kernel_size)
        if close_k > 1:
            cleaned = cv2.morphologyEx(
                cleaned,
                cv2.MORPH_CLOSE,
                np.ones((close_k, close_k), dtype=np.uint8),
            )

        dilate_k = self._odd_kernel(self.config.dilate_kernel_size)
        if dilate_k > 1 and int(self.config.dilate_iterations) > 0:
            cleaned = cv2.dilate(
                cleaned,
                np.ones((dilate_k, dilate_k), dtype=np.uint8),
                iterations=int(self.config.dilate_iterations),
            )

        if mask_small is not None:
            cleaned = cv2.bitwise_and(cleaned, cleaned, mask=mask_small)

        roi_pixel_count = (
            int(np.count_nonzero(mask_small))
            if mask_small is not None
            else int(analysis_w * analysis_h)
        )
        changed_pixel_count = int(np.count_nonzero(cleaned))
        changed_pixel_ratio = float(changed_pixel_count) / float(max(1, roi_pixel_count))

        found = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = list(found[0] if len(found) == 2 else found[1])

        max_contour_area = float(roi_pixel_count) * float(
            self.config.max_contour_area_ratio
        )
        scale_x = float(full_w) / float(analysis_w)
        scale_y = float(full_h) / float(analysis_h)

        regions: list[MotionRegion] = []

        for contour_index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area < float(self.config.min_contour_area):
                continue
            if area > max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < int(self.config.min_bbox_width):
                continue
            if h < int(self.config.min_bbox_height):
                continue

            moments = cv2.moments(contour)
            if abs(float(moments["m00"])) > 1e-6:
                cx_small = float(moments["m10"] / moments["m00"])
                cy_small = float(moments["m01"] / moments["m00"])
            else:
                cx_small = float(x + w / 2.0)
                cy_small = float(y + h / 2.0)

            x_full = int(round(x * scale_x))
            y_full = int(round(y * scale_y))
            w_full = max(1, int(round(w * scale_x)))
            h_full = max(1, int(round(h * scale_y)))

            margin = max(0, int(self.config.roi_margin_px))
            rx0 = max(0, x_full - margin)
            ry0 = max(0, y_full - margin)
            rx1 = min(full_w, x_full + w_full + margin)
            ry1 = min(full_h, y_full + h_full + margin)

            bbox_full: BBox = (
                int(rx0),
                int(ry0),
                int(max(1, rx1 - rx0)),
                int(max(1, ry1 - ry0)),
            )

            centroid_full: PointF = (
                float(cx_small * scale_x),
                float(cy_small * scale_y),
            )

            local_mask = cleaned[y : y + h, x : x + w]
            local_ratio = float(np.count_nonzero(local_mask)) / float(
                max(1, local_mask.size)
            )

            contour_full = np.round(
                contour.astype(np.float32)
                * np.array([[[scale_x, scale_y]]], dtype=np.float32)
            ).astype(np.int32)

            regions.append(
                MotionRegion(
                    region_id=int(contour_index),
                    bbox=bbox_full,
                    centroid=centroid_full,
                    area=area,
                    changed_pixel_ratio=local_ratio,
                    contour=contour_full,
                    debug={
                        "analysis_bbox": (int(x), int(y), int(w), int(h)),
                        "raw_full_bbox": (
                            int(x_full),
                            int(y_full),
                            int(w_full),
                            int(h_full),
                        ),
                    },
                )
            )

        regions.sort(key=lambda r: r.area, reverse=True)
        regions = regions[: max(1, int(self.config.max_regions))]

        if board_center_image is None:
            # Nur Fallback für das isolierte Testtool.
            # In TripleOne selbst geben wir später den kalibrierten Bull-Mittelpunkt mit.
            board_center_image = (full_w / 2.0, full_h / 2.0)

        dart_geometry = self._build_best_dart_geometry(
            regions=regions,
            board_center=board_center_image,
        )

        debug_images: dict[str, np.ndarray] = {}
        if self.config.keep_debug_images:
            debug_images["motion_current_small"] = current_small
            debug_images["motion_reference_small"] = reference_small
            debug_images["motion_diff"] = diff
            debug_images["motion_diff_masked"] = diff_masked
            debug_images["motion_binary"] = binary
            debug_images["motion_cleaned"] = cleaned
            debug_images["motion_overlay"] = self._render_overlay(
                current_frame,
                regions,
                dart_geometry,
                board_center_image,
            )
            if mask_small is not None:
                debug_images["motion_board_mask"] = mask_small

        return DartMotionResult(
            regions=regions,
            changed_pixel_ratio=changed_pixel_ratio,
            motion_detected=bool(regions),
            dart_geometry=dart_geometry,
            metadata={
                "analysis_scale": scale,
                "analysis_size": (int(analysis_w), int(analysis_h)),
                "full_size": (int(full_w), int(full_h)),
                "contour_count": len(contours),
                "region_count": len(regions),
                "roi_pixel_count": roi_pixel_count,
                "changed_pixel_count": changed_pixel_count,
                "board_mask_used": mask_small is not None,
                "board_center_image": board_center_image,
                "dart_geometry_found": dart_geometry is not None,
            },
            debug_images=debug_images,
        )

    def _build_best_dart_geometry(
        self,
        *,
        regions: list[MotionRegion],
        board_center: PointF,
    ) -> Optional[DartGeometry]:
        usable = [
            region
            for region in regions
            if region.contour is not None and len(region.contour) >= 2
        ]
        if not usable:
            return None

        max_group = min(
            max(1, int(self.config.max_fragments_per_dart)),
            len(usable),
        )

        candidates: list[DartGeometry] = []

        for group_size in range(1, max_group + 1):
            for subset in combinations(usable, group_size):
                geometry = self._fit_geometry_to_regions(
                    subset=subset,
                    board_center=board_center,
                )
                if geometry is not None:
                    candidates.append(geometry)

        if not candidates:
            return None

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        return candidates[0]

    def _fit_geometry_to_regions(
        self,
        *,
        subset: tuple[MotionRegion, ...],
        board_center: PointF,
    ) -> Optional[DartGeometry]:
        centroids = np.asarray(
            [region.centroid for region in subset],
            dtype=np.float32,
        )

        # Fragmentabstände begrenzen
        if len(centroids) > 1:
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    distance = float(np.linalg.norm(centroids[i] - centroids[j]))
                    if distance > float(self.config.merge_max_centroid_distance_px):
                        return None

        contour_points = []
        for region in subset:
            assert region.contour is not None
            contour_points.append(
                region.contour.reshape(-1, 2).astype(np.float32)
            )

        points = np.vstack(contour_points)
        if len(points) < 4:
            return None

        center = np.mean(points, axis=0)
        centered = points - center

        covariance = np.cov(centered, rowvar=False)
        if covariance.shape != (2, 2):
            return None

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]

        major_value = float(max(1e-6, eigenvalues[order[0]]))
        minor_value = float(max(1e-6, eigenvalues[order[1]]))

        axis = eigenvectors[:, order[0]].astype(np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            return None
        axis /= axis_norm

        normal = np.array([-axis[1], axis[0]], dtype=np.float32)

        projections = centered @ axis
        perpendicular = centered @ normal

        min_projection = float(np.min(projections))
        max_projection = float(np.max(projections))
        span = max_projection - min_projection

        if span < float(self.config.min_axis_span_px):
            return None

        perpendicular_rms = float(
            np.sqrt(np.mean(np.square(perpendicular)))
        )

        elongation = float(np.sqrt(major_value / minor_value))

        if elongation < float(self.config.min_axis_elongation):
            return None

        # Bei Fragmenten muss auch deren Zentrum ungefähr auf derselben Achse liegen.
        if len(centroids) > 1:
            centroid_offsets = centroids - center
            centroid_perpendicular = np.abs(centroid_offsets @ normal)
            if float(np.max(centroid_perpendicular)) > float(
                self.config.merge_max_perpendicular_distance_px
            ):
                return None

        endpoint_a_np = center + axis * min_projection
        endpoint_b_np = center + axis * max_projection

        endpoint_a = (float(endpoint_a_np[0]), float(endpoint_a_np[1]))
        endpoint_b = (float(endpoint_b_np[0]), float(endpoint_b_np[1]))

        # Das Dartende näher zum Bull ist die Boardseite.
        dist_a_to_board = self._distance(endpoint_a, board_center)
        dist_b_to_board = self._distance(endpoint_b, board_center)

        if dist_a_to_board <= dist_b_to_board:
            impact_point = endpoint_a
            flight_side_point = endpoint_b
            direction = (-float(axis[0]), -float(axis[1]))
        else:
            impact_point = endpoint_b
            flight_side_point = endpoint_a
            direction = (float(axis[0]), float(axis[1]))

        # Confidence bewusst nachvollziehbar halten.
        elongation_score = min(1.0, elongation / 6.0)
        rms_score = max(
            0.0,
            1.0 - perpendicular_rms / max(1.0, float(self.config.merge_max_perpendicular_distance_px)),
        )
        span_score = min(1.0, span / 180.0)
        fragment_bonus = min(1.0, len(subset) / 2.0)

        confidence = (
            0.35 * elongation_score
            + 0.30 * rms_score
            + 0.25 * span_score
            + 0.10 * fragment_bonus
        )

        return DartGeometry(
            region_ids=tuple(int(region.region_id) for region in subset),
            axis_point=(float(center[0]), float(center[1])),
            axis_direction=direction,
            endpoint_a=endpoint_a,
            endpoint_b=endpoint_b,
            impact_point=impact_point,
            flight_side_point=flight_side_point,
            axis_span_px=float(span),
            elongation=float(elongation),
            perpendicular_rms_px=float(perpendicular_rms),
            confidence=float(max(0.0, min(1.0, confidence))),
            debug={
                "dist_a_to_board": dist_a_to_board,
                "dist_b_to_board": dist_b_to_board,
                "region_count": len(subset),
            },
        )

    @staticmethod
    def _distance(a: PointF, b: PointF) -> float:
        return float(np.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))

    @staticmethod
    def _validate_frame(frame: np.ndarray, name: str) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(f"{name} ist ungültig.")

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame.copy()
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"Nicht unterstützte Bildform: {frame.shape}")

    @staticmethod
    def _odd_kernel(value: int) -> int:
        value = max(1, int(value))
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _prepare_mask(
        *,
        board_mask: Optional[np.ndarray],
        full_shape: tuple[int, int],
        analysis_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if board_mask is None:
            return None
        if board_mask.shape[:2] != full_shape:
            raise ValueError("board_mask hat nicht dieselbe Größe wie das Kamerabild.")

        mask = board_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        analysis_h, analysis_w = analysis_shape

        return cv2.resize(
            mask,
            (analysis_w, analysis_h),
            interpolation=cv2.INTER_NEAREST,
        )

    @staticmethod
    def _render_overlay(
        frame: np.ndarray,
        regions: list[MotionRegion],
        dart_geometry: Optional[DartGeometry],
        board_center: PointF,
    ) -> np.ndarray:
        canvas = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2
            else frame.copy()
        )

        for index, region in enumerate(regions):
            x, y, w, h = region.bbox
            cx = int(round(region.centroid[0]))
            cy = int(round(region.centroid[1]))

            cv2.rectangle(
                canvas,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                2,
            )
            cv2.circle(canvas, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"M{index + 1} area={region.area:.0f}",
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Bull-/Boardzentrum für Debug markieren
        bx = int(round(board_center[0]))
        by = int(round(board_center[1]))
        cv2.drawMarker(
            canvas,
            (bx, by),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            1,
        )

        if dart_geometry is not None:
            a = (
                int(round(dart_geometry.endpoint_a[0])),
                int(round(dart_geometry.endpoint_a[1])),
            )
            b = (
                int(round(dart_geometry.endpoint_b[0])),
                int(round(dart_geometry.endpoint_b[1])),
            )
            impact = (
                int(round(dart_geometry.impact_point[0])),
                int(round(dart_geometry.impact_point[1])),
            )

            # MAGENTA = rekonstruierte Dartachse
            cv2.line(
                canvas,
                a,
                b,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            # ROT = geschätzter Board-Einschlagpunkt
            cv2.circle(
                canvas,
                impact,
                8,
                (0, 0, 255),
                2,
            )
            cv2.circle(
                canvas,
                impact,
                3,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                canvas,
                (
                    f"DART conf={dart_geometry.confidence:.2f} "
                    f"span={dart_geometry.axis_span_px:.0f}px "
                    f"regions={dart_geometry.region_ids}"
                ),
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return canvas


__all__ = [
    "PointF",
    "BBox",
    "DartMotionDetectorConfig",
    "MotionRegion",
    "DartGeometry",
    "DartMotionResult",
    "DartMotionDetector",
]
