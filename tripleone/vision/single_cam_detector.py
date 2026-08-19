from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .calibration_geometry import OUTER_DOUBLE_RADIUS_PX, TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y
    from .dart_candidate_detector import CandidateDetectionResult, CandidateDetectorConfig, DartCandidateDetector
    from .dart_keypoint_detector import (
        DartKeypointDetection,
        DartKeypointDetectionResult,
        DartKeypointDetector,
        DartKeypointDetectorConfig,
    )
    from .score_mapper import ScoreMapper, ScoredHit, build_score_mapper
    from .single_cam_observation import SingleCamEstimateObservation, SingleCamObservation
except ImportError:  # pragma: no cover
    from vision.calibration_geometry import OUTER_DOUBLE_RADIUS_PX, TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y  # type: ignore
    from vision.dart_candidate_detector import CandidateDetectionResult, CandidateDetectorConfig, DartCandidateDetector  # type: ignore
    from vision.dart_keypoint_detector import (  # type: ignore
        DartKeypointDetection,
        DartKeypointDetectionResult,
        DartKeypointDetector,
        DartKeypointDetectorConfig,
    )
    from vision.score_mapper import ScoreMapper, ScoredHit, build_score_mapper  # type: ignore
    from vision.single_cam_observation import SingleCamEstimateObservation, SingleCamObservation  # type: ignore

logger = logging.getLogger(__name__)
PointF = tuple[float, float]
BBox = tuple[int, int, int, int]


@dataclass(slots=True)
class SingleCamDetectorConfig:
    detection_backend: str = "keypoint"
    use_change_trigger: bool = False
    require_change_trigger_for_detection: bool = False
    gate_keypoints_with_trigger_boxes: bool = True
    trigger_bbox_margin_px: int = 50
    max_estimates_to_score: int = 3
    score_all_estimates: bool = True
    min_impact_confidence: float = 0.01
    min_combined_confidence: float = 0.01
    weight_candidate_confidence: float = 0.0
    weight_impact_confidence: float = 1.0
    prune_offboard_estimates_before_scoring: bool = True
    max_board_radius_rel_for_scoring: float = 1.03
    fallback_to_unpruned_estimates_if_all_filtered: bool = False
    keep_debug_images: bool = True
    keep_stage_results: bool = True
    render_stage_overlays: bool = True


@dataclass(slots=True)
class KeypointImpactEstimate:
    candidate_id: int
    impact_point: PointF
    method: str
    confidence: float
    source_candidate_confidence: float
    bbox: BBox
    centroid: PointF
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def hypothesis_count(self) -> int:
        return 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "impact_point": self.impact_point,
            "method": self.method,
            "confidence": self.confidence,
            "source_candidate_confidence": self.source_candidate_confidence,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "hypothesis_count": 1,
            "debug": self.debug,
        }


@dataclass(slots=True)
class SingleCamScoredEstimate:
    rank: int
    candidate_id: int
    image_point: PointF
    scored_hit: ScoredHit
    impact_estimate: KeypointImpactEstimate
    candidate_confidence: float
    impact_confidence: float
    combined_confidence: float
    bbox: BBox
    centroid: PointF
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.scored_hit.label

    @property
    def score(self) -> int:
        return self.scored_hit.score

    @property
    def ring(self) -> str:
        return self.scored_hit.ring

    @property
    def segment(self) -> Optional[int]:
        return self.scored_hit.segment

    @property
    def multiplier(self) -> int:
        return self.scored_hit.multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "candidate_id": self.candidate_id,
            "image_point": self.image_point,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "candidate_confidence": self.candidate_confidence,
            "impact_confidence": self.impact_confidence,
            "combined_confidence": self.combined_confidence,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "impact_estimate": self.impact_estimate.to_dict(),
            "scored_hit": self.scored_hit.to_dict(),
            "debug": self.debug,
        }


@dataclass(slots=True)
class SingleCamDetectionResult:
    scored_estimates: list[SingleCamScoredEstimate]
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    candidate_result: Optional[CandidateDetectionResult] = None
    impact_result: Any = None
    keypoint_result: Optional[DartKeypointDetectionResult] = None

    @property
    def best_estimate(self) -> Optional[SingleCamScoredEstimate]:
        return self.scored_estimates[0] if self.scored_estimates else None

    @property
    def best_hit(self) -> Optional[ScoredHit]:
        return None if self.best_estimate is None else self.best_estimate.scored_hit

    @property
    def best_label(self) -> Optional[str]:
        return None if self.best_estimate is None else self.best_estimate.label

    @property
    def best_score(self) -> Optional[int]:
        return None if self.best_estimate is None else self.best_estimate.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "best_label": self.best_label,
            "best_score": self.best_score,
            "scored_estimates": [x.to_dict() for x in self.scored_estimates],
            "candidate_result": None if self.candidate_result is None else self.candidate_result.to_dict(),
            "keypoint_result": None if self.keypoint_result is None else self.keypoint_result.to_dict(),
        }

    def render_debug_overlay(self, frame: np.ndarray, *, max_estimates: Optional[int] = None) -> np.ndarray:
        canvas = _ensure_bgr(frame)
        if self.keypoint_result is not None:
            canvas = self.keypoint_result.render_debug_overlay(canvas)
        count = len(self.scored_estimates)
        if max_estimates is not None:
            count = min(count, max(0, int(max_estimates)))
        for item in self.scored_estimates[:count]:
            x, y = _round_point(item.image_point)
            cv2.putText(
                canvas,
                f"{item.label} score={item.score} KI={item.combined_confidence:.2f}",
                (x + 10, max(18, y + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas


class SingleCamDetector:
    def __init__(
        self,
        *,
        config: Optional[SingleCamDetectorConfig] = None,
        keypoint_detector: Optional[DartKeypointDetector] = None,
        keypoint_detector_config: Optional[DartKeypointDetectorConfig] = None,
        candidate_detector: Optional[DartCandidateDetector] = None,
        candidate_detector_config: Optional[CandidateDetectorConfig] = None,
        impact_estimator: Any = None,
        impact_estimator_config: Any = None,
        score_mapper: Optional[ScoreMapper] = None,
        manual_points: Optional[list[Any]] = None,
        calibration_record: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        image_size: Optional[tuple[int, int]] = None,
        pipeline_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or SingleCamDetectorConfig()
        self.keypoint_detector = keypoint_detector or DartKeypointDetector(config=keypoint_detector_config)
        self.candidate_detector = candidate_detector or DartCandidateDetector(config=candidate_detector_config)
        self.impact_estimator = impact_estimator
        self.impact_estimator_config = impact_estimator_config
        self._pipeline_kwargs = pipeline_kwargs or {}
        self._score_mapper: Optional[ScoreMapper] = None

        if score_mapper is not None:
            self._score_mapper = score_mapper
        elif pipeline is not None or manual_points is not None or calibration_record is not None:
            self._score_mapper = build_score_mapper(
                manual_points=manual_points,
                calibration_record=calibration_record,
                pipeline=pipeline,
                image_size=image_size,
                pipeline_kwargs=self._pipeline_kwargs,
            )

    @property
    def score_mapper(self) -> Optional[ScoreMapper]:
        return self._score_mapper

    def reset_tracking(self) -> None:
        """
        Setzt den aktuellen KI-Dart-Track zurück.

        SingleCamDetector bleibt damit die öffentliche Schnittstelle.
        VisionService muss DartKeypointDetector nicht direkt kennen.
        """
        reset_method = getattr(
            self.keypoint_detector,
            "reset_tracking",
            None,
        )

        if callable(reset_method):
            reset_method()

    def set_score_mapper(self, score_mapper: ScoreMapper) -> None:
        self._score_mapper = score_mapper

    def rebuild_score_mapper_from_manual_points(self, manual_points: list[Any], *, image_size: Optional[tuple[int, int]] = None) -> None:
        self._score_mapper = build_score_mapper(
            manual_points=manual_points,
            image_size=image_size,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    def rebuild_score_mapper_from_record(self, calibration_record: Any) -> None:
        self._score_mapper = build_score_mapper(
            calibration_record=calibration_record,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    def rebuild_score_mapper_from_pipeline(self, pipeline: Any) -> None:
        self._score_mapper = build_score_mapper(
            pipeline=pipeline,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    def detect(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
    ) -> SingleCamDetectionResult:
        self._ensure_score_mapper_ready()
        _validate_frame(frame, "frame")
        _validate_frame(reference_frame, "reference_frame")
        if frame.shape[:2] != reference_frame.shape[:2]:
            raise ValueError("frame und reference_frame müssen dieselbe Größe haben.")

        effective_board_mask = board_mask
        effective_board_polygon = board_polygon
        if effective_board_mask is None and effective_board_polygon is None:
            effective_board_polygon = self._build_auto_board_polygon()

        trigger_result = None
        if self.config.use_change_trigger:
            trigger_result = self._run_change_trigger(
                frame=frame,
                reference_frame=reference_frame,
                board_mask=effective_board_mask,
                board_polygon=effective_board_polygon,
            )
            if self.config.require_change_trigger_for_detection and not trigger_result.candidates:
                return SingleCamDetectionResult(
                    scored_estimates=[],
                    metadata={"backend": "keypoint", "reason": "change_trigger_no_candidate"},
                    candidate_result=trigger_result,
                )

        kp_result = self.keypoint_detector.detect(
            frame,
            reference_frame=reference_frame,
            board_mask=effective_board_mask,
            board_polygon=effective_board_polygon,
        )
        detections = list(kp_result.detections)

        if trigger_result is not None and trigger_result.candidates and self.config.gate_keypoints_with_trigger_boxes:
            gated = self._gate_keypoints_by_trigger_boxes(detections, trigger_result)
            if gated or self.config.require_change_trigger_for_detection:
                detections = gated

        scored = self._score_keypoints(detections)
        metadata = {
            "backend": "keypoint",
            "candidate_count": 0 if trigger_result is None else len(trigger_result.candidates),
            "keypoint_count": len(kp_result.detections),
            "keypoint_count_after_gate": len(detections),
            "impact_count": len(detections),
            "scored_count": len(scored),
            "best_label": None if not scored else scored[0].label,
            "best_score": None if not scored else scored[0].score,
        }

        result = SingleCamDetectionResult(
            scored_estimates=scored,
            metadata=metadata,
            candidate_result=trigger_result if self.config.keep_stage_results else None,
            keypoint_result=kp_result if self.config.keep_stage_results else None,
        )
        if self.config.keep_debug_images:
            if trigger_result is not None:
                result.debug_images.update(trigger_result.debug_images)
            result.debug_images.update(kp_result.debug_images)
            if self.config.render_stage_overlays:
                result.debug_images["single_cam_overlay"] = result.render_debug_overlay(frame)
        return result

    def detect_best_hit(self, frame: np.ndarray, reference_frame: np.ndarray, **kwargs: Any) -> Optional[ScoredHit]:
        return self.detect(frame=frame, reference_frame=reference_frame, **kwargs).best_hit

    def detect_observation(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        camera_index: int,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
        reference_available: bool = True,
    ) -> SingleCamObservation:
        result = self.detect(
            frame=frame,
            reference_frame=reference_frame,
            board_mask=board_mask,
            board_polygon=board_polygon,
        )

        estimates: list[SingleCamEstimateObservation] = []
        for item in result.scored_estimates:
            topdown = _coerce_point(getattr(item.scored_hit, "topdown_point", None))
            if topdown is None:
                topdown = self._project_image_point_to_topdown_safe(item.image_point)
            estimates.append(
                SingleCamEstimateObservation(
                    estimate_rank=int(item.rank),
                    image_point=item.image_point,
                    topdown_point=topdown,
                    label=item.label,
                    score=int(item.score),
                    ring=item.ring,
                    segment=item.segment,
                    multiplier=int(item.multiplier),
                    combined_confidence=float(item.combined_confidence),
                    impact_confidence=float(item.impact_confidence),
                    candidate_confidence=float(item.candidate_confidence),
                    debug=dict(item.debug),
                )
            )

        best = estimates[0] if estimates else None
        return SingleCamObservation(
            camera_index=int(camera_index),
            frame_ok=True,
            detector_ready=self._score_mapper is not None,
            reference_available=bool(reference_available),
            candidate_count=int(result.metadata.get("candidate_count", 0)),
            impact_count=int(result.metadata.get("impact_count", len(estimates))),
            scored_count=len(estimates),
            best_image_point=None if best is None else best.image_point,
            best_topdown_point=None if best is None else best.topdown_point,
            best_label=None if best is None else best.label,
            best_score=None if best is None else best.score,
            best_ring=None if best is None else best.ring,
            best_segment=None if best is None else best.segment,
            best_multiplier=None if best is None else best.multiplier,
            best_combined_confidence=0.0 if best is None else float(best.combined_confidence),
            best_impact_confidence=0.0 if best is None else float(best.impact_confidence),
            best_candidate_confidence=0.0 if best is None else float(best.candidate_confidence),
            estimates=estimates,
            metadata=dict(result.metadata),
            debug={"observation_mode": "keypoint_first"},
            raw_result=result,
        )

    def _score_keypoints(self, detections: list[DartKeypointDetection]) -> list[SingleCamScoredEstimate]:
        assert self._score_mapper is not None
        source = detections[:1] if not self.config.score_all_estimates else detections[:max(1, int(self.config.max_estimates_to_score))]
        scored: list[SingleCamScoredEstimate] = []

        for det in source:
            if det.confidence < float(self.config.min_impact_confidence):
                continue
            topdown = self._project_image_point_to_topdown_safe(det.tip_point)
            if self.config.prune_offboard_estimates_before_scoring and topdown is not None:
                if self._compute_topdown_radius_rel(topdown) > float(self.config.max_board_radius_rel_for_scoring):
                    continue

            scored_hit = self._score_mapper.score_image_point(det.tip_point)
            combined = self._compute_combined_confidence(det.confidence, det.confidence)
            if combined < float(self.config.min_combined_confidence):
                continue

            bbox = _coerce_bbox(det.bbox)
            centroid = _bbox_center_or_point(bbox, det.tip_point)
            adapter = KeypointImpactEstimate(
                candidate_id=int(det.detection_index),
                impact_point=det.tip_point,
                method="yolo_pose_keypoint",
                confidence=float(det.confidence),
                source_candidate_confidence=float(det.confidence),
                bbox=bbox,
                centroid=centroid,
                debug={"keypoint_detection": det.to_dict()},
            )
            scored.append(
                SingleCamScoredEstimate(
                    rank=0,
                    candidate_id=int(det.detection_index),
                    image_point=det.tip_point,
                    scored_hit=scored_hit,
                    impact_estimate=adapter,
                    candidate_confidence=float(det.confidence),
                    impact_confidence=float(det.confidence),
                    combined_confidence=float(combined),
                    bbox=bbox,
                    centroid=centroid,
                    debug={"backend": "keypoint", "method": "yolo_pose_keypoint"},
                )
            )

        scored.sort(key=lambda x: x.combined_confidence, reverse=True)
        for rank, item in enumerate(scored, start=1):
            item.rank = rank
        return scored

    def _run_change_trigger(self, *, frame: np.ndarray, reference_frame: np.ndarray, board_mask: Optional[np.ndarray], board_polygon: Any) -> CandidateDetectionResult:
        center = self._try_get_board_center_image()
        kwargs = {
            "frame": frame,
            "reference_frame": reference_frame,
            "board_mask": board_mask,
            "board_polygon": board_polygon,
        }
        try:
            return self.candidate_detector.detect_candidates(**kwargs, board_center_image=center)
        except TypeError as exc:
            if "board_center_image" not in str(exc):
                raise
            return self.candidate_detector.detect_candidates(**kwargs)

    def _gate_keypoints_by_trigger_boxes(self, detections: list[DartKeypointDetection], trigger_result: CandidateDetectionResult) -> list[DartKeypointDetection]:
        margin = max(0, int(self.config.trigger_bbox_margin_px))
        boxes = [_expand_bbox(_coerce_bbox(c.bbox), margin) for c in trigger_result.candidates]
        return [d for d in detections if any(_point_in_bbox(d.tip_point, b) for b in boxes)]

    def _ensure_score_mapper_ready(self) -> None:
        if self._score_mapper is None:
            raise RuntimeError("SingleCamDetector hat keinen ScoreMapper.")

    def _try_get_board_center_image(self) -> Optional[PointF]:
        return self._project_topdown_point_to_image_safe((float(TOPDOWN_CENTER_X), float(TOPDOWN_CENTER_Y)))

    def _project_topdown_point_to_image_safe(self, point: PointF) -> Optional[PointF]:
        if self._score_mapper is None:
            return None
        for name in ("topdown_point_to_image", "project_topdown_point_to_image", "topdown_to_image"):
            method = getattr(self._score_mapper, name, None)
            if callable(method):
                try:
                    return _coerce_point(method(point))
                except Exception:
                    pass
        return None

    def _project_image_point_to_topdown_safe(self, point: PointF) -> Optional[PointF]:
        if self._score_mapper is None:
            return None
        method = getattr(self._score_mapper, "image_point_to_topdown", None)
        if not callable(method):
            return None
        try:
            return _coerce_point(method(point))
        except Exception:
            return None

    def _build_auto_board_polygon(self, *, point_count: int = 72, radius_scale: float = 1.05) -> Optional[list[PointF]]:
        radius = float(OUTER_DOUBLE_RADIUS_PX) * float(radius_scale)
        polygon: list[PointF] = []
        count = max(24, int(point_count))
        for i in range(count):
            angle = 2.0 * math.pi * i / count
            topdown = (
                float(TOPDOWN_CENTER_X + radius * math.cos(angle)),
                float(TOPDOWN_CENTER_Y + radius * math.sin(angle)),
            )
            image_point = self._project_topdown_point_to_image_safe(topdown)
            if image_point is not None:
                polygon.append(image_point)
        return polygon if len(polygon) >= 12 else None

    def _compute_topdown_radius_rel(self, point: PointF) -> float:
        return float(math.hypot(point[0] - TOPDOWN_CENTER_X, point[1] - TOPDOWN_CENTER_Y) / OUTER_DOUBLE_RADIUS_PX)

    def _compute_combined_confidence(self, candidate_confidence: float, impact_confidence: float) -> float:
        wc = float(self.config.weight_candidate_confidence)
        wi = float(self.config.weight_impact_confidence)
        total = wc + wi
        if total <= 0.0:
            return float(max(0.0, min(1.0, impact_confidence)))
        return float(max(0.0, min(1.0, (wc*candidate_confidence + wi*impact_confidence) / total)))


def build_single_cam_detector(**kwargs: Any) -> SingleCamDetector:
    return SingleCamDetector(**kwargs)


def detect_single_cam(frame: np.ndarray, reference_frame: np.ndarray, *, detector: Optional[SingleCamDetector] = None, **kwargs: Any) -> SingleCamDetectionResult:
    if detector is None:
        detector = SingleCamDetector(**{k: v for k, v in kwargs.items() if k not in {"board_mask", "board_polygon"}})
    return detector.detect(
        frame=frame,
        reference_frame=reference_frame,
        board_mask=kwargs.get("board_mask"),
        board_polygon=kwargs.get("board_polygon"),
    )


def _validate_frame(frame: np.ndarray, name: str) -> None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"{name} ist ungültig.")


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Nicht unterstützte Bildform: {frame.shape}")


def _coerce_point(value: Any) -> Optional[PointF]:
    if value is None:
        return None
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return None


def _coerce_bbox(value: Any) -> BBox:
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) >= 4:
        return tuple(int(round(float(v))) for v in value[:4])  # type: ignore[return-value]
    return (0, 0, 0, 0)


def _bbox_center_or_point(bbox: BBox, point: PointF) -> PointF:
    x, y, w, h = bbox
    return (float(x + w/2.0), float(y + h/2.0)) if w > 0 and h > 0 else point


def _expand_bbox(bbox: BBox, margin: int) -> BBox:
    x, y, w, h = bbox
    return x-margin, y-margin, w+2*margin, h+2*margin


def _point_in_bbox(point: PointF, bbox: BBox) -> bool:
    x, y = point
    bx, by, bw, bh = bbox
    return bx <= x <= bx+bw and by <= y <= by+bh


def _round_point(point: PointF) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


__all__ = [
    "PointF", "BBox", "SingleCamDetectorConfig", "KeypointImpactEstimate",
    "SingleCamScoredEstimate", "SingleCamDetectionResult", "SingleCamDetector",
    "build_single_cam_detector", "detect_single_cam",
]
