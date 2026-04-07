from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from vision.single_cam_observation import (
    SingleCamEstimateObservation,
    SingleCamObservation,
)

PointF = tuple[float, float]


@dataclass(slots=True)
class CameraFusionObservation:
    """
    Eine einzelne verwertbare Beobachtung einer Kamera im gemeinsamen Boardraum.
    """
    camera_index: int
    estimate_rank: int
    image_point: PointF
    topdown_point: PointF
    combined_confidence: float
    impact_confidence: float
    candidate_confidence: float
    label: Optional[str] = None
    score: Optional[int] = None
    ring: Optional[str] = None
    segment: Optional[int] = None
    multiplier: Optional[int] = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "estimate_rank": self.estimate_rank,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "combined_confidence": self.combined_confidence,
            "impact_confidence": self.impact_confidence,
            "candidate_confidence": self.candidate_confidence,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "debug": self.debug,
        }


@dataclass(slots=True)
class FusedBoardImpact:
    """
    Finaler zusammengeführter Treffer im gemeinsamen Boardraum.
    """
    topdown_point: PointF
    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int
    confidence: float
    observations_used: list[CameraFusionObservation] = field(default_factory=list)
    observations_rejected: list[CameraFusionObservation] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topdown_point": self.topdown_point,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "confidence": self.confidence,
            "observations_used": [obs.to_dict() for obs in self.observations_used],
            "observations_rejected": [obs.to_dict() for obs in self.observations_rejected],
            "debug": self.debug,
        }


@dataclass(slots=True)
class MultiCamFusionConfig:
    """
    Konfiguration für die Fusion mehrerer Kamera-Beobachtungen.
    """
    max_estimates_per_camera: int = 2
    cluster_distance_px: float = 28.0
    outlier_distance_px: float = 22.0
    min_cameras_for_fusion: int = 2
    allow_single_camera_fallback: bool = True
    confidence_floor: float = 0.05


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _point_distance(a: PointF, b: PointF) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _weighted_average_point(points: list[PointF], weights: list[float]) -> PointF:
    if not points:
        raise ValueError("points must not be empty")

    if len(points) != len(weights):
        raise ValueError("points and weights must have same length")

    total = float(sum(weights))
    if total <= 1e-9:
        total = float(len(weights))
        weights = [1.0 for _ in weights]

    x = sum(float(point[0]) * float(weight) for point, weight in zip(points, weights)) / total
    y = sum(float(point[1]) * float(weight) for point, weight in zip(points, weights)) / total
    return float(x), float(y)


class MultiCamFusionEngine:
    """
    Führt mehrere SingleCamObservation-Objekte zu einem gemeinsamen Board-Impact zusammen.
    """

    def __init__(self, config: Optional[MultiCamFusionConfig] = None) -> None:
        self.config = config or MultiCamFusionConfig()

    # -------------------------------------------------------------------------
    # Mapper / ScoreMapper Helpers
    # -------------------------------------------------------------------------

    def _get_score_mapper_from_detector(self, detector: Any) -> Any:
        mapper = getattr(detector, "_score_mapper", None)
        if mapper is not None:
            return mapper

        mapper = getattr(detector, "score_mapper", None)
        if mapper is not None:
            return mapper

        raise RuntimeError("Detector has no ScoreMapper configured.")

    def _resolve_score_mapper(
        self,
        *,
        camera_index: int,
        score_mappers_by_camera: Optional[dict[int, Any]],
        detectors_by_camera: Optional[dict[int, Any]],
    ) -> Any:
        if score_mappers_by_camera is not None:
            mapper = score_mappers_by_camera.get(camera_index)
            if mapper is not None:
                return mapper

        if detectors_by_camera is not None:
            detector = detectors_by_camera.get(camera_index)
            if detector is not None:
                return self._get_score_mapper_from_detector(detector)

        raise RuntimeError(f"No ScoreMapper available for camera {camera_index}.")

    # -------------------------------------------------------------------------
    # Observation Extraction
    # -------------------------------------------------------------------------

    def _estimate_to_fusion_observation(
        self,
        *,
        camera_index: int,
        estimate: SingleCamEstimateObservation,
    ) -> Optional[CameraFusionObservation]:
        if estimate.image_point is None or estimate.topdown_point is None:
            return None

        return CameraFusionObservation(
            camera_index=int(camera_index),
            estimate_rank=int(estimate.estimate_rank),
            image_point=(float(estimate.image_point[0]), float(estimate.image_point[1])),
            topdown_point=(float(estimate.topdown_point[0]), float(estimate.topdown_point[1])),
            combined_confidence=float(estimate.combined_confidence),
            impact_confidence=float(estimate.impact_confidence),
            candidate_confidence=float(estimate.candidate_confidence),
            label=estimate.label,
            score=estimate.score,
            ring=estimate.ring,
            segment=estimate.segment,
            multiplier=estimate.multiplier,
            debug=dict(estimate.debug or {}),
        )

    def _extract_camera_observations(
        self,
        observation: SingleCamObservation,
    ) -> list[CameraFusionObservation]:
        out: list[CameraFusionObservation] = []

        if not observation.frame_ok:
            return out

        if not observation.detector_ready:
            return out

        if not observation.reference_available:
            return out

        max_count = max(1, int(self.config.max_estimates_per_camera))

        for estimate in observation.estimates[:max_count]:
            fusion_obs = self._estimate_to_fusion_observation(
                camera_index=observation.camera_index,
                estimate=estimate,
            )
            if fusion_obs is not None:
                out.append(fusion_obs)

        return out

    # -------------------------------------------------------------------------
    # Clustering / Outlier Rejection
    # -------------------------------------------------------------------------

    def _cluster_observations(
        self,
        observations: list[CameraFusionObservation],
    ) -> list[list[CameraFusionObservation]]:
        clusters: list[list[CameraFusionObservation]] = []
        cluster_distance_px = float(self.config.cluster_distance_px)

        for observation in observations:
            placed = False

            for cluster in clusters:
                center = _weighted_average_point(
                    [item.topdown_point for item in cluster],
                    [
                        max(float(item.combined_confidence), float(self.config.confidence_floor))
                        for item in cluster
                    ],
                )

                if _point_distance(observation.topdown_point, center) <= cluster_distance_px:
                    cluster.append(observation)
                    placed = True
                    break

            if not placed:
                clusters.append([observation])

        return clusters

    def _cluster_score(self, cluster: list[CameraFusionObservation]) -> float:
        if not cluster:
            return 0.0

        unique_camera_count = len({item.camera_index for item in cluster})
        confidence_sum = sum(
            max(float(item.combined_confidence), float(self.config.confidence_floor))
            for item in cluster
        )

        # Kamera-Anzahl bewusst dominant
        return float(unique_camera_count * 1000.0 + confidence_sum)

    def _filter_cluster_outliers(
        self,
        cluster: list[CameraFusionObservation],
    ) -> tuple[list[CameraFusionObservation], list[CameraFusionObservation], PointF]:
        if not cluster:
            raise ValueError("cluster must not be empty")

        weights = [
            max(float(item.combined_confidence), float(self.config.confidence_floor))
            for item in cluster
        ]
        center = _weighted_average_point([item.topdown_point for item in cluster], weights)

        used: list[CameraFusionObservation] = []
        rejected: list[CameraFusionObservation] = []

        outlier_distance_px = float(self.config.outlier_distance_px)

        for item in cluster:
            dist = _point_distance(item.topdown_point, center)
            if dist <= outlier_distance_px:
                used.append(item)
            else:
                rejected.append(item)

        if not used:
            best = max(cluster, key=lambda item: float(item.combined_confidence))
            used = [best]
            rejected = [item for item in cluster if item is not best]

        final_center = _weighted_average_point(
            [item.topdown_point for item in used],
            [
                max(float(item.combined_confidence), float(self.config.confidence_floor))
                for item in used
            ],
        )

        return used, rejected, final_center

    # -------------------------------------------------------------------------
    # Fusion
    # -------------------------------------------------------------------------

    def fuse(
        self,
        *,
        observations_by_camera: dict[int, SingleCamObservation],
        score_mappers_by_camera: Optional[dict[int, Any]] = None,
        detectors_by_camera: Optional[dict[int, Any]] = None,
    ) -> Optional[FusedBoardImpact]:
        """
        Führt mehrere SingleCamObservation-Objekte zusammen.

        Du musst entweder:
        - score_mappers_by_camera
        oder
        - detectors_by_camera
        mitgeben.

        Für das finale Scoring reicht ein beliebiger Mapper einer verwendeten Kamera,
        weil alle Beobachtungen bereits im gemeinsamen Topdown-Raum liegen.
        """
        if score_mappers_by_camera is None and detectors_by_camera is None:
            raise RuntimeError(
                "Either score_mappers_by_camera or detectors_by_camera must be provided."
            )

        all_observations: list[CameraFusionObservation] = []

        for camera_index, observation in observations_by_camera.items():
            if observation is None:
                continue

            if int(camera_index) != int(observation.camera_index):
                # defensive Korrektur
                observation = SingleCamObservation(
                    camera_index=int(camera_index),
                    frame_ok=observation.frame_ok,
                    detector_ready=observation.detector_ready,
                    reference_available=observation.reference_available,
                    candidate_count=observation.candidate_count,
                    impact_count=observation.impact_count,
                    scored_count=observation.scored_count,
                    best_image_point=observation.best_image_point,
                    best_topdown_point=observation.best_topdown_point,
                    best_label=observation.best_label,
                    best_score=observation.best_score,
                    best_ring=observation.best_ring,
                    best_segment=observation.best_segment,
                    best_multiplier=observation.best_multiplier,
                    best_combined_confidence=observation.best_combined_confidence,
                    best_impact_confidence=observation.best_impact_confidence,
                    best_candidate_confidence=observation.best_candidate_confidence,
                    estimates=observation.estimates,
                    metadata=observation.metadata,
                    debug=observation.debug,
                    raw_result=observation.raw_result,
                )

            all_observations.extend(self._extract_camera_observations(observation))

        if not all_observations:
            return None

        clusters = self._cluster_observations(all_observations)
        clusters.sort(key=self._cluster_score, reverse=True)

        best_cluster = clusters[0]
        used, rejected, fused_point = self._filter_cluster_outliers(best_cluster)

        used_camera_count = len({item.camera_index for item in used})
        if used_camera_count < int(self.config.min_cameras_for_fusion):
            if not bool(self.config.allow_single_camera_fallback):
                return None

        reference_camera_index = used[0].camera_index
        score_mapper = self._resolve_score_mapper(
            camera_index=reference_camera_index,
            score_mappers_by_camera=score_mappers_by_camera,
            detectors_by_camera=detectors_by_camera,
        )

        if not hasattr(score_mapper, "score_topdown_point"):
            raise RuntimeError("ScoreMapper has no method score_topdown_point(...).")

        scored_hit = score_mapper.score_topdown_point(fused_point)

        weights = [
            max(float(item.combined_confidence), float(self.config.confidence_floor))
            for item in used
        ]
        fusion_confidence = float(sum(weights) / max(1, len(weights)))
        fusion_confidence = max(0.0, min(1.0, fusion_confidence))

        return FusedBoardImpact(
            topdown_point=(float(fused_point[0]), float(fused_point[1])),
            label=str(getattr(scored_hit, "label")),
            score=int(getattr(scored_hit, "score")),
            ring=str(getattr(scored_hit, "ring")),
            segment=None if getattr(scored_hit, "segment", None) is None else int(getattr(scored_hit, "segment")),
            multiplier=int(getattr(scored_hit, "multiplier")),
            confidence=fusion_confidence,
            observations_used=used,
            observations_rejected=rejected,
            debug={
                "total_observations": len(all_observations),
                "cluster_count": len(clusters),
                "best_cluster_size": len(best_cluster),
                "used_camera_count": used_camera_count,
                "clusters": [
                    [item.to_dict() for item in cluster]
                    for cluster in clusters
                ],
            },
        )