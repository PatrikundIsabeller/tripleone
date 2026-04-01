from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import math


PointF = tuple[float, float]


@dataclass(slots=True)
class CameraFusionObservation:
    """
    Ein lokaler Kameratreffer, bereits in den gemeinsamen Topdown-Raum projiziert.
    """
    camera_index: int
    image_point: PointF
    topdown_point: PointF
    combined_confidence: float
    impact_confidence: float
    candidate_confidence: float
    estimate_rank: int
    source_label: Optional[str] = None
    source_score: Optional[int] = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "combined_confidence": self.combined_confidence,
            "impact_confidence": self.impact_confidence,
            "candidate_confidence": self.candidate_confidence,
            "estimate_rank": self.estimate_rank,
            "source_label": self.source_label,
            "source_score": self.source_score,
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
    Konfiguration für 3-Kamera-Fusion im gemeinsamen Boardraum.
    """
    max_estimates_per_camera: int = 2

    # Zwei Beobachtungen gelten als zusammengehörig, wenn sie im Topdown-Raum
    # höchstens so weit auseinanderliegen.
    cluster_distance_px: float = 28.0

    # Innerhalb eines Clusters werden Beobachtungen verworfen, die zu weit vom
    # gewichteten Mittelpunkt abweichen.
    outlier_distance_px: float = 22.0

    # Für eine stabile Fusion wollen wir idealerweise mind. 2 Kameras.
    min_cameras_for_fusion: int = 2

    # Falls nur 1 Kamera etwas Sinnvolles liefert, kann man optional trotzdem
    # einen Fallback zulassen.
    allow_single_camera_fallback: bool = True

    # Gewichtung in der Mittelung
    use_combined_confidence_as_weight: bool = True
    confidence_floor: float = 0.05


def _point_distance(a: PointF, b: PointF) -> float:
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return float(math.hypot(dx, dy))


def _weighted_average_point(points: list[PointF], weights: list[float]) -> PointF:
    if not points:
        raise ValueError("points must not be empty")

    if len(points) != len(weights):
        raise ValueError("points and weights must have same length")

    total = float(sum(weights))
    if total <= 1e-9:
        total = float(len(weights))
        weights = [1.0 for _ in weights]

    x = sum(float(p[0]) * float(w) for p, w in zip(points, weights)) / total
    y = sum(float(p[1]) * float(w) for p, w in zip(points, weights)) / total
    return float(x), float(y)


class MultiCamFusionEngine:
    """
    Führt lokale Single-Cam-Treffer im gemeinsamen Boardraum zusammen.

    Erwartung:
    - pro Kamera existiert ein SingleCamDetector mit ScoreMapper
    - pro Kamera liegt ein SingleCamDetectionResult vor
    - aus scored_estimates.image_point wird per mapper.image_point_to_topdown(...)
      ein gemeinsamer Boardpunkt erzeugt
    """

    def __init__(self, config: Optional[MultiCamFusionConfig] = None) -> None:
        self.config = config or MultiCamFusionConfig()

    def _get_score_mapper(self, detector: Any) -> Any:
        mapper = getattr(detector, "_score_mapper", None)
        if mapper is None:
            raise RuntimeError("Detector has no configured ScoreMapper.")
        return mapper

    def _extract_observations_from_single_cam_result(
        self,
        *,
        camera_index: int,
        detector: Any,
        detection_result: Any,
    ) -> list[CameraFusionObservation]:
        if detection_result is None:
            return []

        scored_estimates = list(getattr(detection_result, "scored_estimates", []) or [])
        if not scored_estimates:
            return []

        mapper = self._get_score_mapper(detector)
        max_count = max(1, int(self.config.max_estimates_per_camera))

        observations: list[CameraFusionObservation] = []

        for estimate in scored_estimates[:max_count]:
            image_point = tuple(estimate.image_point)
            topdown_point = mapper.image_point_to_topdown(image_point)

            observations.append(
                CameraFusionObservation(
                    camera_index=int(camera_index),
                    image_point=(float(image_point[0]), float(image_point[1])),
                    topdown_point=(float(topdown_point[0]), float(topdown_point[1])),
                    combined_confidence=float(getattr(estimate, "combined_confidence", 0.0)),
                    impact_confidence=float(getattr(estimate, "impact_confidence", 0.0)),
                    candidate_confidence=float(getattr(estimate, "candidate_confidence", 0.0)),
                    estimate_rank=int(getattr(estimate, "rank", 0)),
                    source_label=getattr(getattr(estimate, "scored_hit", None), "label", None),
                    source_score=getattr(getattr(estimate, "scored_hit", None), "score", None),
                    debug=dict(getattr(estimate, "debug", {}) or {}),
                )
            )

        return observations

    def _cluster_observations(
        self,
        observations: list[CameraFusionObservation],
    ) -> list[list[CameraFusionObservation]]:
        clusters: list[list[CameraFusionObservation]] = []
        max_dist = float(self.config.cluster_distance_px)

        for obs in observations:
            placed = False

            for cluster in clusters:
                center = _weighted_average_point(
                    [item.topdown_point for item in cluster],
                    [
                        max(float(item.combined_confidence), float(self.config.confidence_floor))
                        for item in cluster
                    ],
                )

                if _point_distance(obs.topdown_point, center) <= max_dist:
                    cluster.append(obs)
                    placed = True
                    break

            if not placed:
                clusters.append([obs])

        return clusters

    def _score_cluster(self, cluster: list[CameraFusionObservation]) -> float:
        if not cluster:
            return 0.0

        unique_cameras = len({obs.camera_index for obs in cluster})
        confidence_sum = sum(max(obs.combined_confidence, self.config.confidence_floor) for obs in cluster)

        # Kameraanzahl stärker belohnen als reine Confidence
        return float(unique_cameras * 1000.0 + confidence_sum)

    def _filter_cluster_outliers(
        self,
        cluster: list[CameraFusionObservation],
    ) -> tuple[list[CameraFusionObservation], list[CameraFusionObservation], PointF]:
        if not cluster:
            raise ValueError("cluster must not be empty")

        weights = [
            max(float(obs.combined_confidence), float(self.config.confidence_floor))
            for obs in cluster
        ]
        center = _weighted_average_point([obs.topdown_point for obs in cluster], weights)

        keep: list[CameraFusionObservation] = []
        reject: list[CameraFusionObservation] = []

        max_outlier_dist = float(self.config.outlier_distance_px)

        for obs in cluster:
            dist = _point_distance(obs.topdown_point, center)
            if dist <= max_outlier_dist:
                keep.append(obs)
            else:
                reject.append(obs)

        if not keep:
            # Fallback: mindestens die beste Observation behalten
            best = max(cluster, key=lambda item: item.combined_confidence)
            keep = [best]
            reject = [item for item in cluster if item is not best]

        final_center = _weighted_average_point(
            [obs.topdown_point for obs in keep],
            [max(float(obs.combined_confidence), float(self.config.confidence_floor)) for obs in keep],
        )

        return keep, reject, final_center

    def fuse(
        self,
        *,
        detectors_by_camera: dict[int, Any],
        detection_results_by_camera: dict[int, Any],
    ) -> Optional[FusedBoardImpact]:
        """
        Führt die lokalen Single-Cam-Treffer zusammen.
        """
        observations: list[CameraFusionObservation] = []

        for camera_index, detection_result in detection_results_by_camera.items():
            detector = detectors_by_camera.get(camera_index)
            if detector is None:
                continue

            observations.extend(
                self._extract_observations_from_single_cam_result(
                    camera_index=int(camera_index),
                    detector=detector,
                    detection_result=detection_result,
                )
            )

        if not observations:
            return None

        clusters = self._cluster_observations(observations)
        clusters.sort(key=self._score_cluster, reverse=True)

        best_cluster = clusters[0]
        used, rejected, fused_point = self._filter_cluster_outliers(best_cluster)

        used_camera_count = len({obs.camera_index for obs in used})

        if used_camera_count < int(self.config.min_cameras_for_fusion):
            if not bool(self.config.allow_single_camera_fallback):
                return None

        # Referenz-Mapper zum finalen Score:
        # irgendein benutzter Camera-Mapper reicht, da Topdown für alle gleich ist.
        reference_detector = detectors_by_camera[used[0].camera_index]
        reference_mapper = self._get_score_mapper(reference_detector)
        scored_hit = reference_mapper.score_topdown_point(fused_point)

        # Fusions-Confidence
        weights = [max(float(obs.combined_confidence), float(self.config.confidence_floor)) for obs in used]
        fusion_confidence = float(sum(weights) / max(1, len(weights)))
        fusion_confidence = max(0.0, min(1.0, fusion_confidence))

        return FusedBoardImpact(
            topdown_point=(float(fused_point[0]), float(fused_point[1])),
            label=str(scored_hit.label),
            score=int(scored_hit.score),
            ring=str(scored_hit.ring),
            segment=None if scored_hit.segment is None else int(scored_hit.segment),
            multiplier=int(scored_hit.multiplier),
            confidence=fusion_confidence,
            observations_used=used,
            observations_rejected=rejected,
            debug={
                "total_observations": len(observations),
                "cluster_count": len(clusters),
                "best_cluster_size": len(best_cluster),
                "used_camera_count": used_camera_count,
                "clusters": [
                    [obs.to_dict() for obs in cluster]
                    for cluster in clusters
                ],
            },
        )