from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


PointF = tuple[float, float]


# -----------------------------------------------------------------------------
# Datenklassen
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SingleCamEstimateObservation:
    """
    Beobachtung für genau EIN scored_estimate aus einem SingleCamDetectionResult.
    Diese Klasse ist bewusst tolerant gebaut, damit sie mit deinem aktuellen
    Projektstand funktioniert, ohne an kleinen Strukturunterschieden zu brechen.
    """
    estimate_rank: int
    image_point: Optional[PointF]
    topdown_point: Optional[PointF]
    label: Optional[str]
    score: Optional[int]
    ring: Optional[str]
    segment: Optional[int]
    multiplier: Optional[int]
    combined_confidence: float
    impact_confidence: float
    candidate_confidence: float
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimate_rank": self.estimate_rank,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "combined_confidence": self.combined_confidence,
            "impact_confidence": self.impact_confidence,
            "candidate_confidence": self.candidate_confidence,
            "debug": self.debug,
        }


@dataclass(slots=True)
class SingleCamObservation:
    """
    Zusammenfassung der kompletten Single-Cam-Pipeline für genau EIN Frame.
    """
    camera_index: int
    frame_ok: bool
    detector_ready: bool
    reference_available: bool
    candidate_count: int
    impact_count: int
    scored_count: int
    best_image_point: Optional[PointF]
    best_topdown_point: Optional[PointF]
    best_label: Optional[str]
    best_score: Optional[int]
    best_ring: Optional[str]
    best_segment: Optional[int]
    best_multiplier: Optional[int]
    best_combined_confidence: float
    best_impact_confidence: float
    best_candidate_confidence: float
    estimates: list[SingleCamEstimateObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    raw_result: Any = None

    @property
    def has_hit(self) -> bool:
        return self.best_image_point is not None and self.best_topdown_point is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "frame_ok": self.frame_ok,
            "detector_ready": self.detector_ready,
            "reference_available": self.reference_available,
            "candidate_count": self.candidate_count,
            "impact_count": self.impact_count,
            "scored_count": self.scored_count,
            "best_image_point": self.best_image_point,
            "best_topdown_point": self.best_topdown_point,
            "best_label": self.best_label,
            "best_score": self.best_score,
            "best_ring": self.best_ring,
            "best_segment": self.best_segment,
            "best_multiplier": self.best_multiplier,
            "best_combined_confidence": self.best_combined_confidence,
            "best_impact_confidence": self.best_impact_confidence,
            "best_candidate_confidence": self.best_candidate_confidence,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
            "metadata": self.metadata,
            "debug": self.debug,
        }


# -----------------------------------------------------------------------------
# Kleine Helpers
# -----------------------------------------------------------------------------

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


def _safe_point(value: Any) -> Optional[PointF]:
    if value is None:
        return None

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
        return None

    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    return None


def _get_score_mapper(detector: Any) -> Any:
    """
    Holt robust den ScoreMapper aus dem SingleCamDetector.
    """
    mapper = getattr(detector, "_score_mapper", None)
    if mapper is not None:
        return mapper

    mapper = getattr(detector, "score_mapper", None)
    if mapper is not None:
        return mapper

    raise RuntimeError("SingleCamDetector has no ScoreMapper configured.")


def _project_image_point_to_topdown(score_mapper: Any, image_point: Optional[PointF]) -> Optional[PointF]:
    if image_point is None:
        return None

    if not hasattr(score_mapper, "image_point_to_topdown"):
        raise RuntimeError("ScoreMapper has no method image_point_to_topdown(...).")

    try:
        topdown = score_mapper.image_point_to_topdown(image_point)
        if topdown is None:
            return None
        return float(topdown[0]), float(topdown[1])
    except Exception:
        return None


def _extract_scored_hit_fields(scored_estimate: Any) -> tuple[Optional[str], Optional[int], Optional[str], Optional[int], Optional[int]]:
    """
    Liest Felder robust entweder direkt vom scored_estimate oder aus scored_hit.
    """
    scored_hit = getattr(scored_estimate, "scored_hit", None)

    label = getattr(scored_estimate, "label", None)
    score = getattr(scored_estimate, "score", None)
    ring = getattr(scored_estimate, "ring", None)
    segment = getattr(scored_estimate, "segment", None)
    multiplier = getattr(scored_estimate, "multiplier", None)

    if scored_hit is not None:
        label = getattr(scored_hit, "label", label)
        score = getattr(scored_hit, "score", score)
        ring = getattr(scored_hit, "ring", ring)
        segment = getattr(scored_hit, "segment", segment)
        multiplier = getattr(scored_hit, "multiplier", multiplier)

    return (
        None if label is None else str(label),
        None if score is None else _safe_int(score),
        None if ring is None else str(ring),
        None if segment is None else _safe_int(segment),
        None if multiplier is None else _safe_int(multiplier),
    )


def _extract_image_point_from_scored_estimate(scored_estimate: Any) -> Optional[PointF]:
    """
    Holt möglichst robust den lokalen Bildpunkt.
    """
    # 1) direkt auf scored_estimate
    direct = _safe_point(getattr(scored_estimate, "image_point", None))
    if direct is not None:
        return direct

    # 2) über impact_estimate
    impact_estimate = getattr(scored_estimate, "impact_estimate", None)
    if impact_estimate is not None:
        point = _safe_point(getattr(impact_estimate, "impact_point", None))
        if point is not None:
            return point

    # 3) Fallback debug
    debug = dict(getattr(scored_estimate, "debug", {}) or {})
    for key in ("image_point", "impact_point"):
        point = _safe_point(debug.get(key))
        if point is not None:
            return point

    return None


def _extract_confidences(scored_estimate: Any) -> tuple[float, float, float]:
    """
    Liest combined / impact / candidate confidence robust.
    """
    combined_confidence = _safe_float(getattr(scored_estimate, "combined_confidence", None), 0.0)
    impact_confidence = _safe_float(getattr(scored_estimate, "impact_confidence", None), 0.0)
    candidate_confidence = _safe_float(getattr(scored_estimate, "candidate_confidence", None), 0.0)

    impact_estimate = getattr(scored_estimate, "impact_estimate", None)
    if impact_estimate is not None:
        if impact_confidence <= 0.0:
            impact_confidence = _safe_float(getattr(impact_estimate, "confidence", None), 0.0)
        if candidate_confidence <= 0.0:
            candidate_confidence = _safe_float(
                getattr(impact_estimate, "source_candidate_confidence", None),
                0.0,
            )

    # Fallback: wenn combined_confidence fehlt, grob aus den beiden ableiten
    if combined_confidence <= 0.0:
        if impact_confidence > 0.0 or candidate_confidence > 0.0:
            combined_confidence = (impact_confidence + candidate_confidence) / 2.0

    return combined_confidence, impact_confidence, candidate_confidence


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

def build_single_cam_observation_from_result(
    *,
    camera_index: int,
    detector: Any,
    detection_result: Any,
    reference_available: Optional[bool] = None,
) -> SingleCamObservation:
    """
    Baut eine robuste SingleCamObservation aus einem existierenden
    SingleCamDetectionResult.
    """
    score_mapper = _get_score_mapper(detector)

    candidate_result = getattr(detection_result, "candidate_result", None)
    impact_result = getattr(detection_result, "impact_result", None)
    scored_estimates = list(getattr(detection_result, "scored_estimates", []) or [])

    candidate_count = len(getattr(candidate_result, "candidates", []) or []) if candidate_result is not None else 0
    impact_count = len(getattr(impact_result, "estimates", []) or []) if impact_result is not None else 0
    scored_count = len(scored_estimates)

    if reference_available is None:
        reference_available = True

    estimate_observations: list[SingleCamEstimateObservation] = []

    for idx, scored_estimate in enumerate(scored_estimates):
        image_point = _extract_image_point_from_scored_estimate(scored_estimate)
        topdown_point = _project_image_point_to_topdown(score_mapper, image_point)

        label, score, ring, segment, multiplier = _extract_scored_hit_fields(scored_estimate)
        combined_confidence, impact_confidence, candidate_confidence = _extract_confidences(scored_estimate)

        estimate_observations.append(
            SingleCamEstimateObservation(
                estimate_rank=_safe_int(getattr(scored_estimate, "rank", idx + 1), idx + 1),
                image_point=image_point,
                topdown_point=topdown_point,
                label=label,
                score=score,
                ring=ring,
                segment=segment,
                multiplier=multiplier,
                combined_confidence=combined_confidence,
                impact_confidence=impact_confidence,
                candidate_confidence=candidate_confidence,
                debug=dict(getattr(scored_estimate, "debug", {}) or {}),
            )
        )

    best = estimate_observations[0] if estimate_observations else None

    return SingleCamObservation(
        camera_index=_safe_int(camera_index),
        frame_ok=True,
        detector_ready=True,
        reference_available=bool(reference_available),
        candidate_count=candidate_count,
        impact_count=impact_count,
        scored_count=scored_count,
        best_image_point=None if best is None else best.image_point,
        best_topdown_point=None if best is None else best.topdown_point,
        best_label=None if best is None else best.label,
        best_score=None if best is None else best.score,
        best_ring=None if best is None else best.ring,
        best_segment=None if best is None else best.segment,
        best_multiplier=None if best is None else best.multiplier,
        best_combined_confidence=0.0 if best is None else best.combined_confidence,
        best_impact_confidence=0.0 if best is None else best.impact_confidence,
        best_candidate_confidence=0.0 if best is None else best.candidate_confidence,
        estimates=estimate_observations,
        metadata=dict(getattr(detection_result, "metadata", {}) or {}),
        debug={
            "has_candidate_result": candidate_result is not None,
            "has_impact_result": impact_result is not None,
            "raw_result_type": type(detection_result).__name__,
        },
        raw_result=detection_result,
    )


def run_single_cam_observation(
    *,
    camera_index: int,
    detector: Any,
    frame: Any,
    reference_frame: Any,
    board_mask: Any = None,
    board_polygon: Any = None,
    reference_available: bool = True,
) -> SingleCamObservation:
    """
    Führt den bestehenden SingleCamDetector aus und baut daraus eine
    SingleCamObservation.

    Erwartung:
    - detector.detect(frame=..., reference_frame=..., ...)
    """
    if detector is None:
        return SingleCamObservation(
            camera_index=_safe_int(camera_index),
            frame_ok=frame is not None,
            detector_ready=False,
            reference_available=bool(reference_available),
            candidate_count=0,
            impact_count=0,
            scored_count=0,
            best_image_point=None,
            best_topdown_point=None,
            best_label=None,
            best_score=None,
            best_ring=None,
            best_segment=None,
            best_multiplier=None,
            best_combined_confidence=0.0,
            best_impact_confidence=0.0,
            best_candidate_confidence=0.0,
            estimates=[],
            metadata={},
            debug={"error": "detector is None"},
            raw_result=None,
        )

    if frame is None:
        return SingleCamObservation(
            camera_index=_safe_int(camera_index),
            frame_ok=False,
            detector_ready=True,
            reference_available=bool(reference_available),
            candidate_count=0,
            impact_count=0,
            scored_count=0,
            best_image_point=None,
            best_topdown_point=None,
            best_label=None,
            best_score=None,
            best_ring=None,
            best_segment=None,
            best_multiplier=None,
            best_combined_confidence=0.0,
            best_impact_confidence=0.0,
            best_candidate_confidence=0.0,
            estimates=[],
            metadata={},
            debug={"error": "frame is None"},
            raw_result=None,
        )

    try:
        detection_result = detector.detect(
            frame=frame,
            reference_frame=reference_frame,
            board_mask=board_mask,
            board_polygon=board_polygon,
        )
    except Exception as exc:
        return SingleCamObservation(
            camera_index=_safe_int(camera_index),
            frame_ok=True,
            detector_ready=True,
            reference_available=bool(reference_available),
            candidate_count=0,
            impact_count=0,
            scored_count=0,
            best_image_point=None,
            best_topdown_point=None,
            best_label=None,
            best_score=None,
            best_ring=None,
            best_segment=None,
            best_multiplier=None,
            best_combined_confidence=0.0,
            best_impact_confidence=0.0,
            best_candidate_confidence=0.0,
            estimates=[],
            metadata={},
            debug={"error": f"detector.detect failed: {exc}"},
            raw_result=None,
        )

    return build_single_cam_observation_from_result(
        camera_index=camera_index,
        detector=detector,
        detection_result=detection_result,
        reference_available=reference_available,
    )