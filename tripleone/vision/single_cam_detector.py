# vision/single_cam_detector.py
# Zweck:
# Diese Datei orchestriert die komplette Single-Camera-Pipeline:
#
# 1) Kandidaten finden
#    -> vision/dart_candidate_detector.py
#
# 2) finalen Impact-Punkt im Bild schätzen
#    -> vision/impact_estimator.py
#
# 3) finalen Bildpunkt auf ein Dartfeld mappen
#    -> vision/score_mapper.py
#
# WICHTIG:
# Diese Datei enthält bewusst KEINE eigene:
# - Board-Geometrie
# - Ring-/Sektorlogik
# - Homography-Berechnung
# - Score-Berechnung
#
# Sie ist nur der Orchestrator zwischen den sauberen Schichten.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# Robuste Imports:
# - normaler Paketbetrieb
# - Fallback für direkte Ausführung / Tests
try:
    from .dart_candidate_detector import (
        CandidateDetectionResult,
        CandidateDetectorConfig,
        DartCandidateDetector,
        detect_dart_candidates,
    )
    from .impact_estimator import (
        ImpactEstimate,
        ImpactEstimationResult,
        ImpactEstimator,
        ImpactEstimatorConfig,
        estimate_impacts_from_detection_result,
    )
    from .score_mapper import (
        ScoreMapper,
        ScoredHit,
        build_score_mapper,
    )
except ImportError:  # pragma: no cover
    from vision.dart_candidate_detector import (  # type: ignore
        CandidateDetectionResult,
        CandidateDetectorConfig,
        DartCandidateDetector,
        detect_dart_candidates,
    )
    from vision.impact_estimator import (  # type: ignore
        ImpactEstimate,
        ImpactEstimationResult,
        ImpactEstimator,
        ImpactEstimatorConfig,
        estimate_impacts_from_detection_result,
    )
    from vision.score_mapper import (  # type: ignore
        ScoreMapper,
        ScoredHit,
        build_score_mapper,
    )

logger = logging.getLogger(__name__)

PointF = tuple[float, float]
BBox = tuple[int, int, int, int]


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SingleCamDetectorConfig:
    """
    Konfiguration für die Orchestrierung der Single-Camera-Pipeline.

    Wichtig:
    Diese Datei trifft keine geometrischen oder scoring-spezifischen Fachentscheidungen.
    Sie entscheidet nur:
    - wie viele Impact-Schätzungen weitergescort werden
    - welche Mindestkonfidenzen gelten
    - wie die finale Ranking-Konfidenz gemischt wird
    """

    # Wie viele Impact-Schätzungen sollen maximal an den ScoreMapper gehen?
    max_estimates_to_score: int = 3

    # Wenn False, wird nur die beste Impact-Schätzung gescort.
    score_all_estimates: bool = True

    # Mindestkonfidenzen
    min_impact_confidence: float = 0.01
    min_combined_confidence: float = 0.01

    # Gewichtung für finale Ranking-Konfidenz
    weight_candidate_confidence: float = 0.40
    weight_impact_confidence: float = 0.60

    # Debug/Ergebnisverhalten
    keep_debug_images: bool = True
    keep_stage_results: bool = True
    render_stage_overlays: bool = True


# -----------------------------------------------------------------------------
# Ergebnisdatenmodelle
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class SingleCamScoredEstimate:
    """
    Ein einzelnes vollständiges Single-Cam-Ergebnis:

    Kandidat -> ImpactEstimate -> ScoredHit -> finales Ranking
    """
    rank: int
    candidate_id: int
    image_point: PointF
    scored_hit: ScoredHit
    impact_estimate: ImpactEstimate
    candidate_confidence: float
    impact_confidence: float
    combined_confidence: float
    bbox: BBox
    centroid: PointF
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """
        Normalisiertes Dart-Label, z. B. D20, T19, SBULL, MISS.
        """
        return self.scored_hit.label

    @property
    def score(self) -> int:
        """
        Numerischer Dart-Score.
        """
        return self.scored_hit.score

    @property
    def ring(self) -> str:
        """
        Ring-Typ, z. B. S, D, T, SBULL, DBULL, MISS.
        """
        return self.scored_hit.ring

    @property
    def segment(self) -> Optional[int]:
        """
        Sektorzahl oder None bei Bull/Miss.
        """
        return self.scored_hit.segment

    @property
    def multiplier(self) -> int:
        """
        Multiplikator des Trefferfelds.
        """
        return self.scored_hit.multiplier

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisierung für Debug/API.
        """
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
    """
    Gesamtergebnis der Single-Cam-Pipeline.
    """
    scored_estimates: list[SingleCamScoredEstimate]
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)
    candidate_result: Optional[CandidateDetectionResult] = None
    impact_result: Optional[ImpactEstimationResult] = None

    @property
    def best_estimate(self) -> Optional[SingleCamScoredEstimate]:
        """
        Bestes Gesamtergebnis.
        """
        if not self.scored_estimates:
            return None
        return self.scored_estimates[0]

    @property
    def best_hit(self) -> Optional[ScoredHit]:
        """
        Direktzugriff auf das beste ScoredHit-Objekt.
        """
        if self.best_estimate is None:
            return None
        return self.best_estimate.scored_hit

    @property
    def best_label(self) -> Optional[str]:
        """
        Direktzugriff auf das beste Hit-Label.
        """
        if self.best_estimate is None:
            return None
        return self.best_estimate.label

    @property
    def best_score(self) -> Optional[int]:
        """
        Direktzugriff auf den besten numerischen Score.
        """
        if self.best_estimate is None:
            return None
        return self.best_estimate.score

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisierung ohne Debug-Bildarrays.
        """
        return {
            "metadata": self.metadata,
            "best_label": self.best_label,
            "best_score": self.best_score,
            "scored_estimates": [estimate.to_dict() for estimate in self.scored_estimates],
            "candidate_result": None if self.candidate_result is None else self.candidate_result.to_dict(),
            "impact_result": None if self.impact_result is None else self.impact_result.to_dict(),
        }

    def render_debug_overlay(
        self,
        frame: np.ndarray,
        *,
        max_estimates: Optional[int] = None,
    ) -> np.ndarray:
        """
        Zeichnet das finale Single-Cam-Ergebnis auf ein Bild.

        Reihenfolge:
        - optional Kandidaten-/Impact-Overlays
        - finale gescorte Treffer
        """
        canvas = _ensure_bgr(frame)

        if self.candidate_result is not None:
            canvas = self.candidate_result.render_debug_overlay(canvas)

        if self.impact_result is not None:
            canvas = self.impact_result.render_debug_overlay(canvas, max_estimates=max_estimates)

        count = len(self.scored_estimates) if max_estimates is None else min(len(self.scored_estimates), max_estimates)

        for estimate in self.scored_estimates[:count]:
            x, y, w, h = estimate.bbox
            px, py = _round_point(estimate.image_point)
            cx, cy = _round_point(estimate.centroid)

            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.circle(canvas, (px, py), 6, (0, 0, 255), 2)
            cv2.circle(canvas, (cx, cy), 3, (255, 255, 0), -1)
            cv2.line(canvas, (cx, cy), (px, py), (0, 165, 255), 1, cv2.LINE_AA)

            text = (
                f"#{estimate.rank} | {estimate.label} | score={estimate.score} | "
                f"comb={estimate.combined_confidence:.2f}"
            )

            cv2.putText(
                canvas,
                text,
                (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas


# -----------------------------------------------------------------------------
# Hauptklasse
# -----------------------------------------------------------------------------

class SingleCamDetector:
    """
    Orchestrator für die Single-Camera-Erkennung.

    Pipeline:
    frame/reference -> candidate detector -> impact estimator -> score mapper

    Diese Klasse darf NICHT:
    - eigene Board-Geometrie erzeugen
    - eigene Ring-/Sektorlogik rechnen
    - eigene Homography rechnen
    """

    def __init__(
        self,
        *,
        config: Optional[SingleCamDetectorConfig] = None,
        candidate_detector: Optional[DartCandidateDetector] = None,
        impact_estimator: Optional[ImpactEstimator] = None,
        score_mapper: Optional[ScoreMapper] = None,
        candidate_detector_config: Optional[CandidateDetectorConfig] = None,
        impact_estimator_config: Optional[ImpactEstimatorConfig] = None,
        manual_points: Optional[list[Any]] = None,
        calibration_record: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        image_size: Optional[tuple[int, int]] = None,
        pipeline_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config or SingleCamDetectorConfig()

        self.candidate_detector = candidate_detector or DartCandidateDetector(
            config=candidate_detector_config
        )
        self.impact_estimator = impact_estimator or ImpactEstimator(
            config=impact_estimator_config
        )

        self._score_mapper: Optional[ScoreMapper] = None
        self._pipeline_kwargs = pipeline_kwargs or {}

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

    # -------------------------------------------------------------------------
    # ScoreMapper-Verwaltung
    # -------------------------------------------------------------------------

    @property
    def score_mapper(self) -> Optional[ScoreMapper]:
        """
        Zugriff auf den aktuellen ScoreMapper.
        """
        return self._score_mapper

    def set_score_mapper(self, score_mapper: ScoreMapper) -> None:
        """
        Setzt einen fertigen ScoreMapper.
        """
        if score_mapper is None:
            raise ValueError("score_mapper must not be None.")
        self._score_mapper = score_mapper

    def rebuild_score_mapper_from_manual_points(
        self,
        manual_points: list[Any],
        *,
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Baut den ScoreMapper aus 4 manuellen Kalibrierpunkten neu auf.
        """
        self._score_mapper = build_score_mapper(
            manual_points=manual_points,
            image_size=image_size,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    def rebuild_score_mapper_from_record(self, calibration_record: Any) -> None:
        """
        Baut den ScoreMapper aus einem Calibration-Record neu auf.
        """
        self._score_mapper = build_score_mapper(
            calibration_record=calibration_record,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    def rebuild_score_mapper_from_pipeline(self, pipeline: Any) -> None:
        """
        Setzt einen bereits vorhandenen Geometrie-Pipeline-Kontext.
        """
        self._score_mapper = build_score_mapper(
            pipeline=pipeline,
            pipeline_kwargs=self._pipeline_kwargs,
        )

    # -------------------------------------------------------------------------
    # Öffentliche Haupt-API
    # -------------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
    ) -> SingleCamDetectionResult:
        """
        Führt die komplette Single-Camera-Pipeline aus.

        Schritte:
        1) Kandidaten finden
        2) Impact-Punkte schätzen
        3) finale Bildpunkte scoren
        4) Ergebnisse sortieren und debugbar zurückgeben
        """
        self._ensure_score_mapper_ready()
        _validate_frame(frame, name="frame")
        _validate_frame(reference_frame, name="reference_frame")

        image_shape = frame.shape[:2]

        # --------------------------------------------------------------
        # Schritt 1: Kandidaten finden
        # --------------------------------------------------------------
        candidate_result = self.candidate_detector.detect_candidates(
            frame=frame,
            reference_frame=reference_frame,
            board_mask=board_mask,
            board_polygon=board_polygon,
        )

        # --------------------------------------------------------------
        # Schritt 2: finale Impact-Punkte schätzen
        # --------------------------------------------------------------
        impact_result = self.impact_estimator.estimate_from_detection_result(
            detection_result=candidate_result,
            image_shape=image_shape,
        )

        # --------------------------------------------------------------
        # Schritt 3: Impact-Punkte scoren
        # --------------------------------------------------------------
        scored_estimates = self._score_impacts(impact_result)

        metadata = {
            "input_shape": tuple(int(v) for v in frame.shape),
            "candidate_count": len(candidate_result.candidates),
            "impact_count": len(impact_result.estimates),
            "scored_count": len(scored_estimates),
            "best_label": scored_estimates[0].label if scored_estimates else None,
            "best_score": scored_estimates[0].score if scored_estimates else None,
            "config": _dataclass_to_dict(self.config),
        }

        debug_images: dict[str, np.ndarray] = {}
        if self.config.keep_debug_images:
            debug_images.update(candidate_result.debug_images)

            if self.config.render_stage_overlays:
                debug_images["candidate_overlay"] = candidate_result.render_debug_overlay(frame)
                debug_images["impact_overlay"] = impact_result.render_debug_overlay(frame)
                final_result_preview = SingleCamDetectionResult(
                    scored_estimates=scored_estimates,
                    metadata=metadata,
                    debug_images={},
                    candidate_result=candidate_result if self.config.keep_stage_results else None,
                    impact_result=impact_result if self.config.keep_stage_results else None,
                )
                debug_images["single_cam_overlay"] = final_result_preview.render_debug_overlay(frame)

        return SingleCamDetectionResult(
            scored_estimates=scored_estimates,
            metadata=metadata,
            debug_images=debug_images,
            candidate_result=candidate_result if self.config.keep_stage_results else None,
            impact_result=impact_result if self.config.keep_stage_results else None,
        )

    def detect_best_hit(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
    ) -> Optional[ScoredHit]:
        """
        Komfortmethode: führt die Pipeline aus und gibt direkt das beste ScoredHit
        zurück.
        """
        result = self.detect(
            frame=frame,
            reference_frame=reference_frame,
            board_mask=board_mask,
            board_polygon=board_polygon,
        )
        return result.best_hit

    # -------------------------------------------------------------------------
    # Interne Hilfslogik
    # -------------------------------------------------------------------------

    def _ensure_score_mapper_ready(self) -> None:
        """
        Stellt sicher, dass ein ScoreMapper konfiguriert ist.
        """
        if self._score_mapper is None:
            raise RuntimeError(
                "SingleCamDetector has no ScoreMapper configured. "
                "Provide one via constructor or rebuild_score_mapper_*()."
            )

    def _score_impacts(
        self,
        impact_result: ImpactEstimationResult,
    ) -> list[SingleCamScoredEstimate]:
        """
        Bewertet die Impact-Schätzungen über den ScoreMapper und erzeugt finale
        SingleCamScoredEstimate-Objekte.
        """
        assert self._score_mapper is not None, "ScoreMapper must be configured before scoring."

        estimates_to_score = list(impact_result.estimates)

        if not self.config.score_all_estimates:
            estimates_to_score = estimates_to_score[:1]
        else:
            max_count = max(1, int(self.config.max_estimates_to_score))
            estimates_to_score = estimates_to_score[:max_count]

        scored_estimates: list[SingleCamScoredEstimate] = []

        for estimate in estimates_to_score:
            if estimate.confidence < self.config.min_impact_confidence:
                continue

            scored_hit = self._score_mapper.score_image_point(estimate.impact_point)

            combined_confidence = self._compute_combined_confidence(
                candidate_confidence=estimate.source_candidate_confidence,
                impact_confidence=estimate.confidence,
            )

            if combined_confidence < self.config.min_combined_confidence:
                continue

            candidate_bbox = _coerce_bbox(estimate.bbox)
            candidate_centroid = _coerce_point(estimate.centroid)

            scored_estimates.append(
                SingleCamScoredEstimate(
                    rank=0,  # wird nach Sortierung gesetzt
                    candidate_id=int(estimate.candidate_id),
                    image_point=_coerce_point(estimate.impact_point),
                    scored_hit=scored_hit,
                    impact_estimate=estimate,
                    candidate_confidence=float(estimate.source_candidate_confidence),
                    impact_confidence=float(estimate.confidence),
                    combined_confidence=float(combined_confidence),
                    bbox=candidate_bbox,
                    centroid=candidate_centroid,
                    debug={
                        "impact_method": estimate.method,
                        "hypothesis_count": estimate.hypothesis_count,
                    },
                )
            )

        scored_estimates.sort(
            key=lambda item: item.combined_confidence,
            reverse=True,
        )

        for rank, estimate in enumerate(scored_estimates, start=1):
            estimate.rank = rank

        return scored_estimates

    def _compute_combined_confidence(
        self,
        *,
        candidate_confidence: float,
        impact_confidence: float,
    ) -> float:
        """
        Mischt Kandidaten- und Impact-Konfidenz für das finale Ranking.
        """
        candidate_part = _clip01(candidate_confidence)
        impact_part = _clip01(impact_confidence)

        combined = (
            self.config.weight_candidate_confidence * candidate_part
            + self.config.weight_impact_confidence * impact_part
        )
        return _clip01(combined)


# -----------------------------------------------------------------------------
# Modulweite Convenience-Funktionen
# -----------------------------------------------------------------------------

def build_single_cam_detector(
    *,
    config: Optional[SingleCamDetectorConfig] = None,
    candidate_detector: Optional[DartCandidateDetector] = None,
    impact_estimator: Optional[ImpactEstimator] = None,
    score_mapper: Optional[ScoreMapper] = None,
    candidate_detector_config: Optional[CandidateDetectorConfig] = None,
    impact_estimator_config: Optional[ImpactEstimatorConfig] = None,
    manual_points: Optional[list[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> SingleCamDetector:
    """
    Bequemer Builder für den SingleCamDetector.
    """
    return SingleCamDetector(
        config=config,
        candidate_detector=candidate_detector,
        impact_estimator=impact_estimator,
        score_mapper=score_mapper,
        candidate_detector_config=candidate_detector_config,
        impact_estimator_config=impact_estimator_config,
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )


def detect_single_cam(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    *,
    board_mask: Optional[np.ndarray] = None,
    board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
    config: Optional[SingleCamDetectorConfig] = None,
    candidate_detector_config: Optional[CandidateDetectorConfig] = None,
    impact_estimator_config: Optional[ImpactEstimatorConfig] = None,
    score_mapper: Optional[ScoreMapper] = None,
    manual_points: Optional[list[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> SingleCamDetectionResult:
    """
    Modulweiter Convenience-Wrapper für die komplette Single-Cam-Pipeline.
    """
    detector = build_single_cam_detector(
        config=config,
        candidate_detector_config=candidate_detector_config,
        impact_estimator_config=impact_estimator_config,
        score_mapper=score_mapper,
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )
    return detector.detect(
        frame=frame,
        reference_frame=reference_frame,
        board_mask=board_mask,
        board_polygon=board_polygon,
    )


# -----------------------------------------------------------------------------
# Interne Hilfsfunktionen
# -----------------------------------------------------------------------------

def _validate_frame(frame: np.ndarray, *, name: str) -> None:
    """
    Prüft, ob ein Eingabebild grundsätzlich gültig ist.
    """
    if frame is None:
        raise ValueError(f"{name} must not be None.")

    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame)!r}.")

    if frame.ndim not in (2, 3):
        raise ValueError(f"{name} must have ndim 2 or 3, got {frame.ndim}.")

    if frame.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """
    Normalisiert ein Bild auf BGR.
    """
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()

    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Unsupported frame shape for BGR conversion: {frame.shape}")


def _coerce_point(value: Any) -> PointF:
    """
    Normalisiert einen 2D-Punkt auf (x, y).
    """
    if value is None:
        raise ValueError("Point must not be None.")

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        raise ValueError(f"Point dict must contain x/y, got {value!r}")

    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        return float(value[0]), float(value[1])

    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)

    raise ValueError(f"Unsupported point value: {value!r}")


def _coerce_bbox(value: Any) -> BBox:
    """
    Normalisiert eine Bounding-Box auf (x, y, w, h).
    """
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 4:
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])

    raise ValueError(f"Unsupported bbox value: {value!r}")


def _round_point(point: PointF) -> tuple[int, int]:
    """
    Rundet einen Float-Punkt für OpenCV-Zeichenoperationen.
    """
    return int(round(point[0])), int(round(point[1]))


def _clip01(value: float) -> float:
    """
    Beschränkt einen Wert auf [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, value)))


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    """
    Kleiner Helfer für Debug-Metadaten.
    """
    if hasattr(value, "__dataclass_fields__"):
        result: dict[str, Any] = {}
        for field_name in value.__dataclass_fields__:
            result[field_name] = getattr(value, field_name)
        return result

    raise TypeError(f"Expected dataclass instance, got {type(value)!r}.")


__all__ = [
    "PointF",
    "BBox",
    "SingleCamDetectorConfig",
    "SingleCamScoredEstimate",
    "SingleCamDetectionResult",
    "SingleCamDetector",
    "build_single_cam_detector",
    "detect_single_cam",
]