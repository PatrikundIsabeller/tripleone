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
import math
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
    from .calibration_geometry import (
        TOPDOWN_CENTER_X,
        TOPDOWN_CENTER_Y,
        OUTER_DOUBLE_RADIUS_PX,
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
    from vision.calibration_geometry import (  # type: ignore
        TOPDOWN_CENTER_X,
        TOPDOWN_CENTER_Y,
        OUTER_DOUBLE_RADIUS_PX,
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
    Diese Datei trifft keine geometrischen oder scoring-spezifischen
    Fachentscheidungen. Sie entscheidet nur:
    - wie viele Impact-Schätzungen weitergescort werden
    - welche Mindestkonfidenzen gelten
    - wie die finale Ranking-Konfidenz gemischt wird
    - ob geometrisch unplausible Off-Board-Kandidaten vor dem Scoring
      verworfen werden
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

    # --------------------------------------------------------------
    # NEU: Geometrisches Candidate-Pruning
    # --------------------------------------------------------------
    prune_offboard_estimates_before_scoring: bool = True

    # Toleranz relativ zum Outer-Double-Radius.
    # 1.00 = exakt Outer Double
    # 1.03 = kleiner Sicherheitspuffer
    max_board_radius_rel_for_scoring: float = 1.03

    # Optionaler Fallback:
    # Wenn alle Estimates weggefiltert werden, kann man die Originalmenge
    # trotzdem weiterreichen. Standardmäßig AUS.
    fallback_to_unpruned_estimates_if_all_filtered: bool = False

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
        if not self.scored_estimates:
            return None
        return self.scored_estimates[0]

    @property
    def best_hit(self) -> Optional[ScoredHit]:
        if self.best_estimate is None:
            return None
        return self.best_estimate.scored_hit

    @property
    def best_label(self) -> Optional[str]:
        if self.best_estimate is None:
            return None
        return self.best_estimate.label

    @property
    def best_score(self) -> Optional[int]:
        if self.best_estimate is None:
            return None
        return self.best_estimate.score

    def to_dict(self) -> dict[str, Any]:
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

    @property
    def score_mapper(self) -> Optional[ScoreMapper]:
        return self._score_mapper

    # -------------------------------------------------------------------------
    # ScoreMapper-Verwaltung
    # -------------------------------------------------------------------------
    def set_score_mapper(self, score_mapper: ScoreMapper) -> None:
        if score_mapper is None:
            raise ValueError("score_mapper must not be None.")
        self._score_mapper = score_mapper

    def rebuild_score_mapper_from_manual_points(
        self,
        manual_points: list[Any],
        *,
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
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

        candidate_result = self._run_candidate_detection(
            frame=frame,
            reference_frame=reference_frame,
            board_mask=board_mask,
            board_polygon=board_polygon,
        )

        # Kandidaten mit Geometrie-/Pipeline-Infos anreichern,
        # damit der ImpactEstimator centerward arbeiten kann.
        if self._score_mapper is not None:
            mapper_pipeline = getattr(self._score_mapper, "pipeline", None)
            has_topdown_to_image = hasattr(self._score_mapper, "topdown_point_to_image")

            for candidate in candidate_result.candidates:
                candidate.debug = dict(getattr(candidate, "debug", {}) or {})

                if mapper_pipeline is not None:
                    candidate.debug["pipeline"] = mapper_pipeline
                    candidate.debug["points_like"] = mapper_pipeline

                board_center_image = candidate_result.metadata.get("board_center_image")
                candidate.debug["board_center_image"] = board_center_image

                if has_topdown_to_image and board_center_image is None:
                    try:
                        computed_center = self._score_mapper.topdown_point_to_image((450.0, 450.0))
                        candidate.debug["board_center_image"] = computed_center
                    except Exception as exc:
                        candidate.debug["board_center_image_error"] = str(exc)

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
        if self._score_mapper is None:
            raise RuntimeError(
                "SingleCamDetector has no ScoreMapper configured. "
                "Provide one via constructor or rebuild_score_mapper_*()."
            )

    def _try_get_board_center_image(self) -> Optional[PointF]:
        """
        Versucht das Board-Zentrum im Bildraum robust zu bestimmen.

        Strategie:
        1. Falls der ScoreMapper eine direkte Projektion von Topdown -> Image kann,
           verwende das Zentrum des standardisierten Topdown-Boards.
        2. Falls das nicht geht, None zurückgeben.

        WICHTIG:
        - Kein Fehler darf die Pipeline killen
        - Das ist nur ein optionaler Hint für den CandidateDetector
        """
        if self._score_mapper is None:
            return None

        topdown_center = (450.0, 450.0)

        for method_name in (
            "topdown_point_to_image",
            "project_topdown_point_to_image",
            "topdown_to_image",
        ):
            method = getattr(self._score_mapper, method_name, None)
            if callable(method):
                try:
                    point = method(topdown_center)
                    coerced = _coerce_point(point)
                    if coerced is not None:
                        return coerced
                except Exception as exc:
                    logger.debug(
                        "Could not get board_center_image via %s: %s",
                        method_name,
                        exc,
                    )

        return None

    def _run_candidate_detection(
        self,
        *,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]] = None,
    ) -> CandidateDetectionResult:
        """
        Führt die Candidate Detection rückwärtskompatibel aus.

        Wichtig:
        Manche Test-Doubles/Fakes unterstützen das neue Argument
        'board_center_image' noch nicht. Deshalb wird der Aufruf
        robust auf beide Signaturen angepasst.
        """
        if self.candidate_detector is None:
            raise RuntimeError("Candidate detector is not configured.")

        board_center_image = self._try_get_board_center_image()

        base_kwargs = {
            "frame": frame,
            "reference_frame": reference_frame,
            "board_mask": board_mask,
            "board_polygon": board_polygon,
        }

        # Neuer Pfad: echter Detector mit board_center_image
        try:
            candidate_result = self.candidate_detector.detect_candidates(
                **base_kwargs,
                board_center_image=board_center_image,
            )
        except TypeError as exc:
            # Nur auf alte Signatur zurückfallen, wenn wirklich das neue
            # Argument das Problem ist.
            if "board_center_image" not in str(exc):
                raise

            candidate_result = self.candidate_detector.detect_candidates(
                **base_kwargs,
            )

        candidate_result.metadata = dict(getattr(candidate_result, "metadata", {}) or {})
        candidate_result.metadata["board_center_image"] = board_center_image

        for candidate in candidate_result.candidates:
            candidate.debug = dict(getattr(candidate, "debug", {}) or {})
            candidate.debug["board_center_image"] = board_center_image

        return candidate_result


    def _project_image_point_to_topdown_safe(
        self,
        point: Any,
    ) -> Optional[PointF]:
        """
        Projiziert einen Bildpunkt robust nach Topdown.
        Gibt None zurück, wenn keine Projektion möglich ist.
        """
        if self._score_mapper is None:
            return None

        try:
            projected = self._score_mapper.image_point_to_topdown(point)
            return _coerce_point(projected)
        except Exception as exc:
            logger.debug("Could not project image point to topdown: %s", exc)
            return None

    def _compute_topdown_radius_rel(
        self,
        topdown_point: PointF,
    ) -> float:
        """
        Berechnet den normierten Board-Radius relativ zum Outer-Double-Radius.
        1.0 = exakt äußerer Double-Rand
        """
        dx = float(topdown_point[0]) - float(TOPDOWN_CENTER_X)
        dy = float(topdown_point[1]) - float(TOPDOWN_CENTER_Y)
        radius_px = math.hypot(dx, dy)
        return float(radius_px / float(OUTER_DOUBLE_RADIUS_PX))

    def _is_estimate_geometrically_plausible(
        self,
        estimate: ImpactEstimate,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Prüft, ob eine Impact-Schätzung geometrisch plausibel auf dem Board liegt.

        Logik:
        - impact_point nach Topdown projizieren
        - normierten Radius berechnen
        - außerhalb max_board_radius_rel_for_scoring => verwerfen

        WICHTIG:
        Wenn die Projektion nicht möglich ist (z. B. in Unit-Tests mit FakeScoreMapper),
        wird der Estimate NICHT verworfen, sondern als 'nicht prüfbar' durchgelassen.
        """
        debug: dict[str, Any] = {
            "checked": False,
            "reason": None,
            "topdown_point": None,
            "radius_rel": None,
            "max_radius_rel": float(self.config.max_board_radius_rel_for_scoring),
        }

        if not self.config.prune_offboard_estimates_before_scoring:
            debug["checked"] = False
            debug["reason"] = "pruning_disabled"
            return True, debug

        topdown_point = self._project_image_point_to_topdown_safe(estimate.impact_point)
        if topdown_point is None:
            debug["checked"] = True
            debug["reason"] = "projection_unavailable_keep"
            return True, debug

        radius_rel = self._compute_topdown_radius_rel(topdown_point)

        debug["checked"] = True
        debug["topdown_point"] = topdown_point
        debug["radius_rel"] = float(radius_rel)

        if radius_rel > float(self.config.max_board_radius_rel_for_scoring):
            debug["reason"] = "offboard_radius"
            return False, debug

        debug["reason"] = "ok"
        return True, debug

    def _prune_estimates_geometrically(
        self,
        estimates: list[ImpactEstimate],
    ) -> tuple[list[ImpactEstimate], list[dict[str, Any]]]:
        """
        Filtert Off-Board-Estimates vor dem Scoring heraus.
        """
        kept: list[ImpactEstimate] = []
        debug_rows: list[dict[str, Any]] = []

        for estimate in estimates:
            keep, prune_debug = self._is_estimate_geometrically_plausible(estimate)
            row = {
                "candidate_id": int(getattr(estimate, "candidate_id", -1)),
                "impact_point": _coerce_point(getattr(estimate, "impact_point")),
                "confidence": float(getattr(estimate, "confidence", 0.0)),
                "keep": bool(keep),
                "prune_debug": prune_debug,
            }
            debug_rows.append(row)

            estimate.debug = dict(getattr(estimate, "debug", {}) or {})
            estimate.debug["geometric_pruning"] = prune_debug

            if keep:
                kept.append(estimate)

        if not kept and estimates and self.config.fallback_to_unpruned_estimates_if_all_filtered:
            logger.warning(
                "All estimates were geometrically pruned. Falling back to unpruned estimates "
                "because fallback_to_unpruned_estimates_if_all_filtered=True."
            )
            return list(estimates), debug_rows

        return kept, debug_rows

    def _score_impacts(
        self,
        impact_result: ImpactEstimationResult,
    ) -> list[SingleCamScoredEstimate]:
        """
        Bewertet die Impact-Schätzungen über den ScoreMapper und erzeugt finale
        SingleCamScoredEstimate-Objekte.

        WICHTIG:
        Geometrisches Pruning passiert jetzt VOR der finalen Begrenzung auf
        max_estimates_to_score, damit nicht zufällig nur die ersten 3 Off-Board-
        Estimates geprüft und alle restlichen ignoriert werden.
        """
        assert self._score_mapper is not None, "ScoreMapper must be configured before scoring."

        all_estimates = list(impact_result.estimates)

        # --------------------------------------------------------------
        # 1) Erst geometrisch plausibel filtern
        # --------------------------------------------------------------
        pruned_estimates, pruning_debug_rows = self._prune_estimates_geometrically(all_estimates)

        # Optionaler Fallback, falls aktiviert
        estimates_after_pruning = pruned_estimates
        if (
            not estimates_after_pruning
            and all_estimates
            and self.config.fallback_to_unpruned_estimates_if_all_filtered
        ):
            estimates_after_pruning = list(all_estimates)

        # --------------------------------------------------------------
        # 2) Erst danach auf max_estimates_to_score begrenzen
        # --------------------------------------------------------------
        if not self.config.score_all_estimates:
            estimates_to_score = estimates_after_pruning[:1]
        else:
            max_count = max(1, int(self.config.max_estimates_to_score))
            estimates_to_score = estimates_after_pruning[:max_count]

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

            pruning_debug = None
            if isinstance(getattr(estimate, "debug", None), dict):
                pruning_debug = estimate.debug.get("geometric_pruning")

            scored_estimates.append(
                SingleCamScoredEstimate(
                    rank=0,
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
                        "hypothesis_count": getattr(
                            estimate,
                            "hypothesis_count",
                            len(getattr(estimate, "hypotheses", []) or []),
                        ),
                        "geometric_pruning": pruning_debug,
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
        weight_candidate = float(self.config.weight_candidate_confidence)
        weight_impact = float(self.config.weight_impact_confidence)

        total_weight = weight_candidate + weight_impact
        if total_weight <= 0.0:
            return 0.0

        score = (
            weight_candidate * float(candidate_confidence)
            + weight_impact * float(impact_confidence)
        ) / total_weight

        return float(max(0.0, min(1.0, score)))


# -----------------------------------------------------------------------------
# Convenience-Funktionen
# -----------------------------------------------------------------------------
def build_single_cam_detector(
    *,
    config: Optional[SingleCamDetectorConfig] = None,
    candidate_detector_config: Optional[CandidateDetectorConfig] = None,
    impact_estimator_config: Optional[ImpactEstimatorConfig] = None,
    manual_points: Optional[list[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> SingleCamDetector:
    return SingleCamDetector(
        config=config,
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
    detector: Optional[SingleCamDetector] = None,
    config: Optional[SingleCamDetectorConfig] = None,
    candidate_detector_config: Optional[CandidateDetectorConfig] = None,
    impact_estimator_config: Optional[ImpactEstimatorConfig] = None,
    manual_points: Optional[list[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> SingleCamDetectionResult:
    if detector is None:
        detector = build_single_cam_detector(
            config=config,
            candidate_detector_config=candidate_detector_config,
            impact_estimator_config=impact_estimator_config,
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
# Hilfsfunktionen
# -----------------------------------------------------------------------------
def _validate_frame(frame: np.ndarray, *, name: str) -> None:
    if frame is None:
        raise ValueError(f"{name} must not be None.")
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame)!r}.")
    if frame.ndim not in (2, 3):
        raise ValueError(f"{name} must have ndim 2 or 3, got {frame.ndim}.")
    if frame.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Unsupported frame shape for BGR conversion: {frame.shape}")


def _coerce_point(value: Any) -> Optional[PointF]:
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        if arr.size >= 2:
            return float(arr[0]), float(arr[1])

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except Exception:
                return None
        if "x_px" in value and "y_px" in value:
            try:
                return float(value["x_px"]), float(value["y_px"])
            except Exception:
                return None

    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return float(value.x), float(value.y)
        except Exception:
            return None

    if hasattr(value, "x_px") and hasattr(value, "y_px"):
        try:
            return float(value.x_px), float(value.y_px)
        except Exception:
            return None

    return None


def _coerce_bbox(value: Any) -> BBox:
    if isinstance(value, np.ndarray):
        arr = value.astype(float).reshape(-1)
        if arr.size >= 4:
            return int(round(arr[0])), int(round(arr[1])), int(round(arr[2])), int(round(arr[3]))

    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return (
            int(round(float(value[0]))),
            int(round(float(value[1]))),
            int(round(float(value[2]))),
            int(round(float(value[3]))),
        )

    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "w", "h")):
            return (
                int(round(float(value["x"]))),
                int(round(float(value["y"]))),
                int(round(float(value["w"]))),
                int(round(float(value["h"]))),
            )

    return (0, 0, 0, 0)


def _round_point(point: PointF) -> tuple[int, int]:
    return int(round(point[0])), int(round(point[1]))


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dataclass_fields__"):
        result = {}
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