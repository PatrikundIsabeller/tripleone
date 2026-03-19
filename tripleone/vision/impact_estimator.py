# vision/impact_estimator.py
# Zweck:
# Diese Datei schätzt für bereits gefundene Dart-Kandidaten den besten finalen
# Einschlagspunkt ("impact point") im BILDKOORDINATENSYSTEM.
#
# Wichtig:
# Diese Datei macht bewusst KEINE:
# - Homography
# - Board-Geometrie
# - Ring-/Sektorlogik
# - Score-Berechnung
#
# Sie beantwortet nur die Frage:
# "Welcher Bildpunkt ist für diesen Kandidaten die beste finale Impact-Schätzung?"
#
# Architekturrolle:
# - Input: DartCandidate(s) aus vision/dart_candidate_detector.py
# - Output: final geschätzte Bildpunkte + Debug-/Konfidenzdaten
# - Danach: single_cam_detector.py kann diesen Punkt an score_mapper.py übergeben

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

# Robuster Import:
# - normaler Paketbetrieb
# - Fallback für direkte Ausführung / Tests
try:
    from .dart_candidate_detector import (
        CandidateDetectionResult,
        DartCandidate,
    )
except ImportError:  # pragma: no cover
    from vision.dart_candidate_detector import (  # type: ignore
        CandidateDetectionResult,
        DartCandidate,
    )

logger = logging.getLogger(__name__)

PointF = tuple[float, float]
PointI = tuple[int, int]
BBox = tuple[int, int, int, int]


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class ImpactEstimatorConfig:
    """
    Konfiguration für die finalen Impact-Schätzungen.

    Wichtig:
    Diese Werte sind absichtlich konservativ und transparent.
    Hier soll nichts "magisch" erraten werden.

    strategy:
    - "blend":
        mehrere Hypothesen werden gewichtet kombiniert
    - "best_hypothesis":
        die stärkste Einzelhypothese gewinnt
    - "candidate_default":
        nimm candidate.impact_point bevorzugt
    - "lowest_contour_point":
        nimm den tiefsten Konturpunkt
    - "major_axis_lower_endpoint":
        nimm den tieferen PCA-/Major-Axis-Endpunkt
    - "directional_contour_tip":
        nimm die entlang der Hauptachse nach unten gerichtete Konturspitze
    """

    strategy: str = "blend"

    # Aktivierung einzelner Hypothesen
    use_candidate_default: bool = True
    use_lowest_contour_point: bool = True
    use_major_axis_lower_endpoint: bool = True
    use_directional_contour_tip: bool = True

    # Basisgewichte der Hypothesen
    weight_candidate_default: float = 0.70
    weight_lowest_contour_point: float = 0.90
    weight_major_axis_lower_endpoint: float = 1.00
    weight_directional_contour_tip: float = 1.10

    # Anforderungen / Heuristik
    min_major_axis_length_for_axis_based_methods: float = 10.0
    min_aspect_ratio_for_axis_based_methods: float = 1.15
    min_candidate_confidence: float = 0.01

    # Richtungsbasierte Konturspitze
    directional_tip_band_fraction: float = 0.15
    directional_tip_top_k_points: int = 8

    # Konsistenzbewertung zwischen Hypothesen
    consistency_distance_scale_px: float = 12.0

    # Gesamt-Konfidenz-Mix
    weight_source_candidate_confidence: float = 0.60
    weight_hypothesis_strength: float = 0.40

    # Nachbearbeitung
    clamp_to_image_bounds: bool = True

    # Debug
    keep_debug_metadata: bool = True


# -----------------------------------------------------------------------------
# Datenmodelle
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class ImpactHypothesis:
    """
    Eine einzelne Impact-Hypothese für einen Kandidaten.
    """
    name: str
    point: PointF
    base_weight: float
    source_quality: float
    consistency_score: float
    final_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisierung für Debug/API.
        """
        return {
            "name": self.name,
            "point": self.point,
            "base_weight": self.base_weight,
            "source_quality": self.source_quality,
            "consistency_score": self.consistency_score,
            "final_weight": self.final_weight,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ImpactEstimate:
    """
    Finale Impact-Schätzung für genau einen Dart-Kandidaten.

    Wichtig:
    'impact_point' ist hier der finale Bildpunkt für die weitere Pipeline.
    Noch immer KEIN Board-Hit / Score.
    """
    candidate_id: int
    impact_point: PointF
    method: str
    confidence: float
    source_candidate_confidence: float
    bbox: BBox
    centroid: PointF
    hypotheses: list[ImpactHypothesis] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    candidate: Optional[Any] = None

    @property
    def hypothesis_count(self) -> int:
        """
        Anzahl verwendeter Hypothesen.
        """
        return len(self.hypotheses)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisierung für Debug/API.
        """
        return {
            "candidate_id": self.candidate_id,
            "impact_point": self.impact_point,
            "method": self.method,
            "confidence": self.confidence,
            "source_candidate_confidence": self.source_candidate_confidence,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "hypothesis_count": self.hypothesis_count,
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "debug": self.debug,
        }


@dataclass(slots=True)
class ImpactEstimationResult:
    """
    Gesamtergebnis über mehrere Kandidaten.
    """
    estimates: list[ImpactEstimate]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_estimate(self) -> Optional[ImpactEstimate]:
        """
        Beste finale Impact-Schätzung.
        """
        if not self.estimates:
            return None
        return self.estimates[0]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialisierung für Debug/API.
        """
        return {
            "metadata": self.metadata,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
        }

    def render_debug_overlay(
        self,
        frame: np.ndarray,
        *,
        max_estimates: Optional[int] = None,
    ) -> np.ndarray:
        """
        Zeichnet finale Impact-Schätzungen auf ein Bild.
        """
        canvas = _ensure_bgr(frame)

        count = len(self.estimates) if max_estimates is None else min(len(self.estimates), max_estimates)

        for estimate in self.estimates[:count]:
            x, y, w, h = estimate.bbox
            ix, iy = _round_point(estimate.impact_point)
            cx, cy = _round_point(estimate.centroid)

            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 1)
            cv2.circle(canvas, (cx, cy), 3, (255, 255, 0), -1)
            cv2.circle(canvas, (ix, iy), 5, (0, 0, 255), -1)

            cv2.line(canvas, (cx, cy), (ix, iy), (0, 165, 255), 1, cv2.LINE_AA)

            cv2.putText(
                canvas,
                f"#{estimate.candidate_id} | {estimate.method} | conf={estimate.confidence:.2f}",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
                cv2.LINE_AA,
            )

            # Hypothesen zusätzlich einzeichnen
            for hypothesis in estimate.hypotheses:
                hx, hy = _round_point(hypothesis.point)
                cv2.circle(canvas, (hx, hy), 2, (0, 255, 0), -1)

        return canvas


# -----------------------------------------------------------------------------
# Hauptklasse
# -----------------------------------------------------------------------------

class ImpactEstimator:
    """
    Schätzt finale Impact-Punkte für Kandidaten.

    Verantwortungsbereich:
    - Hypothesen sammeln
    - Hypothesen konsistent bewerten
    - finalen Bildpunkt bestimmen

    Nicht verantwortlich für:
    - Geometrie / Homography
    - Board-Hit / Score
    - Spielregeln
    """

    def __init__(self, config: Optional[ImpactEstimatorConfig] = None) -> None:
        self.config = config or ImpactEstimatorConfig()

    # -------------------------------------------------------------------------
    # Öffentliche API
    # -------------------------------------------------------------------------

    def estimate_for_candidate(
        self,
        candidate: DartCandidate,
        *,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> Optional[ImpactEstimate]:
        """
        Schätzt den finalen Impact-Punkt für genau einen Kandidaten.
        """
        if candidate is None:
            raise ValueError("candidate must not be None.")

        source_candidate_confidence = float(getattr(candidate, "confidence", 0.0))
        if source_candidate_confidence < self.config.min_candidate_confidence:
            return None

        hypotheses = self._collect_hypotheses(candidate)

        if not hypotheses:
            return None

        hypotheses = self._apply_consistency_scoring(hypotheses)
        chosen_point, method, aggregate_strength, spread_px = self._choose_final_point(hypotheses)

        if image_shape is not None and self.config.clamp_to_image_bounds:
            chosen_point = _clamp_point_to_shape(chosen_point, image_shape)

        confidence = self._compute_final_confidence(
            source_candidate_confidence=source_candidate_confidence,
            aggregate_strength=aggregate_strength,
            spread_px=spread_px,
        )

        debug = {}
        if self.config.keep_debug_metadata:
            debug = {
                "spread_px": float(spread_px),
                "aggregate_strength": float(aggregate_strength),
                "strategy": self.config.strategy,
                "candidate_area": float(getattr(candidate, "area", 0.0)),
                "candidate_aspect_ratio": float(getattr(candidate, "aspect_ratio", 0.0)),
                "candidate_elongation": float(getattr(candidate, "elongation", 0.0)),
            }

        return ImpactEstimate(
            candidate_id=int(getattr(candidate, "candidate_id", -1)),
            impact_point=chosen_point,
            method=method,
            confidence=confidence,
            source_candidate_confidence=source_candidate_confidence,
            bbox=_coerce_bbox(getattr(candidate, "bbox", (0, 0, 0, 0))),
            centroid=_coerce_point(getattr(candidate, "centroid", chosen_point)),
            hypotheses=hypotheses,
            debug=debug,
            candidate=candidate,
        )

    def estimate_for_candidates(
        self,
        candidates: list[DartCandidate],
        *,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> ImpactEstimationResult:
        """
        Schätzt finale Impact-Punkte für mehrere Kandidaten und sortiert sie
        nach finaler Konfidenz.
        """
        estimates: list[ImpactEstimate] = []

        for candidate in candidates:
            estimate = self.estimate_for_candidate(candidate, image_shape=image_shape)
            if estimate is not None:
                estimates.append(estimate)

        estimates.sort(key=lambda estimate: estimate.confidence, reverse=True)

        metadata = {
            "input_candidates": len(candidates),
            "accepted_estimates": len(estimates),
            "config": _dataclass_to_dict(self.config),
        }

        return ImpactEstimationResult(
            estimates=estimates,
            metadata=metadata,
        )

    def estimate_from_detection_result(
        self,
        detection_result: CandidateDetectionResult,
        *,
        image_shape: Optional[tuple[int, int]] = None,
    ) -> ImpactEstimationResult:
        """
        Komfortmethode direkt auf Basis eines CandidateDetectionResult.
        """
        if detection_result is None:
            raise ValueError("detection_result must not be None.")

        if image_shape is None:
            input_shape = detection_result.metadata.get("input_shape")
            if isinstance(input_shape, (tuple, list)) and len(input_shape) >= 2:
                image_shape = (int(input_shape[0]), int(input_shape[1]))

        return self.estimate_for_candidates(
            candidates=list(detection_result.candidates),
            image_shape=image_shape,
        )

    # -------------------------------------------------------------------------
    # Interne Hypothesen-Sammlung
    # -------------------------------------------------------------------------

    def _collect_hypotheses(self, candidate: DartCandidate) -> list[ImpactHypothesis]:
        """
        Sammelt alle aktivierten Impact-Hypothesen für einen Kandidaten.
        """
        hypotheses: list[ImpactHypothesis] = []

        aspect_ratio = float(getattr(candidate, "aspect_ratio", 1.0))
        major_axis_length = float(getattr(candidate, "major_axis_length", 0.0))
        source_candidate_confidence = float(getattr(candidate, "confidence", 0.0))

        debug = getattr(candidate, "debug", {}) or {}

        # ------------------------------------------------------------------
        # 1) Kandidaten-Defaultpunkt
        # ------------------------------------------------------------------
        if self.config.use_candidate_default:
            point = _safe_extract_point(getattr(candidate, "impact_point", None))
            if point is not None:
                source_quality = _clip01(
                    0.55 + 0.45 * source_candidate_confidence
                )
                hypotheses.append(
                    ImpactHypothesis(
                        name="candidate_default",
                        point=point,
                        base_weight=float(self.config.weight_candidate_default),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_candidate_default) * source_quality,
                        metadata={},
                    )
                )

        # ------------------------------------------------------------------
        # 2) Tiefster Konturpunkt
        # ------------------------------------------------------------------
        if self.config.use_lowest_contour_point:
            point = _safe_extract_point(debug.get("contour_lowest_point"))
            if point is None:
                point = _lowest_contour_point_from_candidate(candidate)

            if point is not None:
                source_quality = _clip01(
                    0.50 + 0.30 * source_candidate_confidence + 0.20 * _aspect_quality(aspect_ratio)
                )
                hypotheses.append(
                    ImpactHypothesis(
                        name="lowest_contour_point",
                        point=point,
                        base_weight=float(self.config.weight_lowest_contour_point),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_lowest_contour_point) * source_quality,
                        metadata={},
                    )
                )

        # ------------------------------------------------------------------
        # 3) Tiefer Major-Axis-Endpunkt
        # ------------------------------------------------------------------
        if self.config.use_major_axis_lower_endpoint:
            point = _major_axis_lower_endpoint_from_candidate(candidate)

            if point is not None:
                axis_quality = self._axis_based_quality(
                    aspect_ratio=aspect_ratio,
                    major_axis_length=major_axis_length,
                )
                source_quality = _clip01(
                    0.40 + 0.60 * axis_quality
                )
                hypotheses.append(
                    ImpactHypothesis(
                        name="major_axis_lower_endpoint",
                        point=point,
                        base_weight=float(self.config.weight_major_axis_lower_endpoint),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_major_axis_lower_endpoint) * source_quality,
                        metadata={},
                    )
                )

        # ------------------------------------------------------------------
        # 4) Richtungsbasierte Konturspitze
        # ------------------------------------------------------------------
        if self.config.use_directional_contour_tip:
            directional_tip, directional_meta = self._directional_contour_tip(candidate)

            if directional_tip is not None:
                axis_quality = self._axis_based_quality(
                    aspect_ratio=aspect_ratio,
                    major_axis_length=major_axis_length,
                )
                source_quality = _clip01(
                    0.35 + 0.65 * axis_quality
                )
                hypotheses.append(
                    ImpactHypothesis(
                        name="directional_contour_tip",
                        point=directional_tip,
                        base_weight=float(self.config.weight_directional_contour_tip),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_directional_contour_tip) * source_quality,
                        metadata=directional_meta,
                    )
                )

        return hypotheses

    def _axis_based_quality(
        self,
        *,
        aspect_ratio: float,
        major_axis_length: float,
    ) -> float:
        """
        Bewertet, wie sinnvoll achsenbasierte Methoden für diesen Kandidaten sind.
        """
        aspect_part = _soft_range_score(
            value=float(aspect_ratio),
            good_min=max(1.5, self.config.min_aspect_ratio_for_axis_based_methods),
            good_max=12.0,
            hard_min=1.0,
            hard_max=20.0,
        )

        length_part = _soft_range_score(
            value=float(major_axis_length),
            good_min=max(20.0, self.config.min_major_axis_length_for_axis_based_methods),
            good_max=300.0,
            hard_min=0.0,
            hard_max=500.0,
        )

        return _clip01(0.55 * aspect_part + 0.45 * length_part)

    # -------------------------------------------------------------------------
    # Interne Konsistenz-/Auswahl-Logik
    # -------------------------------------------------------------------------

    def _apply_consistency_scoring(
        self,
        hypotheses: list[ImpactHypothesis],
    ) -> list[ImpactHypothesis]:
        """
        Bewertet, wie gut einzelne Hypothesen räumlich zusammenpassen.

        Idee:
        Wenn mehrere Hypothesen dicht beieinander liegen, steigt deren Gewicht.
        Starke Ausreißer werden abgewertet.
        """
        if len(hypotheses) <= 1:
            return hypotheses

        points = np.asarray([hypothesis.point for hypothesis in hypotheses], dtype=np.float32)
        median_point = np.median(points, axis=0)

        for hypothesis in hypotheses:
            distance = float(np.linalg.norm(np.asarray(hypothesis.point) - median_point))
            consistency_score = math.exp(
                -distance / max(self.config.consistency_distance_scale_px, 1e-6)
            )
            hypothesis.consistency_score = _clip01(consistency_score)
            hypothesis.final_weight = (
                hypothesis.base_weight
                * hypothesis.source_quality
                * hypothesis.consistency_score
            )

        return hypotheses

    def _choose_final_point(
        self,
        hypotheses: list[ImpactHypothesis],
    ) -> tuple[PointF, str, float, float]:
        """
        Wählt den finalen Impact-Punkt aus den Hypothesen.

        Rückgabe:
        - Punkt
        - Methode
        - aggregierte Hypothesenstärke
        - räumliche Streuung der Hypothesen
        """
        if not hypotheses:
            raise ValueError("At least one hypothesis is required.")

        strategy = self.config.strategy.strip().lower()

        available_by_name = {hypothesis.name: hypothesis for hypothesis in hypotheses}

        # Direkte Strategien: eine bestimmte Hypothese bevorzugen
        if strategy in available_by_name:
            chosen = available_by_name[strategy]
            spread_px = _compute_point_spread(chosen.point, [hyp.point for hyp in hypotheses])
            return chosen.point, chosen.name, chosen.final_weight, spread_px

        if strategy == "best_hypothesis":
            chosen = max(hypotheses, key=lambda hypothesis: hypothesis.final_weight)
            spread_px = _compute_point_spread(chosen.point, [hyp.point for hyp in hypotheses])
            return chosen.point, chosen.name, chosen.final_weight, spread_px

        if strategy != "blend":
            raise ValueError(
                "Unsupported impact estimator strategy. Expected one of: "
                "'blend', 'best_hypothesis', 'candidate_default', "
                "'lowest_contour_point', 'major_axis_lower_endpoint', "
                "'directional_contour_tip'."
            )

        # Blend-Strategie:
        # gewichtetes Mittel aller Hypothesen
        weights = np.asarray(
            [max(hypothesis.final_weight, 1e-9) for hypothesis in hypotheses],
            dtype=np.float64,
        )
        points = np.asarray([hypothesis.point for hypothesis in hypotheses], dtype=np.float64)

        weighted_point = np.average(points, axis=0, weights=weights)
        chosen_point = (float(weighted_point[0]), float(weighted_point[1]))

        aggregate_strength = float(np.mean(weights))
        spread_px = _compute_weighted_spread(chosen_point, hypotheses)

        return chosen_point, "blend", aggregate_strength, spread_px

    def _compute_final_confidence(
        self,
        *,
        source_candidate_confidence: float,
        aggregate_strength: float,
        spread_px: float,
    ) -> float:
        """
        Mischt Kandidatenkonfidenz + Hypothesenstärke + Konsistenz.
        """
        source_part = _clip01(source_candidate_confidence)
        strength_part = _clip01(aggregate_strength)

        # Kleine Streuung = bessere Übereinstimmung der Hypothesen
        consistency_part = math.exp(
            -float(spread_px) / max(self.config.consistency_distance_scale_px, 1e-6)
        )
        consistency_part = _clip01(consistency_part)

        base = (
            self.config.weight_source_candidate_confidence * source_part
            + self.config.weight_hypothesis_strength * strength_part
        )

        final_confidence = _clip01(0.75 * base + 0.25 * consistency_part)
        return final_confidence

    # -------------------------------------------------------------------------
    # Interne Hypothesenberechnung
    # -------------------------------------------------------------------------

    def _directional_contour_tip(
        self,
        candidate: DartCandidate,
    ) -> tuple[Optional[PointF], dict[str, Any]]:
        """
        Schätzt eine nach unten gerichtete Konturspitze entlang der Hauptachse.

        Idee:
        - bestimme eine "nach unten" gerichtete Achse
        - projiziere Konturpunkte auf diese Richtung
        - nimm die extremen Punkte im unteren Band
        - bilde daraus einen stabileren Spitzenpunkt

        Das ist bewusst robuster als nur "nimm den tiefsten Einzelpunkt".
        """
        contour = getattr(candidate, "contour", None)
        if contour is None or len(contour) < 3:
            return None, {}

        points = contour.reshape(-1, 2).astype(np.float32)

        major_axis_length = float(getattr(candidate, "major_axis_length", 0.0))
        aspect_ratio = float(getattr(candidate, "aspect_ratio", 1.0))

        if major_axis_length < self.config.min_major_axis_length_for_axis_based_methods:
            return None, {}

        if aspect_ratio < self.config.min_aspect_ratio_for_axis_based_methods:
            return None, {}

        lower_endpoint = _major_axis_lower_endpoint_from_candidate(candidate)
        upper_endpoint = _major_axis_upper_endpoint_from_candidate(candidate)

        # Fallback: Endpunkte direkt aus Kontur berechnen, falls im Candidate nicht verfügbar
        if lower_endpoint is None or upper_endpoint is None:
            upper_endpoint, lower_endpoint = _compute_major_axis_endpoints_from_points(points)

        if lower_endpoint is None or upper_endpoint is None:
            return None, {}

        lower = np.asarray(lower_endpoint, dtype=np.float64)
        upper = np.asarray(upper_endpoint, dtype=np.float64)

        direction = lower - upper
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-9:
            return None, {}

        direction = direction / norm

        projections = (points - upper) @ direction
        max_projection = float(np.max(projections))
        min_projection = float(np.min(projections))
        span = max(max_projection - min_projection, 1e-9)

        band_fraction = _clip01(self.config.directional_tip_band_fraction)
        band_threshold = max_projection - (band_fraction * span)

        band_mask = projections >= band_threshold
        band_points = points[band_mask]

        if len(band_points) == 0:
            return None, {}

        # Zusätzliche Stabilisierung:
        # nimm maximal die projizierten Top-K-Punkte
        sorted_indices = np.argsort(projections)[::-1]
        top_k = max(1, int(self.config.directional_tip_top_k_points))
        top_indices = sorted_indices[:top_k]
        top_points = points[top_indices]
        top_projections = projections[top_indices]

        # Projektion in positive Gewichte umwandeln
        proj_min = float(np.min(top_projections))
        proj_shifted = top_projections - proj_min + 1e-6
        weights = proj_shifted / np.sum(proj_shifted)

        tip = np.sum(top_points * weights[:, None], axis=0)

        metadata = {
            "upper_endpoint": (float(upper[0]), float(upper[1])),
            "lower_endpoint": (float(lower[0]), float(lower[1])),
            "max_projection": max_projection,
            "min_projection": min_projection,
            "band_threshold": band_threshold,
            "band_point_count": int(len(band_points)),
            "top_k_count": int(len(top_points)),
        }

        return (float(tip[0]), float(tip[1])), metadata


# -----------------------------------------------------------------------------
# Modulweite Convenience-Funktionen
# -----------------------------------------------------------------------------

def build_impact_estimator(
    config: Optional[ImpactEstimatorConfig] = None,
) -> ImpactEstimator:
    """
    Bequemer Builder für den ImpactEstimator.
    """
    return ImpactEstimator(config=config)


def estimate_impact_for_candidate(
    candidate: DartCandidate,
    *,
    image_shape: Optional[tuple[int, int]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> Optional[ImpactEstimate]:
    """
    Modulweiter Wrapper für genau einen Kandidaten.
    """
    estimator = build_impact_estimator(config=config)
    return estimator.estimate_for_candidate(candidate, image_shape=image_shape)


def estimate_impacts(
    candidates: list[DartCandidate],
    *,
    image_shape: Optional[tuple[int, int]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> ImpactEstimationResult:
    """
    Modulweiter Wrapper für mehrere Kandidaten.
    """
    estimator = build_impact_estimator(config=config)
    return estimator.estimate_for_candidates(candidates, image_shape=image_shape)


def estimate_impacts_from_detection_result(
    detection_result: CandidateDetectionResult,
    *,
    image_shape: Optional[tuple[int, int]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> ImpactEstimationResult:
    """
    Modulweiter Wrapper direkt auf Basis eines CandidateDetectionResult.
    """
    estimator = build_impact_estimator(config=config)
    return estimator.estimate_from_detection_result(
        detection_result=detection_result,
        image_shape=image_shape,
    )


# -----------------------------------------------------------------------------
# Interne Hilfsfunktionen
# -----------------------------------------------------------------------------

def _safe_extract_point(value: Any) -> Optional[PointF]:
    """
    Extrahiert robust einen 2D-Punkt oder gibt None zurück.
    """
    try:
        return _coerce_point(value)
    except Exception:
        return None


def _coerce_point(value: Any) -> PointF:
    """
    Zwingt verschiedene Punktformate in (x, y).
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
    Zwingt eine Bounding-Box in (x, y, w, h).
    """
    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 4:
        return int(value[0]), int(value[1]), int(value[2]), int(value[3])
    raise ValueError(f"Unsupported bbox value: {value!r}")


def _round_point(point: PointF) -> PointI:
    """
    Rundet einen Float-Punkt für OpenCV-Zeichenoperationen.
    """
    return int(round(point[0])), int(round(point[1]))


def _clip01(value: float) -> float:
    """
    Beschränkt einen Wert auf [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, value)))


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

    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _clamp_point_to_shape(point: PointF, image_shape: tuple[int, int]) -> PointF:
    """
    Beschränkt einen Punkt auf gültige Bildgrenzen.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    x = min(max(float(point[0]), 0.0), max(width - 1, 0))
    y = min(max(float(point[1]), 0.0), max(height - 1, 0))
    return x, y


def _soft_range_score(
    *,
    value: float,
    good_min: float,
    good_max: float,
    hard_min: float,
    hard_max: float,
) -> float:
    """
    Weicher Bereichsscore:
    - innerhalb good-Bereich = 1.0
    - außerhalb hard-Bereich = 0.0
    - dazwischen linear
    """
    value = float(value)

    if value < hard_min or value > hard_max:
        return 0.0

    if good_min <= value <= good_max:
        return 1.0

    if value < good_min:
        denom = max(good_min - hard_min, 1e-9)
        return _clip01((value - hard_min) / denom)

    denom = max(hard_max - good_max, 1e-9)
    return _clip01((hard_max - value) / denom)


def _aspect_quality(aspect_ratio: float) -> float:
    """
    Kleine Hilfsheuristik für längliche Kandidaten.
    """
    return _soft_range_score(
        value=float(aspect_ratio),
        good_min=1.5,
        good_max=12.0,
        hard_min=1.0,
        hard_max=20.0,
    )


def _compute_point_spread(center_point: PointF, points: list[PointF]) -> float:
    """
    Mittlere Distanz einer Punktmenge zu einem Zentrumspunkt.
    """
    if not points:
        return 0.0

    center = np.asarray(center_point, dtype=np.float64)
    arr = np.asarray(points, dtype=np.float64)

    distances = np.linalg.norm(arr - center, axis=1)
    return float(np.mean(distances))


def _compute_weighted_spread(
    center_point: PointF,
    hypotheses: list[ImpactHypothesis],
) -> float:
    """
    Gewichtete mittlere Distanz aller Hypothesen zum finalen Punkt.
    """
    if not hypotheses:
        return 0.0

    center = np.asarray(center_point, dtype=np.float64)
    points = np.asarray([hypothesis.point for hypothesis in hypotheses], dtype=np.float64)
    weights = np.asarray(
        [max(hypothesis.final_weight, 1e-9) for hypothesis in hypotheses],
        dtype=np.float64,
    )

    distances = np.linalg.norm(points - center, axis=1)
    return float(np.average(distances, weights=weights))


def _lowest_contour_point_from_candidate(candidate: DartCandidate) -> Optional[PointF]:
    """
    Bestimmt den tiefsten Konturpunkt direkt aus der Kontur, falls nötig.
    """
    contour = getattr(candidate, "contour", None)
    if contour is None or len(contour) == 0:
        return None

    points = contour.reshape(-1, 2).astype(np.float32)
    point = points[np.argmax(points[:, 1])]
    return float(point[0]), float(point[1])


def _major_axis_lower_endpoint_from_candidate(candidate: DartCandidate) -> Optional[PointF]:
    """
    Holt den tieferen Major-Axis-Endpunkt aus Debugdaten oder aus der Kontur.
    """
    debug = getattr(candidate, "debug", {}) or {}

    a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
    b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

    if a is not None and b is not None:
        return a if a[1] >= b[1] else b

    contour = getattr(candidate, "contour", None)
    if contour is None or len(contour) < 3:
        return None

    points = contour.reshape(-1, 2).astype(np.float32)
    upper, lower = _compute_major_axis_endpoints_from_points(points)
    return lower


def _major_axis_upper_endpoint_from_candidate(candidate: DartCandidate) -> Optional[PointF]:
    """
    Holt den höheren Major-Axis-Endpunkt aus Debugdaten oder aus der Kontur.
    """
    debug = getattr(candidate, "debug", {}) or {}

    a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
    b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

    if a is not None and b is not None:
        return a if a[1] < b[1] else b

    contour = getattr(candidate, "contour", None)
    if contour is None or len(contour) < 3:
        return None

    points = contour.reshape(-1, 2).astype(np.float32)
    upper, lower = _compute_major_axis_endpoints_from_points(points)
    return upper


def _compute_major_axis_endpoints_from_points(points: np.ndarray) -> tuple[Optional[PointF], Optional[PointF]]:
    """
    Berechnet die oberen/unteren Endpunkte der Hauptachse per PCA.

    Rückgabe:
    - oberer Endpunkt
    - unterer Endpunkt
    """
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return None, None

    mean = np.mean(points, axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
    principal_vector = principal_vector / max(np.linalg.norm(principal_vector), 1e-9)

    projections = centered @ principal_vector

    min_idx = int(np.argmin(projections))
    max_idx = int(np.argmax(projections))

    point_a = points[min_idx]
    point_b = points[max_idx]

    a = (float(point_a[0]), float(point_a[1]))
    b = (float(point_b[0]), float(point_b[1]))

    upper = a if a[1] < b[1] else b
    lower = a if a[1] >= b[1] else b

    return upper, lower


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
    "PointI",
    "BBox",
    "ImpactEstimatorConfig",
    "ImpactHypothesis",
    "ImpactEstimate",
    "ImpactEstimationResult",
    "ImpactEstimator",
    "build_impact_estimator",
    "estimate_impact_for_candidate",
    "estimate_impacts",
    "estimate_impacts_from_detection_result",
]