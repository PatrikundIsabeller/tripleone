# vision/impact_estimator.py
# Zweck:
# Diese Datei schätzt aus Dart-Kandidaten einen finalen Impact-Punkt.
#
# WICHTIG:
# - Diese Datei macht KEIN Score-Mapping.
# - Diese Datei macht KEINE eigene Board-Geometrie.
# - Sie nutzt nur Kontur-/Achsen-/Debuginformationen des Kandidaten.
#
# Hauptideen:
# - mehrere Hypothesen pro Kandidat bauen
# - daraus einen finalen Punkt wählen
# - Strategien:
#   - candidate_default
#   - lowest_contour_point
#   - major_axis_lower_endpoint
#   - major_axis_centerward_endpoint
#   - centerward_contour_tip
#   - directional_contour_tip
#   - best_hypothesis
#   - blend

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

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

try:
    from .calibration_geometry import topdown_to_image_point
except ImportError:  # pragma: no cover
    from vision.calibration_geometry import topdown_to_image_point  # type: ignore


PointF = tuple[float, float]


# -----------------------------------------------------------------------------
# Datenklassen
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class ImpactHypothesis:
    """
    Eine einzelne Hypothese für den wahrscheinlichen Impact-Punkt.
    """
    name: str
    point: PointF
    base_weight: float
    source_quality: float
    consistency_score: float
    final_weight: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
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
    Finales Estimate für genau einen DartCandidate.
    """
    candidate_id: int
    impact_point: PointF
    method: str
    confidence: float
    source_candidate_confidence: float
    bbox: tuple[int, int, int, int]
    centroid: PointF
    hypotheses: list[ImpactHypothesis] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    candidate: Optional[DartCandidate] = None

    @property
    def hypothesis_count(self) -> int:
        return len(self.hypotheses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "impact_point": self.impact_point,
            "method": self.method,
            "confidence": self.confidence,
            "source_candidate_confidence": self.source_candidate_confidence,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "hypotheses": [hyp.to_dict() for hyp in self.hypotheses],
            "debug": self.debug,
        }


@dataclass(slots=True)
class ImpactEstimationResult:
    """
    Sammlung aller Impact-Estimates eines Frames.
    """
    estimates: list[ImpactEstimate]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_estimate(self) -> Optional[ImpactEstimate]:
        return self.estimates[0] if self.estimates else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "estimates": [estimate.to_dict() for estimate in self.estimates],
        }

    def render_debug_overlay(
        self,
        image: np.ndarray,
        *,
        max_estimates: Optional[int] = None,
    ) -> np.ndarray:
        """
        Zeichnet Kandidaten-, Hypothesen- und finale Impact-Punkte auf ein Bild.

        Darstellung:
        - Bounding Box des Estimates
        - alle Hypothesenpunkte farbig
        - finaler Impact-Punkt rot
        - Label/Confidence
        """
        if image.ndim == 2:
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            canvas = image.copy()

        estimates = self.estimates
        if max_estimates is not None:
            estimates = estimates[: max(0, int(max_estimates))]

        for i, estimate in enumerate(estimates):
            x, y, w, h = estimate.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 1)

            # Hypothesenpunkte einzeichnen
            for hypothesis in estimate.hypotheses:
                hx = int(round(hypothesis.point[0]))
                hy = int(round(hypothesis.point[1]))
                color = _hypothesis_color(hypothesis.name)

                cv2.circle(canvas, (hx, hy), 4, color, -1)
                cv2.circle(canvas, (hx, hy), 6, color, 1)

                hyp_label = hypothesis.name
                cv2.putText(
                    canvas,
                    hyp_label,
                    (hx + 6, hy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            # Finalen Punkt deutlich markieren
            ix = int(round(estimate.impact_point[0]))
            iy = int(round(estimate.impact_point[1]))
            cv2.circle(canvas, (ix, iy), 6, (0, 0, 255), -1)
            cv2.circle(canvas, (ix, iy), 10, (255, 255, 255), 1)

            label = f"#{i+1} {estimate.method} conf={estimate.confidence:.2f}"
            cv2.putText(
                canvas,
                label,
                (x, max(15, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas


@dataclass(slots=True)
class ImpactEstimatorConfig:
    """
    Konfiguration der Impact-Schätzung.
    """
    strategy: str = "blend"

    use_candidate_default: bool = True
    use_lowest_contour_point: bool = True
    use_major_axis_lower_endpoint: bool = True
    use_major_axis_centerward_endpoint: bool = True
    use_centerward_contour_tip: bool = True
    use_directional_contour_tip: bool = True

    weight_candidate_default: float = 0.15
    weight_lowest_contour_point: float = 0.10
    weight_major_axis_lower_endpoint: float = 0.10
    weight_major_axis_centerward_endpoint: float = 0.35
    weight_centerward_contour_tip: float = 0.35
    weight_directional_contour_tip: float = 2.10

    min_major_axis_length_for_axis_based_methods: float = 10.0
    min_aspect_ratio_for_axis_based_methods: float = 1.15
    min_candidate_confidence: float = 0.01

    directional_tip_band_fraction: float = 0.15
    directional_tip_top_k_points: int = 8

    centerward_tip_top_k_points: int = 12
    centerward_tip_distance_weight: float = 0.70
    centerward_tip_forward_weight: float = 0.30
    centerward_tip_requires_axis_agreement: bool = True
    centerward_tip_max_distance_from_centerward_axis_px: float = 35.0

    consistency_distance_scale_px: float = 12.0
    weight_source_candidate_confidence: float = 0.60
    weight_hypothesis_strength: float = 0.40

    clamp_to_image_bounds: bool = True
    keep_debug_metadata: bool = True

    # --------------------------------------------------------------
    # NEU: boardnahe Konturspitze für große Dartkonturen
    # --------------------------------------------------------------
    use_board_near_contour_tip: bool = True

    # Nur die konturpunkte betrachten, die grob in Richtung Boardzentrum liegen
    board_near_tip_top_k_points: int = 3

    # Wie stark Punkte bevorzugt werden, die näher am Boardzentrum liegen
    board_near_tip_distance_weight: float = 1.40

    # Wie stark Punkte bevorzugt werden, die entlang der Dart-Richtung "nach vorne"
    # in Richtung Boardzentrum zeigen
    board_near_tip_forward_weight: float = 2.20

    # Maximal erlaubter Abstand zur Hauptachsen-Referenz, damit wir nicht auf
    # völlig irrelevante Konturäste springen
    board_near_tip_max_distance_from_axis_px: float = 18.0

    # Gewicht dieser Hypothese im Blend
    weight_board_near_contour_tip: float = 0.35


# -----------------------------------------------------------------------------
# Kleine Helpers
# -----------------------------------------------------------------------------

def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_extract_point(value: Any) -> Optional[PointF]:
    if value is None:
        return None

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
        return None

    if isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        return float(value[0]), float(value[1])

    return None


def _point_distance(a: PointF, b: PointF) -> float:
    """
    Euklidische Distanz zwischen zwei 2D-Punkten.
    """
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(pa - pb))


def _compute_point_spread(reference_point: PointF, points: Sequence[PointF]) -> float:
    """
    Mittlere Distanz eines Referenzpunkts zu einer Menge von Punkten.
    """
    if not points:
        return 0.0

    ref = np.asarray(reference_point, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    dists = np.linalg.norm(pts - ref[None, :], axis=1)
    return float(np.mean(dists))


def _compute_weighted_spread(reference_point: PointF, hypotheses: Sequence[ImpactHypothesis]) -> float:
    """
    Gewichtete Streuung um einen Referenzpunkt.
    """
    if not hypotheses:
        return 0.0

    ref = np.asarray(reference_point, dtype=np.float64)
    points = np.asarray([hyp.point for hyp in hypotheses], dtype=np.float64)
    weights = np.asarray([max(hyp.final_weight, 1e-9) for hyp in hypotheses], dtype=np.float64)

    dists = np.linalg.norm(points - ref[None, :], axis=1)
    return float(np.average(dists, weights=weights))


def _clamp_point_to_image(point: PointF, image_shape: Optional[tuple[int, ...]]) -> PointF:
    """
    Klemmt einen Punkt an die Bildgrenzen.
    """
    if image_shape is None or len(image_shape) < 2:
        return point

    height = int(image_shape[0])
    width = int(image_shape[1])

    x = min(max(float(point[0]), 0.0), max(float(width - 1), 0.0))
    y = min(max(float(point[1]), 0.0), max(float(height - 1), 0.0))
    return x, y


def _hypothesis_color(name: str) -> tuple[int, int, int]:
    """
    Liefert eine feste BGR-Farbe pro Hypothesenname.
    """
    mapping = {
        "candidate_default": (255, 255, 0),              # Cyan
        "lowest_contour_point": (0, 255, 255),          # Gelb
        "major_axis_lower_endpoint": (255, 0, 255),     # Magenta
        "major_axis_centerward_endpoint": (255, 128, 0),# Orange
        "centerward_contour_tip": (0, 165, 255),        # Dunkelorange
        "directional_contour_tip": (0, 255, 0),         # Grün
        "board_near_contour_tip": (0, 0, 255),          # Rot
    }
    return mapping.get(name, (200, 200, 200))


def _compute_major_axis_endpoints_from_points(points: np.ndarray) -> tuple[Optional[PointF], Optional[PointF]]:
    """
    Schätzt zwei Endpunkte der Hauptachse über PCA.
    """
    if points is None or len(points) < 2:
        return None, None

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    mean = np.mean(pts, axis=0)
    centered = pts - mean[None, :]

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    principal_axis = eigvecs[:, order[0]]

    projections = centered @ principal_axis
    min_idx = int(np.argmin(projections))
    max_idx = int(np.argmax(projections))

    endpoint_a = (float(pts[min_idx, 0]), float(pts[min_idx, 1]))
    endpoint_b = (float(pts[max_idx, 0]), float(pts[max_idx, 1]))

    # Rückgabe als "oberer" und "unterer" Punkt in Bildkoordinaten
    if endpoint_a[1] <= endpoint_b[1]:
        return endpoint_a, endpoint_b
    return endpoint_b, endpoint_a


def _major_axis_lower_endpoint_from_candidate(candidate: DartCandidate) -> Optional[PointF]:
    """
    Wählt von zwei Hauptachsen-Endpunkten den tieferen Punkt im Bild.
    """
    debug = getattr(candidate, "debug", {}) or {}

    a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
    b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

    if a is None or b is None:
        contour = getattr(candidate, "contour", None)
        if contour is None or len(contour) < 3:
            return None
        upper, lower = _compute_major_axis_endpoints_from_points(
            contour.reshape(-1, 2).astype(np.float32)
        )
        if upper is None or lower is None:
            return None
        return lower

    return a if a[1] >= b[1] else b


def _major_axis_centerward_endpoint_from_candidate(
    candidate: DartCandidate,
    board_center_image: PointF,
) -> Optional[PointF]:
    """
    Wählt von den beiden Hauptachsen-Endpunkten denjenigen, der näher am
    Boardzentrum liegt.
    """
    debug = getattr(candidate, "debug", {}) or {}

    a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
    b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

    if a is None or b is None:
        contour = getattr(candidate, "contour", None)
        if contour is None or len(contour) < 3:
            return None

        points = contour.reshape(-1, 2).astype(np.float32)
        upper, lower = _compute_major_axis_endpoints_from_points(points)
        if upper is None or lower is None:
            return None
        a, b = upper, lower

    center = np.asarray(board_center_image, dtype=np.float64)
    pa = np.asarray(a, dtype=np.float64)
    pb = np.asarray(b, dtype=np.float64)

    da = float(np.linalg.norm(pa - center))
    db = float(np.linalg.norm(pb - center))

    return a if da <= db else b


def _centerward_contour_tip_from_candidate(
    candidate: DartCandidate,
    board_center_image: PointF,
    *,
    top_k_points: int = 12,
    distance_weight: float = 0.70,
    forward_weight: float = 0.30,
) -> Optional[PointF]:
    """
    Wählt eine Spitzen-Schätzung direkt aus den Konturpunkten.

    Idee:
    - Betrachte alle Konturpunkte des Kandidaten
    - bevorzuge Punkte, die näher am Boardzentrum liegen
    - bevorzuge zusätzlich Punkte, die entlang der Dart-Hauptachse "vorne"
      liegen, also in Richtung Zentrum zeigen
    """
    contour = getattr(candidate, "contour", None)
    if contour is None or len(contour) < 3:
        return None

    points = contour.reshape(-1, 2).astype(np.float64)
    center = np.asarray(board_center_image, dtype=np.float64)

    debug = getattr(candidate, "debug", {}) or {}
    endpoint_a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
    endpoint_b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

    if endpoint_a is None or endpoint_b is None:
        upper, lower = _compute_major_axis_endpoints_from_points(points.astype(np.float32))
        if upper is None or lower is None:
            return None
        endpoint_a, endpoint_b = upper, lower

    a = np.asarray(endpoint_a, dtype=np.float64)
    b = np.asarray(endpoint_b, dtype=np.float64)

    da = float(np.linalg.norm(a - center))
    db = float(np.linalg.norm(b - center))

    centerward_endpoint = a if da <= db else b
    outward_endpoint = b if da <= db else a

    axis_vec = centerward_endpoint - outward_endpoint
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len <= 1e-9:
        return float(centerward_endpoint[0]), float(centerward_endpoint[1])

    axis_dir = axis_vec / axis_len

    distances = np.linalg.norm(points - center[None, :], axis=1)
    forward_scores = (points - outward_endpoint[None, :]) @ axis_dir

    dist_min = float(np.min(distances))
    dist_max = float(np.max(distances))
    if dist_max - dist_min <= 1e-9:
        dist_norm = np.ones(len(points), dtype=np.float64)
    else:
        dist_norm = 1.0 - ((distances - dist_min) / (dist_max - dist_min))

    fwd_min = float(np.min(forward_scores))
    fwd_max = float(np.max(forward_scores))
    if fwd_max - fwd_min <= 1e-9:
        fwd_norm = np.ones(len(points), dtype=np.float64)
    else:
        fwd_norm = (forward_scores - fwd_min) / (fwd_max - fwd_min)

    combined = (
        float(distance_weight) * dist_norm +
        float(forward_weight) * fwd_norm
    )

    k = max(1, min(int(top_k_points), len(points)))
    top_indices = np.argsort(combined)[-k:]
    top_points = points[top_indices]

    chosen = np.mean(top_points, axis=0)
    return float(chosen[0]), float(chosen[1])


def _directional_contour_tip_from_candidate(
    candidate: DartCandidate,
    *,
    band_fraction: float = 0.15,
    top_k_points: int = 8,
) -> Optional[PointF]:
    """
    Sucht eine konturbasierte Spitze in Richtung des unteren Achsenendes.
    """
    contour = getattr(candidate, "contour", None)
    if contour is None or len(contour) < 3:
        return None

    points = contour.reshape(-1, 2).astype(np.float64)
    upper, lower = _compute_major_axis_endpoints_from_points(points.astype(np.float32))
    if upper is None or lower is None:
        return None

    up = np.asarray(upper, dtype=np.float64)
    low = np.asarray(lower, dtype=np.float64)

    axis_vec = low - up
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len <= 1e-9:
        return float(low[0]), float(low[1])

    axis_dir = axis_vec / axis_len

    projections = (points - up[None, :]) @ axis_dir
    max_proj = float(np.max(projections))
    min_keep = max_proj - max(float(band_fraction), 0.0) * axis_len

    mask = projections >= min_keep
    candidates = points[mask]
    if len(candidates) == 0:
        candidates = points

    # untere Spitze bevorzugen
    order = np.argsort(candidates[:, 1])
    chosen_set = candidates[order[-max(1, min(int(top_k_points), len(candidates))):]]
    chosen = np.mean(chosen_set, axis=0)

    return float(chosen[0]), float(chosen[1])


# -----------------------------------------------------------------------------
# Hauptklasse
# -----------------------------------------------------------------------------

class ImpactEstimator:
    def __init__(self, config: Optional[ImpactEstimatorConfig] = None) -> None:
        self.config = config or ImpactEstimatorConfig()

    def estimate_for_candidate(
        self,
        candidate: DartCandidate,
        *,
        image_shape: Optional[tuple[int, ...]] = None,
    ) -> Optional[ImpactEstimate]:
        """
        Baut Hypothesen für genau einen Kandidaten und wählt daraus den finalen
        Impact-Punkt.
        """
        if candidate.confidence < float(self.config.min_candidate_confidence):
            return None

        hypotheses = self._collect_hypotheses(candidate)

        if not hypotheses:
            return None

        chosen_point, method, aggregate_strength, spread_px = self._choose_final_point(hypotheses)

        if self.config.clamp_to_image_bounds:
            chosen_point = _clamp_point_to_image(chosen_point, image_shape)

        confidence = self._combine_confidence(
            candidate_confidence=float(candidate.confidence),
            hypothesis_strength=float(aggregate_strength),
            spread_px=float(spread_px),
        )

        debug = {}
        if self.config.keep_debug_metadata:
            debug = {
                "spread_px": float(spread_px),
                "aggregate_strength": float(aggregate_strength),
                "strategy": self.config.strategy,
            }

        return ImpactEstimate(
            candidate_id=int(candidate.candidate_id),
            impact_point=(float(chosen_point[0]), float(chosen_point[1])),
            method=str(method),
            confidence=float(confidence),
            source_candidate_confidence=float(candidate.confidence),
            bbox=tuple(candidate.bbox),
            centroid=(float(candidate.centroid[0]), float(candidate.centroid[1])),
            hypotheses=hypotheses,
            debug=debug,
            candidate=candidate,
        )

    def estimate_for_candidates(
        self,
        candidates: list[DartCandidate],
        *,
        image_shape: Optional[tuple[int, ...]] = None,
    ) -> ImpactEstimationResult:
        """
        Schätzt Impact-Punkte für mehrere Kandidaten und sortiert nach finaler
        Konfidenz.
        """
        estimates: list[ImpactEstimate] = []

        for candidate in candidates:
            estimate = self.estimate_for_candidate(candidate, image_shape=image_shape)
            if estimate is not None:
                estimates.append(estimate)

        estimates.sort(key=lambda estimate: estimate.confidence, reverse=True)

        return ImpactEstimationResult(
            estimates=estimates,
            metadata={
                "candidate_count": len(candidates),
                "impact_count": len(estimates),
                "best_candidate_id": None if not estimates else estimates[0].candidate_id,
            },
        )

    def estimate_from_detection_result(
        self,
        detection_result: CandidateDetectionResult,
        *,
        image_shape: Optional[tuple[int, ...]] = None,
    ) -> ImpactEstimationResult:
        """
        Baut Estimates direkt aus einem CandidateDetectionResult.
        """
        if image_shape is None:
            metadata = getattr(detection_result, "metadata", {}) or {}
            image_shape = metadata.get("input_shape")

        return self.estimate_for_candidates(
            candidates=list(detection_result.candidates),
            image_shape=image_shape,
        )

    # -------------------------------------------------------------------------
    # interne Logik
    # -------------------------------------------------------------------------

    def _resolve_board_center_image(
        self,
        candidate: DartCandidate,
    ) -> Optional[PointF]:
        """
        Versucht das Boardzentrum im Bildraum zu bestimmen.
        """
        debug = getattr(candidate, "debug", {}) or {}

        direct = _safe_extract_point(debug.get("board_center_image"))
        if direct is not None:
            return direct

        for key in ("pipeline", "points_like", "calibration", "geometry"):
            points_like = debug.get(key)
            if points_like is None:
                continue

            try:
                center = topdown_to_image_point(points_like, 450.0, 450.0)
                if center is not None:
                    return float(center[0]), float(center[1])
            except Exception:
                continue

        return None

    def _axis_based_quality(
        self,
        *,
        aspect_ratio: float,
        major_axis_length: float,
    ) -> float:
        """
        Qualitätsmaß für achsenbasierte Methoden.
        """
        if major_axis_length < float(self.config.min_major_axis_length_for_axis_based_methods):
            return 0.0
        if aspect_ratio < float(self.config.min_aspect_ratio_for_axis_based_methods):
            return 0.0

        ratio_part = _clip01((float(aspect_ratio) - 1.0) / 5.0)
        length_part = _clip01(float(major_axis_length) / 150.0)
        return _clip01(0.5 * ratio_part + 0.5 * length_part)

    def _extract_board_center_image(
        self,
        candidate: DartCandidate,
    ) -> Optional[PointF]:
        """
        Holt das Boardzentrum aus candidate.debug, falls vorhanden.
        """
        debug = getattr(candidate, "debug", {}) or {}
        board_center = debug.get("board_center_image")
        return _safe_extract_point(board_center)

    def _point_to_line_distance_px(
        self,
        point: PointF,
        line_a: PointF,
        line_b: PointF,
    ) -> float:
        """
        Abstand eines Punktes zu einer Linie in Pixeln.
        """
        px, py = point
        ax, ay = line_a
        bx, by = line_b

        dx = bx - ax
        dy = by - ay

        denom = math.hypot(dx, dy)
        if denom <= 1e-9:
            return math.hypot(px - ax, py - ay)

        num = abs(dy * px - dx * py + bx * ay - by * ax)
        return float(num / denom)

    def _estimate_board_near_contour_tip(
        self,
        candidate: DartCandidate,
    ) -> Optional[tuple[PointF, dict[str, Any]]]:
        """
        Sucht auf der Kontur gezielt die boardnahe Spitze.

        Idee:
        - Konturpunkte nehmen
        - Boardzentrum aus candidate.debug lesen
        - Punkte bevorzugen, die:
          1) näher zum Boardzentrum liegen
          2) entlang der Hauptachse plausibel liegen
          3) nicht weit seitlich von der Dartachse abweichen
        """
        if not self.config.use_board_near_contour_tip:
            return None

        contour = getattr(candidate, "contour", None)
        if contour is None:
            return None

        points = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        if len(points) < 3:
            return None

        board_center = self._extract_board_center_image(candidate)
        if board_center is None:
            return None

        debug = getattr(candidate, "debug", {}) or {}

        axis_a = _safe_extract_point(debug.get("major_axis_endpoint_a"))
        axis_b = _safe_extract_point(debug.get("major_axis_endpoint_b"))

        has_axis = axis_a is not None and axis_b is not None

        scored_points: list[tuple[float, PointF]] = []
        centerward_axis: Optional[PointF] = None

        if has_axis:
            da = math.hypot(axis_a[0] - board_center[0], axis_a[1] - board_center[1])
            db = math.hypot(axis_b[0] - board_center[0], axis_b[1] - board_center[1])
            centerward_axis = axis_a if da <= db else axis_b

        axis_vec = None
        axis_len_sq = None
        if has_axis:
            axis_vec = (axis_b[0] - axis_a[0], axis_b[1] - axis_a[1])
            axis_len_sq = axis_vec[0] * axis_vec[0] + axis_vec[1] * axis_vec[1]

        for pt in points:
            point = (float(pt[0]), float(pt[1]))

            dist_to_center = math.hypot(point[0] - board_center[0], point[1] - board_center[1])
            center_score = 1.0 / max(dist_to_center, 1.0)

            axis_score = 1.0
            forward_score = 1.0

            if has_axis and axis_vec is not None and axis_len_sq is not None and axis_len_sq > 1e-9:
                axis_distance = self._point_to_line_distance_px(point, axis_a, axis_b)
                if axis_distance > float(self.config.board_near_tip_max_distance_from_axis_px):
                    continue

                # Nur Punkte auf der boardnahen Achsenhälfte zulassen.
                # Damit fliegen obere/flight-nahe Punkte raus.
                rel = (point[0] - axis_a[0], point[1] - axis_a[1])
                t = (rel[0] * axis_vec[0] + rel[1] * axis_vec[1]) / axis_len_sq

                da = math.hypot(axis_a[0] - board_center[0], axis_a[1] - board_center[1])
                db = math.hypot(axis_b[0] - board_center[0], axis_b[1] - board_center[1])

                # Wenn axis_b boardnaher ist, wollen wir nur den unteren/boardnahen Teil.
                if db <= da:
                    if t < 0.55:
                        continue
                else:
                    if t > 0.45:
                        continue

                axis_score = 1.0 / (1.0 + axis_distance)

                if centerward_axis is not None:
                    forward_dist = math.hypot(
                        point[0] - centerward_axis[0],
                        point[1] - centerward_axis[1],
                    )
                    forward_score = 1.0 / (1.0 + forward_dist)

            score = (
                float(self.config.board_near_tip_distance_weight) * center_score
                + float(self.config.board_near_tip_forward_weight) * forward_score
                + 0.75 * axis_score
            )

            scored_points.append((score, point))

        if not scored_points:
            return None

        scored_points.sort(key=lambda item: item[0], reverse=True)

        top_k = max(1, int(self.config.board_near_tip_top_k_points))
        selected = scored_points[:top_k]

        # Für die boardnahe Spitze darf nicht zu stark über die ganze Kontur gemittelt werden.
        # Deshalb nehmen wir den besten Punkt direkt, oder nur ein sehr lokales Mittel
        # der wenigen besten Punkte.
        if top_k <= 1 or len(selected) == 1:
            best_point = selected[0][1]
        else:
            total_weight = sum(score for score, _ in selected)
            if total_weight <= 1e-9:
                best_point = selected[0][1]
            else:
                x = sum(score * point[0] for score, point in selected) / total_weight
                y = sum(score * point[1] for score, point in selected) / total_weight
                best_point = (float(x), float(y))

        metadata = {
            "board_center_image": board_center,
            "top_k_points": top_k,
            "selected_points": [point for _, point in selected[:8]],
            "best_raw_point": selected[0][1],
            "axis_reference_point": centerward_axis,
        }
        return best_point, metadata

    def _collect_hypotheses(
        self,
        candidate: DartCandidate,
    ) -> list[ImpactHypothesis]:
        """
        Baut alle aktivierten Hypothesen für einen Kandidaten.
        """
        hypotheses: list[ImpactHypothesis] = []

        aspect_ratio = float(getattr(candidate, "aspect_ratio", 0.0))
        major_axis_length = float(getattr(candidate, "major_axis_length", 0.0))

        # ------------------------------------------------------------------
        # 1) Kandidaten-Default
        # ------------------------------------------------------------------
        if self.config.use_candidate_default:
            point = (float(candidate.impact_point[0]), float(candidate.impact_point[1]))
            hypotheses.append(
                ImpactHypothesis(
                    name="candidate_default",
                    point=point,
                    base_weight=float(self.config.weight_candidate_default),
                    source_quality=1.0,
                    consistency_score=1.0,
                    final_weight=float(self.config.weight_candidate_default),
                    metadata={},
                )
            )

        # ------------------------------------------------------------------
        # 2) Tiefster Konturpunkt
        # ------------------------------------------------------------------
        if self.config.use_lowest_contour_point:
            debug = getattr(candidate, "debug", {}) or {}
            point = _safe_extract_point(debug.get("contour_lowest_point"))

            if point is None:
                contour = getattr(candidate, "contour", None)
                if contour is not None and len(contour) >= 1:
                    pts = contour.reshape(-1, 2)
                    idx = int(np.argmax(pts[:, 1]))
                    point = float(pts[idx, 0]), float(pts[idx, 1])

            if point is not None:
                hypotheses.append(
                    ImpactHypothesis(
                        name="lowest_contour_point",
                        point=point,
                        base_weight=float(self.config.weight_lowest_contour_point),
                        source_quality=0.85,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_lowest_contour_point) * 0.85,
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
                source_quality = _clip01(0.45 + 0.55 * axis_quality)
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
        # 3b) Zentrum-näherer Major-Axis-Endpunkt
        # ------------------------------------------------------------------
        if self.config.use_major_axis_centerward_endpoint:
            board_center_image = self._resolve_board_center_image(candidate)

            if board_center_image is not None:
                point = _major_axis_centerward_endpoint_from_candidate(
                    candidate,
                    board_center_image=board_center_image,
                )

                if point is not None:
                    axis_quality = self._axis_based_quality(
                        aspect_ratio=aspect_ratio,
                        major_axis_length=major_axis_length,
                    )
                    source_quality = _clip01(0.45 + 0.55 * axis_quality)
                    hypotheses.append(
                        ImpactHypothesis(
                            name="major_axis_centerward_endpoint",
                            point=point,
                            base_weight=float(self.config.weight_major_axis_centerward_endpoint),
                            source_quality=source_quality,
                            consistency_score=1.0,
                            final_weight=float(self.config.weight_major_axis_centerward_endpoint) * source_quality,
                            metadata={
                                "board_center_image": board_center_image,
                            },
                        )
                    )

        # ------------------------------------------------------------------
        # 3c) Konturbasierte Spitze in Richtung Boardzentrum
        # ------------------------------------------------------------------
        if self.config.use_centerward_contour_tip:
            board_center_image = self._resolve_board_center_image(candidate)

            if board_center_image is not None:
                point = _centerward_contour_tip_from_candidate(
                    candidate,
                    board_center_image=board_center_image,
                    top_k_points=int(self.config.centerward_tip_top_k_points),
                    distance_weight=float(self.config.centerward_tip_distance_weight),
                    forward_weight=float(self.config.centerward_tip_forward_weight),
                )

                if point is not None:
                    is_plausible = True
                    axis_reference_point = None
                    axis_reference_distance = None

                    if self.config.centerward_tip_requires_axis_agreement:
                        axis_reference_point = _major_axis_centerward_endpoint_from_candidate(
                            candidate,
                            board_center_image=board_center_image,
                        )

                        if axis_reference_point is not None:
                            axis_reference_distance = _point_distance(point, axis_reference_point)
                            if axis_reference_distance > float(
                                self.config.centerward_tip_max_distance_from_centerward_axis_px
                            ):
                                is_plausible = False

                    if is_plausible:
                        axis_quality = self._axis_based_quality(
                            aspect_ratio=aspect_ratio,
                            major_axis_length=major_axis_length,
                        )
                        contour_quality = _clip01(0.40 + 0.60 * axis_quality)

                        if axis_reference_distance is not None:
                            max_dist = max(
                                float(self.config.centerward_tip_max_distance_from_centerward_axis_px),
                                1e-6,
                            )
                            agreement = 1.0 - min(axis_reference_distance / max_dist, 1.0)
                            contour_quality = _clip01(
                                contour_quality * (0.75 + 0.25 * agreement)
                            )

                        hypotheses.append(
                            ImpactHypothesis(
                                name="centerward_contour_tip",
                                point=point,
                                base_weight=float(self.config.weight_centerward_contour_tip),
                                source_quality=contour_quality,
                                consistency_score=1.0,
                                final_weight=float(self.config.weight_centerward_contour_tip) * contour_quality,
                                metadata={
                                    "board_center_image": board_center_image,
                                    "top_k_points": int(self.config.centerward_tip_top_k_points),
                                    "axis_reference_point": axis_reference_point,
                                    "axis_reference_distance": axis_reference_distance,
                                    "requires_axis_agreement": bool(
                                        self.config.centerward_tip_requires_axis_agreement
                                    ),
                                },
                            )
                        )

        # ------------------------------------------------------------------
        # 4) Directional contour tip
        # ------------------------------------------------------------------
        if self.config.use_directional_contour_tip:
            point = _directional_contour_tip_from_candidate(
                candidate,
                band_fraction=float(self.config.directional_tip_band_fraction),
                top_k_points=int(self.config.directional_tip_top_k_points),
            )

            if point is not None:
                axis_quality = self._axis_based_quality(
                    aspect_ratio=aspect_ratio,
                    major_axis_length=major_axis_length,
                )
                source_quality = _clip01(0.45 + 0.55 * axis_quality)
                hypotheses.append(
                    ImpactHypothesis(
                        name="directional_contour_tip",
                        point=point,
                        base_weight=float(self.config.weight_directional_contour_tip),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_directional_contour_tip) * source_quality,
                        metadata={},
                    )
                )

        # ------------------------------------------------------------------
        # 5) Board-nahe Konturspitze
        # ------------------------------------------------------------------
        if self.config.use_board_near_contour_tip:
            board_near_tip_result = self._estimate_board_near_contour_tip(candidate)

            if board_near_tip_result is not None:
                board_near_tip_point, board_near_tip_metadata = board_near_tip_result

                axis_quality = self._axis_based_quality(
                    aspect_ratio=aspect_ratio,
                    major_axis_length=major_axis_length,
                )
                source_quality = _clip01(0.45 + 0.55 * axis_quality)

                hypotheses.append(
                    ImpactHypothesis(
                        name="board_near_contour_tip",
                        point=board_near_tip_point,
                        base_weight=float(self.config.weight_board_near_contour_tip),
                        source_quality=source_quality,
                        consistency_score=1.0,
                        final_weight=float(self.config.weight_board_near_contour_tip) * source_quality,
                        metadata=board_near_tip_metadata,
                    )
                )

        return hypotheses

    def _choose_final_point(
        self,
        hypotheses: list[ImpactHypothesis],
    ) -> tuple[PointF, str, float, float]:
        """
        Wählt den finalen Impact-Punkt aus den Hypothesen.
        """
        if not hypotheses:
            raise ValueError("At least one hypothesis is required.")

        strategy = self.config.strategy.strip().lower()

        available_by_name = {hypothesis.name: hypothesis for hypothesis in hypotheses}
        available_names = set(available_by_name.keys())

        supported_direct_strategies = {
            "candidate_default",
            "lowest_contour_point",
            "major_axis_lower_endpoint",
            "major_axis_centerward_endpoint",
            "centerward_contour_tip",
            "directional_contour_tip",
            "board_near_contour_tip",
        }

        if strategy in supported_direct_strategies:
            if strategy not in available_by_name:
                raise ValueError(
                    f"Impact strategy '{strategy}' was requested, but no such hypothesis "
                    f"was generated for this candidate. Available hypotheses: {sorted(available_names)}"
                )

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
                "'major_axis_centerward_endpoint', "
                "'centerward_contour_tip', "
                "'directional_contour_tip'."
                "'board_near_contour_tip'."
            )

        # Blend nur aus den spitzenrelevanten Hypothesen bilden.
        preferred_names = {
            "major_axis_centerward_endpoint",
            "directional_contour_tip",
        }

        preferred_hypotheses = [
            hypothesis
            for hypothesis in hypotheses
            if hypothesis.name in preferred_names
        ]

        # Fallback:
        # Wenn davon zu wenig vorhanden sind, nehme wieder alle Hypothesen.
        active_hypotheses = preferred_hypotheses if len(preferred_hypotheses) >= 2 else hypotheses

        weights = np.asarray(
            [max(hypothesis.final_weight, 1e-9) for hypothesis in active_hypotheses],
            dtype=np.float64,
        )
        points = np.asarray([hypothesis.point for hypothesis in active_hypotheses], dtype=np.float64)

        weighted_point = np.average(points, axis=0, weights=weights)
        chosen_point = (float(weighted_point[0]), float(weighted_point[1]))

        aggregate_strength = float(np.mean(weights))
        spread_px = _compute_weighted_spread(chosen_point, active_hypotheses)

        return chosen_point, "blend", aggregate_strength, spread_px


    def _combine_confidence(
        self,
        *,
        candidate_confidence: float,
        hypothesis_strength: float,
        spread_px: float,
    ) -> float:
        """
        Kombiniert Kandidatenkonfidenz, Hypothesenstärke und Streuung zu einer
        finalen Impact-Konfidenz.
        """
        source_part = _clip01(float(candidate_confidence))
        strength_part = _clip01(float(hypothesis_strength))

        scale = max(float(self.config.consistency_distance_scale_px), 1e-6)
        spread_part = 1.0 / (1.0 + (float(spread_px) / scale))

        confidence = (
            float(self.config.weight_source_candidate_confidence) * source_part +
            float(self.config.weight_hypothesis_strength) * strength_part
        )

        confidence *= float(spread_part)
        return _clip01(confidence)


# -----------------------------------------------------------------------------
# Modulweite Convenience-Wrapper
# -----------------------------------------------------------------------------

def estimate_impact_for_candidate(
    candidate: DartCandidate,
    *,
    image_shape: Optional[tuple[int, ...]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> Optional[ImpactEstimate]:
    estimator = ImpactEstimator(config=config)
    return estimator.estimate_for_candidate(candidate, image_shape=image_shape)


def estimate_impacts(
    candidates: list[DartCandidate],
    *,
    image_shape: Optional[tuple[int, ...]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> ImpactEstimationResult:
    estimator = ImpactEstimator(config=config)
    return estimator.estimate_for_candidates(candidates, image_shape=image_shape)


def estimate_impacts_from_detection_result(
    detection_result: CandidateDetectionResult,
    *,
    image_shape: Optional[tuple[int, ...]] = None,
    config: Optional[ImpactEstimatorConfig] = None,
) -> ImpactEstimationResult:
    estimator = ImpactEstimator(config=config)
    return estimator.estimate_from_detection_result(
        detection_result,
        image_shape=image_shape,
    )