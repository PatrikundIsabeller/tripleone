# vision/dart_candidate_detector.py
# Zweck:
# Diese Datei ist die reine Kandidaten-Detektionsschicht für Triple One.
#
# Sie ist bewusst NICHT für das finale Dart-Scoring verantwortlich.
# Sie macht ausschließlich:
# - Frame/Referenz-Vergleich
# - ROI-/Board-Maskierung
# - Binärmaske und Morphologie
# - Konturfindung
# - Heuristische Bewertung "dart-ähnlicher" Kandidaten
# - Ausgabe von Kandidaten inkl. Impact-Hypothese und Debugdaten
#
# Wichtige Architekturregel:
# Diese Datei besitzt KEINE eigene Board-Geometrie und KEIN eigenes Score-Mapping.
# Sie berechnet NICHT S20/D20/T20/DBULL.
#
# Der Output dieser Datei ist nur:
# "Welche Bildbereiche sind plausible Dart-/Impact-Kandidaten?"
#
# Das finale Feld-Mapping gehört später in:
# - vision/impact_estimator.py
# - vision/single_cam_detector.py

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

PointF = tuple[float, float]
PointI = tuple[int, int]
BBox = tuple[int, int, int, int]


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class CandidateDetectorConfig:
    """
    Konfiguration für die Kandidaten-Detektion.

    Wichtig:
    Diese Werte sind bewusst konservative Startwerte.
    Sie sind dafür gedacht, stabil debugbar zu sein und nicht "magisch"
    auf jeden Fall irgendetwas zu erraten.

    Die Feineinstellung kommt später anhand echter Testbilder/Videos.
    """

    # Vorverarbeitung
    blur_kernel_size: int = 5
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    # Differenz-/Threshold-Logik
    diff_threshold: int = 24
    use_otsu_threshold: bool = False

    # Morphologie
    open_kernel_size: int = 3
    close_kernel_size: int = 5
    dilate_kernel_size: int = 3
    erode_iterations: int = 0
    dilate_iterations: int = 1

    # Kandidatenfilter
    min_contour_area: float = 25.0
    max_contour_area_ratio: float = 0.10
    min_aspect_ratio: float = 1.20
    min_solidity: float = 0.10
    min_extent: float = 0.03
    max_extent: float = 0.95
    min_confidence: float = 0.18
    max_candidates: int = 8

    # Impact-Hypothese
    # Erlaubte Werte:
    # - "lowest_contour_point"
    # - "major_axis_lower_endpoint"
    impact_point_mode: str = "major_axis_lower_endpoint"

    # Referenzbildbehandlung
    allow_reference_resize: bool = False

    # Debug
    keep_debug_images: bool = True
    draw_candidate_ids: bool = True
    draw_confidence: bool = True


# -----------------------------------------------------------------------------
# Datenmodelle für Kandidaten und Ergebnisse
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class ContourMetrics:
    """
    Interne, strukturierte Metriken einer einzelnen Kontur.
    """
    area: float
    perimeter: float
    bbox: BBox
    centroid: PointF
    min_area_rect_center: PointF
    min_area_rect_size: tuple[float, float]
    min_area_rect_angle: float
    aspect_ratio: float
    extent: float
    solidity: float
    circularity: float
    major_axis_length: float
    minor_axis_length: float
    elongation: float
    contour_lowest_point: PointF
    contour_highest_point: PointF
    contour_leftmost_point: PointF
    contour_rightmost_point: PointF
    major_axis_endpoint_a: PointF
    major_axis_endpoint_b: PointF
    touches_image_border: bool


@dataclass(slots=True)
class DartCandidate:
    """
    Ein plausibler Dart-/Impact-Kandidat aus einem Frame.

    Wichtiger Punkt:
    'impact_point' ist hier nur eine Hypothese auf Kandidaten-Ebene,
    noch NICHT das finale Trefferfeld.
    """
    candidate_id: int
    bbox: BBox
    centroid: PointF
    impact_point: PointF
    area: float
    aspect_ratio: float
    solidity: float
    extent: float
    circularity: float
    angle_degrees: float
    major_axis_length: float
    minor_axis_length: float
    elongation: float
    confidence: float
    contour: Optional[np.ndarray] = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_contour: bool = False) -> dict[str, Any]:
        """
        Serialisiert den Kandidaten für Debugging / Logs / API-artige Nutzung.
        """
        data = {
            "candidate_id": self.candidate_id,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "impact_point": self.impact_point,
            "area": self.area,
            "aspect_ratio": self.aspect_ratio,
            "solidity": self.solidity,
            "extent": self.extent,
            "circularity": self.circularity,
            "angle_degrees": self.angle_degrees,
            "major_axis_length": self.major_axis_length,
            "minor_axis_length": self.minor_axis_length,
            "elongation": self.elongation,
            "confidence": self.confidence,
            "debug": self.debug,
        }

        if include_contour and self.contour is not None:
            data["contour"] = self.contour.reshape(-1, 2).tolist()

        return data


@dataclass(slots=True)
class CandidateDetectionResult:
    """
    Gesamtergebnis eines Kandidaten-Detektionslaufs.
    """
    candidates: list[DartCandidate]
    reference_used: bool
    board_mask_used: bool
    board_polygon_used: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def best_candidate(self) -> Optional[DartCandidate]:
        """
        Gibt den Kandidaten mit der höchsten Konfidenz zurück.
        """
        if not self.candidates:
            return None
        return self.candidates[0]

    def to_dict(self, include_contours: bool = False) -> dict[str, Any]:
        """
        Serialisiert das Ergebnis ohne die Debug-Bildarrays.
        """
        return {
            "reference_used": self.reference_used,
            "board_mask_used": self.board_mask_used,
            "board_polygon_used": self.board_polygon_used,
            "metadata": self.metadata,
            "candidates": [
                candidate.to_dict(include_contour=include_contours)
                for candidate in self.candidates
            ],
        }

    def render_debug_overlay(
        self,
        frame: np.ndarray,
        max_candidates: Optional[int] = None,
    ) -> np.ndarray:
        """
        Zeichnet Kandidaten auf ein BGR-Bild für UI/Debug.
        """
        if frame.ndim == 2:
            canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            canvas = frame.copy()

        count = len(self.candidates) if max_candidates is None else min(len(self.candidates), max_candidates)

        for idx, candidate in enumerate(self.candidates[:count]):
            x, y, w, h = candidate.bbox
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 255), 1)

            cx, cy = _round_point(candidate.centroid)
            ix, iy = _round_point(candidate.impact_point)

            cv2.circle(canvas, (cx, cy), 3, (255, 255, 0), -1)
            cv2.circle(canvas, (ix, iy), 4, (0, 0, 255), -1)

            label_parts = [f"#{candidate.candidate_id}"]
            label_parts.append(f"conf={candidate.confidence:.2f}")
            label_parts.append(f"ar={candidate.aspect_ratio:.2f}")
            label_parts.append(f"a={candidate.area:.0f}")

            cv2.putText(
                canvas,
                " | ".join(label_parts),
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if candidate.contour is not None:
                cv2.drawContours(canvas, [candidate.contour], -1, (0, 255, 0), 1)

            # Debug-Linie entlang der Major Axis, falls vorhanden
            a = candidate.debug.get("major_axis_endpoint_a")
            b = candidate.debug.get("major_axis_endpoint_b")
            if a is not None and b is not None:
                cv2.line(
                    canvas,
                    _round_point(a),
                    _round_point(b),
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Linie vom Schwerpunkt zum Impact-Punkt
            cv2.line(
                canvas,
                (cx, cy),
                (ix, iy),
                (0, 165, 255),
                1,
                cv2.LINE_AA,
            )

        return canvas


# -----------------------------------------------------------------------------
# Hauptklasse
# -----------------------------------------------------------------------------

class DartCandidateDetector:
    """
    Reine Kandidaten-Detektion auf Basis von Bilddifferenzen.

    Verantwortungsbereich:
    - mögliche Dart-/Impact-Kandidaten in einem Frame finden
    - Kandidaten strukturieren
    - Debugdaten liefern

    Nicht verantwortlich für:
    - Homography
    - Board-Geometrie
    - Trefferfeldberechnung
    - Spielerlogik
    """

    def __init__(self, config: Optional[CandidateDetectorConfig] = None) -> None:
        self.config = config or CandidateDetectorConfig()

    # -------------------------------------------------------------------------
    # Öffentliche API
    # -------------------------------------------------------------------------

    def detect_candidates(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        *,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[np.ndarray | list[PointI] | list[PointF]] = None,
    ) -> CandidateDetectionResult:
        """
        Hauptmethode:
        Findet plausible Dart-/Impact-Kandidaten in 'frame' im Vergleich zu
        'reference_frame'.

        Parameter:
        - frame:
            aktuelles Bild mit möglichem Dart
        - reference_frame:
            Referenzbild des leeren Boards
        - board_mask:
            optionale Binärmaske (ROI), 0/255
        - board_polygon:
            optionale ROI als Polygon; daraus wird intern eine Maske erzeugt

        Wichtig:
        Es wird KEIN Trefferfeld berechnet.
        """
        _validate_frame(frame, name="frame")
        _validate_frame(reference_frame, name="reference_frame")

        current_bgr, reference_bgr, resized_reference = self._align_frames(frame, reference_frame)

        board_mask_prepared = self._prepare_board_mask(
            image_shape=current_bgr.shape[:2],
            board_mask=board_mask,
            board_polygon=board_polygon,
        )

        gray_current = self._preprocess_to_gray(current_bgr)
        gray_reference = self._preprocess_to_gray(reference_bgr)

        diff = cv2.absdiff(gray_reference, gray_current)

        if board_mask_prepared is not None:
            diff_masked = cv2.bitwise_and(diff, diff, mask=board_mask_prepared)
        else:
            diff_masked = diff

        binary_mask = self._threshold_difference(diff_masked)
        cleaned_mask = self._clean_mask(binary_mask)

        if board_mask_prepared is not None:
            cleaned_mask = cv2.bitwise_and(cleaned_mask, cleaned_mask, mask=board_mask_prepared)

        contours = _find_external_contours(cleaned_mask)
        roi_pixel_count = int(np.count_nonzero(board_mask_prepared)) if board_mask_prepared is not None else int(cleaned_mask.shape[0] * cleaned_mask.shape[1])

        candidates: list[DartCandidate] = []
        rejected_count = 0

        for contour_index, contour in enumerate(contours):
            candidate = self._build_candidate_from_contour(
                contour=contour,
                contour_index=contour_index,
                image_shape=current_bgr.shape[:2],
                roi_pixel_count=roi_pixel_count,
            )
            if candidate is None:
                rejected_count += 1
                continue
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.confidence, reverse=True)
        candidates = candidates[: self.config.max_candidates]

        metadata = {
            "input_shape": tuple(int(v) for v in current_bgr.shape),
            "reference_resized": resized_reference,
            "total_contours": len(contours),
            "rejected_contours": rejected_count,
            "accepted_candidates": len(candidates),
            "roi_pixel_count": roi_pixel_count,
            "config": _dataclass_to_dict(self.config),
        }

        debug_images: dict[str, np.ndarray] = {}
        if self.config.keep_debug_images:
            debug_images["gray_current"] = gray_current
            debug_images["gray_reference"] = gray_reference
            debug_images["difference"] = diff
            debug_images["difference_masked"] = diff_masked
            debug_images["binary_mask"] = binary_mask
            debug_images["cleaned_mask"] = cleaned_mask
            if board_mask_prepared is not None:
                debug_images["board_mask"] = board_mask_prepared

        result = CandidateDetectionResult(
            candidates=candidates,
            reference_used=True,
            board_mask_used=board_mask_prepared is not None,
            board_polygon_used=board_polygon is not None,
            metadata=metadata,
            debug_images=debug_images,
        )

        return result

    # -------------------------------------------------------------------------
    # Interne Frame-/Masken-Vorbereitung
    # -------------------------------------------------------------------------

    def _align_frames(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Stellt sicher, dass Frame und Referenz kompatibel sind.

        Standard:
        - gleiche Auflösung erforderlich

        Optional:
        - Referenz darf auf Frame-Größe resized werden
        """
        current_bgr = _ensure_bgr(frame)
        reference_bgr = _ensure_bgr(reference_frame)

        if current_bgr.shape[:2] == reference_bgr.shape[:2]:
            return current_bgr, reference_bgr, False

        if not self.config.allow_reference_resize:
            raise ValueError(
                "Frame and reference_frame must have identical size. "
                f"Got frame={current_bgr.shape[:2]} vs reference={reference_bgr.shape[:2]}. "
                "If you really want auto-resize, enable config.allow_reference_resize=True."
            )

        resized_reference = cv2.resize(
            reference_bgr,
            (current_bgr.shape[1], current_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        logger.warning(
            "Reference frame was resized from %s to %s.",
            reference_bgr.shape[:2],
            current_bgr.shape[:2],
        )
        return current_bgr, resized_reference, True

    def _prepare_board_mask(
        self,
        *,
        image_shape: tuple[int, int],
        board_mask: Optional[np.ndarray],
        board_polygon: Optional[np.ndarray | list[PointI] | list[PointF]],
    ) -> Optional[np.ndarray]:
        """
        Bereitet optional eine Board-/ROI-Maske vor.

        Wichtig:
        Hier wird keine eigene Geometrie erzeugt.
        Es wird nur eine bereits gelieferte Maske oder ein Polygon verwendet.
        """
        height, width = image_shape

        if board_mask is not None and board_polygon is not None:
            raise ValueError("Pass either board_mask or board_polygon, not both.")

        if board_mask is not None:
            _validate_mask(board_mask, expected_shape=image_shape, name="board_mask")
            return _normalize_mask(board_mask)

        if board_polygon is not None:
            polygon_array = np.asarray(board_polygon, dtype=np.float32)
            if polygon_array.ndim != 2 or polygon_array.shape[1] != 2 or len(polygon_array) < 3:
                raise ValueError(
                    "board_polygon must be an array-like with shape (N, 2) and N >= 3."
                )

            polygon_int = np.round(polygon_array).astype(np.int32)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_int], 255)
            return mask

        return None

    def _preprocess_to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Wandelt ein Bild in ein stabiles Graustufenbild für die Differenzbildung um.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.config.apply_clahe:
            clahe = cv2.createCLAHE(
                clipLimit=self.config.clahe_clip_limit,
                tileGridSize=(self.config.clahe_tile_grid_size, self.config.clahe_tile_grid_size),
            )
            gray = clahe.apply(gray)

        blur_ksize = _ensure_odd_kernel(self.config.blur_kernel_size)
        if blur_ksize > 1:
            gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

        return gray

    def _threshold_difference(self, diff: np.ndarray) -> np.ndarray:
        """
        Erzeugt eine Binärmaske aus dem Differenzbild.
        """
        if self.config.use_otsu_threshold:
            _, binary = cv2.threshold(
                diff,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            return binary

        threshold_value = int(np.clip(self.config.diff_threshold, 0, 255))
        _, binary = cv2.threshold(
            diff,
            threshold_value,
            255,
            cv2.THRESH_BINARY,
        )
        return binary

    def _clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Reinigt die Binärmaske durch Morphologie.
        """
        result = binary_mask.copy()

        open_k = _ensure_odd_kernel(self.config.open_kernel_size)
        close_k = _ensure_odd_kernel(self.config.close_kernel_size)
        dilate_k = _ensure_odd_kernel(self.config.dilate_kernel_size)

        if close_k > 1:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_close)

        if open_k > 1:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel_open)

        erode_iterations = max(0, int(self.config.erode_iterations))
        if erode_iterations > 0:
            kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            result = cv2.erode(result, kernel_erode, iterations=erode_iterations)

        dilate_iterations = max(0, int(self.config.dilate_iterations))
        if dilate_iterations > 0 and dilate_k > 1:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            result = cv2.dilate(result, kernel_dilate, iterations=dilate_iterations)

        return result

    # -------------------------------------------------------------------------
    # Interne Kandidatenbildung
    # -------------------------------------------------------------------------

    def _build_candidate_from_contour(
        self,
        *,
        contour: np.ndarray,
        contour_index: int,
        image_shape: tuple[int, int],
        roi_pixel_count: int,
    ) -> Optional[DartCandidate]:
        """
        Baut aus einer Kontur einen DartCandidate, sofern die Kontur die
        Mindestanforderungen erfüllt.
        """
        metrics = _compute_contour_metrics(contour=contour, image_shape=image_shape)
        if metrics is None:
            return None

        max_contour_area = float(max(1, roi_pixel_count)) * float(self.config.max_contour_area_ratio)

        # Harte Basisfilter, damit grober Müll früh rausfliegt.
        if metrics.area < self.config.min_contour_area:
            return None

        if metrics.area > max_contour_area:
            return None

        if metrics.aspect_ratio < self.config.min_aspect_ratio:
            return None

        if metrics.solidity < self.config.min_solidity:
            return None

        if metrics.extent < self.config.min_extent or metrics.extent > self.config.max_extent:
            return None

        impact_point = self._estimate_candidate_impact_point(metrics)
        confidence, confidence_debug = self._score_candidate(metrics, max_contour_area=max_contour_area)

        if confidence < self.config.min_confidence:
            return None

        candidate = DartCandidate(
            candidate_id=contour_index,
            bbox=metrics.bbox,
            centroid=metrics.centroid,
            impact_point=impact_point,
            area=metrics.area,
            aspect_ratio=metrics.aspect_ratio,
            solidity=metrics.solidity,
            extent=metrics.extent,
            circularity=metrics.circularity,
            angle_degrees=metrics.min_area_rect_angle,
            major_axis_length=metrics.major_axis_length,
            minor_axis_length=metrics.minor_axis_length,
            elongation=metrics.elongation,
            confidence=confidence,
            contour=contour.copy(),
            debug={
                "perimeter": metrics.perimeter,
                "min_area_rect_center": metrics.min_area_rect_center,
                "min_area_rect_size": metrics.min_area_rect_size,
                "touches_image_border": metrics.touches_image_border,
                "contour_lowest_point": metrics.contour_lowest_point,
                "contour_highest_point": metrics.contour_highest_point,
                "contour_leftmost_point": metrics.contour_leftmost_point,
                "contour_rightmost_point": metrics.contour_rightmost_point,
                "major_axis_endpoint_a": metrics.major_axis_endpoint_a,
                "major_axis_endpoint_b": metrics.major_axis_endpoint_b,
                "confidence_parts": confidence_debug,
                "impact_point_mode": self.config.impact_point_mode,
            },
        )

        return candidate

    def _estimate_candidate_impact_point(self, metrics: ContourMetrics) -> PointF:
        """
        Liefert eine Impact-Hypothese auf Kandidaten-Ebene.

        Wichtig:
        Das ist bewusst noch KEINE finale Dart-Spitze im fachlichen Sinn,
        sondern nur eine kontrollierte Heuristik für den nächsten Pipeline-Schritt.
        """
        mode = self.config.impact_point_mode.strip().lower()

        if mode == "lowest_contour_point":
            return metrics.contour_lowest_point

        if mode == "major_axis_lower_endpoint":
            a = metrics.major_axis_endpoint_a
            b = metrics.major_axis_endpoint_b
            return a if a[1] >= b[1] else b

        raise ValueError(
            "Unsupported impact_point_mode. Expected 'lowest_contour_point' "
            "or 'major_axis_lower_endpoint'."
        )

    def _score_candidate(
        self,
        metrics: ContourMetrics,
        *,
        max_contour_area: float,
    ) -> tuple[float, dict[str, float]]:
        """
        Bewertet, wie plausibel eine Kontur als Dart-/Impact-Kandidat ist.

        Wichtig:
        'confidence' bedeutet hier NICHT:
        "Dieses Feld ist sicher getroffen."

        Sondern nur:
        "Diese Kontur sieht eher dart-ähnlich aus als andere Konturen."
        """
        # 1) Flächenscore:
        # Zu klein = eher Rauschen
        # Zu groß = eher globale Bewegung / falsche Maske
        area_ratio = metrics.area / max(max_contour_area, 1.0)
        area_score = _clip01(1.0 - area_ratio) if metrics.area <= max_contour_area else 0.0
        area_score = max(area_score, 0.10)

        # 2) Aspektverhältnis:
        # Darts / Schaft-/Spitzenbereiche sind typischerweise eher länglich als rund.
        aspect_score = _soft_range_score(
            value=metrics.aspect_ratio,
            good_min=1.8,
            good_max=8.0,
            hard_min=self.config.min_aspect_ratio,
            hard_max=20.0,
        )

        # 3) Solidity:
        # Nicht zu zerfranst / nicht zu löchrig.
        solidity_score = _soft_range_score(
            value=metrics.solidity,
            good_min=0.25,
            good_max=0.95,
            hard_min=0.05,
            hard_max=1.0,
        )

        # 4) Extent:
        # Sehr quadratische/kompakte Flächen sind eher schlechter.
        extent_score = _soft_range_score(
            value=metrics.extent,
            good_min=0.08,
            good_max=0.70,
            hard_min=0.01,
            hard_max=0.99,
        )

        # 5) Nicht-rund-Score:
        # Ein Dartkandidat soll tendenziell nicht stark kreisförmig sein.
        non_round_score = _clip01(1.0 - min(metrics.circularity, 1.0))

        # 6) Border-Penalty:
        # Kandidaten am Bildrand sind oft Artefakte.
        border_penalty = 0.20 if metrics.touches_image_border else 0.0

        weighted = (
            0.20 * area_score
            + 0.32 * aspect_score
            + 0.18 * solidity_score
            + 0.15 * extent_score
            + 0.15 * non_round_score
        ) - border_penalty

        confidence = _clip01(weighted)

        confidence_debug = {
            "area_score": float(area_score),
            "aspect_score": float(aspect_score),
            "solidity_score": float(solidity_score),
            "extent_score": float(extent_score),
            "non_round_score": float(non_round_score),
            "border_penalty": float(border_penalty),
            "weighted_confidence": float(confidence),
        }

        return confidence, confidence_debug


# -----------------------------------------------------------------------------
# Modulweite Convenience-Funktionen
# -----------------------------------------------------------------------------

def build_candidate_detector(
    config: Optional[CandidateDetectorConfig] = None,
) -> DartCandidateDetector:
    """
    Bequemer Builder für den Detector.
    """
    return DartCandidateDetector(config=config)


def detect_dart_candidates(
    frame: np.ndarray,
    reference_frame: np.ndarray,
    *,
    board_mask: Optional[np.ndarray] = None,
    board_polygon: Optional[np.ndarray | list[PointI] | list[PointF]] = None,
    config: Optional[CandidateDetectorConfig] = None,
) -> CandidateDetectionResult:
    """
    Modulweiter Convenience-Wrapper.
    """
    detector = build_candidate_detector(config=config)
    return detector.detect_candidates(
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
        raise ValueError(
            f"{name} must have ndim 2 or 3, got {frame.ndim}."
        )

    if frame.size == 0:
        raise ValueError(f"{name} must not be empty.")


def _validate_mask(mask: np.ndarray, *, expected_shape: tuple[int, int], name: str) -> None:
    """
    Prüft, ob eine Binärmaske grundsätzlich zur Bildgröße passt.
    """
    if mask is None:
        raise ValueError(f"{name} must not be None.")

    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(mask)!r}.")

    if mask.ndim != 2:
        raise ValueError(f"{name} must be a single-channel mask.")

    if mask.shape != expected_shape:
        raise ValueError(
            f"{name} shape mismatch. Expected {expected_shape}, got {mask.shape}."
        )


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

    raise ValueError(
        f"Unsupported frame shape for BGR conversion: {frame.shape}"
    )


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Normalisiert beliebige Maskenwerte auf 0/255.
    """
    normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    return normalized


def _ensure_odd_kernel(value: int) -> int:
    """
    Stellt sicher, dass Kernelgrößen gültige ungerade positive Zahlen sind.
    """
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    return value


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


def _soft_range_score(
    *,
    value: float,
    good_min: float,
    good_max: float,
    hard_min: float,
    hard_max: float,
) -> float:
    """
    Weicher Score für Wertebereiche.

    - innerhalb [good_min, good_max] => 1.0
    - außerhalb [hard_min, hard_max] => 0.0
    - dazwischen linearer Übergang

    Das ist absichtlich robust und transparent, nicht "magisch".
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


def _find_external_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Sucht nur externe Konturen.
    """
    contours, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def _compute_contour_metrics(
    *,
    contour: np.ndarray,
    image_shape: tuple[int, int],
) -> Optional[ContourMetrics]:
    """
    Berechnet strukturierte Metriken für eine Kontur.

    Rückgabe:
    - ContourMetrics bei Erfolg
    - None bei degenerierten Konturen
    """
    if contour is None or len(contour) < 3:
        return None

    area = float(cv2.contourArea(contour))
    if area <= 0.0:
        return None

    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)

    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        centroid = (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )
    else:
        centroid = (float(x + w / 2.0), float(y + h / 2.0))

    rect = cv2.minAreaRect(contour)
    (rcx, rcy), (rw, rh), angle = rect

    major_axis_length = float(max(rw, rh))
    minor_axis_length = float(min(rw, rh))
    aspect_ratio = float(major_axis_length / max(minor_axis_length, 1e-6))
    extent = float(area / max(w * h, 1))

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / hull_area) if hull_area > 1e-9 else 0.0

    circularity = float((4.0 * math.pi * area) / max(perimeter * perimeter, 1e-9))
    elongation = float(major_axis_length / max(minor_axis_length, 1e-6))

    points = contour.reshape(-1, 2).astype(np.float32)
    contour_lowest_point = tuple(points[np.argmax(points[:, 1])].tolist())
    contour_highest_point = tuple(points[np.argmin(points[:, 1])].tolist())
    contour_leftmost_point = tuple(points[np.argmin(points[:, 0])].tolist())
    contour_rightmost_point = tuple(points[np.argmax(points[:, 0])].tolist())

    major_axis_endpoint_a, major_axis_endpoint_b = _compute_major_axis_endpoints(points)

    height, width = image_shape
    touches_image_border = (
        x <= 0 or y <= 0 or (x + w) >= (width - 1) or (y + h) >= (height - 1)
    )

    metrics = ContourMetrics(
        area=area,
        perimeter=perimeter,
        bbox=(int(x), int(y), int(w), int(h)),
        centroid=(float(centroid[0]), float(centroid[1])),
        min_area_rect_center=(float(rcx), float(rcy)),
        min_area_rect_size=(float(rw), float(rh)),
        min_area_rect_angle=float(angle),
        aspect_ratio=aspect_ratio,
        extent=extent,
        solidity=solidity,
        circularity=circularity,
        major_axis_length=major_axis_length,
        minor_axis_length=minor_axis_length,
        elongation=elongation,
        contour_lowest_point=(float(contour_lowest_point[0]), float(contour_lowest_point[1])),
        contour_highest_point=(float(contour_highest_point[0]), float(contour_highest_point[1])),
        contour_leftmost_point=(float(contour_leftmost_point[0]), float(contour_leftmost_point[1])),
        contour_rightmost_point=(float(contour_rightmost_point[0]), float(contour_rightmost_point[1])),
        major_axis_endpoint_a=major_axis_endpoint_a,
        major_axis_endpoint_b=major_axis_endpoint_b,
        touches_image_border=touches_image_border,
    )

    return metrics


def _compute_major_axis_endpoints(points: np.ndarray) -> tuple[PointF, PointF]:
    """
    Schätzt zwei Endpunkte entlang der Hauptachse der Kontur mittels PCA.

    Warum das wichtig ist:
    Für längliche Konturen ist das oft stabiler als einfach nur
    linkester/rechtester/oberster/unterster Punkt.

    Diese Endpunkte sind noch KEINE finale Dartspitze,
    aber eine gute Heuristik für den nächsten Pipeline-Schritt.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")

    if len(points) == 1:
        p = (float(points[0, 0]), float(points[0, 1]))
        return p, p

    mean = np.mean(points, axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Größter Eigenwert = Hauptachse
    principal_vector = eigenvectors[:, np.argmax(eigenvalues)]
    principal_vector = principal_vector / max(np.linalg.norm(principal_vector), 1e-9)

    projections = centered @ principal_vector
    min_idx = int(np.argmin(projections))
    max_idx = int(np.argmax(projections))

    point_a = points[min_idx]
    point_b = points[max_idx]

    return (
        (float(point_a[0]), float(point_a[1])),
        (float(point_b[0]), float(point_b[1])),
    )


def _dataclass_to_dict(value: Any) -> dict[str, Any]:
    """
    Kleiner Helfer für Debug-Metadaten.
    """
    if hasattr(value, "__dataclass_fields__"):
        result = {}
        for field_name in value.__dataclass_fields__:
            result[field_name] = getattr(value, field_name)
        return result

    raise TypeError(f"Expected dataclass instance, got {type(value)!r}.")


__all__ = [
    "PointF",
    "PointI",
    "BBox",
    "CandidateDetectorConfig",
    "ContourMetrics",
    "DartCandidate",
    "CandidateDetectionResult",
    "DartCandidateDetector",
    "build_candidate_detector",
    "detect_dart_candidates",
]