# tests/test_impact_estimator.py
# Zweck:
# Diese Tests prüfen ausschließlich die Impact-Schicht.
#
# Es wird bewusst NICHT getestet:
# - Homography / Board-Geometrie
# - Score-Mapping
# - Spiellogik
#
# Es wird getestet:
# - Hypothesenbildung aus DartCandidate
# - finale Impact-Punktwahl
# - Strategien wie candidate_default, lowest_contour_point,
#   major_axis_lower_endpoint, directional_contour_tip, blend
# - Sortierung / best_estimate
# - Clamping an Bildgrenzen
# - Wrapper-Funktionen
# - Debug-/Serialisierung
#
# Die Tests arbeiten mit synthetischen Kandidaten und Konturen, damit die
# Ergebnisse deterministisch bleiben.

from __future__ import annotations

import cv2
import numpy as np
import pytest

import vision.dart_candidate_detector as dcd
import vision.impact_estimator as ie


# -----------------------------------------------------------------------------
# Hilfsfunktionen für synthetische Konturen und Kandidaten
# -----------------------------------------------------------------------------

def _make_contour(points: list[tuple[int, int]]) -> np.ndarray:
    """
    Erstellt eine OpenCV-Kontur aus einer Liste von 2D-Punkten.
    """
    return np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)


def _make_tapered_vertical_contour(
    center_x: int = 100,
    top_y: int = 40,
    bottom_y: int = 180,
) -> np.ndarray:
    """
    Erzeugt eine längliche, nach unten zulaufende Kontur.

    Diese Form eignet sich gut für:
    - directional_contour_tip
    - lowest_contour_point
    - major_axis-basierte Methoden
    """
    mid_y = int((top_y + bottom_y) / 2)

    points = [
        (center_x, top_y),
        (center_x + 8, top_y + 20),
        (center_x + 4, mid_y),
        (center_x, bottom_y),
        (center_x - 4, mid_y),
        (center_x - 8, top_y + 20),
    ]
    return _make_contour(points)


def _contour_centroid(contour: np.ndarray) -> tuple[float, float]:
    """
    Berechnet den Schwerpunkt einer Kontur.
    """
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-9:
        return (
            float(moments["m10"] / moments["m00"]),
            float(moments["m01"] / moments["m00"]),
        )

    x, y, w, h = cv2.boundingRect(contour)
    return float(x + w / 2.0), float(y + h / 2.0)


def _bbox_from_contour(contour: np.ndarray) -> tuple[int, int, int, int]:
    """
    Berechnet die Bounding-Box einer Kontur.
    """
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(w), int(h)


def _build_candidate(
    *,
    candidate_id: int = 0,
    contour: np.ndarray | None = None,
    impact_point: tuple[float, float] = (100.0, 175.0),
    confidence: float = 0.80,
    aspect_ratio: float = 4.0,
    major_axis_length: float = 120.0,
    minor_axis_length: float = 20.0,
    elongation: float | None = None,
    debug: dict | None = None,
) -> dcd.DartCandidate:
    """
    Baut einen synthetischen DartCandidate mit plausiblen Werten.
    """
    if contour is None:
        contour = _make_tapered_vertical_contour()

    if elongation is None:
        elongation = float(major_axis_length / max(minor_axis_length, 1e-6))

    bbox = _bbox_from_contour(contour)
    centroid = _contour_centroid(contour)
    area = float(max(cv2.contourArea(contour), 1.0))

    return dcd.DartCandidate(
        candidate_id=candidate_id,
        bbox=bbox,
        centroid=centroid,
        impact_point=(float(impact_point[0]), float(impact_point[1])),
        area=area,
        aspect_ratio=float(aspect_ratio),
        solidity=0.75,
        extent=0.35,
        circularity=0.25,
        angle_degrees=0.0,
        major_axis_length=float(major_axis_length),
        minor_axis_length=float(minor_axis_length),
        elongation=float(elongation),
        confidence=float(confidence),
        contour=contour.copy(),
        debug=debug.copy() if debug else {},
    )


def _base_config(**overrides) -> ie.ImpactEstimatorConfig:
    """
    Liefert eine stabile Testkonfiguration.
    """
    config = ie.ImpactEstimatorConfig(
        strategy="blend",
        use_candidate_default=True,
        use_lowest_contour_point=True,
        use_major_axis_lower_endpoint=True,
        use_directional_contour_tip=True,
        weight_candidate_default=0.70,
        weight_lowest_contour_point=0.90,
        weight_major_axis_lower_endpoint=1.00,
        weight_directional_contour_tip=1.10,
        min_major_axis_length_for_axis_based_methods=10.0,
        min_aspect_ratio_for_axis_based_methods=1.15,
        min_candidate_confidence=0.01,
        directional_tip_band_fraction=0.15,
        directional_tip_top_k_points=8,
        consistency_distance_scale_px=12.0,
        weight_source_candidate_confidence=0.60,
        weight_hypothesis_strength=0.40,
        clamp_to_image_bounds=True,
        keep_debug_metadata=True,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


# -----------------------------------------------------------------------------
# Grundtests
# -----------------------------------------------------------------------------

def test_estimate_for_candidate_returns_none_below_min_candidate_confidence():
    """
    Kandidaten unterhalb der Mindestkonfidenz dürfen nicht weiterverarbeitet
    werden.
    """
    candidate = _build_candidate(confidence=0.02)

    estimator = ie.ImpactEstimator(
        config=_base_config(min_candidate_confidence=0.10)
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is None


def test_candidate_default_strategy_returns_candidate_impact_point_exactly():
    """
    Im Modus 'candidate_default' muss exakt der im Kandidaten vorhandene
    Impact-Punkt übernommen werden.
    """
    candidate = _build_candidate(
        impact_point=(111.0, 177.0),
        confidence=0.85,
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method == "candidate_default"
    assert estimate.impact_point == pytest.approx((111.0, 177.0), abs=1e-6)
    assert estimate.hypothesis_count == 1
    assert estimate.confidence > 0.0


def test_lowest_contour_point_strategy_uses_debug_lowest_point_exactly():
    """
    Wenn der tiefste Konturpunkt in den Debugdaten vorhanden ist und die
    Strategie darauf gesetzt wird, muss genau dieser Punkt gewählt werden.
    """
    candidate = _build_candidate(
        impact_point=(100.0, 170.0),
        debug={
            "contour_lowest_point": (103.0, 182.0),
        },
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="lowest_contour_point",
            use_candidate_default=False,
            use_lowest_contour_point=True,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method == "lowest_contour_point"
    assert estimate.impact_point == pytest.approx((103.0, 182.0), abs=1e-6)
    assert estimate.hypothesis_count == 1


def test_major_axis_lower_endpoint_strategy_uses_lower_debug_endpoint():
    """
    Wenn zwei Major-Axis-Endpunkte vorliegen, muss der tiefere Endpunkt gewählt
    werden.
    """
    candidate = _build_candidate(
        debug={
            "major_axis_endpoint_a": (98.0, 55.0),
            "major_axis_endpoint_b": (101.0, 176.0),
        }
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="major_axis_lower_endpoint",
            use_candidate_default=False,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=True,
            use_directional_contour_tip=False,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method == "major_axis_lower_endpoint"
    assert estimate.impact_point == pytest.approx((101.0, 176.0), abs=1e-6)
    assert estimate.hypothesis_count == 1


# -----------------------------------------------------------------------------
# Directional-Tip-Tests
# -----------------------------------------------------------------------------

def test_directional_contour_tip_strategy_returns_bottom_tip_for_tapered_shape():
    """
    Mit einer länglichen, nach unten zulaufenden Kontur soll die Richtungsspitze
    nahe der unteren Konturspitze liegen.

    Für maximale Deterministik wird directional_tip_top_k_points=1 gesetzt,
    damit exakt der stärkste Punkt gewählt wird.
    """
    contour = _make_tapered_vertical_contour(center_x=100, top_y=40, bottom_y=180)

    candidate = _build_candidate(
        contour=contour,
        impact_point=(100.0, 170.0),
        confidence=0.90,
        aspect_ratio=5.5,
        major_axis_length=140.0,
        minor_axis_length=18.0,
        debug={
            "major_axis_endpoint_a": (100.0, 40.0),
            "major_axis_endpoint_b": (100.0, 180.0),
        },
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="directional_contour_tip",
            use_candidate_default=False,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=True,
            directional_tip_top_k_points=1,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method == "directional_contour_tip"
    assert estimate.hypothesis_count == 1

    # Untere Spitze der künstlichen Kontur ist exakt (100, 180)
    assert estimate.impact_point == pytest.approx((100.0, 180.0), abs=1e-6)


# -----------------------------------------------------------------------------
# Blend-Strategie
# -----------------------------------------------------------------------------

def test_blend_strategy_combines_multiple_hypotheses_into_reasonable_point():
    """
    Im Blend-Modus muss der finale Punkt innerhalb des Bereichs der einzelnen
    Hypothesen liegen und mehrere Hypothesen berücksichtigen.
    """
    contour = _make_tapered_vertical_contour(center_x=100, top_y=40, bottom_y=180)

    candidate = _build_candidate(
        contour=contour,
        impact_point=(100.0, 178.0),
        confidence=0.90,
        aspect_ratio=5.0,
        major_axis_length=135.0,
        minor_axis_length=18.0,
        debug={
            "contour_lowest_point": (101.0, 180.0),
            "major_axis_endpoint_a": (99.0, 42.0),
            "major_axis_endpoint_b": (100.0, 179.0),
        },
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="blend",
            directional_tip_top_k_points=1,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method == "blend"
    assert estimate.hypothesis_count >= 3
    assert estimate.confidence > 0.0

    xs = [hyp.point[0] for hyp in estimate.hypotheses]
    ys = [hyp.point[1] for hyp in estimate.hypotheses]

    assert min(xs) <= estimate.impact_point[0] <= max(xs)
    assert min(ys) <= estimate.impact_point[1] <= max(ys)

    # Bei dieser Konstruktion muss der Blend im unteren Bereich der Kontur landen.
    assert 175.0 <= estimate.impact_point[1] <= 180.0


# -----------------------------------------------------------------------------
# Sortierung / Sammelverarbeitung
# -----------------------------------------------------------------------------

def test_estimate_for_candidates_sorts_by_final_confidence():
    """
    Mehrere Estimates müssen nach finaler Konfidenz absteigend sortiert werden.
    """
    candidate_low = _build_candidate(
        candidate_id=1,
        impact_point=(90.0, 170.0),
        confidence=0.30,
    )
    candidate_high = _build_candidate(
        candidate_id=2,
        impact_point=(110.0, 175.0),
        confidence=0.90,
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        )
    )
    result = estimator.estimate_for_candidates([candidate_low, candidate_high])

    assert isinstance(result, ie.ImpactEstimationResult)
    assert len(result.estimates) == 2
    assert result.best_estimate is result.estimates[0]

    assert result.estimates[0].candidate_id == 2
    assert result.estimates[1].candidate_id == 1
    assert result.estimates[0].confidence >= result.estimates[1].confidence


def test_estimate_from_detection_result_uses_metadata_input_shape_for_clamping():
    """
    Wenn kein image_shape direkt übergeben wird, soll estimate_from_detection_result
    die Bildgröße aus detection_result.metadata['input_shape'] ableiten und den
    finalen Punkt bei Bedarf clampen.
    """
    candidate = _build_candidate(
        candidate_id=7,
        impact_point=(250.0, 300.0),
        confidence=0.80,
    )

    detection_result = dcd.CandidateDetectionResult(
        candidates=[candidate],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={
            "input_shape": (200, 100, 3),  # height=200, width=100
        },
        debug_images={},
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
            clamp_to_image_bounds=True,
        )
    )
    result = estimator.estimate_from_detection_result(detection_result)

    assert len(result.estimates) == 1
    estimate = result.best_estimate
    assert estimate is not None

    # width=100 -> x max = 99
    # height=200 -> y max = 199
    assert estimate.impact_point == pytest.approx((99.0, 199.0), abs=1e-6)


# -----------------------------------------------------------------------------
# best_hypothesis-Strategie
# -----------------------------------------------------------------------------

def test_best_hypothesis_strategy_prefers_strongest_single_hypothesis():
    """
    Die Strategie 'best_hypothesis' soll die stärkste Einzelhypothese wählen.
    In dieser Konfiguration ist directional_contour_tip am stärksten gewichtet.
    """
    contour = _make_tapered_vertical_contour(center_x=100, top_y=40, bottom_y=180)

    candidate = _build_candidate(
        contour=contour,
        impact_point=(100.0, 170.0),
        confidence=0.90,
        aspect_ratio=5.5,
        major_axis_length=140.0,
        minor_axis_length=18.0,
        debug={
            "contour_lowest_point": (101.0, 180.0),
            "major_axis_endpoint_a": (100.0, 40.0),
            "major_axis_endpoint_b": (100.0, 179.0),
        },
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="best_hypothesis",
            directional_tip_top_k_points=1,
        )
    )
    estimate = estimator.estimate_for_candidate(candidate)

    assert estimate is not None
    assert estimate.method in {
        "candidate_default",
        "lowest_contour_point",
        "major_axis_lower_endpoint",
        "directional_contour_tip",
    }

    # Bei dieser Konfiguration und diesem Kandidaten soll typischerweise
    # directional_contour_tip gewinnen.
    assert estimate.method == "directional_contour_tip"
    assert estimate.impact_point == pytest.approx((100.0, 180.0), abs=1e-6)


# -----------------------------------------------------------------------------
# Overlay / Serialisierung
# -----------------------------------------------------------------------------

def test_render_debug_overlay_returns_same_image_shape():
    """
    Das Debug-Overlay muss ein BGR-Bild mit gleicher Größe zurückgeben.
    """
    candidate = _build_candidate(
        candidate_id=5,
        impact_point=(105.0, 175.0),
        confidence=0.85,
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        )
    )
    result = estimator.estimate_for_candidates([candidate])

    canvas = np.zeros((220, 180, 3), dtype=np.uint8)
    overlay = result.render_debug_overlay(canvas)

    assert overlay.shape == canvas.shape
    assert overlay.ndim == 3


def test_estimate_and_result_to_dict_contain_expected_keys():
    """
    Serialisierung von ImpactEstimate und ImpactEstimationResult soll die
    erwarteten Felder liefern.
    """
    candidate = _build_candidate(
        candidate_id=9,
        impact_point=(111.0, 177.0),
        confidence=0.88,
    )

    estimator = ie.ImpactEstimator(
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        )
    )
    result = estimator.estimate_for_candidates([candidate])

    estimate = result.best_estimate
    assert estimate is not None

    estimate_dict = estimate.to_dict()
    result_dict = result.to_dict()

    assert "candidate_id" in estimate_dict
    assert "impact_point" in estimate_dict
    assert "method" in estimate_dict
    assert "confidence" in estimate_dict
    assert "hypotheses" in estimate_dict
    assert "debug" in estimate_dict

    assert "metadata" in result_dict
    assert "estimates" in result_dict
    assert isinstance(result_dict["estimates"], list)
    assert len(result_dict["estimates"]) == 1


# -----------------------------------------------------------------------------
# Convenience-Wrapper
# -----------------------------------------------------------------------------

def test_module_level_convenience_wrappers_work():
    """
    Die modulweiten Wrapper sollen dieselbe Grundfunktionalität wie die
    Klassen-API liefern.
    """
    candidate_a = _build_candidate(
        candidate_id=11,
        impact_point=(101.0, 171.0),
        confidence=0.55,
    )
    candidate_b = _build_candidate(
        candidate_id=12,
        impact_point=(109.0, 179.0),
        confidence=0.92,
    )

    single = ie.estimate_impact_for_candidate(
        candidate_b,
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        ),
    )
    assert single is not None
    assert single.candidate_id == 12
    assert single.impact_point == pytest.approx((109.0, 179.0), abs=1e-6)

    multi = ie.estimate_impacts(
        [candidate_a, candidate_b],
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        ),
    )
    assert len(multi.estimates) == 2
    assert multi.best_estimate is not None
    assert multi.best_estimate.candidate_id == 12

    detection_result = dcd.CandidateDetectionResult(
        candidates=[candidate_a, candidate_b],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={"input_shape": (240, 320, 3)},
        debug_images={},
    )
    from_detection = ie.estimate_impacts_from_detection_result(
        detection_result,
        config=_base_config(
            strategy="candidate_default",
            use_candidate_default=True,
            use_lowest_contour_point=False,
            use_major_axis_lower_endpoint=False,
            use_directional_contour_tip=False,
        ),
    )
    assert len(from_detection.estimates) == 2
    assert from_detection.best_estimate is not None
    assert from_detection.best_estimate.candidate_id == 12