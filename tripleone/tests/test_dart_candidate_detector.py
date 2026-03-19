# tests/test_dart_candidate_detector.py
# Zweck:
# Diese Tests prüfen bewusst nur die Kandidaten-Detektionsschicht.
#
# Es wird NICHT getestet:
# - finale Board-Geometrie
# - Homography
# - Score-Mapping (S20, D20, DBULL ...)
#
# Es wird getestet:
# - Differenzbildung zwischen Referenzbild und aktuellem Frame
# - Kontur-/Kandidaten-Erkennung
# - Filterlogik (zu klein / zu groß / ROI)
# - Sorting / best_candidate
# - Debugbilder
# - Referenzgrößenbehandlung
# - Impact-Hypothesen-Modi
#
# Die Tests arbeiten bewusst mit synthetischen Bildern, damit die Ergebnisse
# stabil und deterministisch bleiben.

from __future__ import annotations

import cv2
import numpy as np
import pytest

import vision.dart_candidate_detector as dcd


# -----------------------------------------------------------------------------
# Hilfsfunktionen für synthetische Testbilder
# -----------------------------------------------------------------------------

def _make_blank_frame(width: int = 200, height: int = 200) -> np.ndarray:
    """
    Erstellt ein schwarzes BGR-Testbild.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def _draw_filled_rectangle(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Zeichnet ein gefülltes Rechteck in ein Bild.
    """
    result = frame.copy()
    cv2.rectangle(result, top_left, bottom_right, color, thickness=-1)
    return result


def _draw_rotated_rectangle(
    frame: np.ndarray,
    center: tuple[float, float],
    size: tuple[float, float],
    angle_deg: float,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Zeichnet ein gefülltes rotiertes Rechteck in ein Bild.
    """
    result = frame.copy()

    rect = (center, size, angle_deg)
    box = cv2.boxPoints(rect)
    box = np.round(box).astype(np.int32)

    cv2.fillPoly(result, [box], color)
    return result


def _draw_filled_polygon(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """
    Zeichnet ein gefülltes Polygon in ein Bild.
    """
    result = frame.copy()
    polygon = np.asarray(points, dtype=np.int32)
    cv2.fillPoly(result, [polygon], color)
    return result


def _base_config(**overrides) -> dcd.CandidateDetectorConfig:
    """
    Liefert eine bewusst deterministische Testkonfiguration.

    Wichtige Entscheidung:
    Für Tests reduzieren wir Morphologie und Weichzeichnung stark, damit
    die künstlichen Formen möglichst direkt als Konturen erkannt werden.
    """
    config = dcd.CandidateDetectorConfig(
        blur_kernel_size=1,
        apply_clahe=False,
        diff_threshold=10,
        use_otsu_threshold=False,
        open_kernel_size=1,
        close_kernel_size=1,
        dilate_kernel_size=1,
        erode_iterations=0,
        dilate_iterations=0,
        min_contour_area=20.0,
        max_contour_area_ratio=0.25,
        min_aspect_ratio=1.2,
        min_solidity=0.05,
        min_extent=0.01,
        max_extent=1.00,
        min_confidence=0.05,
        max_candidates=8,
        impact_point_mode="major_axis_lower_endpoint",
        allow_reference_resize=False,
        keep_debug_images=True,
        draw_candidate_ids=True,
        draw_confidence=True,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


# -----------------------------------------------------------------------------
# Grundtests: leeres Bild / einfacher Kandidat
# -----------------------------------------------------------------------------

def test_detect_candidates_returns_no_candidates_for_identical_frames():
    """
    Wenn Referenzbild und aktuelles Bild identisch sind, dürfen keine Kandidaten
    gefunden werden.
    """
    reference = _make_blank_frame()
    frame = reference.copy()

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert isinstance(result, dcd.CandidateDetectionResult)
    assert result.candidates == []
    assert result.best_candidate is None
    assert result.reference_used is True
    assert result.board_mask_used is False
    assert result.board_polygon_used is False

    # Debugbilder sollen vorhanden sein, weil keep_debug_images=True
    assert "difference" in result.debug_images
    assert "binary_mask" in result.debug_images
    assert "cleaned_mask" in result.debug_images

    assert result.metadata["accepted_candidates"] == 0
    assert result.metadata["total_contours"] == 0


def test_detect_candidates_finds_single_elongated_candidate():
    """
    Ein klarer länglicher Unterschied zwischen Referenzbild und aktuellem Bild
    soll als plausibler Kandidat erkannt werden.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(100, 100),
        size=(12, 72),
        angle_deg=18,
    )

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert len(result.candidates) >= 1
    candidate = result.best_candidate
    assert candidate is not None

    assert candidate.area > 20.0
    assert candidate.aspect_ratio >= 1.2
    assert candidate.confidence >= 0.05

    # Der Impact-Punkt soll beim "lower endpoint"-Modus tendenziell unterhalb
    # oder auf Höhe des Schwerpunkts liegen.
    assert candidate.impact_point[1] >= candidate.centroid[1]

    # Das Rendern eines Debug-Overlays soll ein Bild gleicher Größe liefern.
    overlay = result.render_debug_overlay(frame)
    assert overlay.shape == frame.shape
    assert overlay.ndim == 3


# -----------------------------------------------------------------------------
# Filtertests: zu klein / zu groß
# -----------------------------------------------------------------------------

def test_small_blob_is_filtered_by_min_contour_area():
    """
    Sehr kleine Unterschiede dürfen nicht als Dart-Kandidat durchgehen.
    """
    reference = _make_blank_frame()
    frame = _draw_filled_rectangle(reference, (50, 50), (52, 52))

    detector = dcd.DartCandidateDetector(
        config=_base_config(min_contour_area=50.0)
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert result.candidates == []
    assert result.best_candidate is None
    assert result.metadata["accepted_candidates"] == 0
    assert result.metadata["total_contours"] >= 1


def test_large_blob_is_filtered_by_max_contour_area_ratio():
    """
    Sehr große Differenzflächen sollen als unplausibel gefiltert werden,
    weil sie eher globale Bewegung / falsche ROI / falsches Referenzbild
    darstellen als einen Dartkandidaten.
    """
    reference = _make_blank_frame()
    frame = _draw_filled_rectangle(reference, (10, 20), (190, 180))

    detector = dcd.DartCandidateDetector(
        config=_base_config(
            max_contour_area_ratio=0.02,
            min_confidence=0.01,
        )
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert result.candidates == []
    assert result.best_candidate is None
    assert result.metadata["accepted_candidates"] == 0
    assert result.metadata["total_contours"] >= 1


# -----------------------------------------------------------------------------
# ROI-/Masken-Tests
# -----------------------------------------------------------------------------

def test_board_mask_excludes_candidate_outside_roi():
    """
    Eine Board-Maske muss Kandidaten außerhalb der ROI zuverlässig unterdrücken.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(45, 100),
        size=(12, 72),
        angle_deg=10,
    )

    # ROI nur auf der rechten Bildhälfte -> linker Kandidat muss verschwinden
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:, 100:] = 255

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(
        frame=frame,
        reference_frame=reference,
        board_mask=mask,
    )

    assert result.board_mask_used is True
    assert result.board_polygon_used is False
    assert result.candidates == []
    assert result.best_candidate is None


def test_board_mask_allows_candidate_inside_roi():
    """
    Eine Board-Maske soll Kandidaten innerhalb der ROI normal zulassen.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(150, 100),
        size=(12, 72),
        angle_deg=10,
    )

    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[:, 100:] = 255

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(
        frame=frame,
        reference_frame=reference,
        board_mask=mask,
    )

    assert result.board_mask_used is True
    assert len(result.candidates) >= 1
    assert result.best_candidate is not None


def test_board_polygon_is_supported_and_creates_internal_mask():
    """
    Statt einer Board-Maske darf auch ein Polygon übergeben werden.
    Daraus muss intern eine Maske erzeugt werden.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(145, 100),
        size=(12, 72),
        angle_deg=15,
    )

    polygon = [
        (100, 10),
        (190, 10),
        (190, 190),
        (100, 190),
    ]

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(
        frame=frame,
        reference_frame=reference,
        board_polygon=polygon,
    )

    assert result.board_polygon_used is True
    assert result.board_mask_used is True
    assert len(result.candidates) >= 1
    assert result.best_candidate is not None


# -----------------------------------------------------------------------------
# Referenzgrößen-Tests
# -----------------------------------------------------------------------------

def test_reference_size_mismatch_raises_without_resize():
    """
    Unterschiedliche Größen von Frame und Referenzbild müssen standardmäßig
    einen Fehler werfen.
    """
    frame = _make_blank_frame(width=200, height=200)
    reference = _make_blank_frame(width=160, height=160)

    detector = dcd.DartCandidateDetector(
        config=_base_config(allow_reference_resize=False)
    )

    with pytest.raises(ValueError):
        detector.detect_candidates(frame=frame, reference_frame=reference)


def test_reference_size_mismatch_can_be_resized_if_enabled():
    """
    Wenn allow_reference_resize=True gesetzt ist, darf das Referenzbild
    automatisch auf Frame-Größe gebracht werden.
    """
    frame = _make_blank_frame(width=200, height=200)
    reference = _make_blank_frame(width=160, height=160)
    frame = _draw_rotated_rectangle(
        frame,
        center=(100, 100),
        size=(12, 72),
        angle_deg=20,
    )

    detector = dcd.DartCandidateDetector(
        config=_base_config(allow_reference_resize=True)
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert result.metadata["reference_resized"] is True
    assert len(result.candidates) >= 1
    assert result.best_candidate is not None


# -----------------------------------------------------------------------------
# Sorting- / Ranking-Tests
# -----------------------------------------------------------------------------

def test_candidates_are_sorted_by_confidence_and_best_candidate_matches():
    """
    Wenn zwei Kandidaten vorhanden sind, soll der plausiblere Kandidat
    vorne stehen und best_candidate darauf zeigen.
    """
    reference = _make_blank_frame()

    # Kandidat A: deutlich länglicher
    frame = _draw_rotated_rectangle(
        reference,
        center=(60, 100),
        size=(10, 78),
        angle_deg=12,
    )

    # Kandidat B: kompakter / weniger dart-ähnlich
    frame = _draw_rotated_rectangle(
        frame,
        center=(145, 100),
        size=(18, 34),
        angle_deg=0,
    )

    detector = dcd.DartCandidateDetector(
        config=_base_config(
            min_confidence=0.01,
            min_aspect_ratio=1.1,
            max_candidates=8,
        )
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert len(result.candidates) >= 2
    assert result.best_candidate is result.candidates[0]
    assert result.candidates[0].confidence >= result.candidates[1].confidence

    # Der beste Kandidat sollte in dieser Konstruktion der länglichere linke sein.
    best = result.best_candidate
    assert best is not None
    assert best.aspect_ratio >= result.candidates[1].aspect_ratio
    assert best.centroid[0] < 100.0


# -----------------------------------------------------------------------------
# Debugbild-Tests
# -----------------------------------------------------------------------------

def test_keep_debug_images_false_returns_empty_debug_images():
    """
    Wenn keep_debug_images=False gesetzt ist, sollen keine Debugbilder
    zurückgegeben werden.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(100, 100),
        size=(12, 72),
        angle_deg=20,
    )

    detector = dcd.DartCandidateDetector(
        config=_base_config(keep_debug_images=False)
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert len(result.candidates) >= 1
    assert result.debug_images == {}


# -----------------------------------------------------------------------------
# Impact-Hypothesen-Tests
# -----------------------------------------------------------------------------

def test_impact_point_mode_lowest_contour_point_uses_lowest_y_from_debug():
    """
    Im Modus 'lowest_contour_point' muss der zurückgegebene Impact-Punkt
    dem in den Debugdaten gespeicherten tiefsten Konturpunkt entsprechen.
    """
    reference = _make_blank_frame()
    frame = _draw_filled_polygon(
        reference,
        points=[(95, 35), (105, 35), (120, 150), (80, 150)],
    )

    detector = dcd.DartCandidateDetector(
        config=_base_config(
            impact_point_mode="lowest_contour_point",
            min_aspect_ratio=1.0,
        )
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert len(result.candidates) >= 1
    candidate = result.best_candidate
    assert candidate is not None

    lowest = candidate.debug["contour_lowest_point"]
    assert candidate.impact_point == pytest.approx(lowest, abs=1e-6)


def test_impact_point_mode_major_axis_lower_endpoint_uses_lower_major_axis_endpoint():
    """
    Im Modus 'major_axis_lower_endpoint' muss der Impact-Punkt dem tieferen
    der beiden PCA-Hauptachsen-Endpunkte entsprechen.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(100, 100),
        size=(12, 80),
        angle_deg=25,
    )

    detector = dcd.DartCandidateDetector(
        config=_base_config(
            impact_point_mode="major_axis_lower_endpoint",
        )
    )
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    assert len(result.candidates) >= 1
    candidate = result.best_candidate
    assert candidate is not None

    a = candidate.debug["major_axis_endpoint_a"]
    b = candidate.debug["major_axis_endpoint_b"]

    expected = a if a[1] >= b[1] else b
    assert candidate.impact_point == pytest.approx(expected, abs=1e-6)


# -----------------------------------------------------------------------------
# Convenience-Wrapper-Tests
# -----------------------------------------------------------------------------

def test_detect_dart_candidates_convenience_wrapper():
    """
    Der modulweite Convenience-Wrapper soll dasselbe Grundverhalten liefern
    wie die Klassen-API.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(100, 100),
        size=(12, 72),
        angle_deg=18,
    )

    result = dcd.detect_dart_candidates(
        frame=frame,
        reference_frame=reference,
        config=_base_config(),
    )

    assert isinstance(result, dcd.CandidateDetectionResult)
    assert len(result.candidates) >= 1
    assert result.best_candidate is not None


# -----------------------------------------------------------------------------
# Ergebnis-Serialisierung
# -----------------------------------------------------------------------------

def test_candidate_and_result_to_dict():
    """
    Serialisierung von Kandidaten und Gesamtergebnis soll sauber funktionieren,
    ohne dass Debugbilder in result.to_dict() landen.
    """
    reference = _make_blank_frame()
    frame = _draw_rotated_rectangle(
        reference,
        center=(100, 100),
        size=(12, 72),
        angle_deg=18,
    )

    detector = dcd.DartCandidateDetector(config=_base_config())
    result = detector.detect_candidates(frame=frame, reference_frame=reference)

    candidate = result.best_candidate
    assert candidate is not None

    candidate_dict = candidate.to_dict(include_contour=False)
    result_dict = result.to_dict(include_contours=False)

    assert "candidate_id" in candidate_dict
    assert "bbox" in candidate_dict
    assert "impact_point" in candidate_dict
    assert "confidence" in candidate_dict

    assert "candidates" in result_dict
    assert "metadata" in result_dict
    assert "reference_used" in result_dict
    assert "debug_images" not in result_dict