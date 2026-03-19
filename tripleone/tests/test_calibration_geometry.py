# tests/test_calibration_geometry.py
# Triple One - Schritt 5:
# Tests für die zentrale Board-Geometrie.
#
# Diese Tests prüfen:
# - 4-Punkt-Homography
# - Bull-Berechnung
# - Bild <-> Top-Down Projektion
# - Score-Mapping
# - Konsistenz der Geometrie
#
# Ausführen mit:
#   pytest tests/test_calibration_geometry.py -v

from __future__ import annotations

import math

import numpy as np

from vision.calibration_geometry import (
    TOPDOWN_CENTER_X,
    TOPDOWN_CENTER_Y,
    OUTER_DOUBLE_RADIUS_PX,
    build_pipeline_points,
    calculate_hit_from_image_point,
    calculate_hit_from_topdown_point,
    compute_bull_from_manual_points,
    compute_homography_image_to_topdown,
    compute_homography_topdown_to_image,
    get_destination_points_topdown,
    image_to_topdown_point,
    project_image_points_to_topdown,
    project_topdown_points_to_image,
    topdown_to_image_point,
)


# ------------------------------------------------------------
# Test-Helfer
# ------------------------------------------------------------

def make_identity_like_manual_points():
    """
    Nimmt genau die Top-Down-Zielpunkte als manuelle Bildpunkte.
    Dadurch wird die Homography praktisch zur Identität.
    """
    dst = get_destination_points_topdown()
    return [
        {"x_px": int(round(dst[0, 0])), "y_px": int(round(dst[0, 1]))},
        {"x_px": int(round(dst[1, 0])), "y_px": int(round(dst[1, 1]))},
        {"x_px": int(round(dst[2, 0])), "y_px": int(round(dst[2, 1]))},
        {"x_px": int(round(dst[3, 0])), "y_px": int(round(dst[3, 1]))},
    ]


def topdown_point_for_segment(
    segment_index: int,
    radius_rel: float,
) -> tuple[float, float]:
    """
    Erzeugt einen Punkt im Zentrum eines Segments
    auf einem gewünschten relativen Radius.

    segment_index:
        0 = 20
        1 = 1
        2 = 18
        ...
    """
    angle_deg = 90.0 - segment_index * 18.0
    angle_rad = math.radians(angle_deg)

    r = OUTER_DOUBLE_RADIUS_PX * radius_rel
    x = TOPDOWN_CENTER_X + math.cos(angle_rad) * r
    y = TOPDOWN_CENTER_Y - math.sin(angle_rad) * r
    return x, y


def approx_equal(a: float, b: float, tol: float = 1.0) -> bool:
    return abs(a - b) <= tol


# ------------------------------------------------------------
# Homography / Projektion
# ------------------------------------------------------------

def test_compute_homography_image_to_topdown_exists():
    points = make_identity_like_manual_points()
    h = compute_homography_image_to_topdown(points)
    assert h is not None
    assert h.shape == (3, 3)


def test_compute_homography_topdown_to_image_exists():
    points = make_identity_like_manual_points()
    h = compute_homography_topdown_to_image(points)
    assert h is not None
    assert h.shape == (3, 3)


def test_identity_like_projection_image_to_topdown():
    points = make_identity_like_manual_points()

    src = np.array(
        [
            [TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y],
            [TOPDOWN_CENTER_X + 50.0, TOPDOWN_CENTER_Y],
            [TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y + 70.0],
        ],
        dtype=np.float32,
    )

    mapped = project_image_points_to_topdown(points, src)
    assert mapped is not None
    assert mapped.shape == (3, 2)

    for i in range(3):
        assert approx_equal(mapped[i, 0], src[i, 0], tol=1.5)
        assert approx_equal(mapped[i, 1], src[i, 1], tol=1.5)


def test_identity_like_projection_topdown_to_image():
    points = make_identity_like_manual_points()

    src = np.array(
        [
            [TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y],
            [TOPDOWN_CENTER_X - 80.0, TOPDOWN_CENTER_Y + 15.0],
            [TOPDOWN_CENTER_X + 30.0, TOPDOWN_CENTER_Y - 120.0],
        ],
        dtype=np.float32,
    )

    mapped = project_topdown_points_to_image(points, src)
    assert mapped is not None
    assert mapped.shape == (3, 2)

    for i in range(3):
        assert approx_equal(mapped[i, 0], src[i, 0], tol=1.5)
        assert approx_equal(mapped[i, 1], src[i, 1], tol=1.5)


def test_roundtrip_image_to_topdown_to_image():
    points = make_identity_like_manual_points()

    img_x = TOPDOWN_CENTER_X + 87.0
    img_y = TOPDOWN_CENTER_Y - 33.0

    top_pt = image_to_topdown_point(points, img_x, img_y)
    assert top_pt is not None

    img_back = topdown_to_image_point(points, top_pt[0], top_pt[1])
    assert img_back is not None

    assert approx_equal(img_back[0], img_x, tol=1.5)
    assert approx_equal(img_back[1], img_y, tol=1.5)


# ------------------------------------------------------------
# Bull / Pipeline
# ------------------------------------------------------------

def test_compute_bull_from_manual_points_identity_like():
    points = make_identity_like_manual_points()
    bull = compute_bull_from_manual_points(points)

    assert approx_equal(bull["x_px"], TOPDOWN_CENTER_X, tol=2.0)
    assert approx_equal(bull["y_px"], TOPDOWN_CENTER_Y, tol=2.0)


def test_build_pipeline_points_adds_bull():
    points = make_identity_like_manual_points()
    pipeline_points = build_pipeline_points(points)

    assert isinstance(pipeline_points, list)
    assert len(pipeline_points) == 5

    bull = pipeline_points[4]
    assert "x_px" in bull
    assert "y_px" in bull


# ------------------------------------------------------------
# Top-Down Score Mapping
# ------------------------------------------------------------

def test_topdown_dbull():
    hit = calculate_hit_from_topdown_point(TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y)
    assert hit.ring_name == "DBULL"
    assert hit.score == 50
    assert hit.label == "DBULL"


def test_topdown_sbull():
    # kleiner Radius knapp außerhalb inner bull, aber innerhalb outer bull
    x = TOPDOWN_CENTER_X + OUTER_DOUBLE_RADIUS_PX * 0.06
    y = TOPDOWN_CENTER_Y
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "SBULL"
    assert hit.score == 25
    assert hit.label == "SBULL"


def test_topdown_single_20():
    # Segmentindex 0 = 20
    x, y = topdown_point_for_segment(segment_index=0, radius_rel=0.40)
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "SINGLE"
    assert hit.score == 20
    assert hit.label == "S20"


def test_topdown_triple_20():
    x, y = topdown_point_for_segment(segment_index=0, radius_rel=0.61)
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "TRIPLE"
    assert hit.score == 60
    assert hit.label == "T20"


def test_topdown_double_20():
    x, y = topdown_point_for_segment(segment_index=0, radius_rel=0.97)
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "DOUBLE"
    assert hit.score == 40
    assert hit.label == "D20"


def test_topdown_single_19():
    # Segmentindex 11 = 19
    x, y = topdown_point_for_segment(segment_index=11, radius_rel=0.40)
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "SINGLE"
    assert hit.score == 19
    assert hit.label == "S19"


def test_topdown_double_2():
    # Segmentindex 8 = 2
    x, y = topdown_point_for_segment(segment_index=8, radius_rel=0.97)
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "DOUBLE"
    assert hit.score == 4
    assert hit.label == "D2"


def test_topdown_miss():
    x = TOPDOWN_CENTER_X + OUTER_DOUBLE_RADIUS_PX * 1.10
    y = TOPDOWN_CENTER_Y
    hit = calculate_hit_from_topdown_point(x, y)

    assert hit.ring_name == "MISS"
    assert hit.score == 0
    assert hit.label == "MISS"


# ------------------------------------------------------------
# Bildpunkt-Scoring über dieselbe Geometrie
# ------------------------------------------------------------

def test_image_point_scoring_matches_topdown_single_20():
    points = make_identity_like_manual_points()

    x, y = topdown_point_for_segment(segment_index=0, radius_rel=0.40)
    hit = calculate_hit_from_image_point(x, y, points)

    assert hit is not None
    assert hit.ring_name == "SINGLE"
    assert hit.score == 20
    assert hit.label == "S20"


def test_image_point_scoring_matches_topdown_double_2():
    points = make_identity_like_manual_points()

    x, y = topdown_point_for_segment(segment_index=8, radius_rel=0.97)
    hit = calculate_hit_from_image_point(x, y, points)

    assert hit is not None
    assert hit.ring_name == "DOUBLE"
    assert hit.score == 4
    assert hit.label == "D2"


def test_image_point_scoring_matches_topdown_single_19():
    points = make_identity_like_manual_points()

    x, y = topdown_point_for_segment(segment_index=11, radius_rel=0.40)
    hit = calculate_hit_from_image_point(x, y, points)

    assert hit is not None
    assert hit.ring_name == "SINGLE"
    assert hit.score == 19
    assert hit.label == "S19"


def test_image_point_miss():
    points = make_identity_like_manual_points()

    x = TOPDOWN_CENTER_X + OUTER_DOUBLE_RADIUS_PX * 1.10
    y = TOPDOWN_CENTER_Y
    hit = calculate_hit_from_image_point(x, y, points)

    assert hit is not None
    assert hit.ring_name == "MISS"
    assert hit.score == 0
    assert hit.label == "MISS"


# ------------------------------------------------------------
# Zusätzliche Konsistenztests
# ------------------------------------------------------------

def test_bull_pipeline_consistency():
    points = make_identity_like_manual_points()
    pipeline_points = build_pipeline_points(points)

    bull = pipeline_points[4]
    hit = calculate_hit_from_image_point(bull["x_px"], bull["y_px"], pipeline_points)

    assert hit is not None
    assert hit.label == "DBULL"
    assert hit.score == 50


def test_topdown_center_projected_back_is_dbull():
    points = make_identity_like_manual_points()

    img_pt = topdown_to_image_point(points, TOPDOWN_CENTER_X, TOPDOWN_CENTER_Y)
    assert img_pt is not None

    hit = calculate_hit_from_image_point(img_pt[0], img_pt[1], points)
    assert hit is not None
    assert hit.label == "DBULL"


def test_manual_points_can_be_passed_as_full_config_dict():
    points = make_identity_like_manual_points()
    config = {
        "frame_width": 1280,
        "frame_height": 720,
        "points": build_pipeline_points(points),
    }

    x, y = topdown_point_for_segment(segment_index=0, radius_rel=0.40)
    hit = calculate_hit_from_image_point(x, y, config)

    assert hit is not None
    assert hit.label == "S20"