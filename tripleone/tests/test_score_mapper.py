# tests/test_score_mapper.py
# Zweck:
# Diese Tests prüfen die saubere Scoring-Schicht auf Basis der zentralen
# Geometrie aus calibration_geometry.py.
#
# WICHTIG:
# score_mapper.py enthält bewusst KEINE eigene Geometrie.
# Die Tests prüfen daher:
# - Label-Normalisierung
# - Score-/Ring-/Segment-/Multiplier-Ableitung
# - Aufbau der "Pipeline" aus 4 Markerpunkten
# - Scoring von Bild- und Top-Down-Punkten
# - Rebuild-Verhalten
# - Wrapper-Funktionen
#
# Wichtiger Kompatibilitätshinweis:
# Der aktuelle score_mapper.py ist auf die echte Legacy-Signatur von
# calibration_geometry.py angepasst. Deshalb prüfen diese Tests auch:
# - build_pipeline_points(...) bekommt Legacy-Punkte {"x_px", "y_px"}
# - project_* und calculate_hit_* werden in Legacy-Form aufgerufen

from __future__ import annotations

import numpy as np
import pytest

import vision.score_mapper as sm


# -----------------------------------------------------------------------------
# Testdaten
# -----------------------------------------------------------------------------

TEST_MANUAL_POINTS = [
    (100.0, 50.0),
    (220.0, 80.0),
    (240.0, 210.0),
    (80.0, 180.0),
]

TEST_IMAGE_SIZE = (640, 480)


def _legacy_points(points):
    """
    Wandelt (x, y)-Punkte in das Legacy-Format der echten Geometry um.
    """
    return [
        {"x_px": int(round(p[0])), "y_px": int(round(p[1]))}
        for p in points
    ]


# -----------------------------------------------------------------------------
# Label-Normalisierung
# -----------------------------------------------------------------------------

def test_normalize_hit_label_variants():
    """
    Unterschiedliche Schreibweisen sollen robust auf ein kanonisches Label
    normalisiert werden.
    """
    assert sm.normalize_hit_label("s20") == "S20"
    assert sm.normalize_hit_label("D20") == "D20"
    assert sm.normalize_hit_label("t19") == "T19"

    assert sm.normalize_hit_label("single20") == "S20"
    assert sm.normalize_hit_label("double20") == "D20"
    assert sm.normalize_hit_label("triple19") == "T19"

    assert sm.normalize_hit_label("miss") == "MISS"
    assert sm.normalize_hit_label("out") == "MISS"

    assert sm.normalize_hit_label("25") == "SBULL"
    assert sm.normalize_hit_label("bull") == "SBULL"
    assert sm.normalize_hit_label("50") == "DBULL"
    assert sm.normalize_hit_label("bullseye") == "DBULL"


def test_normalize_hit_label_invalid_values_raise():
    """
    Ungültige Labels müssen eine Exception werfen.
    """
    with pytest.raises(ValueError):
        sm.normalize_hit_label("xyz")

    with pytest.raises(ValueError):
        sm.normalize_hit_label("D21")

    with pytest.raises(ValueError):
        sm.normalize_hit_label("")


# -----------------------------------------------------------------------------
# Score-/Ring-/Segment-/Multiplier-Ableitung
# -----------------------------------------------------------------------------

def test_hit_label_to_score_ring_segment_and_multiplier():
    """
    Aus einem Label müssen Score, Ring, Segment und Multiplier korrekt
    ableitbar sein.
    """
    assert sm.hit_label_to_score("S20") == 20
    assert sm.hit_label_to_score("D20") == 40
    assert sm.hit_label_to_score("T19") == 57
    assert sm.hit_label_to_score("SBULL") == 25
    assert sm.hit_label_to_score("DBULL") == 50
    assert sm.hit_label_to_score("MISS") == 0

    assert sm.hit_label_to_ring("S20") == "S"
    assert sm.hit_label_to_ring("D20") == "D"
    assert sm.hit_label_to_ring("T19") == "T"
    assert sm.hit_label_to_ring("SBULL") == "SBULL"
    assert sm.hit_label_to_ring("DBULL") == "DBULL"
    assert sm.hit_label_to_ring("MISS") == "MISS"

    assert sm.hit_label_to_segment("S20") == 20
    assert sm.hit_label_to_segment("D20") == 20
    assert sm.hit_label_to_segment("T19") == 19
    assert sm.hit_label_to_segment("SBULL") is None
    assert sm.hit_label_to_segment("DBULL") is None
    assert sm.hit_label_to_segment("MISS") is None

    assert sm.hit_label_to_multiplier("S20") == 1
    assert sm.hit_label_to_multiplier("D20") == 2
    assert sm.hit_label_to_multiplier("T19") == 3
    assert sm.hit_label_to_multiplier("SBULL") == 1
    assert sm.hit_label_to_multiplier("DBULL") == 2
    assert sm.hit_label_to_multiplier("MISS") == 0


# -----------------------------------------------------------------------------
# Pipeline-Aufbau
# -----------------------------------------------------------------------------

def test_score_mapper_builds_pipeline_from_manual_points(monkeypatch):
    """
    ScoreMapper soll die zentrale Geometrie genau einmal mit den 4 Markern
    aufbauen und dabei intern Legacy-Punkte weiterreichen.
    """
    captured = {}

    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        captured["manual_points"] = manual_points
        captured["image_size"] = image_size
        captured["kwargs"] = kwargs
        return {"pipeline_id": "fake_pipeline_1"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    assert mapper.pipeline == {"pipeline_id": "fake_pipeline_1"}
    assert mapper.manual_points == TEST_MANUAL_POINTS
    assert mapper.image_size == TEST_IMAGE_SIZE
    assert captured["manual_points"] == _legacy_points(TEST_MANUAL_POINTS)
    assert captured["image_size"] == TEST_IMAGE_SIZE


def test_score_mapper_rejects_non_four_point_setup():
    """
    Der Mapper unterstützt bewusst nur die 4-Punkt-Kalibrierung.
    """
    with pytest.raises(ValueError):
        sm.ScoreMapper(
            manual_points=[
                (10.0, 10.0),
                (20.0, 20.0),
                (30.0, 30.0),
            ],
            image_size=TEST_IMAGE_SIZE,
        )


def test_score_mapper_can_build_from_calibration_record_dict(monkeypatch):
    """
    Ein Mapper soll auch direkt aus einem Calibration-Record-Dict aufgebaut
    werden können.
    """
    captured = {}

    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        captured["manual_points"] = manual_points
        captured["image_size"] = image_size
        return {"pipeline_id": "from_record"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)

    record = {
        "manual_points": TEST_MANUAL_POINTS,
        "image_size": TEST_IMAGE_SIZE,
    }

    mapper = sm.ScoreMapper(calibration_record=record)

    assert mapper.pipeline == {"pipeline_id": "from_record"}
    assert mapper.manual_points == TEST_MANUAL_POINTS
    assert mapper.image_size == TEST_IMAGE_SIZE
    assert captured["manual_points"] == _legacy_points(TEST_MANUAL_POINTS)
    assert captured["image_size"] == TEST_IMAGE_SIZE


# -----------------------------------------------------------------------------
# Bildpunkt-Scoring
# -----------------------------------------------------------------------------

def test_score_image_point_returns_scoredhit(monkeypatch):
    """
    Prüft den kompletten Weg:
    - Pipeline wird gebaut
    - Bildpunkt wird via Legacy-Projection nach Top-Down projiziert
    - Legacy-Hitfunktion liefert einen Treffer
    - Ergebnis wird als ScoredHit normalisiert
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "image_pipeline"}

    def fake_calculate_hit_from_image_point(x_px, y_px, points_like):
        assert (x_px, y_px) == (321.0, 222.0)
        assert points_like["pipeline_id"] == "image_pipeline"
        return {"label": "d20"}

    def fake_project_image_points_to_topdown(*args, **kwargs):
        if len(args) == 2:
            points_like, image_points = args
        else:
            points_like = kwargs.get("pipeline", kwargs.get("points_like"))
            image_points = kwargs.get("image_points", kwargs.get("points"))

        assert points_like["pipeline_id"] == "image_pipeline"
        pts = np.asarray(image_points, dtype=np.float32)
        assert pts.shape == (1, 2)
        assert tuple(pts[0]) == pytest.approx((321.0, 222.0), abs=1e-6)
        return np.asarray([[111.0, 222.0]], dtype=np.float32)

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_calculate_hit_from_image_point)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    result = mapper.score_image_point((321, 222))

    assert isinstance(result, sm.ScoredHit)
    assert result.label == "D20"
    assert result.score == 40
    assert result.ring == "D"
    assert result.segment == 20
    assert result.multiplier == 2
    assert result.source_space == "image"
    assert result.image_point == (321.0, 222.0)
    assert result.topdown_point == (111.0, 222.0)


def test_score_topdown_point_returns_scoredhit(monkeypatch):
    """
    Prüft den umgekehrten Weg:
    - Top-Down-Punkt wird via Legacy-Topdown-Hitfunktion bewertet
    - Rückprojektion ins Bild erfolgt
    - Ergebnis wird als ScoredHit normalisiert
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "topdown_pipeline"}

    def fake_calculate_hit_from_topdown_point(x_px, y_px):
        assert (x_px, y_px) == (120.0, 60.0)
        return {"ring": "T", "segment": 19}

    def fake_project_topdown_points_to_image(*args, **kwargs):
        if len(args) == 2:
            points_like, topdown_points = args
        else:
            points_like = kwargs.get("pipeline", kwargs.get("points_like"))
            topdown_points = kwargs.get("topdown_points", kwargs.get("points"))

        assert points_like["pipeline_id"] == "topdown_pipeline"
        pts = np.asarray(topdown_points, dtype=np.float32)
        assert pts.shape == (1, 2)
        assert tuple(pts[0]) == pytest.approx((120.0, 60.0), abs=1e-6)
        return np.asarray([[500.0, 300.0]], dtype=np.float32)

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_topdown_point", fake_calculate_hit_from_topdown_point)
    monkeypatch.setattr(sm, "project_topdown_points_to_image", fake_project_topdown_points_to_image)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    result = mapper.score_topdown_point((120, 60))

    assert isinstance(result, sm.ScoredHit)
    assert result.label == "T19"
    assert result.score == 57
    assert result.ring == "T"
    assert result.segment == 19
    assert result.multiplier == 3
    assert result.source_space == "topdown"
    assert result.topdown_point == (120.0, 60.0)
    assert result.image_point == (500.0, 300.0)


def test_score_image_point_handles_tuple_style_raw_hit(monkeypatch):
    """
    Tuple-artige Raw-Hits sollen ebenfalls korrekt normalisiert werden.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "tuple_pipeline"}

    def fake_calculate_hit_from_image_point(x_px, y_px, points_like):
        return ("D", 16)

    def fake_project_image_points_to_topdown(points_like, image_points):
        return np.asarray([[50.0, 60.0]], dtype=np.float32)

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_calculate_hit_from_image_point)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    result = mapper.score_image_point((10, 20))

    assert result.label == "D16"
    assert result.score == 32
    assert result.ring == "D"
    assert result.segment == 16
    assert result.multiplier == 2


def test_score_image_point_handles_bull_and_miss(monkeypatch):
    """
    Bulls und MISS sollen ebenfalls korrekt normalisiert werden.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "bull_pipeline"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(
        sm,
        "project_image_points_to_topdown",
        lambda points_like, image_points: np.asarray([[1.0, 1.0]], dtype=np.float32),
    )

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    monkeypatch.setattr(sm, "calculate_hit_from_image_point", lambda x, y, p: {"label": "SBULL"})
    sbull = mapper.score_image_point((1, 2))
    assert sbull.label == "SBULL"
    assert sbull.score == 25
    assert sbull.multiplier == 1

    monkeypatch.setattr(sm, "calculate_hit_from_image_point", lambda x, y, p: {"label": "DBULL"})
    dbull = mapper.score_image_point((1, 2))
    assert dbull.label == "DBULL"
    assert dbull.score == 50
    assert dbull.multiplier == 2

    monkeypatch.setattr(sm, "calculate_hit_from_image_point", lambda x, y, p: {"label": "MISS"})
    miss = mapper.score_image_point((1, 2))
    assert miss.label == "MISS"
    assert miss.score == 0
    assert miss.multiplier == 0


# -----------------------------------------------------------------------------
# Batch-Methoden
# -----------------------------------------------------------------------------

def test_batch_scoring_methods(monkeypatch):
    """
    Batch-Scoring soll die Einzelmethoden korrekt wiederverwenden.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "batch_pipeline"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    def fake_score_image_point(point):
        x, y = point
        return sm.ScoredHit(
            label="S20",
            score=20,
            ring="S",
            segment=20,
            multiplier=1,
            source_space="image",
            image_point=(float(x), float(y)),
            topdown_point=(1.0, 2.0),
            raw_hit={"label": "S20"},
        )

    def fake_score_topdown_point(point):
        x, y = point
        return sm.ScoredHit(
            label="T19",
            score=57,
            ring="T",
            segment=19,
            multiplier=3,
            source_space="topdown",
            image_point=(5.0, 6.0),
            topdown_point=(float(x), float(y)),
            raw_hit={"label": "T19"},
        )

    monkeypatch.setattr(mapper, "score_image_point", fake_score_image_point)
    monkeypatch.setattr(mapper, "score_topdown_point", fake_score_topdown_point)

    image_results = mapper.score_image_points([(1, 2), (3, 4)])
    assert len(image_results) == 2
    assert all(isinstance(v, sm.ScoredHit) for v in image_results)
    assert [v.label for v in image_results] == ["S20", "S20"]

    topdown_results = mapper.score_topdown_points([(10, 20), (30, 40)])
    assert len(topdown_results) == 2
    assert all(isinstance(v, sm.ScoredHit) for v in topdown_results)
    assert [v.label for v in topdown_results] == ["T19", "T19"]


# -----------------------------------------------------------------------------
# Modulweite Wrapper
# -----------------------------------------------------------------------------

def test_module_level_score_wrappers(monkeypatch):
    """
    Die modulweiten Wrapper sollen ScoreMapper korrekt aufbauen und verwenden.
    """
    created = []

    class DummyMapper:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(kwargs)

        def score_image_point(self, point):
            return sm.ScoredHit(
                label="S5",
                score=5,
                ring="S",
                segment=5,
                multiplier=1,
                source_space="image",
                image_point=(float(point[0]), float(point[1])),
                topdown_point=(1.0, 2.0),
                raw_hit={"label": "S5"},
            )

        def score_topdown_point(self, point):
            return sm.ScoredHit(
                label="D7",
                score=14,
                ring="D",
                segment=7,
                multiplier=2,
                source_space="topdown",
                image_point=(3.0, 4.0),
                topdown_point=(float(point[0]), float(point[1])),
                raw_hit={"label": "D7"},
            )

    monkeypatch.setattr(sm, "ScoreMapper", DummyMapper)

    image_result = sm.score_image_point(
        (12, 34),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert image_result.label == "S5"
    assert image_result.score == 5

    topdown_result = sm.score_topdown_point(
        (56, 78),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert topdown_result.label == "D7"
    assert topdown_result.score == 14

    assert len(created) == 2
    assert created[0]["manual_points"] == TEST_MANUAL_POINTS
    assert created[0]["image_size"] == TEST_IMAGE_SIZE


# -----------------------------------------------------------------------------
# Rebuild
# -----------------------------------------------------------------------------

def test_rebuild_from_manual_points_recreates_pipeline(monkeypatch):
    """
    rebuild_from_manual_points soll die Pipeline neu aufbauen.
    """
    calls = []

    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        calls.append(
            {
                "manual_points": manual_points,
                "image_size": image_size,
            }
        )
        return {"pipeline_id": f"pipeline_{len(calls)}"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    new_points = [
        (10.0, 10.0),
        (20.0, 20.0),
        (30.0, 30.0),
        (40.0, 40.0),
    ]

    mapper.rebuild_from_manual_points(new_points, image_size=(800, 600))

    assert mapper.pipeline == {"pipeline_id": "pipeline_2"}
    assert mapper.manual_points == new_points
    assert mapper.image_size == (800, 600)

    assert len(calls) == 2
    assert calls[0]["manual_points"] == _legacy_points(TEST_MANUAL_POINTS)
    assert calls[0]["image_size"] == TEST_IMAGE_SIZE
    assert calls[1]["manual_points"] == _legacy_points(new_points)
    assert calls[1]["image_size"] == (800, 600)


# -----------------------------------------------------------------------------
# Ergebnisobjekt
# -----------------------------------------------------------------------------

def test_scored_hit_to_dict():
    """
    ScoredHit.to_dict() soll die wichtigsten Felder sauber serialisieren.
    """
    hit = sm.ScoredHit(
        label="D20",
        score=40,
        ring="D",
        segment=20,
        multiplier=2,
        source_space="image",
        image_point=(321.0, 222.0),
        topdown_point=(111.0, 222.0),
        raw_hit={"label": "D20"},
    )

    data = hit.to_dict()

    assert data["label"] == "D20"
    assert data["score"] == 40
    assert data["ring"] == "D"
    assert data["segment"] == 20
    assert data["multiplier"] == 2
    assert data["source_space"] == "image"
    assert data["image_point"] == (321.0, 222.0)
    assert data["topdown_point"] == (111.0, 222.0)
    assert data["is_miss"] is False
    assert data["is_bull"] is False