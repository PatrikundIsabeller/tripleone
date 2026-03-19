# tests/test_score_mapper.py
# Zweck:
# Diese Tests prüfen bewusst die ScoreMapper-Schicht und nicht erneut die komplette
# Geometrie-Implementierung. Die zentrale Geometrie wurde bereits in
# tests/test_calibration_geometry.py getestet.
#
# Hier testen wir vor allem:
# - Label-Normalisierung
# - Score-Berechnung
# - Ring/Segment/Multiplier-Mapping
# - Aufbau des ScoreMapper mit 4 Markerpunkten
# - Konsistenz der Wrapper-Funktionen
# - Verhalten bei Bildpunkt- und Top-Down-Scoring
# - Kompatibilität mit Calibration-Records
#
# Wichtige Designentscheidung:
# Die meisten Tests arbeiten mit monkeypatch/fake geometry functions.
# Dadurch testen wir gezielt score_mapper.py als Schicht und mischen nicht wieder
# Geometriefehler und Mapperfehler zusammen.

from __future__ import annotations

import pytest

import vision.score_mapper as sm


# -----------------------------------------------------------------------------
# Hilfsdaten für wiederverwendbare Testmarker
# -----------------------------------------------------------------------------

# Diese 4 Punkte sind nur Testmarker. Für die ScoreMapper-Unit-Tests ist nicht
# entscheidend, ob sie realistisch auf einem echten Board liegen – wichtig ist,
# dass exakt 4 Punkte verwendet werden.
TEST_MANUAL_POINTS = [
    (100.0, 50.0),
    (200.0, 100.0),
    (180.0, 220.0),
    (80.0, 180.0),
]

TEST_IMAGE_SIZE = (640, 480)


# -----------------------------------------------------------------------------
# Reine Funktions-Tests: Label-Normalisierung und Score-Mapping
# -----------------------------------------------------------------------------

def test_normalize_hit_label_variants():
    """
    Dieser Test prüft, ob verschiedene Schreibweisen sauber auf das interne
    Standardformat normalisiert werden.
    """
    assert sm.normalize_hit_label("20") == "S20"
    assert sm.normalize_hit_label("s20") == "S20"
    assert sm.normalize_hit_label("D20") == "D20"
    assert sm.normalize_hit_label("double20") == "D20"
    assert sm.normalize_hit_label("triple 19") == "T19"

    assert sm.normalize_hit_label("bull") == "SBULL"
    assert sm.normalize_hit_label("25") == "SBULL"
    assert sm.normalize_hit_label("outerbull") == "SBULL"

    assert sm.normalize_hit_label("dbull") == "DBULL"
    assert sm.normalize_hit_label("bullseye") == "DBULL"
    assert sm.normalize_hit_label("50") == "DBULL"

    assert sm.normalize_hit_label("miss") == "MISS"
    assert sm.normalize_hit_label("out") == "MISS"


def test_normalize_hit_label_invalid_values_raise():
    """
    Dieser Test stellt sicher, dass ungültige Labels nicht stillschweigend
    akzeptiert werden.
    """
    with pytest.raises(ValueError):
        sm.normalize_hit_label("S21")

    with pytest.raises(ValueError):
        sm.normalize_hit_label("X20")

    with pytest.raises(ValueError):
        sm.normalize_hit_label("hello")

    with pytest.raises(ValueError):
        sm.normalize_hit_label(None)  # type: ignore[arg-type]


def test_hit_label_to_score_ring_segment_and_multiplier():
    """
    Dieser Test prüft die Umrechnung von Labels in:
    - numerischen Score
    - Ring
    - Segment
    - Multiplikator
    """
    assert sm.hit_label_to_score("MISS") == 0
    assert sm.hit_label_to_score("SBULL") == 25
    assert sm.hit_label_to_score("DBULL") == 50
    assert sm.hit_label_to_score("S20") == 20
    assert sm.hit_label_to_score("D20") == 40
    assert sm.hit_label_to_score("T20") == 60

    assert sm.hit_label_to_ring("MISS") == "MISS"
    assert sm.hit_label_to_ring("SBULL") == "SBULL"
    assert sm.hit_label_to_ring("DBULL") == "DBULL"
    assert sm.hit_label_to_ring("D18") == "D"

    assert sm.hit_label_to_segment("MISS") is None
    assert sm.hit_label_to_segment("SBULL") is None
    assert sm.hit_label_to_segment("DBULL") is None
    assert sm.hit_label_to_segment("S20") == 20
    assert sm.hit_label_to_segment("T19") == 19

    assert sm.hit_label_to_multiplier("MISS") == 0
    assert sm.hit_label_to_multiplier("SBULL") == 1
    assert sm.hit_label_to_multiplier("DBULL") == 2
    assert sm.hit_label_to_multiplier("S18") == 1
    assert sm.hit_label_to_multiplier("D18") == 2
    assert sm.hit_label_to_multiplier("T18") == 3


# -----------------------------------------------------------------------------
# Tests für den Aufbau des ScoreMapper
# -----------------------------------------------------------------------------

def test_score_mapper_builds_pipeline_from_manual_points(monkeypatch):
    """
    Dieser Test prüft, ob ScoreMapper die zentrale Geometrie genau einmal mit
    den 4 Markerpunkten aufbaut und dabei die Bildgröße korrekt weitergibt.
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
    assert captured["manual_points"] == TEST_MANUAL_POINTS
    assert captured["image_size"] == TEST_IMAGE_SIZE


def test_score_mapper_rejects_non_four_point_setup():
    """
    Dieser Test stellt sicher, dass nur die definierte 4-Punkt-Architektur
    akzeptiert wird.
    """
    with pytest.raises(ValueError):
        sm.ScoreMapper(
            manual_points=[(1, 2), (3, 4), (5, 6)],
            image_size=TEST_IMAGE_SIZE,
        )


def test_score_mapper_can_build_from_calibration_record_dict(monkeypatch):
    """
    Dieser Test prüft, ob ein Mapper auch direkt aus einem Calibration-Record-Dict
    aufgebaut werden kann.
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
    assert captured["manual_points"] == TEST_MANUAL_POINTS
    assert captured["image_size"] == TEST_IMAGE_SIZE


# -----------------------------------------------------------------------------
# Tests für Projektion und Bildpunkt-Scoring
# -----------------------------------------------------------------------------

def test_score_image_point_returns_scoredhit(monkeypatch):
    """
    Dieser Test prüft den kompletten Weg:
    - Mapper wird gebaut
    - Bildpunkt wird in Top-Down projiziert
    - Hit wird über die zentrale Geometrie geliefert
    - Ergebnis wird als ScoredHit vereinheitlicht
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "image_pipeline"}

    def fake_calculate_hit_from_image_point(*, image_point, pipeline):
        assert image_point == (321.0, 222.0)
        assert pipeline["pipeline_id"] == "image_pipeline"
        return {"label": "d20"}

    def fake_project_image_points_to_topdown(*, image_points, pipeline):
        assert image_points == [(321.0, 222.0)]
        assert pipeline["pipeline_id"] == "image_pipeline"
        return [(111.0, 222.0)]

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
    assert result.is_miss is False
    assert result.is_bull is False


def test_score_topdown_point_returns_scoredhit(monkeypatch):
    """
    Dieser Test prüft den umgekehrten Weg:
    - Top-Down-Punkt wird bewertet
    - Rückprojektion ins Bild erfolgt
    - Ergebnis wird als ScoredHit normalisiert
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "topdown_pipeline"}

    def fake_calculate_hit_from_topdown_point(*, topdown_point, pipeline):
        assert topdown_point == (120.0, 60.0)
        assert pipeline["pipeline_id"] == "topdown_pipeline"
        return {"ring": "T", "segment": 19}

    def fake_project_topdown_points_to_image(*, topdown_points, pipeline):
        assert topdown_points == [(120.0, 60.0)]
        assert pipeline["pipeline_id"] == "topdown_pipeline"
        return [(500.0, 300.0)]

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
    assert result.is_miss is False
    assert result.is_bull is False


def test_score_image_point_handles_tuple_style_raw_hit(monkeypatch):
    """
    Dieser Test prüft, ob auch tuple-basierte Rückgaben aus der Geometrieschicht
    sauber verarbeitet werden.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "tuple_pipeline"}

    def fake_calculate_hit_from_image_point(*, image_point, pipeline):
        return ("D", 2)

    def fake_project_image_points_to_topdown(*, image_points, pipeline):
        return [(10.0, 20.0)]

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_calculate_hit_from_image_point)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    result = mapper.score_image_point((15, 25))

    assert result.label == "D2"
    assert result.score == 4
    assert result.ring == "D"
    assert result.segment == 2
    assert result.multiplier == 2


def test_score_image_point_handles_bull_and_miss(monkeypatch):
    """
    Dieser Test prüft Sonderfälle wie Bull und Miss.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "bull_pipeline"}

    def fake_project_image_points_to_topdown(*, image_points, pipeline):
        return [(0.0, 0.0)]

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    def fake_hit_dbull(*, image_point, pipeline):
        return "DBULL"

    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_hit_dbull)
    dbull = mapper.score_image_point((1, 2))
    assert dbull.label == "DBULL"
    assert dbull.score == 50
    assert dbull.is_bull is True
    assert dbull.is_miss is False

    def fake_hit_miss(*, image_point, pipeline):
        return "MISS"

    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_hit_miss)
    miss = mapper.score_image_point((3, 4))
    assert miss.label == "MISS"
    assert miss.score == 0
    assert miss.is_bull is False
    assert miss.is_miss is True


# -----------------------------------------------------------------------------
# Tests für Batch-Methoden und Convenience-Wrapper
# -----------------------------------------------------------------------------

def test_batch_scoring_methods(monkeypatch):
    """
    Dieser Test prüft, ob mehrere Punkte nacheinander sauber ausgewertet werden.
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "batch_pipeline"}

    def fake_calculate_hit_from_image_point(*, image_point, pipeline):
        if image_point == (1.0, 1.0):
            return "S20"
        if image_point == (2.0, 2.0):
            return "D20"
        return "MISS"

    def fake_project_image_points_to_topdown(*, image_points, pipeline):
        return [(p[0] + 100.0, p[1] + 100.0) for p in image_points]

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_calculate_hit_from_image_point)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    results = mapper.score_image_points([(1, 1), (2, 2), (3, 3)])

    assert len(results) == 3
    assert [r.label for r in results] == ["S20", "D20", "MISS"]
    assert [r.score for r in results] == [20, 40, 0]


def test_module_level_score_wrappers(monkeypatch):
    """
    Dieser Test prüft die modulweiten Convenience-Wrapper:
    - score_image_point(...)
    - score_topdown_point(...)
    - map_image_point_to_hit(...)
    - map_topdown_point_to_hit(...)
    """
    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        return {"pipeline_id": "module_wrapper_pipeline"}

    def fake_calculate_hit_from_image_point(*, image_point, pipeline):
        return {"label": "T20"}

    def fake_calculate_hit_from_topdown_point(*, topdown_point, pipeline):
        return {"label": "SBULL"}

    def fake_project_image_points_to_topdown(*, image_points, pipeline):
        return [(999.0, 888.0)]

    def fake_project_topdown_points_to_image(*, topdown_points, pipeline):
        return [(777.0, 666.0)]

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)
    monkeypatch.setattr(sm, "calculate_hit_from_image_point", fake_calculate_hit_from_image_point)
    monkeypatch.setattr(sm, "calculate_hit_from_topdown_point", fake_calculate_hit_from_topdown_point)
    monkeypatch.setattr(sm, "project_image_points_to_topdown", fake_project_image_points_to_topdown)
    monkeypatch.setattr(sm, "project_topdown_points_to_image", fake_project_topdown_points_to_image)

    image_result = sm.score_image_point(
        (10, 20),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert image_result.label == "T20"
    assert image_result.score == 60
    assert image_result.topdown_point == (999.0, 888.0)

    topdown_result = sm.score_topdown_point(
        (30, 40),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert topdown_result.label == "SBULL"
    assert topdown_result.score == 25
    assert topdown_result.image_point == (777.0, 666.0)

    image_label = sm.map_image_point_to_hit(
        (10, 20),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert image_label == "T20"

    topdown_label = sm.map_topdown_point_to_hit(
        (30, 40),
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )
    assert topdown_label == "SBULL"


# -----------------------------------------------------------------------------
# Tests für Rebuild / Update-Verhalten
# -----------------------------------------------------------------------------

def test_rebuild_from_manual_points_recreates_pipeline(monkeypatch):
    """
    Dieser Test prüft, ob ein bestehender Mapper mit neuen Markerpunkten korrekt
    neu aufgebaut wird.
    """
    calls = []

    def fake_build_pipeline_points(*, manual_points, image_size=None, **kwargs):
        calls.append(
            {
                "manual_points": list(manual_points),
                "image_size": image_size,
            }
        )
        return {"pipeline_id": f"pipeline_{len(calls)}"}

    monkeypatch.setattr(sm, "build_pipeline_points", fake_build_pipeline_points)

    mapper = sm.ScoreMapper(
        manual_points=TEST_MANUAL_POINTS,
        image_size=TEST_IMAGE_SIZE,
    )

    assert mapper.pipeline == {"pipeline_id": "pipeline_1"}

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
    assert calls[0]["manual_points"] == TEST_MANUAL_POINTS
    assert calls[1]["manual_points"] == new_points
    assert calls[1]["image_size"] == (800, 600)


def test_scored_hit_to_dict():
    """
    Dieser Test prüft die Debug-/API-Ausgabeform von ScoredHit.
    """
    hit = sm.ScoredHit(
        label="D20",
        score=40,
        ring="D",
        segment=20,
        multiplier=2,
        source_space="image",
        image_point=(100.0, 200.0),
        topdown_point=(50.0, 60.0),
        raw_hit={"label": "D20"},
    )

    data = hit.to_dict()

    assert data["label"] == "D20"
    assert data["score"] == 40
    assert data["ring"] == "D"
    assert data["segment"] == 20
    assert data["multiplier"] == 2
    assert data["source_space"] == "image"
    assert data["image_point"] == (100.0, 200.0)
    assert data["topdown_point"] == (50.0, 60.0)
    assert data["is_miss"] is False
    assert data["is_bull"] is False