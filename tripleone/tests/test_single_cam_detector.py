# tests/test_single_cam_detector.py
# Zweck:
# Diese Tests prüfen die Orchestrierung der kompletten Single-Camera-Pipeline.
#
# Es wird bewusst NICHT erneut im Detail getestet:
# - Board-Geometrie
# - ScoreMapper-Interna
# - Candidate-Detector-Interna
# - Impact-Estimator-Interna
#
# Es wird getestet:
# - saubere Verkettung der Schichten
# - Fehlerfall ohne ScoreMapper
# - Ranking / best_hit / best_label / best_score
# - score_all_estimates / max_estimates_to_score
# - keep_stage_results / keep_debug_images
# - Wrapper-Funktionen
# - Render-Overlay / Serialisierung
#
# Die Tests verwenden dafür bewusst Fake-Stage-Objekte, damit wir die
# Orchestrierung deterministisch prüfen können.

from __future__ import annotations

import numpy as np
import pytest

import vision.single_cam_detector as scd
from vision.dart_candidate_detector import CandidateDetectionResult, DartCandidate
from vision.impact_estimator import ImpactEstimate, ImpactEstimationResult
from vision.score_mapper import ScoredHit


# -----------------------------------------------------------------------------
# Hilfsobjekte / Builder
# -----------------------------------------------------------------------------

def _make_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """
    Erstellt ein simples schwarzes BGR-Testbild.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def _build_candidate(
    *,
    candidate_id: int,
    bbox: tuple[int, int, int, int] = (90, 60, 20, 80),
    centroid: tuple[float, float] = (100.0, 100.0),
    impact_point: tuple[float, float] = (100.0, 175.0),
    confidence: float = 0.80,
) -> DartCandidate:
    """
    Baut einen minimalistischen, aber gültigen DartCandidate für Orchestrierungs-Tests.
    """
    contour = np.asarray(
        [
            [[95, 60]],
            [[105, 60]],
            [[108, 140]],
            [[100, 180]],
            [[92, 140]],
        ],
        dtype=np.int32,
    )

    return DartCandidate(
        candidate_id=candidate_id,
        bbox=bbox,
        centroid=centroid,
        impact_point=impact_point,
        area=120.0,
        aspect_ratio=4.0,
        solidity=0.7,
        extent=0.3,
        circularity=0.2,
        angle_degrees=8.0,
        major_axis_length=100.0,
        minor_axis_length=20.0,
        elongation=5.0,
        confidence=confidence,
        contour=contour,
        debug={},
    )


def _build_impact_estimate(
    *,
    candidate_id: int,
    impact_point: tuple[float, float],
    method: str = "blend",
    confidence: float = 0.80,
    source_candidate_confidence: float = 0.75,
    bbox: tuple[int, int, int, int] = (90, 60, 20, 80),
    centroid: tuple[float, float] = (100.0, 100.0),
) -> ImpactEstimate:
    """
    Baut ein minimales ImpactEstimate-Objekt.
    """
    return ImpactEstimate(
        candidate_id=candidate_id,
        impact_point=impact_point,
        method=method,
        confidence=confidence,
        source_candidate_confidence=source_candidate_confidence,
        bbox=bbox,
        centroid=centroid,
        hypotheses=[],
        debug={},
        candidate=None,
    )


def _build_scored_hit(
    *,
    label: str,
    score: int,
    ring: str,
    segment: int | None,
    multiplier: int,
    image_point: tuple[float, float],
    topdown_point: tuple[float, float] | None = None,
) -> ScoredHit:
    """
    Baut ein minimales ScoredHit-Objekt.
    """
    return ScoredHit(
        label=label,
        score=score,
        ring=ring,
        segment=segment,
        multiplier=multiplier,
        source_space="image",
        image_point=image_point,
        topdown_point=topdown_point,
        raw_hit={"label": label},
    )


# -----------------------------------------------------------------------------
# Fake-Stage-Objekte
# -----------------------------------------------------------------------------

class FakeCandidateDetector:
    """
    Fake Candidate Detector für deterministische Pipeline-Tests.
    """

    def __init__(self, result: CandidateDetectionResult):
        self.result = result
        self.calls: list[dict] = []

    def detect_candidates(self, frame, reference_frame, *, board_mask=None, board_polygon=None):
        self.calls.append(
            {
                "frame_shape": frame.shape,
                "reference_shape": reference_frame.shape,
                "board_mask": board_mask,
                "board_polygon": board_polygon,
            }
        )
        return self.result


class FakeImpactEstimator:
    """
    Fake Impact Estimator für deterministische Pipeline-Tests.
    """

    def __init__(self, result: ImpactEstimationResult):
        self.result = result
        self.calls: list[dict] = []

    def estimate_from_detection_result(self, detection_result, *, image_shape=None):
        self.calls.append(
            {
                "detection_result": detection_result,
                "image_shape": image_shape,
            }
        )
        return self.result


class FakeScoreMapper:
    """
    Fake ScoreMapper, der definierte Bildpunkte auf definierte ScoredHits abbildet.
    """

    def __init__(self, mapping: dict[tuple[float, float], ScoredHit]):
        self.mapping = mapping
        self.calls: list[tuple[float, float]] = []

    def score_image_point(self, point):
        normalized = (float(point[0]), float(point[1]))
        self.calls.append(normalized)

        if normalized not in self.mapping:
            raise KeyError(f"No fake score configured for point {normalized}")

        return self.mapping[normalized]


# -----------------------------------------------------------------------------
# Grundtests
# -----------------------------------------------------------------------------

def test_detect_raises_runtime_error_when_no_score_mapper_is_configured():
    """
    Ohne ScoreMapper darf die Single-Cam-Pipeline nicht laufen.
    Das ist wichtig, damit die Orchestrierung nicht stillschweigend ohne
    Geometrie/Scoring weiterarbeitet.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )
    impact_result = ImpactEstimationResult(estimates=[], metadata={})

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=None,
    )

    with pytest.raises(RuntimeError):
        detector.detect(frame=frame, reference_frame=reference)


def test_detect_runs_full_pipeline_and_returns_ranked_results():
    """
    Dieser Test prüft die komplette Verkettung:
    CandidateDetector -> ImpactEstimator -> ScoreMapper -> Ranking.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_1 = _build_candidate(candidate_id=1, confidence=0.70)
    candidate_2 = _build_candidate(candidate_id=2, confidence=0.90)

    candidate_result = CandidateDetectionResult(
        candidates=[candidate_1, candidate_2],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={"some": "value"},
        debug_images={"binary_mask": np.zeros((240, 320), dtype=np.uint8)},
    )

    estimate_1 = _build_impact_estimate(
        candidate_id=1,
        impact_point=(100.0, 175.0),
        confidence=0.60,
        source_candidate_confidence=0.70,
    )
    estimate_2 = _build_impact_estimate(
        candidate_id=2,
        impact_point=(150.0, 170.0),
        confidence=0.95,
        source_candidate_confidence=0.90,
        bbox=(140, 60, 20, 80),
        centroid=(150.0, 100.0),
    )

    impact_result = ImpactEstimationResult(
        estimates=[estimate_1, estimate_2],
        metadata={"impact": "ok"},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 175.0),
            ),
            (150.0, 170.0): _build_scored_hit(
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(150.0, 170.0),
            ),
        }
    )

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert isinstance(result, scd.SingleCamDetectionResult)
    assert len(result.scored_estimates) == 2
    assert result.best_estimate is not None
    assert result.best_hit is not None

    # Zweiter Estimate muss wegen höherer kombinierter Konfidenz vorne liegen
    assert result.best_estimate.candidate_id == 2
    assert result.best_label == "D20"
    assert result.best_score == 40
    assert result.best_hit.label == "D20"
    assert result.best_hit.score == 40

    assert result.scored_estimates[0].rank == 1
    assert result.scored_estimates[1].rank == 2
    assert result.scored_estimates[0].combined_confidence >= result.scored_estimates[1].combined_confidence


# -----------------------------------------------------------------------------
# score_all_estimates / max_estimates_to_score
# -----------------------------------------------------------------------------

def test_score_all_estimates_false_scores_only_best_impact_estimate():
    """
    Wenn score_all_estimates=False gesetzt ist, darf nur das erste ImpactEstimate
    weitergescort werden.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1), _build_candidate(candidate_id=2)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    estimate_1 = _build_impact_estimate(
        candidate_id=1,
        impact_point=(100.0, 175.0),
        confidence=0.90,
        source_candidate_confidence=0.80,
    )
    estimate_2 = _build_impact_estimate(
        candidate_id=2,
        impact_point=(150.0, 175.0),
        confidence=0.85,
        source_candidate_confidence=0.80,
    )

    impact_result = ImpactEstimationResult(
        estimates=[estimate_1, estimate_2],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="T20",
                score=60,
                ring="T",
                segment=20,
                multiplier=3,
                image_point=(100.0, 175.0),
            )
        }
    )

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            score_all_estimates=False,
            max_estimates_to_score=3,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert len(result.scored_estimates) == 1
    assert result.best_label == "T20"
    assert score_mapper.calls == [(100.0, 175.0)]


def test_max_estimates_to_score_limits_number_of_scored_impacts():
    """
    max_estimates_to_score muss die Anzahl der weitergescorten ImpactEstimates
    begrenzen.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[
            _build_candidate(candidate_id=1),
            _build_candidate(candidate_id=2),
            _build_candidate(candidate_id=3),
        ],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(candidate_id=1, impact_point=(10.0, 10.0), confidence=0.95),
            _build_impact_estimate(candidate_id=2, impact_point=(20.0, 20.0), confidence=0.90),
            _build_impact_estimate(candidate_id=3, impact_point=(30.0, 30.0), confidence=0.85),
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (10.0, 10.0): _build_scored_hit(
                label="S1",
                score=1,
                ring="S",
                segment=1,
                multiplier=1,
                image_point=(10.0, 10.0),
            ),
            (20.0, 20.0): _build_scored_hit(
                label="S2",
                score=2,
                ring="S",
                segment=2,
                multiplier=1,
                image_point=(20.0, 20.0),
            ),
        }
    )

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            score_all_estimates=True,
            max_estimates_to_score=2,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert len(result.scored_estimates) == 2
    assert score_mapper.calls == [(10.0, 10.0), (20.0, 20.0)]


# -----------------------------------------------------------------------------
# Konfidenzfilter
# -----------------------------------------------------------------------------

def test_min_impact_confidence_filters_low_impact_estimates():
    """
    ImpactEstimates unterhalb min_impact_confidence dürfen nicht weiter
    verwendet werden.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(
                candidate_id=1,
                impact_point=(100.0, 175.0),
                confidence=0.04,
                source_candidate_confidence=0.95,
            )
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper({})

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            min_impact_confidence=0.10,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert result.scored_estimates == []
    assert result.best_estimate is None
    assert result.best_hit is None
    assert result.best_label is None
    assert result.best_score is None
    assert score_mapper.calls == []


def test_min_combined_confidence_filters_scored_results():
    """
    Auch nach erfolgreichem Scoring muss min_combined_confidence greifen.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(
                candidate_id=1,
                impact_point=(100.0, 175.0),
                confidence=0.20,
                source_candidate_confidence=0.20,
            )
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 175.0),
            )
        }
    )

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            min_combined_confidence=0.90,
            weight_candidate_confidence=0.40,
            weight_impact_confidence=0.60,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert result.scored_estimates == []
    assert result.best_estimate is None


# -----------------------------------------------------------------------------
# Metadata / Stage Results / Debug Images
# -----------------------------------------------------------------------------

def test_keep_stage_results_false_drops_stage_results_from_final_result():
    """
    Wenn keep_stage_results=False gesetzt ist, sollen Candidate- und Impact-Stage
    nicht im Endergebnis mitgetragen werden.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )
    impact_result = ImpactEstimationResult(estimates=[], metadata={})

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            keep_stage_results=False,
            keep_debug_images=False,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=FakeScoreMapper({}),
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert result.candidate_result is None
    assert result.impact_result is None


def test_keep_debug_images_false_returns_empty_debug_images():
    """
    Wenn keep_debug_images=False gesetzt ist, sollen keine Debugbilder im
    Ergebnis landen.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={"binary_mask": np.zeros((240, 320), dtype=np.uint8)},
    )
    impact_result = ImpactEstimationResult(estimates=[], metadata={})

    detector = scd.SingleCamDetector(
        config=scd.SingleCamDetectorConfig(
            keep_debug_images=False,
            keep_stage_results=False,
        ),
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=FakeScoreMapper({}),
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert result.debug_images == {}


def test_metadata_contains_best_label_best_score_and_counts():
    """
    Das Ergebnis-Metadata soll die wichtigsten Pipeline-Zahlen enthalten.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(
                candidate_id=1,
                impact_point=(100.0, 175.0),
                confidence=0.90,
                source_candidate_confidence=0.80,
            )
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="DBULL",
                score=50,
                ring="DBULL",
                segment=None,
                multiplier=2,
                image_point=(100.0, 175.0),
            )
        }
    )

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)

    assert result.metadata["candidate_count"] == 1
    assert result.metadata["impact_count"] == 1
    assert result.metadata["scored_count"] == 1
    assert result.metadata["best_label"] == "DBULL"
    assert result.metadata["best_score"] == 50


# -----------------------------------------------------------------------------
# ScoreMapper-Rebuild / Setter
# -----------------------------------------------------------------------------

def test_set_score_mapper_assigns_mapper_instance():
    """
    set_score_mapper muss einen Mapper sauber übernehmen.
    """
    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(
            CandidateDetectionResult(
                candidates=[],
                reference_used=True,
                board_mask_used=False,
                board_polygon_used=False,
                metadata={},
                debug_images={},
            )
        ),
        impact_estimator=FakeImpactEstimator(
            ImpactEstimationResult(estimates=[], metadata={})
        ),
        score_mapper=FakeScoreMapper({}),
    )

    new_mapper = FakeScoreMapper({})
    detector.set_score_mapper(new_mapper)

    assert detector.score_mapper is new_mapper


def test_rebuild_score_mapper_methods_delegate_to_build_score_mapper(monkeypatch):
    """
    Die Rebuild-Methoden sollen build_score_mapper sauber delegieren.
    """
    created = []

    class DummyMapper:
        pass

    def fake_build_score_mapper(**kwargs):
        created.append(kwargs)
        return DummyMapper()

    monkeypatch.setattr(scd, "build_score_mapper", fake_build_score_mapper)

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(
            CandidateDetectionResult(
                candidates=[],
                reference_used=True,
                board_mask_used=False,
                board_polygon_used=False,
                metadata={},
                debug_images={},
            )
        ),
        impact_estimator=FakeImpactEstimator(
            ImpactEstimationResult(estimates=[], metadata={})
        ),
        score_mapper=FakeScoreMapper({}),
    )

    detector.rebuild_score_mapper_from_manual_points(
        [(1, 2), (3, 4), (5, 6), (7, 8)],
        image_size=(640, 480),
    )
    assert isinstance(detector.score_mapper, DummyMapper)

    detector.rebuild_score_mapper_from_record({"manual_points": [(1, 2), (3, 4), (5, 6), (7, 8)]})
    assert isinstance(detector.score_mapper, DummyMapper)

    detector.rebuild_score_mapper_from_pipeline({"pipeline": "x"})
    assert isinstance(detector.score_mapper, DummyMapper)

    assert len(created) == 3
    assert created[0]["manual_points"] == [(1, 2), (3, 4), (5, 6), (7, 8)]
    assert created[0]["image_size"] == (640, 480)
    assert created[1]["calibration_record"] == {"manual_points": [(1, 2), (3, 4), (5, 6), (7, 8)]}
    assert created[2]["pipeline"] == {"pipeline": "x"}


# -----------------------------------------------------------------------------
# Render / to_dict
# -----------------------------------------------------------------------------

def test_render_debug_overlay_returns_same_shape_image():
    """
    Das finale Overlay muss dieselbe Bildgröße zurückgeben.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(
                candidate_id=1,
                impact_point=(100.0, 175.0),
                confidence=0.90,
                source_candidate_confidence=0.85,
            )
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 175.0),
            )
        }
    )

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)
    overlay = result.render_debug_overlay(frame)

    assert overlay.shape == frame.shape
    assert overlay.ndim == 3


def test_to_dict_contains_expected_keys():
    """
    Serialisierung des SingleCamDetectionResult soll die zentralen Felder liefern.
    """
    frame = _make_frame()
    reference = _make_frame()

    candidate_result = CandidateDetectionResult(
        candidates=[_build_candidate(candidate_id=1)],
        reference_used=True,
        board_mask_used=False,
        board_polygon_used=False,
        metadata={},
        debug_images={},
    )

    impact_result = ImpactEstimationResult(
        estimates=[
            _build_impact_estimate(
                candidate_id=1,
                impact_point=(100.0, 175.0),
                confidence=0.90,
                source_candidate_confidence=0.80,
            )
        ],
        metadata={},
    )

    score_mapper = FakeScoreMapper(
        {
            (100.0, 175.0): _build_scored_hit(
                label="T19",
                score=57,
                ring="T",
                segment=19,
                multiplier=3,
                image_point=(100.0, 175.0),
            )
        }
    )

    detector = scd.SingleCamDetector(
        candidate_detector=FakeCandidateDetector(candidate_result),
        impact_estimator=FakeImpactEstimator(impact_result),
        score_mapper=score_mapper,
    )

    result = detector.detect(frame=frame, reference_frame=reference)
    data = result.to_dict()

    assert "metadata" in data
    assert "best_label" in data
    assert "best_score" in data
    assert "scored_estimates" in data
    assert "candidate_result" in data
    assert "impact_result" in data

    assert data["best_label"] == "T19"
    assert data["best_score"] == 57
    assert len(data["scored_estimates"]) == 1


# -----------------------------------------------------------------------------
# Convenience-Wrapper
# -----------------------------------------------------------------------------

def test_detect_single_cam_convenience_wrapper(monkeypatch):
    """
    Der modulweite Convenience-Wrapper soll einen SingleCamDetector bauen und
    detect(...) korrekt aufrufen.
    """
    frame = _make_frame()
    reference = _make_frame()

    expected_result = scd.SingleCamDetectionResult(
        scored_estimates=[],
        metadata={"ok": True},
        debug_images={},
        candidate_result=None,
        impact_result=None,
    )

    class DummyDetector:
        def __init__(self):
            self.calls = []

        def detect(self, frame, reference_frame, *, board_mask=None, board_polygon=None):
            self.calls.append(
                {
                    "frame_shape": frame.shape,
                    "reference_shape": reference_frame.shape,
                    "board_mask": board_mask,
                    "board_polygon": board_polygon,
                }
            )
            return expected_result

    dummy = DummyDetector()

    def fake_build_single_cam_detector(**kwargs):
        return dummy

    monkeypatch.setattr(scd, "build_single_cam_detector", fake_build_single_cam_detector)

    result = scd.detect_single_cam(
        frame=frame,
        reference_frame=reference,
        board_mask=None,
        board_polygon=None,
    )

    assert result is expected_result
    assert len(dummy.calls) == 1
    assert dummy.calls[0]["frame_shape"] == frame.shape
    assert dummy.calls[0]["reference_shape"] == reference.shape