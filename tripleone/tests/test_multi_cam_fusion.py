# tests/test_multi_cam_fusion.py
# Zweck:
# Diese Tests prüfen ausschließlich die Fusion mehrerer fertiger
# Single-Cam-Ergebnisse.
#
# Es wird bewusst NICHT erneut im Detail getestet:
# - Candidate Detector
# - Impact Estimator
# - ScoreMapper
# - SingleCamDetector-Orchestrierung
#
# Es wird getestet:
# - leerer Input
# - best_single
# - majority_vote
# - score_weighted_vote
# - max_estimates_per_camera
# - Mindestkonfidenzfilter
# - prefer_more_cameras_over_raw_confidence
# - Wrapper / Serialisierung / Debug-Overlay
#
# Die Tests arbeiten bewusst mit synthetischen Single-Cam-Ergebnissen, damit
# die Ergebnisse deterministisch bleiben.

from __future__ import annotations

import numpy as np
import pytest

import vision.multi_cam_fusion as mcf
from vision.impact_estimator import ImpactEstimate
from vision.score_mapper import ScoredHit
from vision.single_cam_detector import (
    SingleCamDetectionResult,
    SingleCamScoredEstimate,
)


# -----------------------------------------------------------------------------
# Hilfsfunktionen / Builder
# -----------------------------------------------------------------------------

def _make_frame(width: int = 320, height: int = 240) -> np.ndarray:
    """
    Erstellt ein schwarzes BGR-Testbild.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def _build_scored_hit(
    *,
    label: str,
    score: int,
    ring: str,
    segment: int | None,
    multiplier: int,
    image_point: tuple[float, float],
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
        topdown_point=None,
        raw_hit={"label": label},
    )


def _build_impact_estimate(
    *,
    candidate_id: int,
    impact_point: tuple[float, float],
    confidence: float,
    source_candidate_confidence: float,
    bbox: tuple[int, int, int, int] = (90, 60, 20, 80),
    centroid: tuple[float, float] = (100.0, 100.0),
) -> ImpactEstimate:
    """
    Baut ein minimales ImpactEstimate-Objekt.
    """
    return ImpactEstimate(
        candidate_id=candidate_id,
        impact_point=impact_point,
        method="blend",
        confidence=confidence,
        source_candidate_confidence=source_candidate_confidence,
        bbox=bbox,
        centroid=centroid,
        hypotheses=[],
        debug={},
        candidate=None,
    )


def _build_single_cam_scored_estimate(
    *,
    rank: int,
    candidate_id: int,
    label: str,
    score: int,
    ring: str,
    segment: int | None,
    multiplier: int,
    image_point: tuple[float, float],
    candidate_confidence: float,
    impact_confidence: float,
    combined_confidence: float,
) -> SingleCamScoredEstimate:
    """
    Baut ein minimales SingleCamScoredEstimate-Objekt.
    """
    impact_estimate = _build_impact_estimate(
        candidate_id=candidate_id,
        impact_point=image_point,
        confidence=impact_confidence,
        source_candidate_confidence=candidate_confidence,
        centroid=(image_point[0], max(0.0, image_point[1] - 60.0)),
    )

    scored_hit = _build_scored_hit(
        label=label,
        score=score,
        ring=ring,
        segment=segment,
        multiplier=multiplier,
        image_point=image_point,
    )

    return SingleCamScoredEstimate(
        rank=rank,
        candidate_id=candidate_id,
        image_point=image_point,
        scored_hit=scored_hit,
        impact_estimate=impact_estimate,
        candidate_confidence=candidate_confidence,
        impact_confidence=impact_confidence,
        combined_confidence=combined_confidence,
        bbox=(int(image_point[0] - 10), int(image_point[1] - 60), 20, 80),
        centroid=(image_point[0], max(0.0, image_point[1] - 40.0)),
        debug={},
    )


def _build_single_cam_result(
    scored_estimates: list[SingleCamScoredEstimate],
) -> SingleCamDetectionResult:
    """
    Baut ein minimales SingleCamDetectionResult.
    """
    best_label = scored_estimates[0].label if scored_estimates else None
    best_score = scored_estimates[0].score if scored_estimates else None

    return SingleCamDetectionResult(
        scored_estimates=scored_estimates,
        metadata={
            "candidate_count": len(scored_estimates),
            "impact_count": len(scored_estimates),
            "scored_count": len(scored_estimates),
            "best_label": best_label,
            "best_score": best_score,
        },
        debug_images={},
        candidate_result=None,
        impact_result=None,
    )


def _base_config(**overrides) -> mcf.MultiCamFusionConfig:
    """
    Liefert eine stabile Testkonfiguration.
    """
    config = mcf.MultiCamFusionConfig(
        fusion_mode="score_weighted_vote",
        max_estimates_per_camera=2,
        min_combined_confidence=0.01,
        weight_camera_estimate_confidence=0.70,
        weight_label_agreement_bonus=0.30,
        agreement_bonus_per_extra_camera=0.10,
        prefer_more_cameras_over_raw_confidence=True,
        keep_debug_metadata=True,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


# -----------------------------------------------------------------------------
# Grundtests
# -----------------------------------------------------------------------------

def test_fuse_empty_input_returns_empty_result():
    """
    Leerer Input muss ein leeres, aber gültiges Fusionsergebnis liefern.
    """
    fusion = mcf.MultiCamFusion(config=_base_config())
    result = fusion.fuse({})

    assert isinstance(result, mcf.MultiCamFusionResult)
    assert result.fused_estimates == []
    assert result.fusion_candidates == []
    assert result.best_estimate is None
    assert result.best_label is None
    assert result.best_score is None

    assert result.metadata["camera_count"] == 0
    assert result.metadata["input_estimate_count"] == 0


def test_fuse_rejects_non_dict_input():
    """
    results_by_camera muss ein dict sein.
    """
    fusion = mcf.MultiCamFusion(config=_base_config())

    with pytest.raises(TypeError):
        fusion.fuse([])  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# best_single
# -----------------------------------------------------------------------------

def test_best_single_returns_highest_confidence_estimate():
    """
    Im Modus best_single muss schlicht das stärkste Einzelresultat gewinnen.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.70,
                impact_confidence=0.75,
                combined_confidence=0.72,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(150.0, 165.0),
                candidate_confidence=0.90,
                impact_confidence=0.95,
                combined_confidence=0.93,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(fusion_mode="best_single"))
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b})

    assert len(result.fused_estimates) == 1
    assert result.best_estimate is not None
    assert result.best_label == "D20"
    assert result.best_score == 40
    assert result.best_estimate.best_camera_key == "cam_b"
    assert result.best_estimate.support_count == 1


# -----------------------------------------------------------------------------
# majority_vote
# -----------------------------------------------------------------------------

def test_majority_vote_prefers_label_with_more_camera_support():
    """
    Im Modus majority_vote muss die Label-Mehrheit gewinnen.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="T20",
                score=60,
                ring="T",
                segment=20,
                multiplier=3,
                image_point=(100.0, 170.0),
                candidate_confidence=0.80,
                impact_confidence=0.85,
                combined_confidence=0.83,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="T20",
                score=60,
                ring="T",
                segment=20,
                multiplier=3,
                image_point=(120.0, 168.0),
                candidate_confidence=0.78,
                impact_confidence=0.82,
                combined_confidence=0.80,
            )
        ]
    )

    cam_c = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=3,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(150.0, 165.0),
                candidate_confidence=0.95,
                impact_confidence=0.97,
                combined_confidence=0.96,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(fusion_mode="majority_vote"))
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b, "cam_c": cam_c})

    assert result.best_estimate is not None
    assert result.best_label == "T20"
    assert result.best_score == 60
    assert result.best_estimate.support_count == 2


def test_majority_vote_breaks_tie_by_confidence():
    """
    Bei gleicher Kamerazahl muss majority_vote nach Konfidenz entscheiden.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.60,
                impact_confidence=0.65,
                combined_confidence=0.63,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(150.0, 170.0),
                candidate_confidence=0.90,
                impact_confidence=0.92,
                combined_confidence=0.91,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(fusion_mode="majority_vote"))
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b})

    assert result.best_estimate is not None
    assert result.best_label == "D20"
    assert result.best_score == 40


# -----------------------------------------------------------------------------
# score_weighted_vote
# -----------------------------------------------------------------------------

def test_score_weighted_vote_aggregates_same_labels_across_cameras():
    """
    Im Modus score_weighted_vote müssen gleiche Labels über mehrere Kameras
    aggregiert werden.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(100.0, 170.0),
                candidate_confidence=0.80,
                impact_confidence=0.90,
                combined_confidence=0.86,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(130.0, 168.0),
                candidate_confidence=0.82,
                impact_confidence=0.88,
                combined_confidence=0.85,
            )
        ]
    )

    cam_c = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=3,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(160.0, 165.0),
                candidate_confidence=0.95,
                impact_confidence=0.96,
                combined_confidence=0.955,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(fusion_mode="score_weighted_vote"))
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b, "cam_c": cam_c})

    assert result.best_estimate is not None
    assert result.best_label == "D20"
    assert result.best_score == 40
    assert result.best_estimate.support_count == 2
    assert len(result.best_estimate.members) == 2


def test_prefer_more_cameras_over_raw_confidence_changes_winner():
    """
    Wenn prefer_more_cameras_over_raw_confidence=True, soll ein von mehr Kameras
    unterstütztes Label vor einem einzelnen, rohen High-Confidence-Label liegen.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.78,
                impact_confidence=0.80,
                combined_confidence=0.79,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(120.0, 168.0),
                candidate_confidence=0.77,
                impact_confidence=0.81,
                combined_confidence=0.79,
            )
        ]
    )

    cam_c = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=3,
                label="T20",
                score=60,
                ring="T",
                segment=20,
                multiplier=3,
                image_point=(160.0, 165.0),
                candidate_confidence=0.98,
                impact_confidence=0.98,
                combined_confidence=0.98,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(
        config=_base_config(
            fusion_mode="score_weighted_vote",
            prefer_more_cameras_over_raw_confidence=True,
        )
    )
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b, "cam_c": cam_c})

    assert result.best_estimate is not None
    assert result.best_label == "S20"
    assert result.best_estimate.support_count == 2


def test_without_prefer_more_cameras_raw_confidence_can_win():
    """
    Wenn prefer_more_cameras_over_raw_confidence=False, darf rohe Fusionskonfidenz
    vor Support-Count priorisiert werden.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.65,
                impact_confidence=0.65,
                combined_confidence=0.65,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(120.0, 168.0),
                candidate_confidence=0.64,
                impact_confidence=0.64,
                combined_confidence=0.64,
            )
        ]
    )

    cam_c = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=3,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(160.0, 165.0),
                candidate_confidence=0.97,
                impact_confidence=0.98,
                combined_confidence=0.975,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(
        config=_base_config(
            fusion_mode="score_weighted_vote",
            prefer_more_cameras_over_raw_confidence=False,
            weight_camera_estimate_confidence=1.0,
            weight_label_agreement_bonus=0.0,
        )
    )
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b, "cam_c": cam_c})

    assert result.best_estimate is not None
    assert result.best_label == "D20"
    assert result.best_score == 40


# -----------------------------------------------------------------------------
# max_estimates_per_camera / min_combined_confidence
# -----------------------------------------------------------------------------

def test_max_estimates_per_camera_limits_flattened_input():
    """
    Pro Kamera dürfen nur die Top-N Estimates in die Fusion eingehen.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.80,
                impact_confidence=0.80,
                combined_confidence=0.80,
            ),
            _build_single_cam_scored_estimate(
                rank=2,
                candidate_id=2,
                label="D20",
                score=40,
                ring="D",
                segment=20,
                multiplier=2,
                image_point=(110.0, 170.0),
                candidate_confidence=0.79,
                impact_confidence=0.79,
                combined_confidence=0.79,
            ),
            _build_single_cam_scored_estimate(
                rank=3,
                candidate_id=3,
                label="T20",
                score=60,
                ring="T",
                segment=20,
                multiplier=3,
                image_point=(120.0, 170.0),
                candidate_confidence=0.78,
                impact_confidence=0.78,
                combined_confidence=0.78,
            ),
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(max_estimates_per_camera=2))
    result = fusion.fuse({"cam_a": cam_a})

    assert result.metadata["input_estimate_count"] == 2
    assert len(result.fusion_candidates) == 2


def test_min_combined_confidence_filters_flattened_estimates():
    """
    Estimates unterhalb min_combined_confidence dürfen gar nicht erst in die
    Fusion eingehen.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.02,
                impact_confidence=0.03,
                combined_confidence=0.025,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(min_combined_confidence=0.10))
    result = fusion.fuse({"cam_a": cam_a})

    assert result.fused_estimates == []
    assert result.fusion_candidates == []
    assert result.best_estimate is None
    assert result.metadata["input_estimate_count"] == 0


# -----------------------------------------------------------------------------
# to_dict / best properties / Overlay
# -----------------------------------------------------------------------------

def test_result_to_dict_contains_expected_keys():
    """
    Serialisierung soll die wichtigsten Felder enthalten.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="DBULL",
                score=50,
                ring="DBULL",
                segment=None,
                multiplier=2,
                image_point=(100.0, 170.0),
                candidate_confidence=0.90,
                impact_confidence=0.95,
                combined_confidence=0.93,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config(fusion_mode="best_single"))
    result = fusion.fuse({"cam_a": cam_a})

    data = result.to_dict()

    assert "metadata" in data
    assert "best_label" in data
    assert "best_score" in data
    assert "fused_estimates" in data
    assert "fusion_candidates" in data

    assert data["best_label"] == "DBULL"
    assert data["best_score"] == 50
    assert len(data["fused_estimates"]) == 1
    assert len(data["fusion_candidates"]) == 1


def test_render_debug_overlay_returns_overlay_per_camera():
    """
    Das Debug-Overlay muss pro übergebenem Frame ein Bild zurückgeben.
    """
    cam_a = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=1,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(100.0, 170.0),
                candidate_confidence=0.85,
                impact_confidence=0.88,
                combined_confidence=0.87,
            )
        ]
    )

    cam_b = _build_single_cam_result(
        [
            _build_single_cam_scored_estimate(
                rank=1,
                candidate_id=2,
                label="S20",
                score=20,
                ring="S",
                segment=20,
                multiplier=1,
                image_point=(120.0, 165.0),
                candidate_confidence=0.84,
                impact_confidence=0.86,
                combined_confidence=0.85,
            )
        ]
    )

    fusion = mcf.MultiCamFusion(config=_base_config())
    result = fusion.fuse({"cam_a": cam_a, "cam_b": cam_b})

    overlays = result.render_debug_overlay(
        {
            "cam_a": _make_frame(),
            "cam_b": _make_frame(),
        }
    )

    assert set(overlays.keys()) == {"cam_a", "cam_b"}
    assert overlays["cam_a"].shape == (240, 320, 3)
    assert overlays["cam_b"].shape == (240, 320, 3)


# -----------------------------------------------------------------------------
# Wrapper
# -----------------------------------------------------------------------------

def test_fuse_multi_cam_results_wrapper(monkeypatch):
    """
    Der modulweite Wrapper soll MultiCamFusion korrekt bauen und fuse(...) aufrufen.
    """
    expected_result = mcf.MultiCamFusionResult(
        fused_estimates=[],
        metadata={"ok": True},
        fusion_candidates=[],
    )

    class DummyFusion:
        def __init__(self):
            self.calls = []

        def fuse(self, results_by_camera):
            self.calls.append(results_by_camera)
            return expected_result

    dummy = DummyFusion()

    def fake_build_multi_cam_fusion(config=None):
        return dummy

    monkeypatch.setattr(mcf, "build_multi_cam_fusion", fake_build_multi_cam_fusion)

    result = mcf.fuse_multi_cam_results({"cam_a": _build_single_cam_result([])})

    assert result is expected_result
    assert len(dummy.calls) == 1
    assert "cam_a" in dummy.calls[0]