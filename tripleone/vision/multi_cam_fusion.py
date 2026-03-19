# vision/multi_cam_fusion.py
# Zweck:
# Diese Datei fusioniert mehrere bereits berechnete Single-Cam-Ergebnisse
# zu einem finalen Multi-Cam-Ergebnis.
#
# Verantwortungsbereich:
# - mehrere Kameraergebnisse einsammeln
# - Konsens / Mehrheitslogik / Konfidenzlogik anwenden
# - konkurrierende Aussagen ranken
# - ein finales Ergebnis liefern
#
# Nicht verantwortlich für:
# - Rohbilddetektion
# - Board-Geometrie
# - Homography
# - Score-Berechnung
#
# Diese Datei arbeitet nur auf den bereits sauberen Outputs aus
# vision/single_cam_detector.py.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .single_cam_detector import (
        SingleCamDetectionResult,
        SingleCamScoredEstimate,
    )
except ImportError:  # pragma: no cover
    from vision.single_cam_detector import (  # type: ignore
        SingleCamDetectionResult,
        SingleCamScoredEstimate,
    )

logger = logging.getLogger(__name__)

PointF = tuple[float, float]


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class MultiCamFusionConfig:
    """
    Konfiguration für die Fusion mehrerer Single-Cam-Ergebnisse.

    fusion_mode:
    - "score_weighted_vote":
        gleiche Labels werden über Konfidenzen aggregiert
    - "best_single":
        nimm einfach das beste Einzelresultat
    - "majority_vote":
        Label-Mehrheit gewinnt, dann Konfidenz
    """

    fusion_mode: str = "score_weighted_vote"

    # Wie viele Top-Estimates pro Kamera dürfen in die Fusion eingehen?
    max_estimates_per_camera: int = 2

    # Mindestkonfidenzen
    min_combined_confidence: float = 0.01

    # Gewichtung innerhalb der Fusion
    weight_camera_estimate_confidence: float = 0.70
    weight_label_agreement_bonus: float = 0.30

    # Bonus für mehrere Kameras mit gleichem Label
    agreement_bonus_per_extra_camera: float = 0.10

    # Wenn mehrere Kameras dasselbe Label liefern:
    # bevorzuge Gruppen mit mehr Kameras
    prefer_more_cameras_over_raw_confidence: bool = True

    # Debug
    keep_debug_metadata: bool = True


# -----------------------------------------------------------------------------
# Datenmodelle
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class CameraEstimateRef:
    """
    Referenz auf ein einzelnes Single-Cam-Ergebnis innerhalb der Fusion.
    """
    camera_key: str
    rank: int
    label: str
    score: int
    combined_confidence: float
    image_point: PointF
    estimate: SingleCamScoredEstimate

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_key": self.camera_key,
            "rank": self.rank,
            "label": self.label,
            "score": self.score,
            "combined_confidence": self.combined_confidence,
            "image_point": self.image_point,
            "estimate": self.estimate.to_dict(),
        }


@dataclass(slots=True)
class FusionCandidate:
    """
    Eine Fusionsgruppe für ein bestimmtes Label, z. B. D20.
    """
    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int
    members: list[CameraEstimateRef] = field(default_factory=list)
    support_count: int = 0
    total_confidence: float = 0.0
    mean_confidence: float = 0.0
    agreement_bonus: float = 0.0
    fused_confidence: float = 0.0
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def best_member(self) -> Optional[CameraEstimateRef]:
        if not self.members:
            return None
        return max(self.members, key=lambda m: m.combined_confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "support_count": self.support_count,
            "total_confidence": self.total_confidence,
            "mean_confidence": self.mean_confidence,
            "agreement_bonus": self.agreement_bonus,
            "fused_confidence": self.fused_confidence,
            "members": [member.to_dict() for member in self.members],
            "debug": self.debug,
        }


@dataclass(slots=True)
class MultiCamFusedEstimate:
    """
    Finales fusioniertes Ergebnis über mehrere Kameras.
    """
    rank: int
    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int
    fused_confidence: float
    support_count: int
    best_camera_key: str
    best_image_point: PointF
    members: list[CameraEstimateRef] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "fused_confidence": self.fused_confidence,
            "support_count": self.support_count,
            "best_camera_key": self.best_camera_key,
            "best_image_point": self.best_image_point,
            "members": [member.to_dict() for member in self.members],
            "debug": self.debug,
        }


@dataclass(slots=True)
class MultiCamFusionResult:
    """
    Gesamtergebnis der Multi-Cam-Fusion.
    """
    fused_estimates: list[MultiCamFusedEstimate]
    metadata: dict[str, Any] = field(default_factory=dict)
    fusion_candidates: list[FusionCandidate] = field(default_factory=list)

    @property
    def best_estimate(self) -> Optional[MultiCamFusedEstimate]:
        if not self.fused_estimates:
            return None
        return self.fused_estimates[0]

    @property
    def best_label(self) -> Optional[str]:
        if self.best_estimate is None:
            return None
        return self.best_estimate.label

    @property
    def best_score(self) -> Optional[int]:
        if self.best_estimate is None:
            return None
        return self.best_estimate.score

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "best_label": self.best_label,
            "best_score": self.best_score,
            "fused_estimates": [estimate.to_dict() for estimate in self.fused_estimates],
            "fusion_candidates": [candidate.to_dict() for candidate in self.fusion_candidates],
        }

    def render_debug_overlay(
        self,
        frames_by_camera: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Zeichnet pro Kamera das beste fusionierte Ergebnis in das jeweilige Bild.

        Wichtiger Hinweis:
        Diese Funktion verändert keine Originalframes.
        """
        overlays: dict[str, np.ndarray] = {}

        for camera_key, frame in frames_by_camera.items():
            canvas = _ensure_bgr(frame)

            if self.best_estimate is not None:
                for member in self.best_estimate.members:
                    if member.camera_key != camera_key:
                        continue

                    px, py = _round_point(member.image_point)

                    cv2.circle(canvas, (px, py), 6, (0, 0, 255), 2)
                    cv2.putText(
                        canvas,
                        f"{self.best_estimate.label} | fused={self.best_estimate.fused_confidence:.2f}",
                        (max(5, px + 8), max(18, py - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            overlays[camera_key] = canvas

        return overlays


# -----------------------------------------------------------------------------
# Hauptklasse
# -----------------------------------------------------------------------------

class MultiCamFusion:
    """
    Fusioniert mehrere Single-Cam-Ergebnisse zu einem finalen Multi-Cam-Ergebnis.
    """

    def __init__(self, config: Optional[MultiCamFusionConfig] = None) -> None:
        self.config = config or MultiCamFusionConfig()

    # -------------------------------------------------------------------------
    # Öffentliche API
    # -------------------------------------------------------------------------

    def fuse(
        self,
        results_by_camera: dict[str, SingleCamDetectionResult],
    ) -> MultiCamFusionResult:
        """
        Führt die Multi-Cam-Fusion aus.

        Erwartet:
        - dict[str, SingleCamDetectionResult]
          z. B. {"cam_0": result0, "cam_1": result1, "cam_2": result2}
        """
        if results_by_camera is None:
            raise ValueError("results_by_camera must not be None.")

        if not isinstance(results_by_camera, dict):
            raise TypeError("results_by_camera must be a dict[str, SingleCamDetectionResult].")

        flat_estimates = self._flatten_estimates(results_by_camera)

        if not flat_estimates:
            return MultiCamFusionResult(
                fused_estimates=[],
                metadata={
                    "camera_count": len(results_by_camera),
                    "input_estimate_count": 0,
                    "fusion_mode": self.config.fusion_mode,
                },
                fusion_candidates=[],
            )

        mode = self.config.fusion_mode.strip().lower()

        if mode == "best_single":
            fused_estimates, fusion_candidates = self._fuse_best_single(flat_estimates)
        elif mode == "majority_vote":
            fused_estimates, fusion_candidates = self._fuse_majority_vote(flat_estimates)
        elif mode == "score_weighted_vote":
            fused_estimates, fusion_candidates = self._fuse_score_weighted_vote(flat_estimates)
        else:
            raise ValueError(
                "Unsupported fusion_mode. Expected one of: "
                "'score_weighted_vote', 'best_single', 'majority_vote'."
            )

        metadata = {
            "camera_count": len(results_by_camera),
            "input_estimate_count": len(flat_estimates),
            "fused_estimate_count": len(fused_estimates),
            "fusion_mode": self.config.fusion_mode,
            "best_label": fused_estimates[0].label if fused_estimates else None,
            "best_score": fused_estimates[0].score if fused_estimates else None,
            "config": _dataclass_to_dict(self.config),
        }

        return MultiCamFusionResult(
            fused_estimates=fused_estimates,
            metadata=metadata,
            fusion_candidates=fusion_candidates,
        )

    # -------------------------------------------------------------------------
    # Flatten / Vorbereitung
    # -------------------------------------------------------------------------

    def _flatten_estimates(
        self,
        results_by_camera: dict[str, SingleCamDetectionResult],
    ) -> list[CameraEstimateRef]:
        """
        Holt die relevanten Top-Estimates pro Kamera in eine flache Liste.
        """
        flat: list[CameraEstimateRef] = []

        limit = max(1, int(self.config.max_estimates_per_camera))

        for camera_key, result in results_by_camera.items():
            if result is None:
                continue

            if not isinstance(camera_key, str):
                raise TypeError("Camera keys must be strings.")

            estimates = list(result.scored_estimates[:limit])

            for estimate in estimates:
                if estimate.combined_confidence < self.config.min_combined_confidence:
                    continue

                flat.append(
                    CameraEstimateRef(
                        camera_key=camera_key,
                        rank=int(estimate.rank),
                        label=str(estimate.label),
                        score=int(estimate.score),
                        combined_confidence=float(estimate.combined_confidence),
                        image_point=_coerce_point(estimate.image_point),
                        estimate=estimate,
                    )
                )

        return flat

    # -------------------------------------------------------------------------
    # Fusion-Modi
    # -------------------------------------------------------------------------

    def _fuse_best_single(
        self,
        flat_estimates: list[CameraEstimateRef],
    ) -> tuple[list[MultiCamFusedEstimate], list[FusionCandidate]]:
        """
        Nimmt einfach das beste Einzelresultat.
        """
        best = max(flat_estimates, key=lambda item: item.combined_confidence)

        candidate = self._build_fusion_candidate(best.label, [best])
        fused = self._fusion_candidate_to_fused_estimate(candidate, rank=1)

        return [fused], [candidate]

    def _fuse_majority_vote(
        self,
        flat_estimates: list[CameraEstimateRef],
    ) -> tuple[list[MultiCamFusedEstimate], list[FusionCandidate]]:
        """
        Mehrheit nach Label gewinnt, bei Gleichstand Konfidenz.
        """
        groups = self._group_by_label(flat_estimates)
        candidates = [self._build_fusion_candidate(label, members) for label, members in groups.items()]

        candidates.sort(
            key=lambda candidate: (
                candidate.support_count,
                candidate.total_confidence,
                candidate.mean_confidence,
            ),
            reverse=True,
        )

        fused_estimates = [
            self._fusion_candidate_to_fused_estimate(candidate, rank=index + 1)
            for index, candidate in enumerate(candidates)
        ]
        return fused_estimates, candidates

    def _fuse_score_weighted_vote(
        self,
        flat_estimates: list[CameraEstimateRef],
    ) -> tuple[list[MultiCamFusedEstimate], list[FusionCandidate]]:
        """
        Aggregiert gleiche Labels über Konfidenz + Agreement-Bonus.
        """
        groups = self._group_by_label(flat_estimates)
        candidates = [self._build_fusion_candidate(label, members) for label, members in groups.items()]

        candidates.sort(
            key=lambda candidate: self._fusion_sort_key(candidate),
            reverse=True,
        )

        fused_estimates = [
            self._fusion_candidate_to_fused_estimate(candidate, rank=index + 1)
            for index, candidate in enumerate(candidates)
        ]
        return fused_estimates, candidates

    # -------------------------------------------------------------------------
    # Gruppierung / Scoring
    # -------------------------------------------------------------------------

    def _group_by_label(
        self,
        flat_estimates: list[CameraEstimateRef],
    ) -> dict[str, list[CameraEstimateRef]]:
        """
        Gruppiert flache Estimates nach Label.
        """
        groups: dict[str, list[CameraEstimateRef]] = {}

        for estimate in flat_estimates:
            groups.setdefault(estimate.label, []).append(estimate)

        return groups

    def _build_fusion_candidate(
        self,
        label: str,
        members: list[CameraEstimateRef],
    ) -> FusionCandidate:
        """
        Baut eine Fusionsgruppe für ein Label.
        """
        if not members:
            raise ValueError("Fusion candidate requires at least one member.")

        best_member = max(members, key=lambda member: member.combined_confidence)
        best_estimate = best_member.estimate

        support_count = len({member.camera_key for member in members})
        total_confidence = float(sum(member.combined_confidence for member in members))
        mean_confidence = total_confidence / max(len(members), 1)

        extra_camera_count = max(0, support_count - 1)
        agreement_bonus = self.config.agreement_bonus_per_extra_camera * extra_camera_count
        agreement_bonus = _clip01(agreement_bonus)

        fused_confidence = self._compute_fused_confidence(
            mean_confidence=mean_confidence,
            agreement_bonus=agreement_bonus,
        )

        debug = {}
        if self.config.keep_debug_metadata:
            debug = {
                "camera_keys": sorted({member.camera_key for member in members}),
                "member_count": len(members),
                "best_member_confidence": float(best_member.combined_confidence),
            }

        return FusionCandidate(
            label=label,
            score=int(best_estimate.score),
            ring=str(best_estimate.ring),
            segment=best_estimate.segment,
            multiplier=int(best_estimate.multiplier),
            members=sorted(members, key=lambda m: m.combined_confidence, reverse=True),
            support_count=support_count,
            total_confidence=total_confidence,
            mean_confidence=mean_confidence,
            agreement_bonus=agreement_bonus,
            fused_confidence=fused_confidence,
            debug=debug,
        )

    def _compute_fused_confidence(
        self,
        *,
        mean_confidence: float,
        agreement_bonus: float,
    ) -> float:
        """
        Mischt mittlere Einzelkonfidenz + Agreement-Bonus.
        """
        fused = (
            self.config.weight_camera_estimate_confidence * _clip01(mean_confidence)
            + self.config.weight_label_agreement_bonus * _clip01(agreement_bonus)
        )
        return _clip01(fused)

    def _fusion_sort_key(self, candidate: FusionCandidate) -> tuple[float, float, float]:
        """
        Sortierschlüssel für score_weighted_vote.
        """
        if self.config.prefer_more_cameras_over_raw_confidence:
            return (
                float(candidate.support_count),
                float(candidate.fused_confidence),
                float(candidate.total_confidence),
            )

        return (
            float(candidate.fused_confidence),
            float(candidate.support_count),
            float(candidate.total_confidence),
        )

    def _fusion_candidate_to_fused_estimate(
        self,
        candidate: FusionCandidate,
        *,
        rank: int,
    ) -> MultiCamFusedEstimate:
        """
        Wandelt eine Fusionsgruppe in ein finales Multi-Cam-Ergebnis um.
        """
        best_member = candidate.best_member
        if best_member is None:
            raise ValueError("Fusion candidate has no best_member.")

        return MultiCamFusedEstimate(
            rank=rank,
            label=candidate.label,
            score=candidate.score,
            ring=candidate.ring,
            segment=candidate.segment,
            multiplier=candidate.multiplier,
            fused_confidence=float(candidate.fused_confidence),
            support_count=int(candidate.support_count),
            best_camera_key=best_member.camera_key,
            best_image_point=_coerce_point(best_member.image_point),
            members=list(candidate.members),
            debug={
                "total_confidence": float(candidate.total_confidence),
                "mean_confidence": float(candidate.mean_confidence),
                "agreement_bonus": float(candidate.agreement_bonus),
            },
        )


# -----------------------------------------------------------------------------
# Modulweite Convenience-Funktionen
# -----------------------------------------------------------------------------

def build_multi_cam_fusion(
    config: Optional[MultiCamFusionConfig] = None,
) -> MultiCamFusion:
    """
    Bequemer Builder für die Multi-Cam-Fusion.
    """
    return MultiCamFusion(config=config)


def fuse_multi_cam_results(
    results_by_camera: dict[str, SingleCamDetectionResult],
    *,
    config: Optional[MultiCamFusionConfig] = None,
) -> MultiCamFusionResult:
    """
    Modulweiter Convenience-Wrapper.
    """
    fusion = build_multi_cam_fusion(config=config)
    return fusion.fuse(results_by_camera)


# -----------------------------------------------------------------------------
# Hilfsfunktionen
# -----------------------------------------------------------------------------

def _clip01(value: float) -> float:
    """
    Beschränkt einen Wert auf [0.0, 1.0].
    """
    return float(max(0.0, min(1.0, value)))


def _coerce_point(value: Any) -> PointF:
    """
    Normalisiert einen 2D-Punkt auf (x, y).
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


def _round_point(point: PointF) -> tuple[int, int]:
    """
    Rundet einen Float-Punkt für OpenCV-Zeichenoperationen.
    """
    return int(round(point[0])), int(round(point[1]))


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

    raise ValueError(f"Unsupported frame shape for BGR conversion: {frame.shape}")


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
    "MultiCamFusionConfig",
    "CameraEstimateRef",
    "FusionCandidate",
    "MultiCamFusedEstimate",
    "MultiCamFusionResult",
    "MultiCamFusion",
    "build_multi_cam_fusion",
    "fuse_multi_cam_results",
]