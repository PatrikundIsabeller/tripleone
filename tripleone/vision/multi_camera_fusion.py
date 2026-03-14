# vision/multi_camera_fusion.py
# Diese Datei enthält die Mehrkamera-Fusion für TripleOne.
#
# Phase 5.0:
# - mehrere Kamera-Treffer zusammenführen
# - robuste Fusion über Board-Koordinaten
# - Median statt Mittelwert gegen Ausreißer
# - finalen Score aus fusioniertem Boardpunkt berechnen

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List

from vision.board_model import calculate_board_hit_from_board_point


@dataclass
class CameraBoardCandidate:
    """
    Einzelner Trefferkandidat einer Kamera im normierten Boardraum.
    """
    camera_index: int
    score_label: str
    score_value: int
    ring_name: str
    board_x: float
    board_y: float
    radius: float


@dataclass
class FusedDartResult:
    """
    Ergebnis einer Mehrkamera-Fusion.
    """
    dart_index: int
    final_label: str
    final_score: int
    final_ring_name: str
    fused_board_x: float
    fused_board_y: float
    fused_radius: float
    camera_candidates: List[CameraBoardCandidate]


def fuse_camera_candidates(
    dart_index: int,
    candidates: List[CameraBoardCandidate],
) -> FusedDartResult:
    """
    Führt mehrere Kamera-Kandidaten über den Median der Board-Koordinaten zusammen.
    """
    if not candidates:
        raise ValueError("Für die Fusion werden mindestens 1 Kamera-Kandidat benötigt.")

    xs = [c.board_x for c in candidates]
    ys = [c.board_y for c in candidates]

    fused_x = float(median(xs))
    fused_y = float(median(ys))

    final_hit = calculate_board_hit_from_board_point(fused_x, fused_y)

    return FusedDartResult(
        dart_index=dart_index,
        final_label=final_hit.label,
        final_score=final_hit.score,
        final_ring_name=final_hit.ring_name,
        fused_board_x=fused_x,
        fused_board_y=fused_y,
        fused_radius=final_hit.radius,
        camera_candidates=candidates,
    )
