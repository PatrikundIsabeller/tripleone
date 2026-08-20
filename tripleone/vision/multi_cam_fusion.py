from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


PointF = tuple[float, float]


# ============================================================================
# Datenmodelle
# ============================================================================


@dataclass(slots=True)
class CameraFusionObservation:
    """
    Eine verwertbare Beobachtung genau einer Kamera.

    Alle topdown_point-Werte liegen bereits im gemeinsamen Boardraum.
    """

    camera_index: int
    estimate_rank: int

    image_point: PointF
    topdown_point: PointF

    combined_confidence: float
    impact_confidence: float
    candidate_confidence: float

    label: Optional[str] = None
    score: Optional[int] = None
    ring: Optional[str] = None
    segment: Optional[int] = None
    multiplier: Optional[int] = None

    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_index": self.camera_index,
            "estimate_rank": self.estimate_rank,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "combined_confidence": self.combined_confidence,
            "impact_confidence": self.impact_confidence,
            "candidate_confidence": self.candidate_confidence,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "debug": self.debug,
        }


@dataclass(slots=True)
class FusedBoardImpact:
    """
    Final bestätigter Treffer aus einer oder mehreren Kameras.
    """

    topdown_point: PointF

    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int

    confidence: float

    observations_used: list[CameraFusionObservation] = field(
        default_factory=list
    )

    observations_rejected: list[CameraFusionObservation] = field(
        default_factory=list
    )

    debug: dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topdown_point": self.topdown_point,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "confidence": self.confidence,
            "observations_used": [
                observation.to_dict()
                for observation in self.observations_used
            ],
            "observations_rejected": [
                observation.to_dict()
                for observation in self.observations_rejected
            ],
            "debug": self.debug,
        }


@dataclass(slots=True)
class MultiCamFusionConfig:
    """
    Fusion für drei komplementäre Kameras.

    Grundprinzip:

    2 gleiche Labels
        -> Mehrheitsentscheidung

    2 gegen 1
        -> Mehrheit gewinnt

    nur 1 Kamera liefert einen TIP
        -> optional Single-Cam-Fallback

    2 oder 3 verschiedene Labels ohne Mehrheit
        -> keine Bestätigung

    fehlender TIP
        -> neutral
    """

    # Pro Kamera normalerweise nur den besten Treffer verwenden.
    max_estimates_per_camera: int = 1

    # Ab dieser Kameraanzahl gilt ein Label als echte Mehrheitsentscheidung.
    min_cameras_for_fusion: int = 2

    # Eine einzelne Kamera darf einen Treffer bestätigen,
    # wenn alle anderen Kameras keinen TIP liefern.
    allow_single_camera_fallback: bool = True

    # Minimale Confidence für Gewichtungen.
    confidence_floor: float = 0.05

    # Räumliche Plausibilitätsprüfung innerhalb derselben Label-Gruppe.
    #
    # Gleiche Labels können aufgrund kleiner Kalibrierungsunterschiede
    # leicht verschiedene Topdown-Punkte liefern.
    spatial_outlier_distance_px: float = 45.0

    # Wenn nur eine Kamera verwendet wird, muss ihre Confidence
    # mindestens diesen Wert erreichen.
    single_camera_min_confidence: float = 0.08


# ============================================================================
# Hilfsfunktionen
# ============================================================================


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        if value is None:
            return float(default)

        return float(value)

    except Exception:
        return float(default)


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        if value is None:
            return int(default)

        return int(value)

    except Exception:
        return int(default)


def _point_distance(
    a: PointF,
    b: PointF,
) -> float:

    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])

    return float(
        (dx * dx + dy * dy) ** 0.5
    )


def _weighted_average_point(
    points: list[PointF],
    weights: list[float],
) -> PointF:

    if not points:
        raise ValueError(
            "points must not be empty"
        )

    if len(points) != len(weights):
        raise ValueError(
            "points and weights must have same length"
        )

    total_weight = float(
        sum(weights)
    )

    if total_weight <= 1e-9:
        weights = [
            1.0
            for _ in points
        ]

        total_weight = float(
            len(points)
        )

    x = sum(
        float(point[0]) * float(weight)
        for point, weight in zip(
            points,
            weights,
        )
    ) / total_weight

    y = sum(
        float(point[1]) * float(weight)
        for point, weight in zip(
            points,
            weights,
        )
    ) / total_weight

    return (
        float(x),
        float(y),
    )


def _normalize_label(
    label: Any,
) -> Optional[str]:

    if label is None:
        return None

    text = str(label).strip().upper()

    if not text:
        return None

    if text == "MISS":
        return None

    return text


# ============================================================================
# Fusion Engine
# ============================================================================


class MultiCamFusionEngine:
    """
    Mehrkamera-Fusion für TripleOne.

    Die Kameras werden bewusst als komplementäre Sensoren behandelt.

    Eine Kamera darf:
    - einen Treffer liefern
    - keinen Treffer liefern

    Kein TIP ist KEIN Fehler.
    """

    def __init__(
        self,
        config: Optional[MultiCamFusionConfig] = None,
    ) -> None:

        self.config = (
            config
            or MultiCamFusionConfig()
        )

    # ========================================================================
    # ScoreMapper
    # ========================================================================

    def _get_score_mapper_from_detector(
        self,
        detector: Any,
    ) -> Any:

        mapper = getattr(
            detector,
            "_score_mapper",
            None,
        )

        if mapper is not None:
            return mapper

        mapper = getattr(
            detector,
            "score_mapper",
            None,
        )

        if mapper is not None:
            return mapper

        raise RuntimeError(
            "Detector has no ScoreMapper configured."
        )

    def _resolve_score_mapper(
        self,
        *,
        camera_index: int,
        score_mappers_by_camera: Optional[
            dict[int, Any]
        ],
        detectors_by_camera: Optional[
            dict[int, Any]
        ],
    ) -> Any:

        if score_mappers_by_camera is not None:
            mapper = score_mappers_by_camera.get(
                camera_index
            )

            if mapper is not None:
                return mapper

        if detectors_by_camera is not None:
            detector = detectors_by_camera.get(
                camera_index
            )

            if detector is not None:
                return (
                    self._get_score_mapper_from_detector(
                        detector
                    )
                )

        raise RuntimeError(
            f"No ScoreMapper available for camera "
            f"{camera_index}."
        )

    # ========================================================================
    # Observation Extraction
    # ========================================================================

    def _estimate_to_fusion_observation(
        self,
        *,
        camera_index: int,
        estimate: Any,
    ) -> Optional[CameraFusionObservation]:

        image_point = getattr(
            estimate,
            "image_point",
            None,
        )

        topdown_point = getattr(
            estimate,
            "topdown_point",
            None,
        )

        if (
            image_point is None
            or topdown_point is None
        ):
            return None

        label = _normalize_label(
            getattr(
                estimate,
                "label",
                None,
            )
        )

        if label is None:
            return None

        return CameraFusionObservation(
            camera_index=int(camera_index),

            estimate_rank=_safe_int(
                getattr(
                    estimate,
                    "estimate_rank",
                    1,
                ),
                1,
            ),

            image_point=(
                float(image_point[0]),
                float(image_point[1]),
            ),

            topdown_point=(
                float(topdown_point[0]),
                float(topdown_point[1]),
            ),

            combined_confidence=_safe_float(
                getattr(
                    estimate,
                    "combined_confidence",
                    0.0,
                )
            ),

            impact_confidence=_safe_float(
                getattr(
                    estimate,
                    "impact_confidence",
                    0.0,
                )
            ),

            candidate_confidence=_safe_float(
                getattr(
                    estimate,
                    "candidate_confidence",
                    0.0,
                )
            ),

            label=label,

            score=getattr(
                estimate,
                "score",
                None,
            ),

            ring=getattr(
                estimate,
                "ring",
                None,
            ),

            segment=getattr(
                estimate,
                "segment",
                None,
            ),

            multiplier=getattr(
                estimate,
                "multiplier",
                None,
            ),

            debug=dict(
                getattr(
                    estimate,
                    "debug",
                    {},
                )
                or {}
            ),
        )

    def _extract_camera_observations(
        self,
        *,
        camera_index: int,
        observation: Any,
    ) -> list[CameraFusionObservation]:

        out: list[
            CameraFusionObservation
        ] = []

        if observation is None:
            return out

        if not bool(
            getattr(
                observation,
                "frame_ok",
                True,
            )
        ):
            return out

        if not bool(
            getattr(
                observation,
                "detector_ready",
                True,
            )
        ):
            return out

        if not bool(
            getattr(
                observation,
                "reference_available",
                True,
            )
        ):
            return out

        estimates = list(
            getattr(
                observation,
                "estimates",
                [],
            )
            or []
        )

        max_count = max(
            1,
            int(
                self.config
                .max_estimates_per_camera
            ),
        )

        for estimate in estimates[:max_count]:
            converted = (
                self._estimate_to_fusion_observation(
                    camera_index=camera_index,
                    estimate=estimate,
                )
            )

            if converted is not None:
                out.append(
                    converted
                )

        return out

    # ========================================================================
    # Gruppenbildung nach Score-Label
    # ========================================================================

    def _group_by_label(
        self,
        observations: list[
            CameraFusionObservation
        ],
    ) -> dict[
        str,
        list[CameraFusionObservation],
    ]:

        groups: dict[
            str,
            list[
                CameraFusionObservation
            ],
        ] = {}

        for observation in observations:
            label = _normalize_label(
                observation.label
            )

            if label is None:
                continue

            groups.setdefault(
                label,
                [],
            ).append(
                observation
            )

        return groups

    def _unique_camera_count(
        self,
        observations: list[
            CameraFusionObservation
        ],
    ) -> int:

        return len(
            {
                observation.camera_index
                for observation in observations
            }
        )

    def _group_confidence(
        self,
        observations: list[
            CameraFusionObservation
        ],
    ) -> float:

        if not observations:
            return 0.0

        values = [
            max(
                float(
                    observation.combined_confidence
                ),
                float(
                    self.config.confidence_floor
                ),
            )
            for observation in observations
        ]

        return float(
            sum(values)
            / len(values)
        )

    # ========================================================================
    # Räumliche Plausibilisierung
    # ========================================================================

    def _filter_spatial_outliers(
        self,
        observations: list[
            CameraFusionObservation
        ],
    ) -> tuple[
        list[CameraFusionObservation],
        list[CameraFusionObservation],
    ]:

        if len(observations) <= 2:
            return (
                list(observations),
                [],
            )

        # ------------------------------------------------------------
        # Bei drei Kameras:
        # den Punkt suchen, der am besten zu den anderen passt.
        # ------------------------------------------------------------

        distance_limit = float(
            self.config
            .spatial_outlier_distance_px
        )

        used: list[
            CameraFusionObservation
        ] = []

        rejected: list[
            CameraFusionObservation
        ] = []

        for observation in observations:

            neighbours = 0

            for other in observations:

                if other is observation:
                    continue

                if (
                    _point_distance(
                        observation.topdown_point,
                        other.topdown_point,
                    )
                    <= distance_limit
                ):
                    neighbours += 1

            if neighbours >= 1:
                used.append(
                    observation
                )
            else:
                rejected.append(
                    observation
                )

        # Falls alle räumlich auseinanderliegen,
        # nicht künstlich alles verwerfen.
        #
        # Das Label selbst hat bereits Mehrheitskonsens.
        if not used:
            return (
                list(observations),
                [],
            )

        return (
            used,
            rejected,
        )

    # ========================================================================
    # Fusion
    # ========================================================================

    def fuse(
        self,
        *,
        observations_by_camera: dict[
            int,
            Any,
        ],
        score_mappers_by_camera: Optional[
            dict[int, Any]
        ] = None,
        detectors_by_camera: Optional[
            dict[int, Any]
        ] = None,
    ) -> Optional[FusedBoardImpact]:

        if (
            score_mappers_by_camera is None
            and detectors_by_camera is None
        ):
            raise RuntimeError(
                "Either score_mappers_by_camera "
                "or detectors_by_camera "
                "must be provided."
            )

        # ------------------------------------------------------------
        # 1. Alle vorhandenen Kamera-Beobachtungen sammeln.
        #
        # Kameras ohne TIP tauchen hier bewusst überhaupt nicht auf.
        # ------------------------------------------------------------

        all_observations: list[
            CameraFusionObservation
        ] = []

        for (
            camera_index,
            observation,
        ) in observations_by_camera.items():

            extracted = (
                self._extract_camera_observations(
                    camera_index=int(
                        camera_index
                    ),
                    observation=observation,
                )
            )

            all_observations.extend(
                extracted
            )

        if not all_observations:
            return None

        # ------------------------------------------------------------
        # 2. Pro Kamera nur die stärkste Observation verwenden.
        #
        # Dadurch kann eine Kamera nicht mehrfach "abstimmen".
        # ------------------------------------------------------------

        best_by_camera: dict[
            int,
            CameraFusionObservation,
        ] = {}

        for observation in all_observations:

            existing = best_by_camera.get(
                observation.camera_index
            )

            if (
                existing is None
                or observation.combined_confidence
                > existing.combined_confidence
            ):
                best_by_camera[
                    observation.camera_index
                ] = observation

        camera_observations = list(
            best_by_camera.values()
        )

        # ------------------------------------------------------------
        # 3. Nach Label gruppieren.
        # ------------------------------------------------------------

        label_groups = (
            self._group_by_label(
                camera_observations
            )
        )

        if not label_groups:
            return None

        # ------------------------------------------------------------
        # 4. Gruppen nach:
        #
        #    1. Anzahl verschiedener Kameras
        #    2. mittlere Confidence
        #
        # sortieren.
        # ------------------------------------------------------------

        ranked_groups = sorted(
            label_groups.items(),
            key=lambda item: (
                self._unique_camera_count(
                    item[1]
                ),
                self._group_confidence(
                    item[1]
                ),
            ),
            reverse=True,
        )

        winning_label, winning_group = (
            ranked_groups[0]
        )

        winning_camera_count = (
            self._unique_camera_count(
                winning_group
            )
        )

        # ------------------------------------------------------------
        # 5. Mehrheitsentscheidung
        # ------------------------------------------------------------

        majority_required = max(
            2,
            int(
                self.config
                .min_cameras_for_fusion
            ),
        )

        fusion_mode = ""

        if (
            winning_camera_count
            >= majority_required
        ):
            # --------------------------------------------------------
            # Beispiel:
            #
            # K1 T20
            # K2 T20
            # K3 S20
            #
            # -> T20
            # --------------------------------------------------------

            fusion_mode = (
                "label_majority"
            )

        else:
            # --------------------------------------------------------
            # Keine Mehrheit.
            #
            # Single-Cam-Fallback ist NUR erlaubt, wenn tatsächlich
            # nur eine einzige Kamera überhaupt einen TIP geliefert hat.
            #
            # Zwei verschiedene Kameras mit zwei verschiedenen Labels
            # dürfen NICHT durch Confidence entschieden werden.
            # --------------------------------------------------------

            unique_reporting_cameras = {
                observation.camera_index
                for observation
                in camera_observations
            }

            if (
                len(
                    unique_reporting_cameras
                )
                != 1
            ):
                return None

            if not bool(
                self.config
                .allow_single_camera_fallback
            ):
                return None

            only_observation = (
                winning_group[0]
            )

            if (
                float(
                    only_observation
                    .combined_confidence
                )
                <
                float(
                    self.config
                    .single_camera_min_confidence
                )
            ):
                return None

            fusion_mode = (
                "single_camera_fallback"
            )

        # ------------------------------------------------------------
        # 6. Räumliche Ausreißer innerhalb der Gewinnergruppe
        # entfernen.
        # ------------------------------------------------------------

        used, spatial_rejected = (
            self._filter_spatial_outliers(
                winning_group
            )
        )

        if not used:
            return None

        # ------------------------------------------------------------
        # 7. Alle Observationen anderer Labels gelten als verworfen.
        # ------------------------------------------------------------

        rejected: list[
            CameraFusionObservation
        ] = []

        rejected.extend(
            spatial_rejected
        )

        for observation in camera_observations:

            if observation in winning_group:
                continue

            rejected.append(
                observation
            )

        # ------------------------------------------------------------
        # 8. Gemeinsamen Topdown-Punkt bilden.
        # ------------------------------------------------------------

        weights = [
            max(
                float(
                    observation
                    .combined_confidence
                ),
                float(
                    self.config
                    .confidence_floor
                ),
            )
            for observation in used
        ]

        fused_point = (
            _weighted_average_point(
                [
                    observation.topdown_point
                    for observation in used
                ],
                weights,
            )
        )

        # ------------------------------------------------------------
        # 9. ScoreMapper benutzen.
        # ------------------------------------------------------------

        reference_camera_index = (
            used[0].camera_index
        )

        score_mapper = (
            self._resolve_score_mapper(
                camera_index=reference_camera_index,
                score_mappers_by_camera=(
                    score_mappers_by_camera
                ),
                detectors_by_camera=(
                    detectors_by_camera
                ),
            )
        )

        if not hasattr(
            score_mapper,
            "score_topdown_point",
        ):
            raise RuntimeError(
                "ScoreMapper has no method "
                "score_topdown_point(...)."
            )

        scored_hit = (
            score_mapper
            .score_topdown_point(
                fused_point
            )
        )

        scored_label = _normalize_label(
            getattr(
                scored_hit,
                "label",
                None,
            )
        )

        # ------------------------------------------------------------
        # 10. Sicherheitsnetz:
        #
        # Wenn der Mittelwert durch einen Draht knapp in ein
        # Nachbarfeld fällt, obwohl die Kameras dasselbe Label
        # gemeldet haben, verwenden wir den stärksten Originalpunkt.
        # ------------------------------------------------------------

        if (
            scored_label is None
            or scored_label != winning_label
        ):
            strongest = max(
                used,
                key=lambda observation: float(
                    observation
                    .combined_confidence
                ),
            )

            fused_point = (
                strongest.topdown_point
            )

            scored_hit = (
                score_mapper
                .score_topdown_point(
                    fused_point
                )
            )

            scored_label = (
                _normalize_label(
                    getattr(
                        scored_hit,
                        "label",
                        None,
                    )
                )
            )

        # ------------------------------------------------------------
        # Wenn selbst der stärkste Originalpunkt nicht zum
        # Mehrheitslabel passt, vertrauen wir dem Kamerakonsens.
        #
        # Score/Ring/Segment stammen dann aus der stärksten
        # Observation der Gewinnergruppe.
        # ------------------------------------------------------------

        strongest = max(
            used,
            key=lambda observation: float(
                observation
                .combined_confidence
            ),
        )

        if scored_label == winning_label:

            final_label = str(
                getattr(
                    scored_hit,
                    "label",
                )
            )

            final_score = int(
                getattr(
                    scored_hit,
                    "score",
                )
            )

            final_ring = str(
                getattr(
                    scored_hit,
                    "ring",
                )
            )

            scored_segment = getattr(
                scored_hit,
                "segment",
                None,
            )

            final_segment = (
                None
                if scored_segment is None
                else int(
                    scored_segment
                )
            )

            final_multiplier = int(
                getattr(
                    scored_hit,
                    "multiplier",
                )
            )

        else:

            final_label = str(
                winning_label
            )

            final_score = _safe_int(
                strongest.score
            )

            final_ring = str(
                strongest.ring
                or ""
            )

            final_segment = (
                None
                if strongest.segment is None
                else int(
                    strongest.segment
                )
            )

            final_multiplier = (
                _safe_int(
                    strongest.multiplier,
                    1,
                )
            )

        # ------------------------------------------------------------
        # 11. Confidence berechnen
        # ------------------------------------------------------------

        confidence_values = [
            max(
                float(
                    observation
                    .combined_confidence
                ),
                float(
                    self.config
                    .confidence_floor
                ),
            )
            for observation in used
        ]

        fusion_confidence = float(
            sum(
                confidence_values
            )
            / len(
                confidence_values
            )
        )

        # Kleine Mehrheits-Belohnung
        if len(
            {
                observation.camera_index
                for observation in used
            }
        ) >= 2:
            fusion_confidence += 0.05

        fusion_confidence = max(
            0.0,
            min(
                1.0,
                fusion_confidence,
            ),
        )

        # ------------------------------------------------------------
        # 12. Finales Ergebnis
        # ------------------------------------------------------------

        return FusedBoardImpact(
            topdown_point=(
                float(
                    fused_point[0]
                ),
                float(
                    fused_point[1]
                ),
            ),

            label=final_label,
            score=final_score,
            ring=final_ring,
            segment=final_segment,
            multiplier=final_multiplier,

            confidence=(
                fusion_confidence
            ),

            observations_used=used,

            observations_rejected=(
                rejected
            ),

            debug={
                "fusion_mode": (
                    fusion_mode
                ),

                "winning_label": (
                    winning_label
                ),

                "winning_camera_count": (
                    winning_camera_count
                ),

                "reporting_camera_count": len(
                    {
                        observation.camera_index
                        for observation
                        in camera_observations
                    }
                ),

                "all_labels": {
                    label: [
                        observation.camera_index
                        for observation
                        in observations
                    ]
                    for label, observations
                    in label_groups.items()
                },

                "used_cameras": [
                    observation.camera_index
                    for observation in used
                ],

                "rejected_cameras": [
                    observation.camera_index
                    for observation in rejected
                ],

                "used_observations": [
                    observation.to_dict()
                    for observation in used
                ],

                "rejected_observations": [
                    observation.to_dict()
                    for observation in rejected
                ],
            },
        )