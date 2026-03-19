# vision/score_mapper.py
# Zweck:
# Diese Datei bildet eine saubere Scoring-Schicht auf Basis der zentralen Geometrie.
# Sie enthält bewusst KEINE eigene Board-Geometrie, sondern verwendet ausschließlich
# die Funktionen aus vision/calibration_geometry.py.
#
# Aufgaben dieser Datei:
# - zentrale Treffer-Normalisierung (MISS, S20, D20, T20, SBULL, DBULL)
# - Umrechnung von Hit-Labels in numerische Scores
# - Scoring für Bildpunkte und Top-Down-Punkte
# - optionale Projektion Bild <-> Top-Down für Debugging / UI
# - dünne, robuste Wrapper-Schicht für andere Module
#
# Wichtige Regel:
# Diese Datei darf niemals anfangen, eigene Ringgrenzen oder Sektorlogik zu berechnen.
# Sobald das passiert, ist die "eine Wahrheit" wieder kaputt.

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# Import robust halten:
# - relative Importe für normalen Paketbetrieb
# - absoluter Fallback für direkte Ausführung / Tests
try:
    from .calibration_geometry import (
        build_pipeline_points,
        calculate_hit_from_image_point,
        calculate_hit_from_topdown_point,
        project_image_points_to_topdown,
        project_topdown_points_to_image,
    )
except ImportError:  # pragma: no cover
    from vision.calibration_geometry import (  # type: ignore
        build_pipeline_points,
        calculate_hit_from_image_point,
        calculate_hit_from_topdown_point,
        project_image_points_to_topdown,
        project_topdown_points_to_image,
    )


# --------------------------------------------------------------------------------------
# Basis-Typen
# --------------------------------------------------------------------------------------

Point = tuple[float, float]


# --------------------------------------------------------------------------------------
# Kleine Hilfsfunktionen für robuste Punkt- und Daten-Normalisierung
# --------------------------------------------------------------------------------------

def _coerce_point(value: Any) -> Point:
    """
    Wandelt unterschiedliche Punktformate in ein (x, y)-Tuple um.

    Unterstützte Formate:
    - (x, y)
    - [x, y]
    - {"x": ..., "y": ...}
    - Objekt mit .x und .y
    """
    if value is None:
        raise ValueError("Point is None.")

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        raise ValueError(f"Point dict must contain 'x' and 'y', got: {value}")

    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Point sequence must have length 2, got: {value}")
        return float(value[0]), float(value[1])

    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)

    raise ValueError(f"Unsupported point format: {value!r}")


def _coerce_points(values: Iterable[Any]) -> list[Point]:
    """
    Wandelt eine iterable Punktliste in eine saubere Liste von (x, y)-Tuples um.
    """
    return [_coerce_point(v) for v in values]


def _extract_manual_points_from_record(record: Any) -> list[Point]:
    """
    Extrahiert die 4 manuellen Marker aus einem CameraCalibrationRecord oder Dict.

    Wichtige Designentscheidung:
    score_mapper.py normalisiert NICHT still heimlich alte Formate.
    Dafür ist calibration_storage.py zuständig.
    Hier wird absichtlich streng gearbeitet, damit Fehler früh sichtbar werden.
    """
    candidate_names = [
        "manual_points",
        "marker_points",
        "markers",
        "image_points",
        "points",
    ]

    # Dict-Fall
    if isinstance(record, dict):
        for name in candidate_names:
            if name in record and record[name]:
                points = _coerce_points(record[name])
                if len(points) != 4:
                    raise ValueError(
                        f"Expected exactly 4 manual points in record['{name}'], got {len(points)}. "
                        "Normalize old formats in calibration_storage.py first."
                    )
                return points

        # Fallback: p1..p4
        p_names = ["p1", "p2", "p3", "p4"]
        if all(name in record for name in p_names):
            points = _coerce_points([record[name] for name in p_names])
            if len(points) != 4:
                raise ValueError("Record p1..p4 extraction failed.")
            return points

        raise ValueError("Could not extract manual points from calibration record dict.")

    # Objekt-Fall
    for name in candidate_names:
        if hasattr(record, name):
            value = getattr(record, name)
            if value:
                points = _coerce_points(value)
                if len(points) != 4:
                    raise ValueError(
                        f"Expected exactly 4 manual points in record.{name}, got {len(points)}. "
                        "Normalize old formats in calibration_storage.py first."
                    )
                return points

    p_names = ["p1", "p2", "p3", "p4"]
    if all(hasattr(record, name) for name in p_names):
        points = _coerce_points([getattr(record, name) for name in p_names])
        if len(points) != 4:
            raise ValueError("Record p1..p4 extraction failed.")
        return points

    raise ValueError("Could not extract manual points from calibration record object.")


def _extract_optional_image_size(record: Any) -> Optional[tuple[int, int]]:
    """
    Extrahiert optional eine Bildgröße aus einem Record.
    Diese Information wird nur weitergereicht, wenn build_pipeline_points sie unterstützt.
    """
    candidate_names = [
        "image_size",
        "frame_size",
        "reference_image_size",
    ]

    if isinstance(record, dict):
        for name in candidate_names:
            if name in record and record[name]:
                value = record[name]
                if isinstance(value, (tuple, list)) and len(value) == 2:
                    return int(value[0]), int(value[1])

        width = record.get("image_width")
        height = record.get("image_height")
        if width is not None and height is not None:
            return int(width), int(height)
        return None

    for name in candidate_names:
        if hasattr(record, name):
            value = getattr(record, name)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return int(value[0]), int(value[1])

    width = getattr(record, "image_width", None)
    height = getattr(record, "image_height", None)
    if width is not None and height is not None:
        return int(width), int(height)

    return None


# --------------------------------------------------------------------------------------
# Robuste Funktionsaufrufe gegen calibration_geometry.py
# --------------------------------------------------------------------------------------

def _filter_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filtert nur die kwargs, die eine Ziel-Funktion tatsächlich unterstützt.
    Das macht score_mapper.py toleranter gegenüber kleinen API-Änderungen
    in calibration_geometry.py.
    """
    sig = inspect.signature(func)
    supported = {}

    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ) and name in kwargs:
            supported[name] = kwargs[name]

    return supported


def _try_named_then_positional(
    func: Any,
    named_candidates: Sequence[dict[str, Any]],
    positional_candidates: Sequence[tuple[Any, ...]],
) -> Any:
    """
    Versucht mehrere Call-Varianten nacheinander.
    Das ist bewusst defensiv, damit kleine Signatur-Unterschiede nicht
    die ganze Architektur blockieren.
    """
    last_error: Optional[Exception] = None

    for kwargs in named_candidates:
        try:
            filtered = _filter_supported_kwargs(func, kwargs)
            return func(**filtered)
        except TypeError as exc:
            last_error = exc

    for args in positional_candidates:
        try:
            return func(*args)
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        f"Could not call function '{getattr(func, '__name__', repr(func))}' "
        f"with supported arguments. Last error: {last_error}"
    )


def _build_pipeline(
    *,
    manual_points: Sequence[Point],
    image_size: Optional[tuple[int, int]] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Baut die zentrale Pipeline-Geometrie über calibration_geometry.build_pipeline_points().

    Wichtig:
    - diese Funktion rechnet NICHT selbst Geometrie
    - sie reicht nur die 4 Marker an die zentrale Geometrie weiter
    """
    if len(manual_points) != 4:
        raise ValueError(
            f"Expected exactly 4 manual points, got {len(manual_points)}. "
            "score_mapper.py supports only the 4-point calibration architecture."
        )

    manual_points = _coerce_points(manual_points)
    extra_kwargs = extra_kwargs or {}

    width = None
    height = None
    if image_size:
        width, height = int(image_size[0]), int(image_size[1])

    named_candidates = [
        {
            "manual_points": manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "image_points": manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "marker_points": manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "markers": manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "points": manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
    ]

    positional_candidates = [
        (manual_points,),
    ]

    return _try_named_then_positional(
        build_pipeline_points,
        named_candidates=named_candidates,
        positional_candidates=positional_candidates,
    )


def _project_image_points(
    points: Sequence[Point],
    pipeline: Any,
) -> Any:
    """
    Projiziert Bildpunkte nach Top-Down über die zentrale Geometrie.
    """
    named_candidates = [
        {"image_points": points, "pipeline": pipeline},
        {"points": points, "pipeline": pipeline},
        {"pts": points, "pipeline": pipeline},
        {"image_points": points, "geometry": pipeline},
        {"points": points, "geometry": pipeline},
        {"image_points": points, "context": pipeline},
        {"points": points, "context": pipeline},
    ]

    positional_candidates = [
        (points, pipeline),
        (pipeline, points),
    ]

    return _try_named_then_positional(
        project_image_points_to_topdown,
        named_candidates=named_candidates,
        positional_candidates=positional_candidates,
    )


def _project_topdown_points(
    points: Sequence[Point],
    pipeline: Any,
) -> Any:
    """
    Projiziert Top-Down-Punkte zurück ins Kamerabild über die zentrale Geometrie.
    """
    named_candidates = [
        {"topdown_points": points, "pipeline": pipeline},
        {"points": points, "pipeline": pipeline},
        {"pts": points, "pipeline": pipeline},
        {"topdown_points": points, "geometry": pipeline},
        {"points": points, "geometry": pipeline},
        {"topdown_points": points, "context": pipeline},
        {"points": points, "context": pipeline},
    ]

    positional_candidates = [
        (points, pipeline),
        (pipeline, points),
    ]

    return _try_named_then_positional(
        project_topdown_points_to_image,
        named_candidates=named_candidates,
        positional_candidates=positional_candidates,
    )


def _calculate_hit_from_image(
    point: Point,
    pipeline: Any,
) -> Any:
    """
    Berechnet einen Treffer für einen Bildpunkt ausschließlich über calibration_geometry.
    """
    named_candidates = [
        {"image_point": point, "pipeline": pipeline},
        {"point": point, "pipeline": pipeline},
        {"pt": point, "pipeline": pipeline},
        {"image_point": point, "geometry": pipeline},
        {"point": point, "geometry": pipeline},
        {"image_point": point, "context": pipeline},
        {"point": point, "context": pipeline},
    ]

    positional_candidates = [
        (point, pipeline),
        (pipeline, point),
    ]

    return _try_named_then_positional(
        calculate_hit_from_image_point,
        named_candidates=named_candidates,
        positional_candidates=positional_candidates,
    )


def _calculate_hit_from_topdown(
    point: Point,
    pipeline: Any,
) -> Any:
    """
    Berechnet einen Treffer für einen Top-Down-Punkt ausschließlich über calibration_geometry.
    """
    named_candidates = [
        {"topdown_point": point, "pipeline": pipeline},
        {"point": point, "pipeline": pipeline},
        {"pt": point, "pipeline": pipeline},
        {"topdown_point": point, "geometry": pipeline},
        {"point": point, "geometry": pipeline},
        {"topdown_point": point, "context": pipeline},
        {"point": point, "context": pipeline},
    ]

    positional_candidates = [
        (point, pipeline),
        (pipeline, point),
    ]

    return _try_named_then_positional(
        calculate_hit_from_topdown_point,
        named_candidates=named_candidates,
        positional_candidates=positional_candidates,
    )


# --------------------------------------------------------------------------------------
# Hit-Label-Normalisierung und Score-Umrechnung
# --------------------------------------------------------------------------------------

_HIT_RE = re.compile(r"^(S|D|T)?(\d{1,2})$")


def normalize_hit_label(label: str) -> str:
    """
    Normalisiert Hit-Labels auf das interne Standardformat.

    Erlaubte Zielformate:
    - MISS
    - SBULL
    - DBULL
    - S1..S20
    - D1..D20
    - T1..T20

    Beispiele:
    - "20" -> "S20"
    - "d20" -> "D20"
    - "double20" -> "D20"
    - "bull" -> "SBULL"
    """
    if label is None:
        raise ValueError("Hit label is None.")

    clean = str(label).strip().upper().replace(" ", "").replace("-", "")
    clean = clean.replace("DOUBLE", "D")
    clean = clean.replace("TRIPLE", "T")
    clean = clean.replace("SINGLE", "S")
    clean = clean.replace("OUTERBULL", "SBULL")
    clean = clean.replace("INNERBULL", "DBULL")

    if clean in {"MISS", "M", "OUT"}:
        return "MISS"

    if clean in {"BULL", "SBULL", "S25", "25"}:
        return "SBULL"

    if clean in {"DBULL", "BULLSEYE", "BULL50", "D50", "50"}:
        return "DBULL"

    match = _HIT_RE.match(clean)
    if not match:
        raise ValueError(f"Unsupported hit label format: {label!r}")

    ring = match.group(1) or "S"
    segment = int(match.group(2))

    if not 1 <= segment <= 20:
        raise ValueError(f"Segment out of range in hit label: {label!r}")

    return f"{ring}{segment}"


def hit_label_to_score(label: str) -> int:
    """
    Rechnet ein Hit-Label in den numerischen Dart-Score um.
    """
    normalized = normalize_hit_label(label)

    if normalized == "MISS":
        return 0
    if normalized == "SBULL":
        return 25
    if normalized == "DBULL":
        return 50

    ring = normalized[0]
    segment = int(normalized[1:])

    if ring == "S":
        return segment
    if ring == "D":
        return segment * 2
    if ring == "T":
        return segment * 3

    raise ValueError(f"Unsupported normalized hit label: {normalized}")


def hit_label_to_multiplier(label: str) -> int:
    """
    Liefert den Multiplikator für ein Hit-Label.
    """
    normalized = normalize_hit_label(label)

    if normalized == "MISS":
        return 0
    if normalized == "SBULL":
        return 1
    if normalized == "DBULL":
        return 2

    ring = normalized[0]
    if ring == "S":
        return 1
    if ring == "D":
        return 2
    if ring == "T":
        return 3

    raise ValueError(f"Unsupported normalized hit label: {normalized}")


def hit_label_to_ring(label: str) -> str:
    """
    Liefert den Ring-Typ aus einem Hit-Label.
    """
    normalized = normalize_hit_label(label)

    if normalized == "MISS":
        return "MISS"
    if normalized == "SBULL":
        return "SBULL"
    if normalized == "DBULL":
        return "DBULL"

    return normalized[0]


def hit_label_to_segment(label: str) -> Optional[int]:
    """
    Liefert die Sektorzahl für ein Hit-Label.
    Für MISS / Bulls gibt es keinen normalen Sektor.
    """
    normalized = normalize_hit_label(label)

    if normalized in {"MISS", "SBULL", "DBULL"}:
        return None

    return int(normalized[1:])


def _extract_label_from_raw_hit(raw_hit: Any) -> str:
    """
    Extrahiert best-effort ein Hit-Label aus unterschiedlichen Rückgabeformaten
    von calibration_geometry.py.

    Unterstützte typische Formen:
    - "D20"
    - {"label": "D20"}
    - {"hit": "D20"}
    - {"ring": "D", "segment": 20}
    - {"multiplier": 2, "segment": 20}
    - ("D", 20)
    - Objekt mit .label / .ring / .segment / .multiplier
    """
    # Direktes Label
    if isinstance(raw_hit, str):
        return normalize_hit_label(raw_hit)

    # Tuple / Liste
    if isinstance(raw_hit, (tuple, list)):
        if len(raw_hit) == 1:
            return normalize_hit_label(str(raw_hit[0]))

        if len(raw_hit) >= 2:
            first = raw_hit[0]
            second = raw_hit[1]

            if isinstance(first, str) and isinstance(second, (int, float)):
                first_clean = first.strip().upper()
                if first_clean in {"S", "D", "T"}:
                    return normalize_hit_label(f"{first_clean}{int(second)}")
                return normalize_hit_label(first_clean)

    # Dict
    if isinstance(raw_hit, dict):
        for key in ("label", "hit", "code", "result", "field"):
            if key in raw_hit and raw_hit[key] is not None:
                return normalize_hit_label(str(raw_hit[key]))

        ring = raw_hit.get("ring")
        segment = raw_hit.get("segment", raw_hit.get("number"))
        if ring is not None and segment is not None:
            return normalize_hit_label(f"{str(ring)}{int(segment)}")

        multiplier = raw_hit.get("multiplier")
        segment = raw_hit.get("segment", raw_hit.get("number"))
        if multiplier is not None and segment is not None:
            multiplier = int(multiplier)
            segment = int(segment)

            if multiplier == 1:
                return normalize_hit_label(f"S{segment}")
            if multiplier == 2:
                return normalize_hit_label(f"D{segment}")
            if multiplier == 3:
                return normalize_hit_label(f"T{segment}")

    # Objekt mit Attributen
    for attr in ("label", "hit", "code", "result", "field"):
        if hasattr(raw_hit, attr):
            value = getattr(raw_hit, attr)
            if value is not None:
                return normalize_hit_label(str(value))

    ring = getattr(raw_hit, "ring", None)
    segment = getattr(raw_hit, "segment", getattr(raw_hit, "number", None))
    if ring is not None and segment is not None:
        return normalize_hit_label(f"{str(ring)}{int(segment)}")

    multiplier = getattr(raw_hit, "multiplier", None)
    segment = getattr(raw_hit, "segment", getattr(raw_hit, "number", None))
    if multiplier is not None and segment is not None:
        multiplier = int(multiplier)
        segment = int(segment)
        if multiplier == 1:
            return normalize_hit_label(f"S{segment}")
        if multiplier == 2:
            return normalize_hit_label(f"D{segment}")
        if multiplier == 3:
            return normalize_hit_label(f"T{segment}")

    raise ValueError(f"Could not extract hit label from raw hit result: {raw_hit!r}")


# --------------------------------------------------------------------------------------
# Ergebnis-Modell
# --------------------------------------------------------------------------------------

@dataclass(slots=True)
class ScoredHit:
    """
    Einheitliches Ergebnisobjekt für einen Treffer.
    """
    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int
    source_space: str  # "image" oder "topdown"
    image_point: Optional[Point] = None
    topdown_point: Optional[Point] = None
    raw_hit: Any = None

    @property
    def is_miss(self) -> bool:
        return self.label == "MISS"

    @property
    def is_bull(self) -> bool:
        return self.label in {"SBULL", "DBULL"}

    def to_dict(self) -> dict[str, Any]:
        """
        Praktische Debug-/API-Darstellung.
        """
        return {
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "source_space": self.source_space,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "is_miss": self.is_miss,
            "is_bull": self.is_bull,
        }


def _build_scored_hit(
    *,
    raw_hit: Any,
    source_space: str,
    image_point: Optional[Point] = None,
    topdown_point: Optional[Point] = None,
) -> ScoredHit:
    """
    Baut aus einem Roh-Resultat ein stabiles ScoredHit-Objekt.
    """
    label = _extract_label_from_raw_hit(raw_hit)

    return ScoredHit(
        label=label,
        score=hit_label_to_score(label),
        ring=hit_label_to_ring(label),
        segment=hit_label_to_segment(label),
        multiplier=hit_label_to_multiplier(label),
        source_space=source_space,
        image_point=image_point,
        topdown_point=topdown_point,
        raw_hit=raw_hit,
    )


# --------------------------------------------------------------------------------------
# Hauptklasse: ScoreMapper
# --------------------------------------------------------------------------------------

class ScoreMapper:
    """
    Dünne Mapping-Schicht auf Basis der zentralen Geometrie.

    Diese Klasse cached die Pipeline-Geometrie und bietet stabile Methoden für:
    - Bildpunkt -> Treffer
    - Top-Down-Punkt -> Treffer
    - Bildpunkt -> Top-Down
    - Top-Down -> Bildpunkt

    Wichtige Architekturregel:
    ScoreMapper besitzt keine eigene Board-Logik.
    """

    def __init__(
        self,
        *,
        manual_points: Optional[Sequence[Any]] = None,
        calibration_record: Optional[Any] = None,
        pipeline: Optional[Any] = None,
        image_size: Optional[tuple[int, int]] = None,
        pipeline_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        self._pipeline_kwargs = pipeline_kwargs or {}

        if pipeline is not None:
            self._pipeline = pipeline
            self._manual_points = _coerce_points(manual_points) if manual_points is not None else None
            self._image_size = image_size
            return

        if calibration_record is not None:
            manual_points = _extract_manual_points_from_record(calibration_record)
            if image_size is None:
                image_size = _extract_optional_image_size(calibration_record)

        if manual_points is None:
            raise ValueError(
                "ScoreMapper requires either 'pipeline', 'manual_points', or 'calibration_record'."
            )

        self._manual_points = _coerce_points(manual_points)
        self._image_size = image_size
        self._pipeline = _build_pipeline(
            manual_points=self._manual_points,
            image_size=self._image_size,
            extra_kwargs=self._pipeline_kwargs,
        )

    # ------------------------------------------------------------------
    # Öffentliche Properties
    # ------------------------------------------------------------------

    @property
    def pipeline(self) -> Any:
        """
        Gibt die gecachte zentrale Pipeline-Geometrie zurück.
        """
        return self._pipeline

    @property
    def manual_points(self) -> Optional[list[Point]]:
        """
        Gibt die 4 verwendeten manuellen Marker zurück, falls bekannt.
        """
        if self._manual_points is None:
            return None
        return list(self._manual_points)

    @property
    def image_size(self) -> Optional[tuple[int, int]]:
        """
        Optionale Bildgröße, falls beim Erzeugen vorhanden.
        """
        return self._image_size

    @property
    def bull_topdown(self) -> Optional[Point]:
        """
        Best-effort Zugriff auf den Bullpunkt im Top-Down-System.
        """
        pipeline = self._pipeline

        if isinstance(pipeline, dict):
            for key in ("bull_topdown", "bull", "bull_center", "center"):
                value = pipeline.get(key)
                if value is not None:
                    try:
                        return _coerce_point(value)
                    except Exception:
                        pass

        for attr in ("bull_topdown", "bull", "bull_center", "center"):
            if hasattr(pipeline, attr):
                value = getattr(pipeline, attr)
                if value is not None:
                    try:
                        return _coerce_point(value)
                    except Exception:
                        pass

        return None

    # ------------------------------------------------------------------
    # Rebuild / Update
    # ------------------------------------------------------------------

    def rebuild_from_manual_points(
        self,
        manual_points: Sequence[Any],
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """
        Baut die Pipeline mit 4 neuen Markern neu auf.
        """
        self._manual_points = _coerce_points(manual_points)
        self._image_size = image_size or self._image_size
        self._pipeline = _build_pipeline(
            manual_points=self._manual_points,
            image_size=self._image_size,
            extra_kwargs=self._pipeline_kwargs,
        )

    def rebuild_from_record(self, calibration_record: Any) -> None:
        """
        Baut die Pipeline aus einem Calibration-Record neu auf.
        """
        manual_points = _extract_manual_points_from_record(calibration_record)
        image_size = _extract_optional_image_size(calibration_record)

        self.rebuild_from_manual_points(manual_points, image_size=image_size)

    # ------------------------------------------------------------------
    # Projektionen
    # ------------------------------------------------------------------

    def image_point_to_topdown(self, point: Any) -> Point:
        """
        Projiziert genau einen Bildpunkt ins Top-Down-System.
        """
        point_xy = _coerce_point(point)
        result = _project_image_points([point_xy], self._pipeline)

        if isinstance(result, (list, tuple)) and len(result) > 0:
            return _coerce_point(result[0])

        # Falls die Geometrie-Funktion direkt nur einen Punkt zurückgibt
        return _coerce_point(result)

    def topdown_point_to_image(self, point: Any) -> Point:
        """
        Projiziert genau einen Top-Down-Punkt zurück ins Bild.
        """
        point_xy = _coerce_point(point)
        result = _project_topdown_points([point_xy], self._pipeline)

        if isinstance(result, (list, tuple)) and len(result) > 0:
            return _coerce_point(result[0])

        return _coerce_point(result)

    def image_points_to_topdown(self, points: Sequence[Any]) -> list[Point]:
        """
        Projiziert mehrere Bildpunkte ins Top-Down-System.
        """
        points_xy = _coerce_points(points)
        result = _project_image_points(points_xy, self._pipeline)

        if isinstance(result, (list, tuple)):
            return [_coerce_point(p) for p in result]

        return [_coerce_point(result)]

    def topdown_points_to_image(self, points: Sequence[Any]) -> list[Point]:
        """
        Projiziert mehrere Top-Down-Punkte zurück ins Bild.
        """
        points_xy = _coerce_points(points)
        result = _project_topdown_points(points_xy, self._pipeline)

        if isinstance(result, (list, tuple)):
            return [_coerce_point(p) for p in result]

        return [_coerce_point(result)]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_image_point(self, point: Any) -> ScoredHit:
        """
        Bewertet einen Bildpunkt und liefert ein einheitliches ScoredHit-Ergebnis.
        """
        image_point = _coerce_point(point)
        raw_hit = _calculate_hit_from_image(image_point, self._pipeline)

        topdown_point = None
        try:
            topdown_point = self.image_point_to_topdown(image_point)
        except Exception as exc:
            logger.debug("Could not project image point to topdown: %s", exc)

        return _build_scored_hit(
            raw_hit=raw_hit,
            source_space="image",
            image_point=image_point,
            topdown_point=topdown_point,
        )

    def score_topdown_point(self, point: Any) -> ScoredHit:
        """
        Bewertet einen Top-Down-Punkt und liefert ein einheitliches ScoredHit-Ergebnis.
        """
        topdown_point = _coerce_point(point)
        raw_hit = _calculate_hit_from_topdown(topdown_point, self._pipeline)

        image_point = None
        try:
            image_point = self.topdown_point_to_image(topdown_point)
        except Exception as exc:
            logger.debug("Could not project topdown point to image: %s", exc)

        return _build_scored_hit(
            raw_hit=raw_hit,
            source_space="topdown",
            image_point=image_point,
            topdown_point=topdown_point,
        )

    def score_image_points(self, points: Sequence[Any]) -> list[ScoredHit]:
        """
        Bewertet mehrere Bildpunkte.
        """
        return [self.score_image_point(p) for p in points]

    def score_topdown_points(self, points: Sequence[Any]) -> list[ScoredHit]:
        """
        Bewertet mehrere Top-Down-Punkte.
        """
        return [self.score_topdown_point(p) for p in points]

    # ------------------------------------------------------------------
    # Kompakte Convenience-Methoden
    # ------------------------------------------------------------------

    def score_image_point_label(self, point: Any) -> str:
        """
        Liefert nur das Label für einen Bildpunkt.
        """
        return self.score_image_point(point).label

    def score_image_point_value(self, point: Any) -> int:
        """
        Liefert nur den numerischen Score für einen Bildpunkt.
        """
        return self.score_image_point(point).score

    def score_topdown_point_label(self, point: Any) -> str:
        """
        Liefert nur das Label für einen Top-Down-Punkt.
        """
        return self.score_topdown_point(point).label

    def score_topdown_point_value(self, point: Any) -> int:
        """
        Liefert nur den numerischen Score für einen Top-Down-Punkt.
        """
        return self.score_topdown_point(point).score


# --------------------------------------------------------------------------------------
# Modulweite Convenience-Funktionen
# --------------------------------------------------------------------------------------

def build_score_mapper(
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> ScoreMapper:
    """
    Praktischer Builder für einen ScoreMapper.
    """
    return ScoreMapper(
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )


def score_image_point(
    point: Any,
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> ScoredHit:
    """
    Stateless Convenience-Wrapper für genau einen Bildpunkt.
    """
    mapper = build_score_mapper(
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )
    return mapper.score_image_point(point)


def score_topdown_point(
    point: Any,
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> ScoredHit:
    """
    Stateless Convenience-Wrapper für genau einen Top-Down-Punkt.
    """
    mapper = build_score_mapper(
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )
    return mapper.score_topdown_point(point)


# --------------------------------------------------------------------------------------
# Kompatibilitäts-Wrapper für ältere Modulaufrufe
# --------------------------------------------------------------------------------------

def map_image_point_to_hit(
    point: Any,
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Kompatibilitätsfunktion:
    Gibt nur das normalisierte Hit-Label für einen Bildpunkt zurück.
    """
    return score_image_point(
        point,
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    ).label


def map_topdown_point_to_hit(
    point: Any,
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> str:
    """
    Kompatibilitätsfunktion:
    Gibt nur das normalisierte Hit-Label für einen Top-Down-Punkt zurück.
    """
    return score_topdown_point(
        point,
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    ).label


__all__ = [
    "Point",
    "ScoredHit",
    "ScoreMapper",
    "normalize_hit_label",
    "hit_label_to_score",
    "hit_label_to_multiplier",
    "hit_label_to_ring",
    "hit_label_to_segment",
    "build_score_mapper",
    "score_image_point",
    "score_topdown_point",
    "map_image_point_to_hit",
    "map_topdown_point_to_hit",
]