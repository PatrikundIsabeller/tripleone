# vision/score_mapper.py
# Zweck:
# Diese Datei bildet eine saubere Scoring-Schicht auf Basis der zentralen Geometrie.
#
# WICHTIG:
# Diese Datei besitzt bewusst KEINE eigene Board-Geometrie.
# Sie verwendet ausschließlich vision/calibration_geometry.py als Quelle der Wahrheit.
#
# Verantwortlichkeiten:
# - Aufbau / Halten eines Geometry-Kontexts ("pipeline")
# - Projektion Bild <-> Top-Down
# - Bildpunkt -> HitResult / Label / numerischer Score
# - Label-Normalisierung
# - stabile Ergebnisobjekte für Debugging und Weiterverarbeitung
#
# Wichtiger Kompatibilitätshinweis:
# Die aktuelle echte calibration_geometry.py im Repo arbeitet noch mit einer
# Legacy-Signatur wie:
#   calculate_hit_from_image_point(x_px, y_px, points_like)
#   project_image_points_to_topdown(points_like, image_points)
#   project_topdown_points_to_image(points_like, topdown_points)
#   build_pipeline_points(points_like)
#
# Deshalb hält score_mapper.py intern "pipeline" aktuell als points_like-kompatible
# Struktur (Liste von {"x_px", "y_px"}), nicht als komplexes Pipeline-Objekt.

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np

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

logger = logging.getLogger(__name__)

Point = tuple[float, float]


# --------------------------------------------------------------------------------------
# Punkt-/Record-Normalisierung
# --------------------------------------------------------------------------------------

def _coerce_point(value: Any) -> Point:
    """
    Wandelt unterschiedliche Punktformate in ein (x, y)-Tuple um.

    Unterstützt:
    - (x, y)
    - [x, y]
    - {"x": ..., "y": ...}
    - {"x_px": ..., "y_px": ...}
    - Objekt mit .x und .y
    """
    if value is None:
        raise ValueError("Point is None.")

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if "x_px" in value and "y_px" in value:
            return float(value["x_px"]), float(value["y_px"])
        raise ValueError(
            f"Point dict must contain either 'x'/'y' or 'x_px'/'y_px', got: {value}"
        )

    if isinstance(value, (tuple, list, np.ndarray)):
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


def _points_to_legacy_point_dicts(values: Sequence[Any]) -> list[dict[str, int]]:
    """
    Wandelt Punktformate in das Legacy-Format der echten calibration_geometry.py um:
    [{"x_px": ..., "y_px": ...}, ...]
    """
    pts = _coerce_points(values)
    return [
        {"x_px": int(round(x)), "y_px": int(round(y))}
        for x, y in pts
    ]


def _extract_manual_points_from_record(record: Any) -> list[Point]:
    """
    Extrahiert die 4 manuellen Marker aus einem Calibration-Record oder Dict.

    Unterstützte Quellen:
    - manual_points
    - marker_points
    - markers
    - image_points
    - points
    - p1..p4
    """
    candidate_names = [
        "manual_points",
        "marker_points",
        "markers",
        "image_points",
        "points",
    ]

    if isinstance(record, dict):
        for name in candidate_names:
            if name in record and record[name]:
                points = _coerce_points(record[name])
                if len(points) < 4:
                    raise ValueError(
                        f"Expected at least 4 manual points in record['{name}'], got {len(points)}."
                    )
                return points[:4]

        p_names = ["p1", "p2", "p3", "p4"]
        if all(name in record for name in p_names):
            points = _coerce_points([record[name] for name in p_names])
            if len(points) != 4:
                raise ValueError("Record p1..p4 extraction failed.")
            return points

        raise ValueError("Could not extract manual points from calibration record dict.")

    for name in candidate_names:
        if hasattr(record, name):
            value = getattr(record, name)
            if value:
                points = _coerce_points(value)
                if len(points) < 4:
                    raise ValueError(
                        f"Expected at least 4 manual points in record.{name}, got {len(points)}."
                    )
                return points[:4]

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

        width = record.get("image_width", record.get("width", record.get("frame_width")))
        height = record.get("image_height", record.get("height", record.get("frame_height")))
        if width is not None and height is not None:
            return int(width), int(height)
        return None

    for name in candidate_names:
        if hasattr(record, name):
            value = getattr(record, name)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return int(value[0]), int(value[1])

    width = getattr(record, "image_width", getattr(record, "width", getattr(record, "frame_width", None)))
    height = getattr(record, "image_height", getattr(record, "height", getattr(record, "frame_height", None)))
    if width is not None and height is not None:
        return int(width), int(height)

    return None


# --------------------------------------------------------------------------------------
# Robuste Helfer für Funktionsaufrufe
# --------------------------------------------------------------------------------------

def _filter_supported_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Filtert nur kwargs, die eine Ziel-Funktion tatsächlich unterstützt.
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


# --------------------------------------------------------------------------------------
# Geometry-Adapter
# --------------------------------------------------------------------------------------

def _build_pipeline(
    *,
    manual_points: Sequence[Point],
    image_size: Optional[tuple[int, int]] = None,
    extra_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """
    Baut die zentrale Pipeline-Geometrie über calibration_geometry.build_pipeline_points().

    WICHTIG:
    In deinem echten Repo liefert build_pipeline_points(points_like) aktuell ein
    kompatibles 5-Punkte-Format zurück. Das behandeln wir hier als "pipeline".
    """
    if len(manual_points) != 4:
        raise ValueError(
            f"Expected exactly 4 manual points, got {len(manual_points)}. "
            "score_mapper.py supports only the 4-point calibration architecture."
        )

    manual_points = _coerce_points(manual_points)
    legacy_manual_points = _points_to_legacy_point_dicts(manual_points)
    extra_kwargs = extra_kwargs or {}

    width = None
    height = None
    if image_size:
        width, height = int(image_size[0]), int(image_size[1])

    named_candidates = [
        {
            "manual_points": legacy_manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "image_points": legacy_manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "marker_points": legacy_manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "markers": legacy_manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
        {
            "points": legacy_manual_points,
            "image_size": image_size,
            "image_width": width,
            "image_height": height,
            **extra_kwargs,
        },
    ]

    positional_candidates = [
        (legacy_manual_points,),
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
    Projiziert Bildpunkte nach Top-Down.

    Unterstützt die echte Triple-One-Signatur:
    project_image_points_to_topdown(points_like, image_points)
    """
    pts = np.asarray(points, dtype=np.float32)

    try:
        return project_image_points_to_topdown(pipeline, pts)
    except TypeError:
        pass

    named_candidates = [
        {"image_points": pts, "pipeline": pipeline},
        {"points": pts, "pipeline": pipeline},
        {"pts": pts, "pipeline": pipeline},
        {"image_points": pts, "geometry": pipeline},
        {"points": pts, "geometry": pipeline},
        {"image_points": pts, "context": pipeline},
        {"points": pts, "context": pipeline},
    ]

    positional_candidates = [
        (pts, pipeline),
        (pipeline, pts),
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
    Projiziert Top-Down-Punkte zurück ins Bild.

    Unterstützt die echte Triple-One-Signatur:
    project_topdown_points_to_image(points_like, topdown_points)
    """
    pts = np.asarray(points, dtype=np.float32)

    try:
        return project_topdown_points_to_image(pipeline, pts)
    except TypeError:
        pass

    named_candidates = [
        {"topdown_points": pts, "pipeline": pipeline},
        {"points": pts, "pipeline": pipeline},
        {"pts": pts, "pipeline": pipeline},
        {"topdown_points": pts, "geometry": pipeline},
        {"points": pts, "geometry": pipeline},
        {"topdown_points": pts, "context": pipeline},
        {"points": pts, "context": pipeline},
    ]

    positional_candidates = [
        (pts, pipeline),
        (pipeline, pts),
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
    Berechnet einen Treffer für einen Bildpunkt.

    Unterstützt die echte Triple-One-Signatur:
    calculate_hit_from_image_point(x_px, y_px, points_like)
    """
    x, y = float(point[0]), float(point[1])

    try:
        return calculate_hit_from_image_point(x, y, pipeline)
    except TypeError:
        pass

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
    Berechnet einen Treffer für einen Top-Down-Punkt.

    Unterstützt die echte Triple-One-Signatur:
    calculate_hit_from_topdown_point(x_px, y_px)
    """
    x, y = float(point[0]), float(point[1])

    try:
        return calculate_hit_from_topdown_point(x, y)
    except TypeError:
        pass

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
# Label-Normalisierung / Score
# --------------------------------------------------------------------------------------

_HIT_RE = re.compile(r"^(S|D|T)?(\d{1,2})$")


def normalize_hit_label(label: str) -> str:
    """
    Normalisiert Hit-Labels auf:
    - MISS
    - SBULL
    - DBULL
    - S1..S20
    - D1..D20
    - T1..T20
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
    normalized = normalize_hit_label(label)

    if normalized == "MISS":
        return "MISS"
    if normalized == "SBULL":
        return "SBULL"
    if normalized == "DBULL":
        return "DBULL"

    return normalized[0]


def hit_label_to_segment(label: str) -> Optional[int]:
    normalized = normalize_hit_label(label)

    if normalized in {"MISS", "SBULL", "DBULL"}:
        return None

    return int(normalized[1:])


# --------------------------------------------------------------------------------------
# Raw-Hit-Extraktion
# --------------------------------------------------------------------------------------

def _extract_label_from_raw_hit(raw_hit: Any) -> str:
    """
    Extrahiert robust ein Hit-Label aus unterschiedlichen Rückgabeformaten.

    Unterstützt unter anderem:
    - "D20"
    - {"label": "D20"}
    - {"ring": "D", "segment": 20}
    - ("D", 20)
    - HitResult-Objekt aus calibration_geometry.py
    """
    if raw_hit is None:
        raise ValueError("Raw hit result is None.")

    if isinstance(raw_hit, str):
        return normalize_hit_label(raw_hit)

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

    # Objekt-Attribute
    for attr in ("label", "hit", "code", "result", "field"):
        if hasattr(raw_hit, attr):
            value = getattr(raw_hit, attr)
            if value is not None:
                return normalize_hit_label(str(value))

    ring = getattr(raw_hit, "ring", getattr(raw_hit, "ring_name", None))
    segment = getattr(raw_hit, "segment", getattr(raw_hit, "segment_value", getattr(raw_hit, "number", None)))

    # Falls ring_name aus alter Geometry kommt:
    # "DOUBLE"/"TRIPLE"/"SINGLE" + segment_value -> D/T/S
    if ring is not None and segment is not None:
        ring_str = str(ring).strip().upper()
        if ring_str in {"DOUBLE", "D"}:
            return normalize_hit_label(f"D{int(segment)}")
        if ring_str in {"TRIPLE", "T"}:
            return normalize_hit_label(f"T{int(segment)}")
        if ring_str in {"SINGLE", "S"}:
            return normalize_hit_label(f"S{int(segment)}")
        return normalize_hit_label(f"{ring_str}{int(segment)}")

    multiplier = getattr(raw_hit, "multiplier", None)
    segment = getattr(raw_hit, "segment", getattr(raw_hit, "segment_value", getattr(raw_hit, "number", None)))
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
# Ergebnisobjekt
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
# Hauptklasse
# --------------------------------------------------------------------------------------

class ScoreMapper:
    """
    Dünne Mapping-Schicht auf Basis der zentralen Geometrie.

    Aktuell hält self._pipeline eine points_like-kompatible Struktur, die zur
    echten calibration_geometry.py passt.
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

    @property
    def pipeline(self) -> Any:
        return self._pipeline

    @property
    def manual_points(self) -> Optional[list[Point]]:
        if self._manual_points is None:
            return None
        return list(self._manual_points)

    @property
    def image_size(self) -> Optional[tuple[int, int]]:
        return self._image_size

    def rebuild_from_manual_points(
        self,
        manual_points: Sequence[Any],
        image_size: Optional[tuple[int, int]] = None,
    ) -> None:
        self._manual_points = _coerce_points(manual_points)
        self._image_size = image_size or self._image_size
        self._pipeline = _build_pipeline(
            manual_points=self._manual_points,
            image_size=self._image_size,
            extra_kwargs=self._pipeline_kwargs,
        )

    def rebuild_from_record(self, calibration_record: Any) -> None:
        manual_points = _extract_manual_points_from_record(calibration_record)
        image_size = _extract_optional_image_size(calibration_record)
        self.rebuild_from_manual_points(manual_points, image_size=image_size)

    # ------------------------------------------------------------------
    # Projektionen
    # ------------------------------------------------------------------

    def image_point_to_topdown(self, point: Any) -> Point:
        point_xy = _coerce_point(point)
        result = _project_image_points([point_xy], self._pipeline)

        if result is None:
            raise ValueError("Projection image -> topdown returned None.")

        if isinstance(result, np.ndarray):
            if result.ndim == 2 and len(result) > 0:
                return _coerce_point(result[0])
            return _coerce_point(result)

        if isinstance(result, (list, tuple)) and len(result) > 0:
            return _coerce_point(result[0])

        return _coerce_point(result)

    def topdown_point_to_image(self, point: Any) -> Point:
        point_xy = _coerce_point(point)
        result = _project_topdown_points([point_xy], self._pipeline)

        if result is None:
            raise ValueError("Projection topdown -> image returned None.")

        if isinstance(result, np.ndarray):
            if result.ndim == 2 and len(result) > 0:
                return _coerce_point(result[0])
            return _coerce_point(result)

        if isinstance(result, (list, tuple)) and len(result) > 0:
            return _coerce_point(result[0])

        return _coerce_point(result)

    def image_points_to_topdown(self, points: Sequence[Any]) -> list[Point]:
        points_xy = _coerce_points(points)
        result = _project_image_points(points_xy, self._pipeline)

        if result is None:
            return []

        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                return [_coerce_point(p) for p in result.tolist()]
            return [_coerce_point(result.tolist())]

        if isinstance(result, (list, tuple)):
            return [_coerce_point(p) for p in result]

        return [_coerce_point(result)]

    def topdown_points_to_image(self, points: Sequence[Any]) -> list[Point]:
        points_xy = _coerce_points(points)
        result = _project_topdown_points(points_xy, self._pipeline)

        if result is None:
            return []

        if isinstance(result, np.ndarray):
            if result.ndim == 2:
                return [_coerce_point(p) for p in result.tolist()]
            return [_coerce_point(result.tolist())]

        if isinstance(result, (list, tuple)):
            return [_coerce_point(p) for p in result]

        return [_coerce_point(result)]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_image_point(self, point: Any) -> ScoredHit:
        image_point = _coerce_point(point)
        raw_hit = _calculate_hit_from_image(image_point, self._pipeline)

        if raw_hit is None:
            raise ValueError(
                f"Geometry returned None for image point {image_point}. "
                "This usually indicates a points_like/pipeline format mismatch."
            )

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
        topdown_point = _coerce_point(point)
        raw_hit = _calculate_hit_from_topdown(topdown_point, self._pipeline)

        if raw_hit is None:
            raise ValueError(
                f"Geometry returned None for topdown point {topdown_point}."
            )

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
        return [self.score_image_point(p) for p in points]

    def score_topdown_points(self, points: Sequence[Any]) -> list[ScoredHit]:
        return [self.score_topdown_point(p) for p in points]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score_image_point_label(self, point: Any) -> str:
        return self.score_image_point(point).label

    def score_image_point_value(self, point: Any) -> int:
        return self.score_image_point(point).score

    def score_topdown_point_label(self, point: Any) -> str:
        return self.score_topdown_point(point).label

    def score_topdown_point_value(self, point: Any) -> int:
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
    mapper = build_score_mapper(
        manual_points=manual_points,
        calibration_record=calibration_record,
        pipeline=pipeline,
        image_size=image_size,
        pipeline_kwargs=pipeline_kwargs,
    )
    return mapper.score_topdown_point(point)


def map_image_point_to_hit(
    point: Any,
    *,
    manual_points: Optional[Sequence[Any]] = None,
    calibration_record: Optional[Any] = None,
    pipeline: Optional[Any] = None,
    image_size: Optional[tuple[int, int]] = None,
    pipeline_kwargs: Optional[dict[str, Any]] = None,
) -> str:
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