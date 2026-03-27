# tools/manual_recalibrate_4point.py
# ------------------------------------------------------------
# Dieses Tool dient zum manuellen Neusetzen der 4 Primärmarker
# auf dem Referenzbild einer Kamera.
#
# Ziel:
# - P1 = 20|1  am äußeren Double-Draht
# - P2 = 6|10  am äußeren Double-Draht
# - P3 = 3|19  am äußeren Double-Draht
# - P4 = 11|14 am äußeren Double-Draht
#
# Wichtige Eigenschaften:
# - lädt calibration.json
# - lädt ein Referenzbild
# - zeigt vorhandene Marker
# - Marker können per Maus verschoben werden
# - speichert zurück in calibration.json
# - überschreibt standardmäßig nur die ersten 4 Punkte
# - ein evtl. vorhandener alter 5. Bullpunkt bleibt erhalten
#
# Bedienung:
# - Linksklick nahe an einem Marker: Marker auswählen / ziehen
# - Rechtsklick: ausgewählten Marker an Position setzen
# - Taste 1..4: aktiven Marker wählen
# - Taste R: Marker auf ursprünglich geladene Werte zurücksetzen
# - Taste S: Speichern
# - Taste ESC oder Q: Beenden ohne Speichern
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


# ------------------------------------------------------------
# Projektpfad ergänzen, damit Import aus tools/ funktioniert
# ------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ------------------------------------------------------------
# Optional: echte Bull-Berechnung aus Projekt nutzen
# Fallback unten, falls Import lokal fehlschlägt.
# ------------------------------------------------------------
try:
    from vision.calibration_geometry import compute_bull_from_manual_points
except Exception:
    compute_bull_from_manual_points = None  # type: ignore


WINDOW_NAME = "Triple One - 4 Point Recalibration"

MARKER_INFO = [
    ("P1", "20|1"),
    ("P2", "6|10"),
    ("P3", "3|19"),
    ("P4", "11|14"),
]

MARKER_COLORS = [
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (255, 255, 0),   # Cyan
    (0, 255, 0),     # Green
]


# ------------------------------------------------------------
# Grundfunktionen
# ------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Bild konnte nicht geladen werden: {path}")
    return image


def _to_xy_tuple(value: Any) -> Optional[tuple[float, float]]:
    if value is None:
        return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except Exception:
            return None

    if isinstance(value, dict):
        if "x_px" in value and "y_px" in value:
            try:
                return float(value["x_px"]), float(value["y_px"])
            except Exception:
                return None
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except Exception:
                return None

    if hasattr(value, "x_px") and hasattr(value, "y_px"):
        try:
            return float(value.x_px), float(value.y_px)
        except Exception:
            return None

    if hasattr(value, "x") and hasattr(value, "y"):
        try:
            return float(value.x), float(value.y)
        except Exception:
            return None

    return None


def _point_to_dict(point: tuple[float, float]) -> dict[str, int]:
    return {
        "x_px": int(round(point[0])),
        "y_px": int(round(point[1])),
    }


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extract_camera_payload(raw: dict[str, Any], camera_index: int) -> dict[str, Any]:
    cameras = raw.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError("calibration.json enthält kein 'cameras'-Array.")

    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(
            f"Kameraindex {camera_index} ist ungültig. "
            f"Verfügbar: 0 bis {len(cameras) - 1}"
        )

    payload = cameras[camera_index]
    if not isinstance(payload, dict):
        raise ValueError(f"Eintrag für Kamera {camera_index} ist kein Objekt.")

    return payload


def _extract_first_4_points(camera_payload: dict[str, Any]) -> list[tuple[float, float]]:
    raw_points = camera_payload.get("points", [])
    if not isinstance(raw_points, list):
        raise ValueError("camera_payload['points'] ist keine Liste.")

    points: list[tuple[float, float]] = []
    for item in raw_points[:4]:
        pt = _to_xy_tuple(item)
        if pt is None:
            raise ValueError(f"Ungültiger Punkt in calibration.json: {item}")
        points.append(pt)

    if len(points) < 4:
        raise ValueError(
            "Zu wenige Punkte in calibration.json. Erwartet mindestens 4 Punkte."
        )

    return points


def _compute_bull(points: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
    """
    Nutzt bevorzugt die echte Projektfunktion.
    Falls diese nicht importiert werden kann, wird als Fallback der
    Schnittpunkt der Diagonalen P1-P3 und P2-P4 angenähert.
    """
    if len(points) < 4:
        return None

    manual_points = [_point_to_dict(p) for p in points[:4]]

    if compute_bull_from_manual_points is not None:
        try:
            bull = compute_bull_from_manual_points(manual_points)
            pt = _to_xy_tuple(bull)
            if pt is not None:
                return pt
        except Exception:
            pass

    # Fallback über Diagonalenschnitt
    return _intersect_lines(points[0], points[2], points[1], points[3])


def _intersect_lines(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    p4: tuple[float, float],
) -> Optional[tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-9:
        return None

    px = (
        (x1 * y2 - y1 * x2) * (x3 - x4)
        - (x1 - x2) * (x3 * y4 - y3 * x4)
    ) / denom
    py = (
        (x1 * y2 - y1 * x2) * (y3 - y4)
        - (y1 - y2) * (x3 * y4 - y3 * x4)
    ) / denom

    return float(px), float(py)


# ------------------------------------------------------------
# UI / Editor State
# ------------------------------------------------------------
class CalibrationEditor:
    """
    Hält den Zustand des kleinen Kalibrier-Editors und verarbeitet
    Maus-/Tastatur-Interaktion.
    """

    def __init__(
        self,
        image: np.ndarray,
        json_data: dict[str, Any],
        calibration_path: Path,
        camera_index: int,
    ) -> None:
        self.base_image = image
        self.json_data = json_data
        self.calibration_path = calibration_path
        self.camera_index = camera_index

        self.camera_payload = _extract_camera_payload(self.json_data, self.camera_index)

        self.original_points = _extract_first_4_points(self.camera_payload)
        self.points = copy.deepcopy(self.original_points)

        self.active_index = 0
        self.dragging_index: Optional[int] = None
        self.hover_index: Optional[int] = None
        self.drag_threshold_px = 25.0

        self.saved = False

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------
    def render(self) -> np.ndarray:
        canvas = self.base_image.copy()

        # Verbindungslinien zwischen den Punkten
        poly = np.array(
            [[int(round(x)), int(round(y))] for x, y in self.points],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, (120, 120, 120), 1, cv2.LINE_AA)

        # Marker und Labels
        for idx, point in enumerate(self.points):
            label_short, label_semantic = MARKER_INFO[idx]
            color = MARKER_COLORS[idx]

            x, y = int(round(point[0])), int(round(point[1]))

            radius = 9 if idx == self.active_index else 7
            thickness = -1

            cv2.circle(canvas, (x, y), radius, color, thickness, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 13, color, 1, cv2.LINE_AA)

            if idx == self.active_index:
                cv2.circle(canvas, (x, y), 18, (255, 255, 255), 1, cv2.LINE_AA)

            label = f"{label_short} = {label_semantic}"
            self._draw_label(canvas, label, (x + 10, y - 10), color)

        # Bull
        bull = _compute_bull(self.points)
        if bull is not None:
            bx, by = int(round(bull[0])), int(round(bull[1]))
            cv2.circle(canvas, (bx, by), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.line(canvas, (bx - 10, by), (bx + 10, by), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (bx, by - 10), (bx, by + 10), (0, 255, 255), 2, cv2.LINE_AA)
            self._draw_label(canvas, "Bull (auto)", (bx + 10, by - 10), (0, 255, 255))

        # Hilfe / Status
        help_lines = [
            f"Kamera: {self.camera_index}",
            f"Aktiv: {MARKER_INFO[self.active_index][0]} = {MARKER_INFO[self.active_index][1]}",
            "Linksklick nahe Marker: auswählen / ziehen",
            "Rechtsklick: aktiven Marker setzen",
            "Tasten 1-4: aktiven Marker wählen",
            "R: Reset  |  S: Speichern  |  Q/ESC: Beenden",
        ]
        self._draw_help_box(canvas, help_lines)

        return canvas

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1

        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        cv2.rectangle(
            image,
            (x - 3, y - h - 4),
            (x + w + 4, y + baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            text,
            (x, y),
            font,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_help_box(self, image: np.ndarray, lines: list[str]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 1
        padding = 10
        line_gap = 8

        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        box_w = max(w for w, _ in sizes) + padding * 2
        box_h = sum(h for _, h in sizes) + line_gap * (len(lines) - 1) + padding * 2

        x1, y1 = 10, 10
        x2, y2 = x1 + box_w, y1 + box_h

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (80, 80, 80), 1)

        y = y1 + padding + sizes[0][1]
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (x1 + padding, y),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )
            if i < len(lines) - 1:
                y += sizes[i + 1][1] + line_gap

    # --------------------------------------------------------
    # Mauslogik
    # --------------------------------------------------------
    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        mouse_point = (float(x), float(y))

        nearest_index = self._find_nearest_marker(mouse_point)
        self.hover_index = nearest_index

        if event == cv2.EVENT_LBUTTONDOWN:
            if nearest_index is not None:
                self.active_index = nearest_index
                self.dragging_index = nearest_index

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging_index is not None:
                self.points[self.dragging_index] = mouse_point

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_index = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points[self.active_index] = mouse_point

    def _find_nearest_marker(self, mouse_point: tuple[float, float]) -> Optional[int]:
        best_index: Optional[int] = None
        best_distance = float("inf")

        for idx, point in enumerate(self.points):
            dist = _distance(mouse_point, point)
            if dist < best_distance:
                best_distance = dist
                best_index = idx

        if best_distance <= self.drag_threshold_px:
            return best_index

        return None

    # --------------------------------------------------------
    # Tastenlogik
    # --------------------------------------------------------
    def handle_key(self, key: int) -> bool:
        """
        Gibt True zurück, wenn das Programm beendet werden soll.
        """
        if key in (27, ord("q"), ord("Q")):
            return True

        if key == ord("1"):
            self.active_index = 0
        elif key == ord("2"):
            self.active_index = 1
        elif key == ord("3"):
            self.active_index = 2
        elif key == ord("4"):
            self.active_index = 3
        elif key in (ord("r"), ord("R")):
            self.points = copy.deepcopy(self.original_points)
        elif key in (ord("s"), ord("S")):
            self.save_points()
            self.saved = True
            return True

        return False

    # --------------------------------------------------------
    # Speichern
    # --------------------------------------------------------
    def save_points(self) -> None:
        """
        Schreibt die neuen 4 Primärmarker zurück nach calibration.json.

        Wichtige Entscheidung:
        - erste 4 Punkte werden überschrieben
        - falls ein 5. Legacy-Punkt existiert, bleibt er erhalten
        """
        points_payload = self.camera_payload.get("points")
        if not isinstance(points_payload, list):
            points_payload = []

        new_primary_points = [_point_to_dict(p) for p in self.points]

        if len(points_payload) >= 5:
            # Legacy-Bullpunkt hinten erhalten
            legacy_rest = points_payload[4:]
            self.camera_payload["points"] = new_primary_points + legacy_rest
        else:
            self.camera_payload["points"] = new_primary_points

        _write_json(self.calibration_path, self.json_data)
        print("")
        print("[DONE] calibration.json gespeichert.")
        print(f"[DONE] Datei: {self.calibration_path}")
        print("[DONE] Neue 4 Marker:")
        for idx, point in enumerate(self.points):
            label_short, label_semantic = MARKER_INFO[idx]
            print(
                f"  {label_short} ({label_semantic}): "
                f"x={int(round(point[0]))}, y={int(round(point[1]))}"
            )


# ------------------------------------------------------------
# OpenCV-Maus-Bridge
# ------------------------------------------------------------
_EDITOR_INSTANCE: Optional[CalibrationEditor] = None


def _mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global _EDITOR_INSTANCE
    if _EDITOR_INSTANCE is not None:
        _EDITOR_INSTANCE.on_mouse(event, x, y, flags)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triple One - Tool zum manuellen Neusetzen der 4 Kalibrier-Marker"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Pfad zum Referenzbild / leeren Board-Bild",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="Pfad zur calibration.json",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Kameraindex in calibration.json",
    )

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> int:
    global _EDITOR_INSTANCE

    args = parse_args()

    calibration_path = Path(args.calibration)
    image_path = Path(args.image)

    json_data = _read_json(calibration_path)
    image = _load_image(image_path)

    editor = CalibrationEditor(
        image=image,
        json_data=json_data,
        calibration_path=calibration_path,
        camera_index=args.camera_index,
    )
    _EDITOR_INSTANCE = editor

    print("")
    print("Triple One - 4 Point Recalibration")
    print("----------------------------------")
    print("Setze exakt diese 4 Punkte auf dem äußeren Double-Draht:")
    for short_label, semantic_label in MARKER_INFO:
        print(f"  {short_label} = {semantic_label}")
    print("")
    print("Bedienung:")
    print("  Linksklick nahe Marker  -> auswählen / ziehen")
    print("  Rechtsklick             -> aktiven Marker setzen")
    print("  Tasten 1..4             -> aktiven Marker wählen")
    print("  R                       -> Reset")
    print("  S                       -> Speichern")
    print("  Q / ESC                 -> Beenden ohne Speichern")
    print("")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback)

    while True:
        canvas = editor.render()
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(16) & 0xFF
        if key != 255:
            should_quit = editor.handle_key(key)
            if should_quit:
                break

    cv2.destroyAllWindows()

    if editor.saved:
        return 0

    print("")
    print("[INFO] Beendet ohne Speichern.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())