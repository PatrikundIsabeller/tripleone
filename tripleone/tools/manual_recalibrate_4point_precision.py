# tools/manual_recalibrate_4point_precision.py
# ------------------------------------------------------------
# Präzisions-Kalibriertool für die 4 Primärmarker:
# P1 = 20|1
# P2 = 6|10
# P3 = 3|19
# P4 = 11|14
#
# Verbesserungen gegenüber der einfachen Version:
# - Zoom-Lupe für den aktiven Marker
# - Fadenkreuz im Zoomfenster
# - Feine Verschiebung per Pfeiltasten
# - Shift + Pfeiltasten = 5 px
# - Tab = nächster Marker
# - 1..4 = Marker direkt wählen
# - Maus für grobe Platzierung, Tastatur für Feinarbeit
#
# Speichern:
# - überschreibt die ersten 4 Punkte in calibration.json
# - lässt ggf. vorhandenen 5. Legacy-Punkt stehen
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


CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from vision.calibration_geometry import compute_bull_from_manual_points
except Exception:
    compute_bull_from_manual_points = None  # type: ignore


WINDOW_NAME = "Triple One - 4 Point Precision Recalibration"

MARKER_INFO = [
    ("P1", "20|1"),
    ("P2", "6|10"),
    ("P3", "3|19"),
    ("P4", "11|14"),
]

MARKER_COLORS = [
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Gelb
    (255, 255, 0),   # Cyan
    (0, 255, 0),     # Grün
]


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

    return None


def _point_to_dict(point: tuple[float, float]) -> dict[str, int]:
    return {"x_px": int(round(point[0])), "y_px": int(round(point[1]))}


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extract_camera_payload(raw: dict[str, Any], camera_index: int) -> dict[str, Any]:
    cameras = raw.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError("calibration.json enthält kein 'cameras'-Array.")

    if camera_index < 0 or camera_index >= len(cameras):
        raise IndexError(f"Kameraindex {camera_index} ungültig. Verfügbar: 0 bis {len(cameras)-1}")

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
        raise ValueError("Zu wenige Punkte in calibration.json. Erwartet mindestens 4 Punkte.")

    return points


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

    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return float(px), float(py)


def _compute_bull(points: list[tuple[float, float]]) -> Optional[tuple[float, float]]:
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

    return _intersect_lines(points[0], points[2], points[1], points[3])


class CalibrationEditor:
    """
    Editor für präzises Setzen der 4 Marker.
    """

    def __init__(
        self,
        image: np.ndarray,
        json_data: dict[str, Any],
        calibration_path: Path,
        camera_index: int,
    ) -> None:
        self.base_image = image
        self.image_h, self.image_w = image.shape[:2]

        self.json_data = json_data
        self.calibration_path = calibration_path
        self.camera_index = camera_index

        self.camera_payload = _extract_camera_payload(self.json_data, self.camera_index)

        self.original_points = _extract_first_4_points(self.camera_payload)
        self.points = copy.deepcopy(self.original_points)

        self.active_index = 0
        self.dragging_index: Optional[int] = None
        self.drag_threshold_px = 25.0
        self.saved = False

        # Zoom-Einstellungen
        self.zoom_crop_half_size = 30
        self.zoom_scale = 6
        self.zoom_box_padding = 12

    def render(self) -> np.ndarray:
        canvas = self.base_image.copy()

        # Polygon
        poly = np.array(
            [[int(round(x)), int(round(y))] for x, y in self.points],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, (120, 120, 120), 1, cv2.LINE_AA)

        # Marker
        for idx, point in enumerate(self.points):
            short_label, semantic_label = MARKER_INFO[idx]
            color = MARKER_COLORS[idx]
            x, y = int(round(point[0])), int(round(point[1]))

            radius = 9 if idx == self.active_index else 7
            cv2.circle(canvas, (x, y), radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 14, color, 1, cv2.LINE_AA)

            if idx == self.active_index:
                cv2.circle(canvas, (x, y), 20, (255, 255, 255), 1, cv2.LINE_AA)

            label = f"{short_label} = {semantic_label}"
            self._draw_label(canvas, label, (x + 10, y - 10), color)

        # Bull
        bull = _compute_bull(self.points)
        if bull is not None:
            bx, by = int(round(bull[0])), int(round(bull[1]))
            cv2.circle(canvas, (bx, by), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.line(canvas, (bx - 10, by), (bx + 10, by), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (bx, by - 10), (bx, by + 10), (0, 255, 255), 2, cv2.LINE_AA)
            self._draw_label(canvas, "Bull (auto)", (bx + 10, by - 10), (0, 255, 255))

        # Hilfe
        help_lines = [
            f"Kamera: {self.camera_index}",
            f"Aktiv: {MARKER_INFO[self.active_index][0]} = {MARKER_INFO[self.active_index][1]}",
            "Maus: grob setzen / ziehen",
            "Pfeile: 1 px  |  Shift+Pfeile: 5 px",
            "1..4 Marker wählen  |  TAB nächster Marker",
            "R Reset  |  S Speichern  |  Q/ESC Beenden",
        ]
        self._draw_help_box(canvas, help_lines)

        # Zoom-Lupe
        self._draw_zoom_inset(canvas)

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
        cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

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
            cv2.putText(image, line, (x1 + padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            if i < len(lines) - 1:
                y += sizes[i + 1][1] + line_gap

    def _draw_zoom_inset(self, image: np.ndarray) -> None:
        """
        Zeichnet eine vergrößerte Lupe um den aktiven Marker.
        """
        point = self.points[self.active_index]
        x, y = int(round(point[0])), int(round(point[1]))

        hs = self.zoom_crop_half_size
        x1 = max(0, x - hs)
        y1 = max(0, y - hs)
        x2 = min(self.image_w, x + hs + 1)
        y2 = min(self.image_h, y + hs + 1)

        crop = self.base_image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return

        # Marker und Linien auch im Crop anzeigen
        for idx, p in enumerate(self.points):
            px, py = int(round(p[0])) - x1, int(round(p[1])) - y1
            if 0 <= px < crop.shape[1] and 0 <= py < crop.shape[0]:
                color = MARKER_COLORS[idx]
                r = 4 if idx == self.active_index else 3
                cv2.circle(crop, (px, py), r, color, -1, cv2.LINE_AA)

        zoom = cv2.resize(
            crop,
            (crop.shape[1] * self.zoom_scale, crop.shape[0] * self.zoom_scale),
            interpolation=cv2.INTER_NEAREST,
        )

        zh, zw = zoom.shape[:2]
        pad = self.zoom_box_padding

        dest_x1 = image.shape[1] - zw - pad
        dest_y1 = pad
        dest_x2 = dest_x1 + zw
        dest_y2 = dest_y1 + zh

        if dest_x1 < 0 or dest_y2 > image.shape[0]:
            return

        # Rahmen
        cv2.rectangle(image, (dest_x1 - 2, dest_y1 - 2), (dest_x2 + 2, dest_y2 + 2), (255, 255, 255), -1)
        image[dest_y1:dest_y2, dest_x1:dest_x2] = zoom
        cv2.rectangle(image, (dest_x1 - 2, dest_y1 - 2), (dest_x2 + 2, dest_y2 + 2), (0, 0, 0), 1)

        # Fadenkreuz in der Mitte
        cx = dest_x1 + zw // 2
        cy = dest_y1 + zh // 2
        cv2.line(image, (cx - 25, cy), (cx + 25, cy), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(image, (cx, cy - 25), (cx, cy + 25), (0, 255, 255), 1, cv2.LINE_AA)

        self._draw_label(
            image,
            f"Zoom {MARKER_INFO[self.active_index][0]}",
            (dest_x1, dest_y2 + 20),
            (255, 255, 255),
        )

    def on_mouse(self, event: int, x: int, y: int, flags: int) -> None:
        mouse_point = (float(x), float(y))

        if event == cv2.EVENT_LBUTTONDOWN:
            nearest_index = self._find_nearest_marker(mouse_point)
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

    def move_active_marker(self, dx: int, dy: int) -> None:
        x, y = self.points[self.active_index]
        nx = min(max(0.0, x + dx), float(self.image_w - 1))
        ny = min(max(0.0, y + dy), float(self.image_h - 1))
        self.points[self.active_index] = (nx, ny)

    def handle_key(self, key: int) -> bool:
        """
        True => beenden
        """
        # ESC / Q
        if key in (27, ord("q"), ord("Q")):
            return True

        # Zahlen 1..4
        if key == ord("1"):
            self.active_index = 0
            return False
        if key == ord("2"):
            self.active_index = 1
            return False
        if key == ord("3"):
            self.active_index = 2
            return False
        if key == ord("4"):
            self.active_index = 3
            return False

        # TAB
        if key == 9:
            self.active_index = (self.active_index + 1) % 4
            return False

        # Reset
        if key in (ord("r"), ord("R")):
            self.points = copy.deepcopy(self.original_points)
            return False

        # Speichern
        if key in (ord("s"), ord("S")):
            self.save_points()
            self.saved = True
            return True

        # Normale Pfeiltasten (Windows OpenCV typische Codes)
        # links 2424832, oben 2490368, rechts 2555904, unten 2621440
        arrow_map = {
            2424832: (-1, 0),
            2490368: (0, -1),
            2555904: (1, 0),
            2621440: (0, 1),
        }

        # Shift + Pfeiltasten (häufig dieselben Codes schwer unterscheidbar je nach OpenCV/Windows)
        # Deshalb zusätzliche große Schritte auf I/J/K/L
        if key in arrow_map:
            dx, dy = arrow_map[key]
            self.move_active_marker(dx, dy)
            return False

        # Alternative Feinsteuerung
        if key == ord("j"):
            self.move_active_marker(-1, 0)
            return False
        if key == ord("l"):
            self.move_active_marker(1, 0)
            return False
        if key == ord("i"):
            self.move_active_marker(0, -1)
            return False
        if key == ord("k"):
            self.move_active_marker(0, 1)
            return False

        # Grobe Schritte 5 px
        if key == ord("J"):
            self.move_active_marker(-5, 0)
            return False
        if key == ord("L"):
            self.move_active_marker(5, 0)
            return False
        if key == ord("I"):
            self.move_active_marker(0, -5)
            return False
        if key == ord("K"):
            self.move_active_marker(0, 5)
            return False

        return False

    def save_points(self) -> None:
        points_payload = self.camera_payload.get("points")
        if not isinstance(points_payload, list):
            points_payload = []

        new_primary_points = [_point_to_dict(p) for p in self.points]

        if len(points_payload) >= 5:
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
            short_label, semantic_label = MARKER_INFO[idx]
            print(
                f"  {short_label} ({semantic_label}): "
                f"x={int(round(point[0]))}, y={int(round(point[1]))}"
            )


_EDITOR_INSTANCE: Optional[CalibrationEditor] = None


def _mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    global _EDITOR_INSTANCE
    if _EDITOR_INSTANCE is not None:
        _EDITOR_INSTANCE.on_mouse(event, x, y, flags)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triple One - Präzisions-Tool zum manuellen Neusetzen der 4 Kalibrier-Marker"
    )
    parser.add_argument("--image", required=True, help="Pfad zum Referenzbild / leeren Board-Bild")
    parser.add_argument("--calibration", required=True, help="Pfad zur calibration.json")
    parser.add_argument("--camera-index", type=int, default=0, help="Kameraindex in calibration.json")
    return parser.parse_args()


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
    print("Triple One - 4 Point Precision Recalibration")
    print("-------------------------------------------")
    print("Setze exakt diese 4 Punkte auf dem äußeren Double-Draht:")
    for short_label, semantic_label in MARKER_INFO:
        print(f"  {short_label} = {semantic_label}")
    print("")
    print("Bedienung:")
    print("  Maus: grob setzen / ziehen")
    print("  Pfeile oder i/j/k/l: 1 px")
    print("  I/J/K/L: 5 px")
    print("  TAB: nächster Marker")
    print("  Tasten 1..4: aktiven Marker wählen")
    print("  R: Reset")
    print("  S: Speichern")
    print("  Q / ESC: Beenden ohne Speichern")
    print("")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_callback)

    while True:
        canvas = editor.render()
        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKeyEx(16)
        if key != -1:
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