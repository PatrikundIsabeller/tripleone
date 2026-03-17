# app/widgets/calibration_preview.py
# Reset K1:
# - nur 4 manuelle Marker
# - Bullzentrum wird automatisch berechnet
# - Präzisionslupe für aktiven Marker
# - feines Justieren per Maus + Pfeiltasten
#
# Marker-Bedeutung:
# P1 = 20|1 auf äußerem Double-Draht
# P2 = 6|10 auf äußerem Double-Draht
# P3 = 3|19 auf äußerem Double-Draht
# P4 = 11|14 auf äußerem Double-Draht

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QWidget


class CalibrationPreview(QWidget):
    """
    Reset K1:
    - 4 manuelle Marker
    - Bull wird automatisch aus Homography berechnet
    - Rechtsklick = Testpunkt
    - aktive Marker-Lupe
    - Pfeiltasten = Feinjustage
    """

    points_changed = pyqtSignal(list)
    test_point_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._frame: Optional[QImage] = None
        self._overlay_config: Dict = {}
        self._manual_points: List[Dict[str, int]] = []
        self._test_point: Optional[Tuple[int, int]] = None
        self._status_text: str = ""

        self._active_point_index: Optional[int] = None
        self._dragging: bool = False
        self._drag_anchor_img: Optional[Tuple[float, float]] = None
        self._drag_anchor_point: Optional[Tuple[float, float]] = None

        self._loupe_zoom: int = 10
        self._loupe_radius_px: int = 14
        self._marker_radius_px: int = 10
        self._selection_distance_px: int = 22

        self._label_font = QFont("Arial", 10, QFont.Weight.Bold)
        self._small_font = QFont("Arial", 8)

    # ------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------

    def set_frame(self, image: QImage) -> None:
        self._frame = image
        self.update()

    def clear_frame(self) -> None:
        self._frame = None
        self.update()

    def set_status_text(self, text: str) -> None:
        self._status_text = text
        self.update()

    def set_test_point(self, x_px: int, y_px: int) -> None:
        self._test_point = (int(x_px), int(y_px))
        self.update()

    def clear_test_point(self) -> None:
        self._test_point = None
        self.update()

    def set_overlay_config(self, config: Dict) -> None:
        self._overlay_config = deepcopy(config)

        points = deepcopy(config.get("points", []))
        # Reset K1:
        # nur die ersten 4 Punkte sind manuell ziehbar
        self._manual_points = points[:4]

        self.update()

    # ------------------------------------------------------------
    # Geometrie / Hilfsfunktionen
    # ------------------------------------------------------------

    def _image_size(self) -> Tuple[int, int]:
        if self._frame is None:
            return 1280, 720
        return self._frame.width(), self._frame.height()

    def _frame_rect(self) -> QRectF:
        """
        Rechteck, in dem das Bild letterboxed dargestellt wird.
        """
        w = self.width()
        h = self.height()

        img_w, img_h = self._image_size()
        if img_w <= 0 or img_h <= 0:
            return QRectF(0, 0, w, h)

        scale = min(w / img_w, h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (w - draw_w) / 2.0
        y = (h - draw_h) / 2.0
        return QRectF(x, y, draw_w, draw_h)

    def _widget_to_image(self, widget_pos: QPointF) -> Optional[Tuple[float, float]]:
        if self._frame is None:
            return None

        rect = self._frame_rect()
        if not rect.contains(widget_pos):
            return None

        img_w, img_h = self._image_size()
        rel_x = (widget_pos.x() - rect.x()) / rect.width()
        rel_y = (widget_pos.y() - rect.y()) / rect.height()

        x_px = rel_x * img_w
        y_px = rel_y * img_h
        return x_px, y_px

    def _image_to_widget(self, x_px: float, y_px: float) -> QPointF:
        rect = self._frame_rect()
        img_w, img_h = self._image_size()

        if img_w <= 0 or img_h <= 0:
            return QPointF(0, 0)

        x = rect.x() + (x_px / img_w) * rect.width()
        y = rect.y() + (y_px / img_h) * rect.height()
        return QPointF(x, y)

    def _clamp_to_image(self, x_px: float, y_px: float) -> Tuple[int, int]:
        img_w, img_h = self._image_size()
        x_px = max(0, min(img_w - 1, round(x_px)))
        y_px = max(0, min(img_h - 1, round(y_px)))
        return int(x_px), int(y_px)

    # ------------------------------------------------------------
    # Feste 4-Punkt-Geometrie
    # ------------------------------------------------------------

    def _dest_point_on_outer_double(self, boundary_angle_deg: float) -> Tuple[float, float]:
        s = 900.0
        cx = s / 2.0
        cy = s / 2.0
        r = s * 0.36

        a = math.radians(boundary_angle_deg)
        x = cx + math.cos(a) * r
        y = cy - math.sin(a) * r
        return x, y

    def _topdown_destination_points(self) -> np.ndarray:
        """
        Feste Referenzpunkte auf dem äußeren Double-Draht.
        """
        p1 = self._dest_point_on_outer_double(81.0)   # 20|1
        p2 = self._dest_point_on_outer_double(351.0)  # 6|10
        p3 = self._dest_point_on_outer_double(261.0)  # 3|19
        p4 = self._dest_point_on_outer_double(171.0)  # 11|14
        return np.array([p1, p2, p3, p4], dtype=np.float32)

    def _src_points_array(self) -> Optional[np.ndarray]:
        if len(self._manual_points) != 4:
            return None

        src = []
        for item in self._manual_points:
            src.append([float(item["x_px"]), float(item["y_px"])])
        return np.array(src, dtype=np.float32)

    def _homography_image_to_topdown(self) -> Optional[np.ndarray]:
        src = self._src_points_array()
        if src is None:
            return None
        dst = self._topdown_destination_points()
        return cv2.getPerspectiveTransform(src, dst)

    def _homography_topdown_to_image(self) -> Optional[np.ndarray]:
        src = self._src_points_array()
        if src is None:
            return None
        dst = self._topdown_destination_points()
        return cv2.getPerspectiveTransform(dst, src)

    def _project_topdown_points_to_image(self, pts_topdown: np.ndarray) -> Optional[np.ndarray]:
        """
        pts_topdown: shape (N, 2)
        """
        h_inv = self._homography_topdown_to_image()
        if h_inv is None:
            return None

        pts = pts_topdown.reshape(-1, 1, 2).astype(np.float32)
        mapped = cv2.perspectiveTransform(pts, h_inv)
        return mapped.reshape(-1, 2)

    def _computed_bull_image_point(self) -> Optional[Tuple[float, float]]:
        """
        Bull wird aus Homography berechnet, nicht manuell gezogen.
        """
        center_top = np.array([[450.0, 450.0]], dtype=np.float32)
        mapped = self._project_topdown_points_to_image(center_top)
        if mapped is None:
            return None
        return float(mapped[0, 0]), float(mapped[0, 1])

    def _computed_config_points(self) -> List[Dict[str, int]]:
        """
        Liefert 5 Punkte zurück:
        - 4 manuelle
        - 1 berechneter Bull
        Damit bleibt die restliche Pipeline kompatibel.
        """
        points = deepcopy(self._manual_points)

        bull = self._computed_bull_image_point()
        if bull is None:
            points.append({"x_px": 0, "y_px": 0})
        else:
            bx, by = self._clamp_to_image(*bull)
            points.append({"x_px": bx, "y_px": by})

        return points

    # ------------------------------------------------------------
    # Overlay-Erzeugung
    # ------------------------------------------------------------

    def _relative_radii(self) -> List[float]:
        """
        Standardisierte Ringradien relativ zum äußeren Double-Radius.
        """
        return [
            1.000000,            # äußerer Double-Rand
            162.0 / 170.0,       # innerer Double-Rand
            107.0 / 170.0,       # äußerer Triple-Rand
            99.0 / 170.0,        # innerer Triple-Rand
            15.9 / 170.0,        # äußerer Bull-Rand
            6.35 / 170.0,        # innerer Bull-Rand
        ]

    def _ring_polylines_image(self) -> List[np.ndarray]:
        polylines = []

        rel_radii = self._relative_radii()
        outer_r = 900.0 * 0.36
        cx = 450.0
        cy = 450.0

        for rel_r in rel_radii:
            r = outer_r * rel_r
            pts = []
            for deg in range(0, 361, 3):
                a = math.radians(deg)
                x = cx + math.cos(a) * r
                y = cy - math.sin(a) * r
                pts.append([x, y])

            pts_np = np.array(pts, dtype=np.float32)
            mapped = self._project_topdown_points_to_image(pts_np)
            if mapped is not None:
                polylines.append(mapped)

        return polylines

    def _sector_lines_image(self) -> List[np.ndarray]:
        lines = []

        cx = 450.0
        cy = 450.0
        outer_r = 900.0 * 0.36

        # Segmentgrenzen alle 18°
        # Startgrenze 20|1 bei 81°
        for i in range(20):
            angle_deg = 81.0 - i * 18.0
            a = math.radians(angle_deg)

            p0 = [cx, cy]
            p1 = [cx + math.cos(a) * outer_r, cy - math.sin(a) * outer_r]

            pts_np = np.array([p0, p1], dtype=np.float32)
            mapped = self._project_topdown_points_to_image(pts_np)
            if mapped is not None:
                lines.append(mapped)

        return lines

    def _segment_label_positions(self) -> List[Tuple[str, Tuple[float, float]]]:
        """
        Zahlen leicht außerhalb des Double-Rings.
        """
        order = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5]

        cx = 450.0
        cy = 450.0
        label_r = 900.0 * 0.405

        positions = []
        for i, value in enumerate(order):
            center_angle_deg = 72.0 - i * 18.0
            a = math.radians(center_angle_deg)
            x = cx + math.cos(a) * label_r
            y = cy - math.sin(a) * label_r
            positions.append((str(value), (x, y)))

        return positions

    def _sector_wedge_image(
        self,
        start_angle_deg: float,
        end_angle_deg: float,
        inner_radius_rel: float = 0.0,
        outer_radius_rel: float = 1.0,
        steps: int = 48,
    ) -> Optional[np.ndarray]:
        """
        Erzeugt einen Sektor als Polygon im Bildraum.
        Winkel:
        0° = rechts, 90° = oben

        Für das 20er-Segment:
        Grenzen sind 99° und 81°.
        """
        cx = 450.0
        cy = 450.0
        outer_r = 900.0 * 0.36
        r0 = outer_r * inner_radius_rel
        r1 = outer_r * outer_radius_rel

        if end_angle_deg > start_angle_deg:
            angles_outer = np.linspace(start_angle_deg, end_angle_deg, steps)
        else:
            angles_outer = np.linspace(start_angle_deg, end_angle_deg, steps)

        outer_pts = []
        for deg in angles_outer:
            a = math.radians(deg)
            x = cx + math.cos(a) * r1
            y = cy - math.sin(a) * r1
            outer_pts.append([x, y])

        inner_pts = []
        for deg in reversed(angles_outer):
            a = math.radians(deg)
            x = cx + math.cos(a) * r0
            y = cy - math.sin(a) * r0
            inner_pts.append([x, y])

        poly_top = np.array(outer_pts + inner_pts, dtype=np.float32)
        mapped = self._project_topdown_points_to_image(poly_top)
        return mapped    

    def _draw_highlighted_twenty_segment(self, painter: QPainter) -> None:
        """
        Zeichnet das 20er-Segment rot/pink als visuelle Referenz.
        Das ist nur optisch und beeinflusst die Kalibrierung nicht.
        """
        if len(self._manual_points) != 4:
            return

        alpha = float(self._overlay_config.get("overlay_alpha", 0.35))

        # 20-Segment liegt zwischen den Grenzen:
        # 20|5 = 99°
        # 20|1 = 81°
        poly = self._sector_wedge_image(
            start_angle_deg=99.0,
            end_angle_deg=81.0,
            inner_radius_rel=0.0,
            outer_radius_rel=1.0,
            steps=64,
        )
        if poly is None or len(poly) < 3:
            return

        path = QPainterPath()
        first = self._image_to_widget(poly[0, 0], poly[0, 1])
        path.moveTo(first)

        for pt in poly[1:]:
            wpt = self._image_to_widget(pt[0], pt[1])
            path.lineTo(wpt)

        path.closeSubpath()

        # etwas kräftiger als das übrige Overlay
        fill_color = QColor(255, 0, 80, int(255 * max(0.20, alpha * 0.85)))
        border_color = QColor(255, 120, 170, int(255 * max(0.30, alpha * 0.95)))

        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(fill_color)
        painter.drawPath(path)

    # ------------------------------------------------------------
    # Zeichnen
    # ------------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(12, 12, 12))

        if self._frame is None:
            self._draw_empty_state(painter)
            return

        frame_rect = self._frame_rect()
        painter.drawImage(frame_rect, self._frame)

        self._draw_overlay(painter)
        self._draw_test_point(painter)
        self._draw_points(painter)
        self._draw_status(painter)
        self._draw_loupe(painter)

    def _draw_empty_state(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(self._label_font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Kamerabild")

    def _draw_overlay(self, painter: QPainter) -> None:
        if len(self._manual_points) != 4:
            return

        alpha = float(self._overlay_config.get("overlay_alpha", 0.35))
        show_sector_lines = bool(self._overlay_config.get("show_sector_lines", True))
        show_numbers = bool(self._overlay_config.get("show_numbers", True))

        line_color = QColor(120, 220, 255, int(255 * alpha))
        ring_pen = QPen(line_color, 1.5)
        sector_pen = QPen(QColor(180, 240, 255, int(255 * alpha)), 1.2)

        # 20er-Segment als rote Referenzfläche
        self._draw_highlighted_twenty_segment(painter)
        # Ringe
        for poly in self._ring_polylines_image():
            path = QPainterPath()
            first = self._image_to_widget(poly[0, 0], poly[0, 1])
            path.moveTo(first)
            for pt in poly[1:]:
                wpt = self._image_to_widget(pt[0], pt[1])
                path.lineTo(wpt)

            painter.setPen(ring_pen)
            painter.drawPath(path)

        # Segmentlinien
        if show_sector_lines:
            painter.setPen(sector_pen)
            for line in self._sector_lines_image():
                p0 = self._image_to_widget(line[0, 0], line[0, 1])
                p1 = self._image_to_widget(line[1, 0], line[1, 1])
                painter.drawLine(p0, p1)

        # Zahlen
        if show_numbers:
            painter.setFont(self._small_font)
            painter.setPen(QPen(QColor(230, 255, 255, 220), 1))
            for text, pos in self._segment_label_positions():
                mapped = self._project_topdown_points_to_image(
                    np.array([[pos[0], pos[1]]], dtype=np.float32)
                )
                if mapped is None:
                    continue
                wx = self._image_to_widget(mapped[0, 0], mapped[0, 1])
                painter.drawText(
                    QRectF(wx.x() - 10, wx.y() - 8, 20, 16),
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )

        # Bull berechnet anzeigen
        bull = self._computed_bull_image_point()
        if bull is not None:
            self._draw_crosshair(
                painter,
                bull[0],
                bull[1],
                QColor(0, 255, 255, 230),
                radius=11,
                thickness=2,
                label="C",
            )

    def _draw_points(self, painter: QPainter) -> None:
        labels = ["P1", "P2", "P3", "P4"]

        for idx, point in enumerate(self._manual_points):
            color = QColor(255, 255, 255, 230)
            border = QColor(30, 30, 30, 255)

            if self._active_point_index == idx:
                color = QColor(255, 210, 60, 250)

            wpt = self._image_to_widget(point["x_px"], point["y_px"])

            painter.setPen(QPen(border, 2))
            painter.setBrush(color)
            painter.drawEllipse(wpt, self._marker_radius_px, self._marker_radius_px)

            label_rect = QRectF(wpt.x() + 10, wpt.y() - 12, 54, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 30, 30, 190))
            painter.drawRoundedRect(label_rect, 4, 4)

            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(self._label_font)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, labels[idx])

    def _draw_test_point(self, painter: QPainter) -> None:
        if self._test_point is None:
            return

        self._draw_crosshair(
            painter,
            self._test_point[0],
            self._test_point[1],
            QColor(40, 255, 120, 240),
            radius=12,
            thickness=2,
            label=None,
        )

    def _draw_crosshair(
        self,
        painter: QPainter,
        x_px: float,
        y_px: float,
        color: QColor,
        radius: int = 10,
        thickness: int = 2,
        label: Optional[str] = None,
    ) -> None:
        wpt = self._image_to_widget(x_px, y_px)
        pen = QPen(color, thickness)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.drawEllipse(wpt, radius, radius)
        painter.drawLine(QPointF(wpt.x() - radius - 6, wpt.y()), QPointF(wpt.x() + radius + 6, wpt.y()))
        painter.drawLine(QPointF(wpt.x(), wpt.y() - radius - 6), QPointF(wpt.x(), wpt.y() + radius + 6))

        if label:
            rect = QRectF(wpt.x() + 12, wpt.y() - 12, 26, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 30, 30, 190))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_status(self, painter: QPainter) -> None:
        if not self._status_text:
            return

        rect = QRectF(10, self.height() - 28, self.width() - 20, 20)
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(self._small_font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._status_text)

    def _draw_loupe(self, painter: QPainter) -> None:
        """
        Große Präzisionslupe für aktiven Marker.
        """
        if self._frame is None:
            return
        if self._active_point_index is None:
            return
        if not (0 <= self._active_point_index < len(self._manual_points)):
            return

        point = self._manual_points[self._active_point_index]
        cx = int(point["x_px"])
        cy = int(point["y_px"])

        src_size = self._loupe_radius_px * 2 + 1
        left = max(0, cx - self._loupe_radius_px)
        top = max(0, cy - self._loupe_radius_px)

        img_w, img_h = self._image_size()
        left = min(left, img_w - src_size)
        top = min(top, img_h - src_size)
        left = max(0, left)
        top = max(0, top)

        cropped = self._frame.copy(left, top, min(src_size, img_w - left), min(src_size, img_h - top))
        if cropped.isNull():
            return

        loupe_w = cropped.width() * self._loupe_zoom
        loupe_h = cropped.height() * self._loupe_zoom

        target_x = 18
        target_y = 18

        # Wenn links oben zu eng ist, rechts oben anzeigen
        if target_x + loupe_w > self.width() - 20:
            target_x = self.width() - loupe_w - 20

        rect = QRectF(target_x, target_y, loupe_w, loupe_h)

        pix = QPixmap.fromImage(cropped)
        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        painter.setBrush(QColor(10, 10, 10, 210))
        painter.drawRoundedRect(rect.adjusted(-6, -26, 6, 6), 10, 10)

        painter.drawPixmap(rect.toRect(), pix)

        # Lupe-Titel
        painter.setFont(self._small_font)
        painter.setPen(QPen(QColor(255, 240, 180), 1))
        title = f"Präzisionslupe {self._loupe_zoom}x – {['P1','P2','P3','P4'][self._active_point_index]}"
        painter.drawText(
            QRectF(rect.x(), rect.y() - 22, rect.width(), 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            title,
        )

        # Fadenkreuz exakt auf aktuellem Punkt
        local_x = (cx - left) * self._loupe_zoom + (self._loupe_zoom / 2.0)
        local_y = (cy - top) * self._loupe_zoom + (self._loupe_zoom / 2.0)

        lx = rect.x() + local_x
        ly = rect.y() + local_y

        painter.setPen(QPen(QColor(60, 255, 120), 2))
        painter.drawLine(QPointF(rect.x(), ly), QPointF(rect.x() + rect.width(), ly))
        painter.drawLine(QPointF(lx, rect.y()), QPointF(lx, rect.y() + rect.height()))
        painter.drawEllipse(QPointF(lx, ly), 7, 7)

        # 1px-Raster für Präzision
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        for i in range(cropped.width() + 1):
            x = rect.x() + i * self._loupe_zoom
            painter.drawLine(QPointF(x, rect.y()), QPointF(x, rect.y() + rect.height()))
        for j in range(cropped.height() + 1):
            y = rect.y() + j * self._loupe_zoom
            painter.drawLine(QPointF(rect.x(), y), QPointF(rect.x() + rect.width(), y))

        # Hinweise
        hint_rect = QRectF(rect.x(), rect.y() + rect.height() + 6, rect.width(), 34)
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawText(
            hint_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "Pfeiltasten: 1 px   |   Shift + Pfeiltasten: 5 px\nAlt beim Ziehen: langsamer",
        )

    # ------------------------------------------------------------
    # Interaktion
    # ------------------------------------------------------------

    def _nearest_point_index(self, img_x: float, img_y: float) -> Optional[int]:
        best_idx = None
        best_dist = None

        for idx, point in enumerate(self._manual_points):
            dx = point["x_px"] - img_x
            dy = point["y_px"] - img_y
            dist = math.sqrt(dx * dx + dy * dy)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist is not None and best_dist <= self._selection_distance_px:
            return best_idx
        return None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame is None:
            return

        self.setFocus()

        img_pos = self._widget_to_image(event.position())
        if img_pos is None:
            return

        img_x, img_y = img_pos

        if event.button() == Qt.MouseButton.RightButton:
            x_px, y_px = self._clamp_to_image(img_x, img_y)
            self._test_point = (x_px, y_px)
            self.test_point_selected.emit(x_px, y_px)
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._nearest_point_index(img_x, img_y)
            if idx is not None:
                self._active_point_index = idx
                self._dragging = True
                self._drag_anchor_img = (img_x, img_y)
                self._drag_anchor_point = (
                    float(self._manual_points[idx]["x_px"]),
                    float(self._manual_points[idx]["y_px"]),
                )
                self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if not self._dragging or self._active_point_index is None:
            return
        if self._drag_anchor_img is None or self._drag_anchor_point is None:
            return

        img_pos = self._widget_to_image(event.position())
        if img_pos is None:
            return

        img_x, img_y = img_pos
        start_img_x, start_img_y = self._drag_anchor_img
        start_px, start_py = self._drag_anchor_point

        dx = img_x - start_img_x
        dy = img_y - start_img_y

        # Alt = langsame Präzisionsbewegung
        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            dx *= 0.20
            dy *= 0.20

        new_x, new_y = self._clamp_to_image(start_px + dx, start_py + dy)

        self._manual_points[self._active_point_index]["x_px"] = new_x
        self._manual_points[self._active_point_index]["y_px"] = new_y

        self.points_changed.emit(self._computed_config_points())
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._drag_anchor_img = None
        self._drag_anchor_point = None
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._active_point_index is None:
            return
        if not (0 <= self._active_point_index < len(self._manual_points)):
            return

        step = 1
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            step = 5

        x_px = self._manual_points[self._active_point_index]["x_px"]
        y_px = self._manual_points[self._active_point_index]["y_px"]

        changed = False

        if event.key() == Qt.Key.Key_Left:
            x_px -= step
            changed = True
        elif event.key() == Qt.Key.Key_Right:
            x_px += step
            changed = True
        elif event.key() == Qt.Key.Key_Up:
            y_px -= step
            changed = True
        elif event.key() == Qt.Key.Key_Down:
            y_px += step
            changed = True
        elif event.key() == Qt.Key.Key_Plus:
            self._loupe_zoom = min(20, self._loupe_zoom + 1)
            changed = True
        elif event.key() == Qt.Key.Key_Minus:
            self._loupe_zoom = max(4, self._loupe_zoom - 1)
            changed = True
        elif event.key() == Qt.Key.Key_Tab:
            self._active_point_index = (self._active_point_index + 1) % 4
            changed = True

        if changed:
            x_px, y_px = self._clamp_to_image(x_px, y_px)
            self._manual_points[self._active_point_index]["x_px"] = x_px
            self._manual_points[self._active_point_index]["y_px"] = y_px
            self.points_changed.emit(self._computed_config_points())
            self.update()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._active_point_index is None:
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self._loupe_zoom = min(20, self._loupe_zoom + 1)
        elif delta < 0:
            self._loupe_zoom = max(4, self._loupe_zoom - 1)
        self.update()