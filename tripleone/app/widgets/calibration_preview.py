# app/widgets/calibration_preview.py
# Triple One - Schritt 2:
# Calibration Preview baut vollständig auf vision/calibration_geometry.py auf.
#
# Ziel:
# - nur 4 manuelle Marker
# - Bull wird automatisch aus der 4-Punkt-Homography berechnet
# - Overlay und Testpunkt nutzen dieselbe Geometrie
# - Präzisionslupe für Marker und Pending-Testpunkt
# - Rechtsklick = Pending-Testpunkt
# - Enter/Space = Testpunkt bestätigen
# - Esc = Pending-Testpunkt verwerfen
# - Pfeiltasten = 1 px, Shift = 5 px
# - Mausrad / +/- = Lupenzoom

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
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
)
from PyQt6.QtWidgets import QWidget

from vision.calibration_geometry import (
    build_pipeline_points,
    compute_bull_from_manual_points,
    generate_number_positions_image,
    generate_ring_polylines_image,
    generate_sector_lines_image,
    generate_twenty_segment_polygon_image,
    get_manual_labels,
)


class CalibrationPreview(QWidget):
    """
    Reines Preview-/Interaktions-Widget.

    Diese Klasse darf:
    - Bild anzeigen
    - Overlay zeichnen
    - Marker bewegen
    - Lupe anzeigen
    - Pending-Testpunkt setzen/bestätigen

    Diese Klasse darf NICHT:
    - eigene Homography rechnen
    - eigene Scorelogik rechnen
    - eigene Ring-/Winkelgeometrie erfinden
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

        self._status_text: str = ""

        self._active_point_index: Optional[int] = None
        self._dragging = False
        self._drag_anchor_img: Optional[Tuple[float, float]] = None
        self._drag_anchor_point: Optional[Tuple[float, float]] = None

        self._confirmed_test_point: Optional[Tuple[int, int]] = None
        self._pending_test_point: Optional[Tuple[int, int]] = None

        self._loupe_zoom = 10
        self._loupe_radius_px = 24
        self._marker_radius_px = 10
        self._selection_distance_px = 22

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
        self._confirmed_test_point = (int(x_px), int(y_px))
        self._pending_test_point = None
        self.update()

    def clear_test_point(self) -> None:
        self._confirmed_test_point = None
        self._pending_test_point = None
        self.update()

    def set_overlay_config(self, config: Dict) -> None:
        self._overlay_config = deepcopy(config)

        points = deepcopy(config.get("points", []))
        self._manual_points = points[:4]

        if self._active_point_index is None and len(self._manual_points) >= 1:
            self._active_point_index = 0
        elif self._active_point_index is not None and self._active_point_index >= len(self._manual_points):
            self._active_point_index = max(0, len(self._manual_points) - 1) if self._manual_points else None

        self.update()

    # ------------------------------------------------------------
    # Geometrie / Koordinaten
    # ------------------------------------------------------------

    def _image_size(self) -> Tuple[int, int]:
        if self._frame is None:
            return 1280, 720
        return self._frame.width(), self._frame.height()

    def _frame_rect(self) -> QRectF:
        """
        Letterboxed Rechteck, in dem das Bild dargestellt wird.
        """
        widget_w = self.width()
        widget_h = self.height()

        img_w, img_h = self._image_size()
        if img_w <= 0 or img_h <= 0:
            return QRectF(0, 0, widget_w, widget_h)

        scale = min(widget_w / img_w, widget_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale

        x = (widget_w - draw_w) / 2.0
        y = (widget_h - draw_h) / 2.0
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
            return QPointF(0.0, 0.0)

        x = rect.x() + (x_px / img_w) * rect.width()
        y = rect.y() + (y_px / img_h) * rect.height()
        return QPointF(x, y)

    def _clamp_to_image(self, x_px: float, y_px: float) -> Tuple[int, int]:
        img_w, img_h = self._image_size()
        x_px = max(0, min(img_w - 1, round(x_px)))
        y_px = max(0, min(img_h - 1, round(y_px)))
        return int(x_px), int(y_px)

    def _pipeline_points(self) -> List[Dict[str, int]]:
        return build_pipeline_points(self._manual_points)

    # ------------------------------------------------------------
    # Overlay-Zeichnung
    # ------------------------------------------------------------

    def _draw_empty_state(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(self._label_font)
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Kamerabild")

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

        painter.setPen(QPen(color, thickness))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(wpt, radius, radius)
        painter.drawLine(
            QPointF(wpt.x() - radius - 6, wpt.y()),
            QPointF(wpt.x() + radius + 6, wpt.y()),
        )
        painter.drawLine(
            QPointF(wpt.x(), wpt.y() - radius - 6),
            QPointF(wpt.x(), wpt.y() + radius + 6),
        )

        if label:
            rect = QRectF(wpt.x() + 12, wpt.y() - 12, 28, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 30, 30, 200))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_overlay(self, painter: QPainter) -> None:
        if len(self._manual_points) != 4:
            return

        pipeline_points = self._pipeline_points()

        alpha = float(self._overlay_config.get("overlay_alpha", 0.35))
        show_sector_lines = bool(self._overlay_config.get("show_sector_lines", True))
        show_numbers = bool(self._overlay_config.get("show_numbers", True))

        # 20er-Segment rot
        poly20 = generate_twenty_segment_polygon_image(pipeline_points)
        if poly20 is not None and len(poly20) >= 3:
            path = QPainterPath()
            first = self._image_to_widget(poly20[0, 0], poly20[0, 1])
            path.moveTo(first)

            for pt in poly20[1:]:
                wpt = self._image_to_widget(pt[0], pt[1])
                path.lineTo(wpt)

            path.closeSubpath()

            painter.setPen(QPen(QColor(255, 140, 170, int(255 * max(0.30, alpha))), 1.2))
            painter.setBrush(QColor(255, 0, 80, int(255 * max(0.18, alpha * 0.90))))
            painter.drawPath(path)

        ring_pen = QPen(QColor(120, 220, 255, int(255 * alpha)), 1.5)
        sector_pen = QPen(QColor(180, 240, 255, int(255 * alpha)), 1.2)

        for poly in generate_ring_polylines_image(pipeline_points):
            path = QPainterPath()
            first = self._image_to_widget(poly[0, 0], poly[0, 1])
            path.moveTo(first)

            for pt in poly[1:]:
                wpt = self._image_to_widget(pt[0], pt[1])
                path.lineTo(wpt)

            painter.setPen(ring_pen)
            painter.drawPath(path)

        if show_sector_lines:
            painter.setPen(sector_pen)
            for line in generate_sector_lines_image(pipeline_points):
                p0 = self._image_to_widget(line[0, 0], line[0, 1])
                p1 = self._image_to_widget(line[1, 0], line[1, 1])
                painter.drawLine(p0, p1)

        if show_numbers:
            painter.setFont(self._small_font)
            painter.setPen(QPen(QColor(230, 255, 255, 220), 1))

            for text, pos in generate_number_positions_image(pipeline_points):
                wp = self._image_to_widget(pos[0], pos[1])
                rect = QRectF(wp.x() - 10, wp.y() - 8, 20, 16)
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Automatisch berechneter Bull
        bull = compute_bull_from_manual_points(self._manual_points)
        self._draw_crosshair(
            painter,
            bull["x_px"],
            bull["y_px"],
            QColor(0, 255, 255, 230),
            radius=10,
            thickness=2,
            label="C",
        )

    def _draw_manual_points(self, painter: QPainter) -> None:
        labels = get_manual_labels()

        for idx, point in enumerate(self._manual_points):
            color = QColor(255, 255, 255, 235)
            if self._active_point_index == idx:
                color = QColor(255, 210, 60, 250)

            wpt = self._image_to_widget(point["x_px"], point["y_px"])

            painter.setPen(QPen(QColor(20, 20, 20, 255), 2))
            painter.setBrush(color)
            painter.drawEllipse(wpt, self._marker_radius_px, self._marker_radius_px)

            label_rect = QRectF(wpt.x() + 10, wpt.y() - 12, 88, 20)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(30, 30, 30, 190))
            painter.drawRoundedRect(label_rect, 4, 4)

            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(self._label_font)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, labels[idx])

    def _draw_confirmed_test_point(self, painter: QPainter) -> None:
        if self._confirmed_test_point is None:
            return

        self._draw_crosshair(
            painter,
            self._confirmed_test_point[0],
            self._confirmed_test_point[1],
            QColor(40, 255, 120, 240),
            radius=12,
            thickness=2,
            label=None,
        )

    def _draw_pending_test_point(self, painter: QPainter) -> None:
        if self._pending_test_point is None:
            return

        self._draw_crosshair(
            painter,
            self._pending_test_point[0],
            self._pending_test_point[1],
            QColor(255, 190, 40, 240),
            radius=12,
            thickness=2,
            label="T",
        )

    def _draw_status(self, painter: QPainter) -> None:
        if not self._status_text:
            return

        rect = QRectF(10, self.height() - 28, self.width() - 20, 20)
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(self._small_font)
        painter.drawText(
            rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self._status_text,
        )

    # ------------------------------------------------------------
    # Lupe
    # ------------------------------------------------------------

    def _loupe_target(self) -> Optional[Tuple[int, int, str]]:
        if self._pending_test_point is not None:
            return self._pending_test_point[0], self._pending_test_point[1], "Präzisions-Testpunkt"

        if self._active_point_index is not None and 0 <= self._active_point_index < len(self._manual_points):
            p = self._manual_points[self._active_point_index]
            return p["x_px"], p["y_px"], get_manual_labels()[self._active_point_index]

        return None

    def _draw_loupe(self, painter: QPainter) -> None:
        if self._frame is None:
            return

        target = self._loupe_target()
        if target is None:
            return

        cx, cy, title = target

        src_size = self._loupe_radius_px * 2 + 1
        left = max(0, cx - self._loupe_radius_px)
        top = max(0, cy - self._loupe_radius_px)

        img_w, img_h = self._image_size()
        left = min(left, max(0, img_w - src_size))
        top = min(top, max(0, img_h - src_size))

        cropped = self._frame.copy(
            left,
            top,
            min(src_size, img_w - left),
            min(src_size, img_h - top),
        )
        if cropped.isNull():
            return

        loupe_w = cropped.width() * self._loupe_zoom
        loupe_h = cropped.height() * self._loupe_zoom

        # Hochwertig hochskalieren, damit die Lupe weniger pixelig wirkt
        scaled_loupe = cropped.scaled(
            int(loupe_w),
            int(loupe_h),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        target_x = 18
        target_y = 18
        if target_x + loupe_w > self.width() - 20:
            target_x = self.width() - loupe_w - 20

        rect = QRectF(target_x, target_y, loupe_w, loupe_h)

        painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
        painter.setBrush(QColor(10, 10, 10, 215))
        painter.drawRoundedRect(rect.adjusted(-6, -26, 6, 46), 10, 10)

        painter.drawImage(rect, scaled_loupe)

        painter.setFont(self._small_font)
        painter.setPen(QPen(QColor(255, 240, 180), 1))
        painter.drawText(
            QRectF(rect.x(), rect.y() - 22, rect.width(), 18),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"{title} – {self._loupe_zoom}x",
        )

        local_x = (cx - left) * self._loupe_zoom + (self._loupe_zoom / 2.0)
        local_y = (cy - top) * self._loupe_zoom + (self._loupe_zoom / 2.0)

        lx = rect.x() + local_x
        ly = rect.y() + local_y

        painter.setPen(QPen(QColor(60, 255, 120), 2))
        painter.drawLine(QPointF(rect.x(), ly), QPointF(rect.x() + rect.width(), ly))
        painter.drawLine(QPointF(lx, rect.y()), QPointF(lx, rect.y() + rect.height()))
        painter.drawEllipse(QPointF(lx, ly), 7, 7)

        # 1px Raster
        painter.setPen(QPen(QColor(255, 255, 255, 35), 1))
        for i in range(cropped.width() + 1):
            x = rect.x() + i * self._loupe_zoom
            painter.drawLine(QPointF(x, rect.y()), QPointF(x, rect.y() + rect.height()))
        for j in range(cropped.height() + 1):
            y = rect.y() + j * self._loupe_zoom
            painter.drawLine(QPointF(rect.x(), y), QPointF(rect.x() + rect.width(), y))

        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawText(
            QRectF(rect.x(), rect.y() + rect.height() + 6, rect.width(), 34),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            "Pfeiltasten: 1 px   |   Shift: 5 px\nEnter/Space: Test bestätigen   |   Esc: verwerfen",
        )

    # ------------------------------------------------------------
    # Paint
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
        self._draw_manual_points(painter)
        self._draw_confirmed_test_point(painter)
        self._draw_pending_test_point(painter)
        self._draw_status(painter)
        self._draw_loupe(painter)

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
            self._pending_test_point = (x_px, y_px)
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

        if event.modifiers() & Qt.KeyboardModifier.AltModifier:
            dx *= 0.20
            dy *= 0.20

        new_x, new_y = self._clamp_to_image(start_px + dx, start_py + dy)

        self._manual_points[self._active_point_index]["x_px"] = new_x
        self._manual_points[self._active_point_index]["y_px"] = new_y

        self.points_changed.emit(build_pipeline_points(self._manual_points))
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._dragging = False
        self._drag_anchor_img = None
        self._drag_anchor_point = None
        self.update()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        # Direkte Auswahl des aktiven Kalibrierpunkts per Tastatur:
        # 1 = P1, 2 = P2, 3 = P3, 4 = P4
        if event.key() == Qt.Key.Key_1:
            if len(self._manual_points) >= 1:
                self._active_point_index = 0
                self.update()
            return

        if event.key() == Qt.Key.Key_2:
            if len(self._manual_points) >= 2:
                self._active_point_index = 1
                self.update()
            return

        if event.key() == Qt.Key.Key_3:
            if len(self._manual_points) >= 3:
                self._active_point_index = 2
                self.update()
            return

        if event.key() == Qt.Key.Key_4:
            if len(self._manual_points) >= 4:
                self._active_point_index = 3
                self.update()
            return
        # Pending-Testpunkt hat Vorrang
        if self._pending_test_point is not None:
            x_px, y_px = self._pending_test_point
            step = 5 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1

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
            elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
                self._confirmed_test_point = self._pending_test_point
                self._pending_test_point = None
                self.test_point_selected.emit(
                    self._confirmed_test_point[0],
                    self._confirmed_test_point[1],
                )
                self.update()
                return
            elif event.key() == Qt.Key.Key_Escape:
                self._pending_test_point = None
                self.update()
                return
            elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self._loupe_zoom = min(20, self._loupe_zoom + 1)
                self.update()
                return
            elif event.key() == Qt.Key.Key_Minus:
                self._loupe_zoom = max(4, self._loupe_zoom - 1)
                self.update()
                return

            if changed:
                self._pending_test_point = self._clamp_to_image(x_px, y_px)
                self.update()
                return

        # Marker-Feinjustage
        if self._active_point_index is None or not (0 <= self._active_point_index < len(self._manual_points)):
            return

        x_px = self._manual_points[self._active_point_index]["x_px"]
        y_px = self._manual_points[self._active_point_index]["y_px"]
        step = 5 if (event.modifiers() & Qt.KeyboardModifier.ShiftModifier) else 1

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
        elif event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._loupe_zoom = min(20, self._loupe_zoom + 1)
            self.update()
            return
        elif event.key() == Qt.Key.Key_Minus:
            self._loupe_zoom = max(4, self._loupe_zoom - 1)
            self.update()
            return
        elif event.key() == Qt.Key.Key_Tab:
            self._active_point_index = (self._active_point_index + 1) % 4
            self.update()
            return

        if changed:
            x_px, y_px = self._clamp_to_image(x_px, y_px)
            self._manual_points[self._active_point_index]["x_px"] = x_px
            self._manual_points[self._active_point_index]["y_px"] = y_px
            self.points_changed.emit(build_pipeline_points(self._manual_points))
            self.update()

    def wheelEvent(self, event) -> None:
        if self._pending_test_point is None and self._active_point_index is None:
            return

        delta = event.angleDelta().y()
        if delta > 0:
            self._loupe_zoom = min(20, self._loupe_zoom + 1)
        elif delta < 0:
            self._loupe_zoom = max(4, self._loupe_zoom - 1)

        self.update()