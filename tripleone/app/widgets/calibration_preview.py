# app/widgets/calibration_preview.py
# Diese Datei enthält das Vorschau-Widget für die Kalibrierungsseite.
# Es zeigt das Livebild der Kamera und zeichnet darüber ein per Homography
# verzerrtes Dartboard-Overlay.
#
# Phase 3.4b:
# - 4 Boundary-Punkte + 1 Center-Punkt
# - echtes Homography-Overlay
# - Zoom-Lupe für Hover / Drag
# - konsistente Geometrie mit vision/board_model.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QWidget

from vision.board_model import (
    SECTOR_ORDER,
    RING_RADII,
    BOUNDARY_ANGLES_DEG,
    build_overlay_to_image_homography,
)


POINT_LABELS = ["P1", "P2", "P3", "P4", "C"]


class CalibrationPreview(QWidget):
    points_changed = pyqtSignal(list)
    test_point_selected = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMouseTracking(True)

        self._image: Optional[QImage] = None
        self._status_text = "Keine Vorschau"

        self._overlay_config: Dict = {
            "frame_width": 1280,
            "frame_height": 720,
            "overlay_alpha": 0.90,
            "show_numbers": True,
            "show_sector_lines": True,
            "points": [
                {"x_px": 930, "y_px": 233},
                {"x_px": 1051, "y_px": 609},
                {"x_px": 462, "y_px": 478},
                {"x_px": 621, "y_px": 217},
                {"x_px": 748, "y_px": 361},
            ],
        }

        self._dragging_index: Optional[int] = None
        self._hover_index: Optional[int] = None
        self._test_point_px: Optional[Tuple[int, int]] = None
        self._last_mouse_frame_pos: Optional[Tuple[int, int]] = None

        self._loupe_source_half_size = 28
        self._loupe_size_px = 220
        self._loupe_margin_px = 14

    def set_status_text(self, text: str) -> None:
        self._status_text = text
        self.update()

    def set_frame(self, image: QImage) -> None:
        self._image = image
        self.update()

    def clear_frame(self) -> None:
        self._image = None
        self._test_point_px = None
        self._last_mouse_frame_pos = None
        self.update()

    def set_overlay_config(self, config: Dict) -> None:
        self._overlay_config = dict(config)
        self.update()

    def set_test_point(self, x_px: int, y_px: int) -> None:
        self._test_point_px = (x_px, y_px)
        self.update()

    def clear_test_point(self) -> None:
        self._test_point_px = None
        self.update()

    def _get_image_rect(self) -> Optional[QRectF]:
        if self._image is None or self._image.width() <= 0 or self._image.height() <= 0:
            return None

        widget_w = self.width()
        widget_h = self.height()
        image_w = self._image.width()
        image_h = self._image.height()

        scale = min(widget_w / image_w, widget_h / image_h)
        draw_w = image_w * scale
        draw_h = image_h * scale

        x = (widget_w - draw_w) / 2.0
        y = (widget_h - draw_h) / 2.0

        return QRectF(x, y, draw_w, draw_h)

    def _frame_to_widget(self, image_rect: QRectF, x_px: float, y_px: float) -> Tuple[float, float]:
        frame_width = max(1, int(self._overlay_config.get("frame_width", 1280)))
        frame_height = max(1, int(self._overlay_config.get("frame_height", 720)))

        x = image_rect.left() + (x_px / frame_width) * image_rect.width()
        y = image_rect.top() + (y_px / frame_height) * image_rect.height()
        return x, y

    def _widget_to_frame(self, image_rect: QRectF, x_widget: float, y_widget: float) -> Tuple[int, int]:
        frame_width = max(1, int(self._overlay_config.get("frame_width", 1280)))
        frame_height = max(1, int(self._overlay_config.get("frame_height", 720)))

        x_px = int(round(((x_widget - image_rect.left()) / image_rect.width()) * frame_width))
        y_px = int(round(((y_widget - image_rect.top()) / image_rect.height()) * frame_height))

        x_px = max(0, min(x_px, frame_width - 1))
        y_px = max(0, min(y_px, frame_height - 1))
        return x_px, y_px

    def _get_points_frame(self) -> List[Tuple[int, int]]:
        points = self._overlay_config.get("points", [])
        result = []

        for i in range(5):
            raw = points[i] if i < len(points) and isinstance(points[i], dict) else {"x_px": 0, "y_px": 0}
            result.append((int(raw.get("x_px", 0)), int(raw.get("y_px", 0))))
        return result

    def _set_point(self, index: int, x_px: int, y_px: int) -> None:
        frame_width = max(1, int(self._overlay_config.get("frame_width", 1280)))
        frame_height = max(1, int(self._overlay_config.get("frame_height", 720)))

        x_px = max(0, min(x_px, frame_width - 1))
        y_px = max(0, min(y_px, frame_height - 1))

        points = list(self._overlay_config.get("points", []))
        while len(points) < 5:
            points.append({"x_px": 0, "y_px": 0})

        points[index] = {"x_px": x_px, "y_px": y_px}
        self._overlay_config["points"] = points

    def _find_near_point(self, image_rect: QRectF, x_widget: float, y_widget: float) -> Optional[int]:
        points = self._get_points_frame()

        best_idx = None
        best_dist = 999999.0

        for i, (x_px, y_px) in enumerate(points):
            xw, yw = self._frame_to_widget(image_rect, x_px, y_px)
            dx = x_widget - xw
            dy = y_widget - yw
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 24.0 and dist < best_dist:
                best_idx = i
                best_dist = dist

        return best_idx

    def _draw_text_with_outline(
        self,
        img: np.ndarray,
        text: str,
        x: int,
        y: int,
        font_scale: float,
        fill_bgra: Tuple[int, int, int, int],
        outline_bgra: Tuple[int, int, int, int],
        fill_thickness: int = 2,
        outline_thickness: int = 8,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (x, y), font, font_scale, outline_bgra, outline_thickness, lineType=cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, fill_bgra, fill_thickness, lineType=cv2.LINE_AA)

    def _draw_line_with_shadow(
        self,
        img: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        line_bgra: Tuple[int, int, int, int],
        shadow_bgra: Tuple[int, int, int, int],
        line_thickness: int = 3,
        shadow_thickness: int = 7,
    ) -> None:
        cv2.line(img, p1, p2, shadow_bgra, shadow_thickness, lineType=cv2.LINE_AA)
        cv2.line(img, p1, p2, line_bgra, line_thickness, lineType=cv2.LINE_AA)

    def _draw_circle_with_shadow(
        self,
        img: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        line_bgra: Tuple[int, int, int, int],
        shadow_bgra: Tuple[int, int, int, int],
        line_thickness: int,
        shadow_thickness: int,
    ) -> None:
        cv2.circle(img, center, radius, shadow_bgra, shadow_thickness, lineType=cv2.LINE_AA)
        cv2.circle(img, center, radius, line_bgra, line_thickness, lineType=cv2.LINE_AA)

    def _create_canonical_overlay_rgba(self, size: int = 1600) -> np.ndarray:
        """
        Erstellt die perfekte Draufsicht des Boards.
        """
        img = np.zeros((size, size, 4), dtype=np.uint8)

        alpha_factor = float(self._overlay_config.get("overlay_alpha", 0.90))
        alpha = max(0, min(255, int(alpha_factor * 255)))

        show_numbers = bool(self._overlay_config.get("show_numbers", True))
        show_sector_lines = bool(self._overlay_config.get("show_sector_lines", True))

        cx = size // 2
        cy = size // 2
        r = int(size * 0.40)

        board_fill_blue = (255, 120, 40, int(alpha * 0.45))
        sector_highlight = (255, 0, 110, int(alpha * 0.85))
        ring_white = (245, 245, 245, min(255, alpha))
        shadow = (10, 10, 10, min(220, alpha))
        text_fill = (255, 255, 255, 255)
        text_outline = (0, 0, 0, 255)

        # Grundfläche
        cv2.circle(img, (cx, cy), r, board_fill_blue, -1, lineType=cv2.LINE_AA)

        # Demo-Segment links oben leicht hervorheben, ähnlich Autodarts-Screenshot
        highlight_index = 0  # ungefähr links
        start_deg = -90.0 - 9.0 + highlight_index * 18.0
        end_deg = start_deg + 18.0

        pts = [(cx, cy)]
        for a in np.linspace(start_deg, end_deg, 20):
            rad = math.radians(a)
            pts.append((int(cx + math.cos(rad) * r), int(cy + math.sin(rad) * r)))
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillConvexPoly(img, pts_np, sector_highlight, lineType=cv2.LINE_AA)

        # Ringe
        ring_order = [
            RING_RADII["inner_bull"],
            RING_RADII["outer_bull"],
            RING_RADII["triple_inner"],
            RING_RADII["triple_outer"],
            RING_RADII["double_inner"],
            RING_RADII["double_outer"],
        ]

        for rel_radius in ring_order:
            rr = int(round(r * rel_radius))
            self._draw_circle_with_shadow(img, (cx, cy), rr, ring_white, shadow, 3, 7)

        # Sektorlinien
        if show_sector_lines:
            line_start_radius = 0
            line_end_radius = r
            start_angle_deg = -90.0 - 9.0

            for i in range(20):
                angle_deg = start_angle_deg + i * 18.0
                angle_rad = math.radians(angle_deg)
                x1 = int(cx + math.cos(angle_rad) * line_start_radius)
                y1 = int(cy + math.sin(angle_rad) * line_start_radius)
                x2 = int(cx + math.cos(angle_rad) * line_end_radius)
                y2 = int(cy + math.sin(angle_rad) * line_end_radius)
                self._draw_line_with_shadow(img, (x1, y1), (x2, y2), ring_white, shadow, 2, 6)

        # Zahlen
        if show_numbers:
            number_radius = int(r * 1.19)
            for i, value in enumerate(SECTOR_ORDER):
                angle_deg = -90.0 + i * 18.0
                angle_rad = math.radians(angle_deg)

                x = int(cx + math.cos(angle_rad) * number_radius)
                y = int(cy + math.sin(angle_rad) * number_radius)

                text = str(value)
                font_scale = 1.05
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)

                tx = x - tw // 2
                ty = y + th // 2

                self._draw_text_with_outline(
                    img=img,
                    text=text,
                    x=tx,
                    y=ty,
                    font_scale=font_scale,
                    fill_bgra=text_fill,
                    outline_bgra=text_outline,
                    fill_thickness=2,
                    outline_thickness=7,
                )

        # Center-Marker
        cv2.circle(img, (cx, cy), 5, (0, 255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 12, (0, 255, 255, 220), 1, lineType=cv2.LINE_AA)

        return img

    def _warp_overlay_to_frame(self) -> Optional[np.ndarray]:
        frame_width = max(1, int(self._overlay_config.get("frame_width", 1280)))
        frame_height = max(1, int(self._overlay_config.get("frame_height", 720)))

        overlay = self._create_canonical_overlay_rgba(size=1600)
        h = build_overlay_to_image_homography(self._overlay_config, overlay.shape[0])
        if h is None:
            return None

        try:
            warped = cv2.warpPerspective(
                overlay,
                h,
                (frame_width, frame_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )
        except cv2.error:
            return None

        return warped

    def _numpy_rgba_to_qimage(self, rgba_img: np.ndarray) -> QImage:
        h, w, ch = rgba_img.shape
        bytes_per_line = ch * w
        return QImage(rgba_img.data, w, h, bytes_per_line, QImage.Format.Format_RGBA8888).copy()

    def _frame_qimage_to_bgr(self) -> Optional[np.ndarray]:
        if self._image is None:
            return None

        image = self._image.convertToFormat(QImage.Format.Format_RGBA8888)
        width = image.width()
        height = image.height()

        ptr = image.bits()
        ptr.setsize(height * width * 4)

        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        return bgr

    def _draw_loupe(self, painter: QPainter, image_rect: QRectF) -> None:
        active_index = self._dragging_index if self._dragging_index is not None else self._hover_index
        if active_index is None:
            return

        frame_bgr = self._frame_qimage_to_bgr()
        if frame_bgr is None:
            return

        points = self._get_points_frame()
        if active_index >= len(points):
            return

        px, py = points[active_index]
        half = self._loupe_source_half_size

        h, w = frame_bgr.shape[:2]
        x1 = max(0, px - half)
        y1 = max(0, py - half)
        x2 = min(w, px + half + 1)
        y2 = min(h, py + half + 1)

        crop = frame_bgr[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return

        loupe = cv2.resize(crop, (self._loupe_size_px, self._loupe_size_px), interpolation=cv2.INTER_NEAREST)

        cx = self._loupe_size_px // 2
        cy = self._loupe_size_px // 2

        cv2.line(loupe, (0, cy), (self._loupe_size_px, cy), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.line(loupe, (cx, 0), (cx, self._loupe_size_px), (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(loupe, (cx, cy), 7, (0, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(loupe, (cx, cy), 2, (0, 255, 255), -1, lineType=cv2.LINE_AA)

        cv2.rectangle(loupe, (0, 0), (self._loupe_size_px - 1, self._loupe_size_px - 1), (255, 255, 255), 2, lineType=cv2.LINE_AA)

        label = POINT_LABELS[active_index]
        cv2.rectangle(loupe, (0, 0), (76, 28), (20, 20, 20), -1, lineType=cv2.LINE_AA)
        cv2.putText(loupe, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        loupe_rgb = cv2.cvtColor(loupe, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            loupe_rgb.data,
            loupe_rgb.shape[1],
            loupe_rgb.shape[0],
            loupe_rgb.shape[1] * 3,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(qimg)

        point_widget_x, point_widget_y = self._frame_to_widget(image_rect, px, py)
        loupe_x = int(point_widget_x + self._loupe_margin_px)
        loupe_y = int(point_widget_y - self._loupe_size_px - self._loupe_margin_px)

        if loupe_x + self._loupe_size_px > self.width() - 6:
            loupe_x = int(point_widget_x - self._loupe_size_px - self._loupe_margin_px)

        if loupe_y < 6:
            loupe_y = int(point_widget_y + self._loupe_margin_px)

        loupe_x = max(6, min(loupe_x, self.width() - self._loupe_size_px - 6))
        loupe_y = max(6, min(loupe_y, self.height() - self._loupe_size_px - 6))

        painter.fillRect(QRectF(loupe_x + 4, loupe_y + 4, self._loupe_size_px, self._loupe_size_px), QColor(0, 0, 0, 110))
        painter.drawPixmap(loupe_x, loupe_y, pixmap)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        painter.fillRect(self.rect(), QColor("#111111"))

        image_rect = self._get_image_rect()
        if image_rect is not None and self._image is not None:
            painter.drawImage(image_rect, self._image)

            warped_overlay = self._warp_overlay_to_frame()
            if warped_overlay is not None:
                qimg = self._numpy_rgba_to_qimage(warped_overlay)
                painter.drawImage(image_rect, qimg)

            self._draw_test_point(painter, image_rect)
            self._draw_point_handles(painter, image_rect)
            self._draw_loupe(painter, image_rect)
        else:
            painter.setPen(QColor("#9a9a9a"))
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._status_text)

        painter.end()

    def _draw_test_point(self, painter: QPainter, image_rect: QRectF) -> None:
        if self._test_point_px is None:
            return

        x_px, y_px = self._test_point_px
        xw, yw = self._frame_to_widget(image_rect, x_px, y_px)

        pen = QPen(QColor(0, 255, 180, 255))
        pen.setWidthF(2.2)
        painter.setPen(pen)
        painter.drawLine(int(xw - 10), int(yw), int(xw + 10), int(yw))
        painter.drawLine(int(xw), int(yw - 10), int(xw), int(yw + 10))

        ring_pen = QPen(QColor(0, 255, 180, 200))
        ring_pen.setWidthF(1.5)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(xw - 8, yw - 8, 16, 16))

    def _draw_point_handles(self, painter: QPainter, image_rect: QRectF) -> None:
        points = self._get_points_frame()

        for i, (x_px, y_px) in enumerate(points):
            xw, yw = self._frame_to_widget(image_rect, x_px, y_px)

            base_color = QColor(0, 255, 255) if i == 4 else QColor(255, 255, 255)

            if self._dragging_index == i:
                color = QColor(255, 180, 0)
                radius = 12
            elif self._hover_index == i:
                color = QColor(0, 220, 255)
                radius = 11
            else:
                color = base_color
                radius = 10 if i == 4 else 9

            pen = QPen(color)
            pen.setWidthF(2.2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(xw - radius, yw - radius, radius * 2, radius * 2))

            if i == 4:
                painter.drawLine(int(xw - 8), int(yw), int(xw + 8), int(yw))
                painter.drawLine(int(xw), int(yw - 8), int(xw), int(yw + 8))

            font = QFont("Segoe UI", 9)
            font.setBold(True)
            painter.setFont(font)

            label_rect = QRectF(xw + 8, yw - 14, 34, 22)
            painter.fillRect(label_rect, QColor(20, 20, 20, 180))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, POINT_LABELS[i])

    def mousePressEvent(self, event: QMouseEvent) -> None:
        image_rect = self._get_image_rect()
        if image_rect is None:
            super().mousePressEvent(event)
            return

        pos = event.position()
        self._last_mouse_frame_pos = self._widget_to_frame(image_rect, pos.x(), pos.y())

        if event.button() == Qt.MouseButton.RightButton:
            x_px, y_px = self._last_mouse_frame_pos
            self._test_point_px = (x_px, y_px)
            self.test_point_selected.emit(x_px, y_px)
            self.update()
            event.accept()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            idx = self._find_near_point(image_rect, pos.x(), pos.y())
            if idx is not None:
                self._dragging_index = idx
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        image_rect = self._get_image_rect()
        if image_rect is None:
            super().mouseMoveEvent(event)
            return

        pos = event.position()
        self._last_mouse_frame_pos = self._widget_to_frame(image_rect, pos.x(), pos.y())

        if self._dragging_index is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            x_px, y_px = self._last_mouse_frame_pos
            self._set_point(self._dragging_index, x_px, y_px)
            self.points_changed.emit(self._overlay_config["points"])
            self.update()
            event.accept()
            return

        hover_idx = self._find_near_point(image_rect, pos.x(), pos.y())
        self._hover_index = hover_idx

        if hover_idx is not None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging_index is not None:
            self._dragging_index = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            event.accept()
            return

        super().mouseReleaseEvent(event)