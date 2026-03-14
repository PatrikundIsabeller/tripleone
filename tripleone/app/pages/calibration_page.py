# app/pages/calibration_page.py
# Diese Datei enthält die Kalibrierungsseite.
# Hier wird pro Kamera das Livebild mit Homography-Overlay angezeigt.
#
# Phase 3.4b:
# - 4 Boundary-Punkte + 1 Center-Punkt
# - echter Homography-Workflow
#
# Phase 3.5:
# - Klick -> Score
# - visuelle Trefferanzeige
#
# Phase 4.1:
# - bis zu 3 Darts nacheinander erkennen
# - inkrementelle Referenz
# - aktuelle Trefferliste + Gesamtscore

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDoubleSpinBox,
    QCheckBox,
    QFrame,
    QMessageBox,
    QSizePolicy,
)

from vision.camera_manager import CameraWorker
from vision.board_model import calculate_board_hit_from_image_point
from vision.dart_detector import DartDetector, DartDetectionResult
from app.widgets.calibration_preview import CalibrationPreview


class CalibrationCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("CalibrationCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(360)

        self.setStyleSheet("""
            QFrame#CalibrationCard {
                background-color: #1f1f1f;
                border: 1px solid #333333;
                border-radius: 12px;
            }
            QLabel {
                color: #f2f2f2;
            }
            QDoubleSpinBox {
                background-color: #2b2b2b;
                color: #f2f2f2;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 4px;
                min-height: 28px;
            }
            QCheckBox {
                color: #f2f2f2;
            }
            QPushButton {
                background-color: #2d6cdf;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a78e8;
            }
        """)

        self.worker: Optional[CameraWorker] = None
        self.camera_config: Dict = {}
        self._syncing_from_preview = False
        self._points: List[Dict[str, int]] = []

        self.detector = DartDetector()
        self.last_frame_bgr: Optional[np.ndarray] = None

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.device_info_label = QLabel("Gerät: keine Kamera gespeichert")
        self.device_info_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self.help_label = QLabel(
            "Tipp:\n"
            "- Ziehe P1 bis P4 exakt auf den äußeren Double-Ring\n"
            "- P1 = Grenze 20|1\n"
            "- P2 = Grenze 6|10\n"
            "- P3 = Grenze 3|19\n"
            "- P4 = Grenze 11|14\n"
            "- C  = Bull-Mittelpunkt\n"
            "- Rechtsklick ins Bild = Score-Test\n"
            "- Phase 4.1: Leeres Board speichern -> Auto-Erkennung scharf -> 3 Darts werfen"
        )
        self.help_label.setStyleSheet("font-size: 12px; color: #d8d8d8;")

        self.preview = CalibrationPreview()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.10, 1.00)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(0.90)

        self.show_numbers_check = QCheckBox("Zahlen anzeigen")
        self.show_numbers_check.setChecked(True)

        self.show_sector_lines_check = QCheckBox("Sektorlinien anzeigen")
        self.show_sector_lines_check.setChecked(True)

        self.reset_button = QPushButton("5 Punkte zurücksetzen")
        self.reset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_button.clicked.connect(self.reset_points)

        self.capture_reference_button = QPushButton("Leeres Board speichern")
        self.capture_reference_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.capture_reference_button.clicked.connect(self.capture_reference_frame)

        self.arm_detector_button = QPushButton("Auto-Erkennung scharf")
        self.arm_detector_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arm_detector_button.clicked.connect(self.arm_detector)

        self.reset_detector_button = QPushButton("3-Dart-Runde zurücksetzen")
        self.reset_detector_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_detector_button.clicked.connect(self.reset_detector_round)

        self.point_info_label = QLabel("Punkte: P1 bis P4 + C noch nicht angepasst")
        self.point_info_label.setWordWrap(True)
        self.point_info_label.setStyleSheet("font-size: 12px; color: #8ee6ff; font-weight: 600;")

        self.test_result_label = QLabel("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setStyleSheet("font-size: 12px; color: #8effc9; font-weight: 700;")

        self.throw_list_label = QLabel("Darts: - / - / -")
        self.throw_list_label.setWordWrap(True)
        self.throw_list_label.setStyleSheet("font-size: 12px; color: #ffdb8a; font-weight: 700;")

        self.detector_status_label = QLabel("Auto-Erkennung: keine Referenz")
        self.detector_status_label.setWordWrap(True)
        self.detector_status_label.setStyleSheet("font-size: 12px; color: #ffdb8a; font-weight: 700;")

        self.status_label = QLabel("Status: nicht gestartet")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self._build_ui()
        self._connect_signals()
        self._push_overlay_to_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self.title_label)
        layout.addWidget(self.device_info_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.preview, 1)

        row_1 = QHBoxLayout()
        row_1.setSpacing(8)
        row_1.addWidget(QLabel("Alpha:"))
        row_1.addWidget(self.alpha_spin)
        row_1.addStretch()
        row_1.addWidget(self.reset_button)

        row_2 = QHBoxLayout()
        row_2.setSpacing(8)
        row_2.addWidget(self.show_numbers_check)
        row_2.addStretch()
        row_2.addWidget(self.show_sector_lines_check)

        row_3 = QHBoxLayout()
        row_3.setSpacing(8)
        row_3.addWidget(self.capture_reference_button)
        row_3.addWidget(self.arm_detector_button)
        row_3.addWidget(self.reset_detector_button)

        layout.addLayout(row_1)
        layout.addLayout(row_2)
        layout.addLayout(row_3)
        layout.addWidget(self.point_info_label)
        layout.addWidget(self.test_result_label)
        layout.addWidget(self.throw_list_label)
        layout.addWidget(self.detector_status_label)
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        self.alpha_spin.valueChanged.connect(self._push_overlay_to_preview)
        self.show_numbers_check.toggled.connect(self._push_overlay_to_preview)
        self.show_sector_lines_check.toggled.connect(self._push_overlay_to_preview)
        self.preview.points_changed.connect(self._handle_points_changed)
        self.preview.test_point_selected.connect(self._handle_test_point_selected)

    def _default_points(self) -> List[Dict[str, int]]:
        frame_width = self._get_frame_width()
        frame_height = self._get_frame_height()
        from config.calibration_settings import _default_points as get_default_points
        return get_default_points(frame_width, frame_height)

    def _handle_points_changed(self, points: list) -> None:
        self._syncing_from_preview = True
        try:
            self._points = deepcopy(points)
            self._update_point_info_label()
        finally:
            self._syncing_from_preview = False

    def _handle_test_point_selected(self, x_px: int, y_px: int) -> None:
        result = calculate_board_hit_from_image_point(x_px, y_px, self.get_calibration_config())

        if result is None:
            self.test_result_label.setText(f"Testpunkt: Fehler bei Rückprojektion | Punkt=({x_px}, {y_px})")
            return

        score_text = f"{result.label} = {result.score}"

        self.test_result_label.setText(
            f"Testpunkt: {score_text} | "
            f"Ring: {result.ring_name} | "
            f"Radius: {result.radius:.3f} | "
            f"Board=({result.board_x:.3f}, {result.board_y:.3f}) | "
            f"Bild=({result.image_x_px}, {result.image_y_px})"
        )

        self.preview.set_test_point(x_px, y_px)

    def _update_throw_list_label(self) -> None:
        labels = [dart.score_label for dart in self.detector.detected_darts]
        while len(labels) < 3:
            labels.append("-")

        total = sum(d.score_value for d in self.detector.detected_darts)
        self.throw_list_label.setText(
            f"Darts: {labels[0]} / {labels[1]} / {labels[2]} | Gesamt: {total}"
        )

    def _handle_auto_detection_result(self, result: DartDetectionResult) -> None:
        self.preview.set_test_point(result.x_px, result.y_px)

        self.test_result_label.setText(
            f"Auto-Treffer: {result.score_label} = {result.score_value} | "
            f"Ring: {result.ring_name} | "
            f"Area: {result.contour_area:.1f} | "
            f"Board=({result.board_x:.3f}, {result.board_y:.3f}) | "
            f"Bild=({result.x_px}, {result.y_px})"
        )

        self._update_throw_list_label()

        if len(self.detector.detected_darts) >= self.detector.max_darts_per_round:
            self.detector_status_label.setText("Auto-Erkennung: 3 Darts erkannt, Runde beendet")
        else:
            remaining = self.detector.max_darts_per_round - len(self.detector.detected_darts)
            self.detector_status_label.setText(
                f"Auto-Erkennung: Treffer erkannt, warte auf nächsten Dart ({remaining} offen)"
            )

    def _update_point_info_label(self) -> None:
        points = self._points if self._points else self._default_points()
        labels = ["P1", "P2", "P3", "P4", "C"]
        parts = []

        for i, point in enumerate(points):
            x_px = int(point.get("x_px", 0))
            y_px = int(point.get("y_px", 0))
            parts.append(f"{labels[i]}=({x_px},{y_px})")

        self.point_info_label.setText("Punkte: " + " | ".join(parts))

    def _get_frame_width(self) -> int:
        return max(320, int(self.camera_config.get("width", 1280)))

    def _get_frame_height(self) -> int:
        return max(240, int(self.camera_config.get("height", 720)))

    def _push_overlay_to_preview(self) -> None:
        if self._syncing_from_preview:
            return
        self.preview.set_overlay_config(self.get_calibration_config())

    def reset_points(self) -> None:
        self._points = self._default_points()
        self.preview.set_overlay_config(self.get_calibration_config())
        self.preview.clear_test_point()
        self.test_result_label.setText("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self._update_point_info_label()

    def reset_detector_round(self) -> None:
        self.detector.reset_round()
        self.preview.clear_test_point()
        self.test_result_label.setText("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self._update_throw_list_label()

        if self.detector.reference_gray is not None:
            self.detector_status_label.setText("Auto-Erkennung: Runde zurückgesetzt, wieder scharf schaltbar")
        else:
            self.detector_status_label.setText("Auto-Erkennung: keine Referenz")

    def capture_reference_frame(self) -> None:
        if self.last_frame_bgr is None:
            self.detector_status_label.setText("Auto-Erkennung: kein Livebild für Referenz vorhanden")
            return

        self.detector.set_reference_frame(self.last_frame_bgr)
        self.preview.clear_test_point()
        self.test_result_label.setText("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self._update_throw_list_label()
        self.detector_status_label.setText("Auto-Erkennung: Referenz gespeichert (leeres Board)")

    def arm_detector(self) -> None:
        ok = self.detector.arm()
        if ok:
            self.detector_status_label.setText("Auto-Erkennung: scharf, jetzt bis zu 3 Darts werfen")
        else:
            self.detector_status_label.setText("Auto-Erkennung: zuerst Referenzbild speichern")

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")
        self.preview.set_status_text(text)

    def set_camera_config(self, camera_config: Dict) -> None:
        self.camera_config = dict(camera_config)

        device_id = int(camera_config.get("device_id", -1))
        enabled = bool(camera_config.get("enabled", True))

        if not enabled:
            self.device_info_label.setText("Gerät: deaktiviert")
        elif device_id < 0:
            self.device_info_label.setText("Gerät: keine Kamera ausgewählt")
        else:
            self.device_info_label.setText(
                f"Gerät: Index {device_id} – {camera_config.get('width', 1280)}x{camera_config.get('height', 720)} @ {camera_config.get('fps', 30)} FPS"
            )

    def set_calibration_config(self, calibration_config: Dict) -> None:
        self.alpha_spin.setValue(float(calibration_config.get("overlay_alpha", 0.90)))
        self.show_numbers_check.setChecked(bool(calibration_config.get("show_numbers", True)))
        self.show_sector_lines_check.setChecked(bool(calibration_config.get("show_sector_lines", True)))
        self._points = deepcopy(calibration_config.get("points", self._default_points()))
        self.preview.set_overlay_config(self.get_calibration_config())
        self._update_point_info_label()
        self._update_throw_list_label()

    def get_calibration_config(self) -> Dict:
        points = deepcopy(self._points if self._points else self._default_points())

        return {
            "frame_width": self._get_frame_width(),
            "frame_height": self._get_frame_height(),
            "overlay_alpha": float(self.alpha_spin.value()),
            "show_numbers": bool(self.show_numbers_check.isChecked()),
            "show_sector_lines": bool(self.show_sector_lines_check.isChecked()),
            "points": points,
        }

    def _qimage_to_bgr(self, image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        width = converted.width()
        height = converted.height()

        ptr = converted.bits()
        ptr.setsize(height * width * 4)

        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def update_preview(self, image: QImage) -> None:
        self.preview.set_frame(image)

        try:
            self.last_frame_bgr = self._qimage_to_bgr(image)
        except Exception:
            self.last_frame_bgr = None
            return

        if self.last_frame_bgr is None:
            return

        detection = self.detector.process_frame(self.last_frame_bgr, self.get_calibration_config())
        if detection is not None:
            self._handle_auto_detection_result(detection)

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.preview.clear_frame()
        self.last_frame_bgr = None
        self.test_result_label.setText("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self.set_status("gestoppt")

    def start_worker(self) -> None:
        self.stop_worker()

        enabled = bool(self.camera_config.get("enabled", True))
        device_id = int(self.camera_config.get("device_id", -1))

        if not enabled:
            self.preview.clear_frame()
            self.set_status("deaktiviert")
            return

        if device_id < 0:
            self.preview.clear_frame()
            self.set_status("keine Kamera gewählt")
            return

        self.worker = CameraWorker(
            device_id=device_id,
            width=int(self.camera_config.get("width", 1280)),
            height=int(self.camera_config.get("height", 720)),
            fps=int(self.camera_config.get("fps", 30)),
            rotation=int(self.camera_config.get("rotation", 0)),
            flip=bool(self.camera_config.get("flip", False)),
        )

        self.worker.frame_ready.connect(self.update_preview)
        self.worker.status_changed.connect(self.set_status)
        self.worker.start()


class CalibrationPage(QWidget):
    def __init__(
        self,
        camera_config: Dict,
        calibration_config: Dict,
        save_callback,
        parent=None,
    ):
        super().__init__(parent)

        self.camera_config = deepcopy(camera_config)
        self.calibration_config = deepcopy(calibration_config)
        self.save_callback = save_callback

        self.title_label = QLabel("TripleOne – Kalibrierung / Phase 4.1")
        self.title_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.info_label = QLabel(
            "Phase 4.1:\n"
            "3 Darts nacheinander mit inkrementeller Referenz erkennen.\n"
            "Workflow: Leeres Board speichern -> Auto-Erkennung scharf -> bis zu 3 Darts werfen."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 13px; color: #cccccc;")

        self.card_1 = CalibrationCard("Kamera 1")
        self.card_2 = CalibrationCard("Kamera 2")
        self.card_3 = CalibrationCard("Kamera 3")
        self.cards: List[CalibrationCard] = [self.card_1, self.card_2, self.card_3]

        self.start_button = QPushButton("Livebilder starten / aktualisieren")
        self.start_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_button.clicked.connect(self.apply_preview)

        self.stop_button = QPushButton("Alle Kameras stoppen")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_all_cameras)

        self.save_button = QPushButton("Kalibrierung speichern")
        self.save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_button.clicked.connect(self.save_settings)

        self.reset_all_button = QPushButton("Alle 5-Punkt-Overlays zurücksetzen")
        self.reset_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_all_button.clicked.connect(self.reset_all_overlays)

        self._build_ui()
        self._load_data_into_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet("""
            QPushButton {
                background-color: #2d6cdf;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #3a78e8;
            }
        """)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.reset_all_button)
        buttons_layout.addStretch()

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(14)
        cards_layout.addWidget(self.card_1, 1)
        cards_layout.addWidget(self.card_2, 1)
        cards_layout.addWidget(self.card_3, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(cards_layout, 1)

    def _load_data_into_ui(self) -> None:
        cam_list = self.camera_config.get("cameras", [])
        cal_list = self.calibration_config.get("cameras", [])

        for i, card in enumerate(self.cards):
            if i < len(cam_list):
                card.set_camera_config(cam_list[i])
            if i < len(cal_list):
                card.set_calibration_config(cal_list[i])

    def update_camera_config(self, new_camera_config: Dict) -> None:
        self.camera_config = deepcopy(new_camera_config)
        cam_list = self.camera_config.get("cameras", [])

        for i, card in enumerate(self.cards):
            if i < len(cam_list):
                card.set_camera_config(cam_list[i])

    def collect_calibration_config(self) -> Dict:
        new_data = {"cameras": []}
        for idx, card in enumerate(self.cards):
            item = {"name": f"Kamera {idx + 1}"}
            item.update(card.get_calibration_config())
            new_data["cameras"].append(item)
        return new_data

    def apply_preview(self) -> None:
        for card in self.cards:
            card.start_worker()

    def save_settings(self) -> None:
        self.calibration_config = self.collect_calibration_config()
        self.save_callback(self.calibration_config)

        QMessageBox.information(
            self,
            "Gespeichert",
            "Die Kalibrierung wurde gespeichert.",
        )

    def reset_all_overlays(self) -> None:
        for card in self.cards:
            card.reset_points()

    def stop_all_cameras(self) -> None:
        for card in self.cards:
            card.stop_worker()
