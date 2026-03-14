# app/pages/calibration_page.py
# Phase 5.4:
# - globales Referenzbild / global scharf
# - zentraler Event-Manager
# - pro Kamera Debug-Ausgabe für Detection
# - Fusion erst nach Eventende

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
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
from vision.dart_event_manager import DartEventManager, ClosedDartEvent
from vision.multi_camera_fusion import (
    CameraBoardCandidate,
    FusedDartResult,
    fuse_camera_candidates,
)
from app.widgets.calibration_preview import CalibrationPreview


class CalibrationCard(QFrame):
    def __init__(
        self,
        title: str,
        camera_slot: int,
        auto_detection_callback: Optional[Callable[[int, DartDetectionResult], None]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("CalibrationCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(360)

        self.camera_slot = camera_slot
        self.auto_detection_callback = auto_detection_callback

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
        
        # Kameraspezifisches Tuning der Detector-Parameter
        # Kamera 3 ist aktuell zu empfindlich und zählt Nachbewegungen doppelt.
        if self.camera_slot == 0:
            # Kamera 1: läuft schon recht gut
            self.detector.diff_threshold = 22
            self.detector.min_contour_area = 110.0
            self.detector.required_stable_frames = 3
            self.detector.post_hit_cooldown_seconds = 0.9

        elif self.camera_slot == 1:
            # Kamera 2: läuft schon recht gut
            self.detector.diff_threshold = 22
            self.detector.min_contour_area = 110.0
            self.detector.required_stable_frames = 3
            self.detector.post_hit_cooldown_seconds = 0.9

        elif self.camera_slot == 2:
            # Kamera 3: strenger machen gegen Schatten / Nachtrigger
            self.detector.diff_threshold = 28
            self.detector.min_contour_area = 180.0
            self.detector.required_stable_frames = 4
            self.detector.post_hit_cooldown_seconds = 1.4

        self.last_frame_bgr: Optional[np.ndarray] = None

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.device_info_label = QLabel("Gerät: keine Kamera gespeichert")
        self.device_info_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self.help_label = QLabel(
            "Tipp:\n"
            "- Ziehe P1 bis P4 exakt auf den äußeren Double-Ring\n"
            "- Rechtsklick = Score-Test\n"
            "- Referenz / Scharfstellung global oben\n"
            "- Unten siehst du jetzt die Debug-Werte der Erkennung"
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
        self.reset_button.clicked.connect(self.reset_points)

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

        self.debug_label = QLabel("Debug: -")
        self.debug_label.setWordWrap(True)
        self.debug_label.setStyleSheet("font-size: 11px; color: #d0d0d0;")

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

        layout.addLayout(row_1)
        layout.addLayout(row_2)
        layout.addWidget(self.point_info_label)
        layout.addWidget(self.test_result_label)
        layout.addWidget(self.throw_list_label)
        layout.addWidget(self.detector_status_label)
        layout.addWidget(self.debug_label)
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

        self.test_result_label.setText(
            f"Testpunkt: {result.label} = {result.score} | "
            f"Ring: {result.ring_name} | "
            f"Board=({result.board_x:.3f}, {result.board_y:.3f}) | "
            f"Bild=({result.image_x_px}, {result.image_y_px})"
        )
        self.preview.set_test_point(x_px, y_px)

    def _update_throw_list_label(self) -> None:
        labels = [dart.score_label for dart in self.detector.detected_darts]
        while len(labels) < 3:
            labels.append("-")

        total = sum(d.score_value for d in self.detector.detected_darts)
        self.throw_list_label.setText(f"Darts: {labels[0]} / {labels[1]} / {labels[2]} | Gesamt: {total}")

    def _update_debug_label(self) -> None:
        dbg = self.detector.get_debug_snapshot()

        area = f"{dbg.chosen_area:.1f}" if dbg.chosen_area is not None else "-"
        aspect = f"{dbg.chosen_aspect_ratio:.2f}" if dbg.chosen_aspect_ratio is not None else "-"
        fill = f"{dbg.chosen_fill_ratio:.2f}" if dbg.chosen_fill_ratio is not None else "-"
        tip = (
            f"({dbg.chosen_tip_x},{dbg.chosen_tip_y})"
            if dbg.chosen_tip_x is not None and dbg.chosen_tip_y is not None
            else "-"
        )
        tip_radius = f"{dbg.chosen_tip_radius:.3f}" if dbg.chosen_tip_radius is not None else "-"

        self.debug_label.setText(
            "Debug: "
            f"armed={dbg.is_armed} | "
            f"diff={dbg.diff_nonzero_pixels} | "
            f"contours={dbg.contour_count_total} | "
            f"valid={dbg.contour_count_valid} | "
            f"area={area} | "
            f"aspect={aspect} | "
            f"fill={fill} | "
            f"tip={tip} | "
            f"tip_r={tip_radius} | "
            f"reason={dbg.reject_reason} | "
            f"info={dbg.info_text}"
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

        if self.auto_detection_callback is not None:
            self.auto_detection_callback(self.camera_slot, result)

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
        self._update_debug_label()

        if self.detector.reference_gray is not None:
            self.detector_status_label.setText("Auto-Erkennung: Runde zurückgesetzt, wieder scharf schaltbar")
        else:
            self.detector_status_label.setText("Auto-Erkennung: keine Referenz")

    def capture_reference_frame(self) -> bool:
        if self.last_frame_bgr is None:
            self.detector_status_label.setText("Auto-Erkennung: kein Livebild für Referenz vorhanden")
            return False

        self.detector.set_reference_frame(self.last_frame_bgr)
        self.preview.clear_test_point()
        self.test_result_label.setText("Testpunkt / Auto-Treffer: noch keiner gesetzt")
        self._update_throw_list_label()
        self._update_debug_label()
        self.detector_status_label.setText("Auto-Erkennung: Referenz gespeichert")
        return True

    def arm_detector(self) -> bool:
        ok = self.detector.arm()
        self._update_debug_label()
        if ok:
            self.detector_status_label.setText("Auto-Erkennung: global scharf")
        else:
            self.detector_status_label.setText("Auto-Erkennung: zuerst globale Referenz speichern")
        return ok

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
        self._update_debug_label()

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
        self._update_debug_label()

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
        self._update_debug_label()

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

        self.event_manager = DartEventManager()
        self.fused_darts: List[FusedDartResult] = []

        self.title_label = QLabel("TripleOne – Kalibrierung / Phase 5.4 Debug")
        self.title_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.info_label = QLabel(
            "Phase 5.4:\n"
            "Jetzt siehst du pro Kamera die Debug-Werte der Detection. "
            "Damit können wir endlich erkennen, warum eine Kamera nichts oder nur falsche Treffer liefert."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 13px; color: #cccccc;")

        self.card_1 = CalibrationCard("Kamera 1", camera_slot=0, auto_detection_callback=self.handle_camera_auto_detection)
        self.card_2 = CalibrationCard("Kamera 2", camera_slot=1, auto_detection_callback=self.handle_camera_auto_detection)
        self.card_3 = CalibrationCard("Kamera 3", camera_slot=2, auto_detection_callback=self.handle_camera_auto_detection)
        self.cards: List[CalibrationCard] = [self.card_1, self.card_2, self.card_3]

        self.start_button = QPushButton("Livebilder starten / aktualisieren")
        self.start_button.clicked.connect(self.apply_preview)

        self.stop_button = QPushButton("Alle Kameras stoppen")
        self.stop_button.clicked.connect(self.stop_all_cameras)

        self.save_button = QPushButton("Kalibrierung speichern")
        self.save_button.clicked.connect(self.save_settings)

        self.reset_all_button = QPushButton("Alle 5-Punkt-Overlays zurücksetzen")
        self.reset_all_button.clicked.connect(self.reset_all_overlays)

        self.global_reference_button = QPushButton("Leeres Board global speichern")
        self.global_reference_button.clicked.connect(self.capture_reference_global)

        self.global_arm_button = QPushButton("Auto-Erkennung global scharf")
        self.global_arm_button.clicked.connect(self.arm_detectors_global)

        self.global_reset_round_button = QPushButton("Runde global zurücksetzen")
        self.global_reset_round_button.clicked.connect(self.reset_round_global)

        self.global_status_label = QLabel("Global: noch keine Aktion ausgeführt")
        self.global_status_label.setWordWrap(True)
        self.global_status_label.setStyleSheet("font-size: 13px; color: #8effc9; font-weight: 700;")

        self.event_status_label = QLabel("Event-Manager: nicht scharf")
        self.event_status_label.setWordWrap(True)
        self.event_status_label.setStyleSheet("font-size: 13px; color: #7fe6ff; font-weight: 700;")

        self.fusion_status_label = QLabel("Fusion: noch keine gemeinsamen Treffer")
        self.fusion_status_label.setWordWrap(True)
        self.fusion_status_label.setStyleSheet("font-size: 13px; color: #7fe6ff; font-weight: 700;")

        self.fusion_darts_label = QLabel("Finale Darts: - / - / -")
        self.fusion_darts_label.setWordWrap(True)
        self.fusion_darts_label.setStyleSheet("font-size: 13px; color: #ffd27f; font-weight: 700;")

        self.fusion_detail_label = QLabel("Fusion-Details: -")
        self.fusion_detail_label.setWordWrap(True)
        self.fusion_detail_label.setStyleSheet("font-size: 12px; color: #d0d0d0;")

        self.event_timer = QTimer(self)
        self.event_timer.timeout.connect(self._poll_event_manager)
        self.event_timer.start(50)

        self._build_ui()
        self._load_data_into_ui()
        self._update_status_labels()

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

        row_main = QHBoxLayout()
        row_main.setSpacing(10)
        row_main.addWidget(self.start_button)
        row_main.addWidget(self.stop_button)
        row_main.addWidget(self.save_button)
        row_main.addWidget(self.reset_all_button)
        row_main.addStretch()

        row_global = QHBoxLayout()
        row_global.setSpacing(10)
        row_global.addWidget(self.global_reference_button)
        row_global.addWidget(self.global_arm_button)
        row_global.addWidget(self.global_reset_round_button)
        row_global.addStretch()

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(14)
        cards_layout.addWidget(self.card_1, 1)
        cards_layout.addWidget(self.card_2, 1)
        cards_layout.addWidget(self.card_3, 1)

        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)
        info_layout.addWidget(self.global_status_label)
        info_layout.addWidget(self.event_status_label)
        info_layout.addWidget(self.fusion_status_label)
        info_layout.addWidget(self.fusion_darts_label)
        info_layout.addWidget(self.fusion_detail_label)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.info_label)
        main_layout.addLayout(row_main)
        main_layout.addLayout(row_global)
        main_layout.addLayout(info_layout)
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
        QMessageBox.information(self, "Gespeichert", "Die Kalibrierung wurde gespeichert.")

    def reset_all_overlays(self) -> None:
        for card in self.cards:
            card.reset_points()

    def stop_all_cameras(self) -> None:
        for card in self.cards:
            card.stop_worker()

    def _enabled_camera_slots(self) -> List[int]:
        cam_list = self.camera_config.get("cameras", [])
        slots: List[int] = []
        for idx, cam in enumerate(cam_list[:3]):
            if bool(cam.get("enabled", True)) and int(cam.get("device_id", -1)) >= 0:
                slots.append(idx)
        return slots

    def _minimum_candidates_for_fusion(self) -> int:
        active = len(self._enabled_camera_slots())
        if active <= 1:
            return 1
        return 2

    def capture_reference_global(self) -> None:
        active_slots = self._enabled_camera_slots()
        if not active_slots:
            self.global_status_label.setText("Global: keine aktiven Kameras vorhanden")
            return

        success = 0
        for slot in active_slots:
            if self.cards[slot].capture_reference_frame():
                success += 1

        self.reset_round_global(update_global_status=False)
        self.global_status_label.setText(
            f"Global: Referenz gespeichert auf {success}/{len(active_slots)} aktiven Kameras"
        )
        self._update_status_labels()

    def arm_detectors_global(self) -> None:
        active_slots = self._enabled_camera_slots()
        if not active_slots:
            self.global_status_label.setText("Global: keine aktiven Kameras vorhanden")
            return

        success = 0
        for slot in active_slots:
            if self.cards[slot].arm_detector():
                success += 1

        if success == len(active_slots):
            self.event_manager.arm()
            self.fused_darts = []
            self.global_status_label.setText(
                f"Global: Auto-Erkennung auf {success}/{len(active_slots)} aktiven Kameras scharf"
            )
        else:
            self.global_status_label.setText(
                f"Global: nur {success}/{len(active_slots)} Kameras scharf – zuerst globale Referenz speichern"
            )

        self._update_status_labels()

    def reset_round_global(self, update_global_status: bool = True) -> None:
        self.event_manager.reset_round()
        self.fused_darts = []

        for card in self.cards:
            card.reset_detector_round()

        if update_global_status:
            self.global_status_label.setText("Global: Runde und Event-Manager zurückgesetzt")

        self._update_status_labels()

    def handle_camera_auto_detection(self, camera_index: int, result: DartDetectionResult) -> None:
        accepted = self.event_manager.add_candidate(camera_index, result)
        if not accepted:
            return
        self._update_status_labels()

    def _poll_event_manager(self) -> None:
        closed_event = self.event_manager.poll_closed_event()
        if closed_event is None:
            self.event_status_label.setText(self.event_manager.current_status_text())
            return

        self._fuse_closed_event(closed_event)
        self._update_status_labels()

    def _fuse_closed_event(self, closed_event: ClosedDartEvent) -> None:
        candidates: List[CameraBoardCandidate] = []

        for item in closed_event.candidates:
            det = item.detection
            candidates.append(
                CameraBoardCandidate(
                    camera_index=item.camera_index,
                    score_label=det.score_label,
                    score_value=det.score_value,
                    ring_name=det.ring_name,
                    board_x=det.board_x,
                    board_y=det.board_y,
                    radius=det.radius,
                )
            )

        if len(candidates) < self._minimum_candidates_for_fusion():
            self.fusion_status_label.setText(
                f"Fusion: Event {closed_event.event_index} verworfen – zu wenige Kandidaten ({len(candidates)})"
            )
            return

        fused = fuse_camera_candidates(
            dart_index=closed_event.event_index,
            candidates=candidates,
        )
        self.fused_darts.append(fused)

    def _update_status_labels(self) -> None:
        self.event_status_label.setText(self.event_manager.current_status_text())

        if not self.fused_darts:
            self.fusion_status_label.setText("Fusion: noch keine gemeinsamen Treffer")
            self.fusion_darts_label.setText("Finale Darts: - / - / -")
            self.fusion_detail_label.setText("Fusion-Details: -")
            return

        labels = [d.final_label for d in self.fused_darts]
        while len(labels) < 3:
            labels.append("-")

        total = sum(d.final_score for d in self.fused_darts)

        self.fusion_status_label.setText(
            f"Fusion: {len(self.fused_darts)} finale Darts aus Eventfenstern berechnet"
        )
        self.fusion_darts_label.setText(
            f"Finale Darts: {labels[0]} / {labels[1]} / {labels[2]} | Gesamt: {total}"
        )

        last = self.fused_darts[-1]
        cam_parts = []
        for c in last.camera_candidates:
            cam_parts.append(f"K{c.camera_index + 1}:{c.score_label}")

        self.fusion_detail_label.setText(
            f"Fusion-Details letzter Dart: {' | '.join(cam_parts)} "
            f"-> Final: {last.final_label} = {last.final_score} "
            f"| FusedBoard=({last.fused_board_x:.3f}, {last.fused_board_y:.3f})"
        )
