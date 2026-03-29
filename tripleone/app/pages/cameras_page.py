# app/pages/cameras_page.py
# Diese Datei enthält die Kameraseite der App.
# Hier können 3 Kameras ausgewählt, konfiguriert, gestartet und gespeichert werden.
# Diese Version enthält ein stabiles Layout, damit sich die Kamera-Karten beim Start
# der Live-Vorschau nicht ungewollt extrem in die Breite ziehen.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import cv2

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QFrame,
    QMessageBox,
    QSizePolicy
)

from vision.camera_manager import CameraWorker
from vision.vision_service import (
    STATUS_BOARD_NOT_REFERENCED,
    STATUS_COOLDOWN,
    STATUS_DISARMED,
    STATUS_ERROR,
    STATUS_HIT_DETECTED,
    STATUS_NO_HIT,
    STATUS_READY,
    STATUS_WAITING_FOR_CLEAR,
    VisionService,
    VisionServiceConfig,
)
from vision.single_cam_detector import SingleCamDetector


class CameraCard(QFrame):
    """
    Ein einzelnes Kamera-Panel mit:
    - Vorschau
    - Auswahl der Kamera
    - Auflösung / FPS
    - Rotation / Flip
    - Statusanzeige
    """

    def __init__(self, title: str, camera_index: int, detector: Optional[SingleCamDetector] = None, parent=None):
        super().__init__(parent)
        self.setObjectName("CameraCard")
        self.setStyleSheet("""
            QFrame#CameraCard {
                background-color: #1f1f1f;
                border: 1px solid #333333;
                border-radius: 12px;
            }
            QLabel {
                color: #f2f2f2;
            }
            QComboBox, QSpinBox {
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
        """)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumWidth(320)

        self.worker: Optional[CameraWorker] = None
        self._last_image: Optional[QImage] = None
        self.camera_index = camera_index
        self.detector = detector
        self.vision_service = VisionService(
            config=VisionServiceConfig(
                auto_arm_on_reference_save=False,
                require_board_clear_after_hit=True,
                min_seconds_between_hits=0.80,
                clear_board_diff_threshold=18,
                clear_board_changed_ratio_threshold=0.0045,
                clear_board_blur_kernel_size=5,
                clear_board_required_consecutive_frames=2,
                use_board_mask_for_clear_check=False,
                keep_debug_images=True,
            ),
            default_detector=self.detector,
        )
        self._last_raw_frame = None
        self._last_detection_result = None
        

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        self.preview_container = QFrame()
        self.preview_container.setStyleSheet("""
            QFrame {
                background-color: #111111;
                border: 1px solid #2d2d2d;
                border-radius: 8px;
            }
        """)
        self.preview_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.preview_container.setFixedHeight(240)

        self.preview_label = QLabel("Keine Vorschau")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("""
            QLabel {
                background: transparent;
                border: none;
                color: #aaaaaa;
                font-size: 14px;
            }
        """)

        preview_layout = QVBoxLayout(self.preview_container)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        preview_layout.addWidget(self.preview_label)

        self.device_combo = QComboBox()
        self.device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(320, 3840)
        self.width_spin.setValue(1280)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(240, 2160)
        self.height_spin.setValue(720)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)

        self.rotation_combo = QComboBox()
        self.rotation_combo.addItems(["0", "90", "180", "270"])

        self.flip_check = QCheckBox("Horizontal spiegeln")

        self.enabled_check = QCheckBox("Aktiv")
        self.enabled_check.setChecked(True)
        self.show_overlay_check = QCheckBox("Debug-Overlay anzeigen")
        self.show_overlay_check.setChecked(True)
        self.show_overlay_check.toggled.connect(self._refresh_preview_from_last_raw_frame)

        self.status_label = QLabel("Status: nicht gestartet")
        self.save_reference_button = QPushButton("Leeres Board speichern")
        self.save_reference_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_reference_button.clicked.connect(self.save_reference_frame)

        self.arm_button = QPushButton("Erkennung aktivieren")
        self.arm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.arm_button.clicked.connect(self.arm_detection)

        self.disarm_button = QPushButton("Erkennung deaktivieren")
        self.disarm_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.disarm_button.clicked.connect(self.disarm_detection)

        self.hit_label = QLabel("Treffer: -")
        self.hit_label.setStyleSheet("font-size: 13px; color: #7CFC98;")
        self.hit_label.setWordWrap(True)

        self.vision_status_label = QLabel("Vision: keine Referenz")
        self.vision_status_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")
        self.vision_status_label.setWordWrap(True)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_container)

        row_1 = QHBoxLayout()
        row_1.setSpacing(8)
        row_1.addWidget(QLabel("Kamera:"))
        row_1.addWidget(self.device_combo, 1)

        row_2 = QHBoxLayout()
        row_2.setSpacing(8)
        row_2.addWidget(QLabel("Breite:"))
        row_2.addWidget(self.width_spin)
        row_2.addWidget(QLabel("Höhe:"))
        row_2.addWidget(self.height_spin)

        row_3 = QHBoxLayout()
        row_3.setSpacing(8)
        row_3.addWidget(QLabel("FPS:"))
        row_3.addWidget(self.fps_spin)
        row_3.addWidget(QLabel("Rotation:"))
        row_3.addWidget(self.rotation_combo)

        row_4 = QHBoxLayout()
        row_4.setSpacing(8)
        row_4.addWidget(self.enabled_check)
        row_4.addWidget(self.show_overlay_check)
        row_4.addStretch()
        row_4.addWidget(self.flip_check)

        layout.addLayout(row_1)
        layout.addLayout(row_2)
        layout.addLayout(row_3)
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(8)
        buttons_row.addWidget(self.save_reference_button)
        buttons_row.addWidget(self.arm_button)
        buttons_row.addWidget(self.disarm_button)

        layout.addLayout(row_4)
        layout.addLayout(buttons_row)
        layout.addWidget(self.status_label)
        layout.addWidget(self.vision_status_label)
        layout.addWidget(self.hit_label)
        layout.addStretch()

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")

    def clear_preview(self, text: str = "Keine Vorschau") -> None:
        self._last_image = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(text)

    def _bgr_to_qimage(self, frame_bgr) -> QImage:
        """
        Wandelt ein OpenCV-BGR-Bild in QImage um.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(
            rgb.data,
            w,
            h,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()

    def _render_detection_overlay_to_qimage(self, frame_bgr) -> QImage:
        """
        Rendert – falls vorhanden – das Detection-Debug-Overlay auf das BGR-Bild
        und gibt ein QImage für die Preview zurück.
        """
        display_frame = frame_bgr.copy()

        if self.show_overlay_check.isChecked():
            detection_result = self._last_detection_result
            if detection_result is not None and hasattr(detection_result, "render_debug_overlay"):
                try:
                    display_frame = detection_result.render_debug_overlay(display_frame)
                except Exception as exc:
                    self.vision_status_label.setText(f"Vision: Overlay-Fehler – {exc}")

        return self._bgr_to_qimage(display_frame)

    def _refresh_preview_from_last_raw_frame(self) -> None:
        """
        Rendert die Vorschau aus dem letzten Raw-Frame neu.
        Nützlich, wenn das Debug-Overlay ein-/ausgeschaltet wird.
        """
        if self._last_raw_frame is None:
            return

        if self.detector is None:
            self._last_image = self._bgr_to_qimage(self._last_raw_frame)
        else:
            self._last_image = self._render_detection_overlay_to_qimage(self._last_raw_frame)

        self._render_last_image()

    def update_preview(self, image: QImage) -> None:
        """
        Normale Vorschau vom Worker.
        Wird nur verwendet, wenn noch kein Raw-Frame-Overlay aktiv gezeichnet wurde.
        """
        if self._last_raw_frame is None:
            self._last_image = image
            self._render_last_image()

    def _render_last_image(self) -> None:
        if self._last_image is None:
            return

        target_size = self.preview_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        pixmap = QPixmap.fromImage(self._last_image)
        scaled = pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        self._render_last_image()
        super().resizeEvent(event)

    def set_detector(self, detector: Optional[SingleCamDetector]) -> None:
        self.detector = detector
        self.vision_service.set_default_detector(detector)

        if detector is None:
            self.vision_status_label.setText("Vision: kein Detector gesetzt")
        else:
            self.vision_status_label.setText("Vision: Detector gesetzt")

    def save_reference_frame(self) -> None:
        if self._last_raw_frame is None:
            QMessageBox.warning(self, "Referenz", "Noch kein Kameraframe verfügbar.")
            return

        self.vision_service.set_reference_frame(self.camera_index, self._last_raw_frame)

        if self.detector is None:
            self.vision_status_label.setText("Vision: Referenz gespeichert, aber kein Detector gesetzt")
        else:
            self.vision_status_label.setText("Vision: Referenz gespeichert")

        self.hit_label.setText("Treffer: -")

    def arm_detection(self) -> None:
        state = self.vision_service.get_state(self.camera_index)

        if self.detector is None:
            QMessageBox.warning(self, "Erkennung", "Kein SingleCamDetector gesetzt.")
            return

        if state.reference_frame is None:
            QMessageBox.warning(self, "Erkennung", "Bitte zuerst ein leeres Board speichern.")
            return

        self.vision_service.arm(self.camera_index)
        self.vision_status_label.setText("Vision: armed")

    def disarm_detection(self) -> None:
        self.vision_service.disarm(self.camera_index)
        self.vision_status_label.setText("Vision: disarmed")

    def handle_raw_frame(self, frame_bgr) -> None:
        """
        Nimmt rohe OpenCV-Frames entgegen, verarbeitet sie über VisionService
        und rendert optional das Debug-Overlay direkt in die Vorschau.
        """
        self._last_raw_frame = frame_bgr

        if self.detector is None:
            self._last_detection_result = None
            self.vision_status_label.setText("Vision: kein Detector gesetzt")

            # Dann einfach das rohe Bild anzeigen
            self._last_image = self._bgr_to_qimage(frame_bgr)
            self._render_last_image()
            return

        result = self.vision_service.process_frame(
            camera_id=self.camera_index,
            frame=frame_bgr,
        )
        self._last_detection_result = result.detection_result

        # Preview-Bild mit oder ohne Overlay erzeugen
        self._last_image = self._render_detection_overlay_to_qimage(frame_bgr)
        self._render_last_image()

        if result.status == STATUS_HIT_DETECTED and result.hit_event is not None:
            self.vision_status_label.setText(f"Vision: Treffer erkannt ({result.hit_event.label})")
            self.hit_label.setText(
                f"Treffer: {result.hit_event.label} | "
                f"Score: {result.hit_event.score} | "
                f"Segment: {result.hit_event.segment}"
            )
        elif result.status == STATUS_WAITING_FOR_CLEAR:
            self.vision_status_label.setText("Vision: warte auf freies Board")
        elif result.status == STATUS_READY:
            self.vision_status_label.setText("Vision: bereit")
        elif result.status == STATUS_NO_HIT:
            self.vision_status_label.setText("Vision: kein Treffer")
        elif result.status == STATUS_BOARD_NOT_REFERENCED:
            self.vision_status_label.setText("Vision: keine Referenz")
        elif result.status == STATUS_DISARMED:
            self.vision_status_label.setText("Vision: deaktiviert")
        elif result.status == STATUS_COOLDOWN:
            self.vision_status_label.setText("Vision: cooldown")
        elif result.status == STATUS_ERROR:
            self.vision_status_label.setText(f"Vision: Fehler – {result.message}")

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self._last_raw_frame = None
        self._last_detection_result = None
        self._last_image = None
        self.clear_preview("Keine Vorschau")
        self.set_status("gestoppt")
        self.vision_status_label.setText("Vision: gestoppt")
        self.hit_label.setText("Treffer: -")

    def start_worker(self, config: Dict) -> None:
        self.stop_worker()

        self.hit_label.setText("Treffer: -")
        self.vision_status_label.setText("Vision: starte Kamera ...")

        if not config.get("enabled", True):
            self.clear_preview("Deaktiviert")
            self.set_status("deaktiviert")
            return

        device_id = config.get("device_id", -1)
        if device_id < 0:
            self.clear_preview("Keine Kamera ausgewählt")
            self.set_status("keine Kamera gewählt")
            return

        self.worker = CameraWorker(
            device_id=device_id,
            width=config.get("width", 1280),
            height=config.get("height", 720),
            fps=config.get("fps", 30),
            rotation=config.get("rotation", 0),
            flip=config.get("flip", False)
        )

        self.worker.frame_ready.connect(self.update_preview)
        self.worker.raw_frame_ready.connect(self.handle_raw_frame)
        self.worker.status_changed.connect(self.set_status)
        self.worker.start()

    def get_config(self, idx: int) -> Dict:
        return {
            "name": f"Kamera {idx + 1}",
            "device_id": int(self.device_combo.currentData()),
            "width": int(self.width_spin.value()),
            "height": int(self.height_spin.value()),
            "fps": int(self.fps_spin.value()),
            "rotation": int(self.rotation_combo.currentText()),
            "flip": bool(self.flip_check.isChecked()),
            "enabled": bool(self.enabled_check.isChecked())
        }

    def set_config(self, config: Dict) -> None:
        self.width_spin.setValue(config.get("width", 1280))
        self.height_spin.setValue(config.get("height", 720))
        self.fps_spin.setValue(config.get("fps", 30))
        self.rotation_combo.setCurrentText(str(config.get("rotation", 0)))
        self.flip_check.setChecked(config.get("flip", False))
        self.enabled_check.setChecked(config.get("enabled", True))


class CamerasPage(QWidget):
    """
    Hauptseite für die Kamerakonfiguration.
    Enthält 3 Kamera-Karten und Buttons zum Starten/Speichern/Aktualisieren.
    """

    def __init__(
        self,
        config_data: Dict,
        save_callback,
        refresh_cameras_callback,
        detectors: Optional[List[Optional[SingleCamDetector]]] = None,
        parent=None
    ):
        super().__init__(parent)
        self.config_data = deepcopy(config_data)
        self.save_callback = save_callback
        self.refresh_cameras_callback = refresh_cameras_callback
        self.available_cameras: List[Dict[str, int]] = []
        self.detectors = detectors or [None, None, None]
        while len(self.detectors) < 3:
            self.detectors.append(None)

        self.title_label = QLabel("TripleOne – Kameras")
        self.title_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.card_1 = CameraCard("Kamera 1", camera_index=0, detector=self.detectors[0])
        self.card_2 = CameraCard("Kamera 2", camera_index=1, detector=self.detectors[1])
        self.card_3 = CameraCard("Kamera 3", camera_index=2, detector=self.detectors[2])
        self.cards = [self.card_1, self.card_2, self.card_3]

        self.refresh_button = QPushButton("Kameras neu erkennen")
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.clicked.connect(self.refresh_cameras_callback)

        self.apply_button = QPushButton("Vorschau starten / aktualisieren")
        self.apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.apply_button.clicked.connect(self.apply_preview)

        self.stop_button = QPushButton("Alle Kameras stoppen")
        self.stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_all_cameras)

        self.save_button = QPushButton("Einstellungen speichern")
        self.save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_button.clicked.connect(self.save_settings)

        self._build_ui()
        self.load_config_into_ui()

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
        buttons_layout.addWidget(self.refresh_button)
        buttons_layout.addWidget(self.apply_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.save_button)
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
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(cards_layout, 1)

    def set_available_cameras(self, cameras: List[Dict[str, int]]) -> None:
        self.available_cameras = cameras

        for card in self.cards:
            previous_device_id = card.device_combo.currentData()

            card.device_combo.blockSignals(True)
            card.device_combo.clear()
            card.device_combo.addItem("Keine Kamera", -1)

            for cam in cameras:
                card.device_combo.addItem(f"{cam['name']} (Index {cam['index']})", cam["index"])

            found_index = card.device_combo.findData(previous_device_id)
            if found_index >= 0:
                card.device_combo.setCurrentIndex(found_index)

            card.device_combo.blockSignals(False)

    def load_config_into_ui(self) -> None:
        cameras_config = self.config_data.get("cameras", [])
        for index, card in enumerate(self.cards):
            if index < len(cameras_config):
                card.set_config(cameras_config[index])

    def set_detectors(self, detectors: List[Optional[SingleCamDetector]]) -> None:
        self.detectors = list(detectors)
        while len(self.detectors) < 3:
            self.detectors.append(None)

        for idx, card in enumerate(self.cards):
            card.set_detector(self.detectors[idx])

    def apply_config_to_ui_device_selection(self) -> None:
        cameras_config = self.config_data.get("cameras", [])

        for index, card in enumerate(self.cards):
            if index >= len(cameras_config):
                continue

            device_id = cameras_config[index].get("device_id", -1)
            combo_index = card.device_combo.findData(device_id)
            if combo_index >= 0:
                card.device_combo.setCurrentIndex(combo_index)
            else:
                card.device_combo.setCurrentIndex(0)

    def collect_ui_config(self) -> Dict:
        new_config = deepcopy(self.config_data)
        new_config["cameras"] = []

        for idx, card in enumerate(self.cards):
            new_config["cameras"].append(card.get_config(idx))

        return new_config

    def validate_camera_config(self, config: Dict) -> Tuple[bool, str]:
        used_device_ids = []

        for cam in config.get("cameras", []):
            if not cam.get("enabled", True):
                continue

            device_id = cam.get("device_id", -1)
            if device_id < 0:
                continue

            if device_id in used_device_ids:
                return False, (
                    f"Die Kamera mit Geräte-Index {device_id} wurde mehrfach ausgewählt. "
                    f"Bitte verwende jede physische Kamera nur einmal."
                )

            used_device_ids.append(device_id)

        return True, ""

    def apply_preview(self) -> None:
        new_config = self.collect_ui_config()
        is_valid, error_text = self.validate_camera_config(new_config)

        if not is_valid:
            QMessageBox.warning(self, "Ungültige Kamera-Auswahl", error_text)
            return

        self.config_data = new_config

        for index, card in enumerate(self.cards):
            card.start_worker(self.config_data["cameras"][index])

    def save_settings(self) -> None:
        new_config = self.collect_ui_config()
        is_valid, error_text = self.validate_camera_config(new_config)

        if not is_valid:
            QMessageBox.warning(self, "Ungültige Kamera-Auswahl", error_text)
            return

        self.config_data = new_config
        self.save_callback(self.config_data)
        self.apply_preview()

        QMessageBox.information(
            self,
            "Gespeichert",
            "Die Kamera-Einstellungen wurden gespeichert."
        )

    def stop_all_cameras(self) -> None:
        for card in self.cards:
            card.stop_worker()