# app/pages/camera_page.py
# Zweck:
# Diese Seite zeigt ein Live-Kamerabild und integriert den VisionService
# für den echten Produktfluss:
#
# 1. Leeres Board speichern
# 2. Erkennung aktivieren
# 3. Liveframes verarbeiten
# 4. Treffer anzeigen
# 5. Auf freies Board warten
#
# WICHTIG:
# - Diese Datei baut KEINE eigene Geometrie
# - Diese Datei baut KEIN eigenes Scoring
# - Diese Datei orchestriert nur Kamera + VisionService + UI
#
# Erwartete Projektmodule:
# - vision.vision_service
# - vision.single_cam_detector

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

try:
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
except ImportError:  # pragma: no cover
    from ..vision.vision_service import (  # type: ignore
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
    from ..vision.single_cam_detector import SingleCamDetector  # type: ignore


class CameraPage(QWidget):
    """
    Einfache Live-Kameraseite für den Produktfluss mit VisionService.

    Erwartete Nutzung:
    - Detector extern bauen
    - CameraPage(detector=dein_detector) instanziieren
    - Kamera auswählen / starten
    - "Leeres Board speichern"
    - "Erkennung aktivieren"
    """

    def __init__(
        self,
        detector: Optional[SingleCamDetector] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        # --------------------------------------------------------------
        # Detector + VisionService
        # --------------------------------------------------------------
        self.detector: Optional[SingleCamDetector] = detector
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

        # --------------------------------------------------------------
        # Kamerastatus
        # --------------------------------------------------------------
        self.camera_id: int = 0
        self.capture: Optional[cv2.VideoCapture] = None
        self.current_frame_bgr: Optional[np.ndarray] = None
        self.camera_running: bool = False

        # --------------------------------------------------------------
        # UI / Timing
        # --------------------------------------------------------------
        self.frame_timer = QTimer(self)
        self.frame_timer.setInterval(33)  # ~30 FPS
        self.frame_timer.timeout.connect(self._update_camera_loop)

        self._build_ui()
        self._populate_camera_selector()

    # -----------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.setObjectName("camera_page")

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # --------------------------------------------------------------
        # Kopfzeile
        # --------------------------------------------------------------
        header = QHBoxLayout()
        self.title_label = QLabel("Kamera / Vision")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        header.addWidget(self.title_label)
        header.addStretch(1)
        root.addLayout(header)

        # --------------------------------------------------------------
        # Oberer Steuerbereich
        # --------------------------------------------------------------
        controls_group = QGroupBox("Steuerung")
        controls_layout = QGridLayout(controls_group)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(10)

        self.camera_selector = QComboBox()
        self.refresh_cameras_button = QPushButton("Kameras aktualisieren")
        self.start_camera_button = QPushButton("Kamera starten")
        self.stop_camera_button = QPushButton("Kamera stoppen")
        self.stop_camera_button.setEnabled(False)

        self.save_reference_button = QPushButton("Leeres Board speichern")
        self.save_reference_button.setEnabled(False)

        self.arm_button = QPushButton("Erkennung aktivieren")
        self.arm_button.setEnabled(False)

        self.disarm_button = QPushButton("Erkennung deaktivieren")
        self.disarm_button.setEnabled(False)

        self.show_overlay_checkbox = QCheckBox("Overlay anzeigen")
        self.show_overlay_checkbox.setChecked(True)

        controls_layout.addWidget(QLabel("Kamera:"), 0, 0)
        controls_layout.addWidget(self.camera_selector, 0, 1)
        controls_layout.addWidget(self.refresh_cameras_button, 0, 2)
        controls_layout.addWidget(self.start_camera_button, 0, 3)
        controls_layout.addWidget(self.stop_camera_button, 0, 4)

        controls_layout.addWidget(self.save_reference_button, 1, 0, 1, 2)
        controls_layout.addWidget(self.arm_button, 1, 2)
        controls_layout.addWidget(self.disarm_button, 1, 3)
        controls_layout.addWidget(self.show_overlay_checkbox, 1, 4)

        root.addWidget(controls_group)

        # --------------------------------------------------------------
        # Hauptbereich
        # --------------------------------------------------------------
        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # Livebild
        self.video_frame = QFrame()
        self.video_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.video_frame.setStyleSheet("background: #101010; border-radius: 10px;")
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(8, 8, 8, 8)

        self.video_label = QLabel("Kein Kamerabild")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(900, 520)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet("color: #cccccc; background: #1b1b1b; border-radius: 8px;")
        video_layout.addWidget(self.video_label)

        body.addWidget(self.video_frame, 3)

        # Statusbereich rechts
        sidebar = QVBoxLayout()
        sidebar.setSpacing(12)
        body.addLayout(sidebar, 1)

        status_group = QGroupBox("Status")
        status_layout = QGridLayout(status_group)

        self.status_value = QLabel("-")
        self.status_value.setStyleSheet("font-weight: 700;")
        self.reference_value = QLabel("Nein")
        self.armed_value = QLabel("Nein")
        self.last_hit_value = QLabel("-")
        self.last_score_value = QLabel("-")
        self.last_ring_value = QLabel("-")
        self.last_segment_value = QLabel("-")
        self.last_confidence_value = QLabel("-")
        self.board_clear_value = QLabel("-")
        self.changed_ratio_value = QLabel("-")

        status_layout.addWidget(QLabel("Status:"), 0, 0)
        status_layout.addWidget(self.status_value, 0, 1)

        status_layout.addWidget(QLabel("Referenz gesetzt:"), 1, 0)
        status_layout.addWidget(self.reference_value, 1, 1)

        status_layout.addWidget(QLabel("Armed:"), 2, 0)
        status_layout.addWidget(self.armed_value, 2, 1)

        status_layout.addWidget(QLabel("Letzter Treffer:"), 3, 0)
        status_layout.addWidget(self.last_hit_value, 3, 1)

        status_layout.addWidget(QLabel("Score:"), 4, 0)
        status_layout.addWidget(self.last_score_value, 4, 1)

        status_layout.addWidget(QLabel("Ring:"), 5, 0)
        status_layout.addWidget(self.last_ring_value, 5, 1)

        status_layout.addWidget(QLabel("Segment:"), 6, 0)
        status_layout.addWidget(self.last_segment_value, 6, 1)

        status_layout.addWidget(QLabel("Confidence:"), 7, 0)
        status_layout.addWidget(self.last_confidence_value, 7, 1)

        status_layout.addWidget(QLabel("Board frei:"), 8, 0)
        status_layout.addWidget(self.board_clear_value, 8, 1)

        status_layout.addWidget(QLabel("Changed Ratio:"), 9, 0)
        status_layout.addWidget(self.changed_ratio_value, 9, 1)

        sidebar.addWidget(status_group)

        info_group = QGroupBox("Hinweise")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel(
            "Ablauf:\n"
            "1. Kamera starten\n"
            "2. Leeres Board speichern\n"
            "3. Erkennung aktivieren\n"
            "4. Wurf erkennen lassen\n"
            "5. Auf freies Board warten"
        )
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        sidebar.addWidget(info_group)

        sidebar.addStretch(1)

        # --------------------------------------------------------------
        # Signalverbindungen
        # --------------------------------------------------------------
        self.refresh_cameras_button.clicked.connect(self._populate_camera_selector)
        self.start_camera_button.clicked.connect(self.start_camera)
        self.stop_camera_button.clicked.connect(self.stop_camera)
        self.save_reference_button.clicked.connect(self.save_reference_frame)
        self.arm_button.clicked.connect(self.arm_detection)
        self.disarm_button.clicked.connect(self.disarm_detection)

    # -----------------------------------------------------------------
    # Öffentliche API
    # -----------------------------------------------------------------
    def set_detector(self, detector: SingleCamDetector) -> None:
        """
        Setzt oder ersetzt den aktuellen Detector.
        """
        self.detector = detector
        self.vision_service.set_default_detector(detector)
        self._refresh_state_labels()

    # -----------------------------------------------------------------
    # Kamera
    # -----------------------------------------------------------------
    def _populate_camera_selector(self) -> None:
        self.camera_selector.clear()

        available_indices = []
        for index in range(5):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            ok, _frame = cap.read()
            cap.release()
            if ok:
                available_indices.append(index)

        if not available_indices:
            self.camera_selector.addItem("Keine Kamera gefunden", -1)
            self.start_camera_button.setEnabled(False)
            return

        for index in available_indices:
            self.camera_selector.addItem(f"Kamera {index}", index)

        self.start_camera_button.setEnabled(True)

    def start_camera(self) -> None:
        selected_data = self.camera_selector.currentData()
        if selected_data is None or int(selected_data) < 0:
            QMessageBox.warning(self, "Kamera", "Keine gültige Kamera ausgewählt.")
            return

        self.stop_camera()

        self.camera_id = int(selected_data)
        self.capture = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)

        if self.capture is None or not self.capture.isOpened():
            self.capture = None
            QMessageBox.critical(self, "Kamera", f"Kamera {self.camera_id} konnte nicht geöffnet werden.")
            return

        self.camera_running = True
        self.frame_timer.start()

        self.start_camera_button.setEnabled(False)
        self.stop_camera_button.setEnabled(True)
        self.save_reference_button.setEnabled(True)
        self.arm_button.setEnabled(self.detector is not None)
        self.disarm_button.setEnabled(True)

        self._set_status_text("Kamera läuft.")
        self._refresh_state_labels()

    def stop_camera(self) -> None:
        self.frame_timer.stop()

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.camera_running = False
        self.current_frame_bgr = None

        self.video_label.setText("Kein Kamerabild")
        self.video_label.setPixmap(QPixmap())

        self.start_camera_button.setEnabled(True)
        self.stop_camera_button.setEnabled(False)
        self.save_reference_button.setEnabled(False)

        state = self.vision_service.get_state(self.camera_id)
        self.arm_button.setEnabled(state.reference_frame is not None and self.detector is not None)
        self.disarm_button.setEnabled(False)

        self._set_status_text("Kamera gestoppt.")
        self._refresh_state_labels()

    # -----------------------------------------------------------------
    # Produktfluss: Referenz / Arm / Disarm
    # -----------------------------------------------------------------
    def save_reference_frame(self) -> None:
        """
        Speichert das aktuelle Liveframe als Referenz für diese Kamera.
        """
        if self.current_frame_bgr is None:
            QMessageBox.warning(self, "Referenz", "Es ist kein aktuelles Kamerabild verfügbar.")
            return

        self.vision_service.set_reference_frame(self.camera_id, self.current_frame_bgr)
        self._set_status_text("Leeres Board gespeichert.")
        self.arm_button.setEnabled(self.detector is not None)
        self.disarm_button.setEnabled(True)
        self._refresh_state_labels()

    def arm_detection(self) -> None:
        """
        Aktiviert die Erkennung.
        """
        if self.detector is None:
            QMessageBox.warning(self, "Erkennung", "Es ist kein Detector gesetzt.")
            return

        state = self.vision_service.get_state(self.camera_id)
        if state.reference_frame is None:
            QMessageBox.warning(self, "Erkennung", "Zuerst ein leeres Board speichern.")
            return

        self.vision_service.arm(self.camera_id)
        self._set_status_text("Erkennung aktiviert.")
        self._refresh_state_labels()

    def disarm_detection(self) -> None:
        """
        Deaktiviert die Erkennung.
        """
        self.vision_service.disarm(self.camera_id)
        self._set_status_text("Erkennung deaktiviert.")
        self._refresh_state_labels()

    # -----------------------------------------------------------------
    # Live-Loop
    # -----------------------------------------------------------------
    def _update_camera_loop(self) -> None:
        """
        Holt das nächste Kamerabild, zeigt es an und verarbeitet optional
        die Trefferlogik über den VisionService.
        """
        if self.capture is None or not self.camera_running:
            return

        ok, frame_bgr = self.capture.read()
        if not ok or frame_bgr is None:
            self._set_status_text("Kamerabild konnte nicht gelesen werden.")
            return

        self.current_frame_bgr = frame_bgr

        # --------------------------------------------------------------
        # VisionService verarbeiten
        # --------------------------------------------------------------
        result = self.vision_service.process_frame(
            camera_id=self.camera_id,
            frame=frame_bgr,
        )

        display_frame = frame_bgr.copy()

        # Overlay anzeigen, wenn verfügbar und gewünscht
        if self.show_overlay_checkbox.isChecked():
            detection_result = result.detection_result
            if detection_result is not None and hasattr(detection_result, "render_debug_overlay"):
                try:
                    display_frame = detection_result.render_debug_overlay(display_frame)
                except Exception:
                    pass

        self._show_frame(display_frame)
        self._apply_result_to_ui(result)

    # -----------------------------------------------------------------
    # UI-Update
    # -----------------------------------------------------------------
    def _apply_result_to_ui(self, result: Any) -> None:
        self._set_status_text(result.message)

        if result.status == STATUS_HIT_DETECTED and result.hit_event is not None:
            self.last_hit_value.setText(str(result.hit_event.label))
            self.last_score_value.setText(str(result.hit_event.score))
            self.last_ring_value.setText(str(result.hit_event.ring))
            self.last_segment_value.setText("-" if result.hit_event.segment is None else str(result.hit_event.segment))
            self.last_confidence_value.setText(
                "-" if result.hit_event.confidence is None else f"{result.hit_event.confidence:.3f}"
            )

        if result.board_is_clear is None:
            self.board_clear_value.setText("-")
        else:
            self.board_clear_value.setText("Ja" if result.board_is_clear else "Nein")

        if result.board_changed_ratio is None:
            self.changed_ratio_value.setText("-")
        else:
            self.changed_ratio_value.setText(f"{result.board_changed_ratio:.6f}")

        self._refresh_state_labels(result.status)

    def _refresh_state_labels(self, current_status: Optional[str] = None) -> None:
        state = self.vision_service.get_state(self.camera_id)

        if current_status is None:
            current_status = state.last_status

        self.status_value.setText(str(current_status or "-"))
        self.reference_value.setText("Ja" if state.reference_frame is not None else "Nein")
        self.armed_value.setText("Ja" if state.armed else "Nein")

        # Farben für schnellen Überblick
        status_color = "#d0d0d0"

        if current_status == STATUS_HIT_DETECTED:
            status_color = "#4CAF50"
        elif current_status == STATUS_WAITING_FOR_CLEAR:
            status_color = "#FFB300"
        elif current_status == STATUS_READY:
            status_color = "#2196F3"
        elif current_status == STATUS_NO_HIT:
            status_color = "#d0d0d0"
        elif current_status == STATUS_ERROR:
            status_color = "#E53935"
        elif current_status in (STATUS_BOARD_NOT_REFERENCED, STATUS_DISARMED):
            status_color = "#9E9E9E"
        elif current_status == STATUS_COOLDOWN:
            status_color = "#FB8C00"

        self.status_value.setStyleSheet(f"font-weight: 700; color: {status_color};")

    def _set_status_text(self, text: str) -> None:
        self.info_label.setText(str(text))

    # -----------------------------------------------------------------
    # Bilddarstellung
    # -----------------------------------------------------------------
    def _show_frame(self, frame_bgr: np.ndarray) -> None:
        pixmap = self._bgr_to_pixmap(frame_bgr)
        if pixmap.isNull():
            return

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.current_frame_bgr is not None:
            self._show_frame(self.current_frame_bgr)

    def _bgr_to_pixmap(self, frame_bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------
    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.stop_camera()
        super().closeEvent(event)