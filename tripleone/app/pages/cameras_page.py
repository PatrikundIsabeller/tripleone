# app/pages/cameras_page.py
# Diese Datei enthält die Kameraseite der App.
# Hier können 3 Kameras ausgewählt, konfiguriert, gestartet und gespeichert werden.
# Diese Version enthält ein stabiles Layout, damit sich die Kamera-Karten beim Start
# der Live-Vorschau nicht ungewollt extrem in die Breite ziehen.

from __future__ import annotations

import threading

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2

from PyQt6.QtCore import (
    Qt,
    QObject,
    QThread,
    QTimer,
    pyqtSignal,
    pyqtSlot,
)
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
from vision.multi_cam_fusion import MultiCamFusionEngine, MultiCamFusionConfig

# --------------------------------------------------------------
# TEMPORÄRER TESTMODUS
# --------------------------------------------------------------
# True:
# - nur Kamera 1 sichtbar
# - Kamera 2 und 3 deaktiviert
# - Kamera 1 bekommt eine große Vorschau
#
# Später einfach wieder auf False setzen.
SINGLE_CAMERA_TEST_MODE = False

class VisionInferenceWorker(QObject):
    """
    Führt die schwere Vision-/YOLO-Auswertung außerhalb des GUI-Threads aus.

    Wichtig:
    - Kamera-Vorschau bleibt flüssig.
    - PyTorch/Ultralytics dürfen hier blockieren, ohne Qt einzufrieren.
    - Es wird immer nur ein Frame gleichzeitig verarbeitet.
    """

    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    processing_finished = pyqtSignal()

    def __init__(
        self,
        *,
        camera_index: int,
        vision_service: VisionService,
        vision_lock: threading.RLock,
    ) -> None:
        super().__init__()

        self.camera_index = int(camera_index)
        self.vision_service = vision_service
        self.vision_lock = vision_lock

    @pyqtSlot(object)
    def process_frame(self, frame_bgr) -> None:
        try:
            with self.vision_lock:
                result = self.vision_service.process_frame(
                    camera_id=self.camera_index,
                    frame=frame_bgr,
                )

            self.result_ready.emit(result)

        except Exception as exc:
            print(
                f"[VISION ERROR] "
                f"K{self.camera_index + 1} "
                f"{type(exc).__name__}: {exc}"
            )

            self.error_occurred.emit(
                f"{type(exc).__name__}: {exc}"
            )

        finally:
            self.processing_finished.emit()


class CameraCard(QFrame):
    inference_frame_requested = pyqtSignal(object)

    """
    Ein einzelnes Kamera-Panel mit:
    - Vorschau
    - Auswahl der Kamera
    - Auflösung / FPS
    - Rotation / Flip
    - Statusanzeige
    """

    def shutdown(self) -> None:
        """
        Beendet Kamera und Vision-Thread vollständig.
        """
        self.stop_worker()

        if self._inference_timer.isActive():
            self._inference_timer.stop()

        if self._inference_thread.isRunning():
            self._inference_thread.quit()
            self._inference_thread.wait(5000)

    def __init__(
        self,
        title: str,
        camera_index: int,
        detector: Optional[SingleCamDetector] = None,
        fusion_update_callback=None,
        parent=None
    ):
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
                confirm_hit_required_consecutive_frames=2,
                min_hit_confidence=0.08,
                confirm_same_label_required=False,
                confirm_max_topdown_distance_px=24.0,
                confirm_max_image_distance_px=40.0,
                pending_hit_max_age_seconds=0.75,
                clear_board_diff_threshold=18,
                clear_board_changed_ratio_threshold=0.0045,
                clear_board_blur_kernel_size=5,
                clear_board_required_consecutive_frames=2,
                use_board_mask_for_clear_check=True,
                keep_debug_images=True,
            ),
            default_detector=self.detector,
        )

        # --------------------------------------------------------------
        # Vision/YOLO läuft NICHT im GUI-Thread
        # --------------------------------------------------------------
        self._vision_lock = threading.RLock()

        self._inference_thread = QThread(self)

        self._inference_worker = VisionInferenceWorker(
            camera_index=self.camera_index,
            vision_service=self.vision_service,
            vision_lock=self._vision_lock,
        )

        self._inference_worker.moveToThread(
            self._inference_thread
        )

        self.inference_frame_requested.connect(
            self._inference_worker.process_frame,
            Qt.ConnectionType.QueuedConnection,
        )

        self._inference_worker.result_ready.connect(
            self._handle_vision_result,
            Qt.ConnectionType.QueuedConnection,
        )

        self._inference_worker.error_occurred.connect(
            self._handle_vision_error,
            Qt.ConnectionType.QueuedConnection,
        )

        self._inference_worker.processing_finished.connect(
            self._handle_inference_finished,
            Qt.ConnectionType.QueuedConnection,
        )

        self._inference_thread.start()

        # True, solange YOLO gerade einen Frame verarbeitet.
        self._inference_busy = False

        # Aktuellster Kameraframe.
        # Alte Frames werden NICHT aufgestaut.
        self._pending_inference_frame = None

        # Maximal ca. 10 Inferences pro Sekunde.
        # Die Vorschau selbst darf weiter mit 30 FPS laufen.
        self._inference_timer = QTimer(self)
        self._inference_timer.setInterval(100)
        self._inference_timer.timeout.connect(
            self._dispatch_latest_frame_to_inference
        )
        self._inference_timer.start()

        self._last_raw_frame = None
        self._last_detection_result = None
        self._last_observation = None
        self._last_result_status = None
        self._last_image = None
        self._fusion_update_callback = fusion_update_callback

        # Bestätigter Treffer für UI
        self._last_confirmed_hit_event = None

        # Harte, gelatchte Fusion-Observation:
        # bleibt erhalten, bis das Board wirklich wieder frei ist.
        self._latched_fusion_observation = None

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
        #self.preview_container.setFixedHeight(700)
        # Fenster vergrößern

        if SINGLE_CAMERA_TEST_MODE and self.camera_index == 0:
            self.preview_container.setFixedHeight(650)
        else:
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
        if self._last_raw_frame is None:
            return

        if self.detector is None:
            self._last_image = self._bgr_to_qimage(self._last_raw_frame)
        else:
            self._last_image = self._render_detection_overlay_to_qimage(self._last_raw_frame)

        self._render_last_image()

    def update_preview(self, image: QImage) -> None:
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

    def _format_hit_text(self, hit_event) -> str:
        if hit_event is None:
            return "Treffer: -"

        image_point = getattr(hit_event, "image_point", None)
        topdown_point = getattr(hit_event, "topdown_point", None)

        if image_point is not None:
            image_text = (
                f"IMG=({float(image_point[0]):.1f}, "
                f"{float(image_point[1]):.1f})"
            )
        else:
            image_text = "IMG=(-,-)"

        if topdown_point is not None:
            topdown_text = (
                f"TOP=({float(topdown_point[0]):.1f}, "
                f"{float(topdown_point[1]):.1f})"
            )
        else:
            topdown_text = "TOP=(-,-)"

        return (
            f"Treffer: {hit_event.label} | "
            f"Score: {hit_event.score} | "
            f"Segment: {hit_event.segment} | "
            f"Ring: {getattr(hit_event, 'ring', '-')} | "
            f"{image_text} | "
            f"{topdown_text}"
        )

    def _build_fusion_observation_from_hit_event(self, hit_event):
        """
        Baut eine einfache Observation direkt aus dem bestätigten HitEvent.
        """
        if hit_event is None:
            return None

        image_point = getattr(hit_event, "image_point", None)
        topdown_point = getattr(hit_event, "topdown_point", None)

        if image_point is None or topdown_point is None:
            return None

        confidence = getattr(hit_event, "confidence", None)
        confidence = 0.0 if confidence is None else float(confidence)

        estimate = SimpleNamespace(
            estimate_rank=1,
            image_point=image_point,
            topdown_point=topdown_point,
            label=str(getattr(hit_event, "label", "")),
            score=int(getattr(hit_event, "score", 0)),
            ring=str(getattr(hit_event, "ring", "")),
            segment=getattr(hit_event, "segment", None),
            multiplier=int(getattr(hit_event, "multiplier", 0)),
            combined_confidence=confidence,
            impact_confidence=confidence,
            candidate_confidence=confidence,
            debug={
                "source": "vision_service_hit_event",
                "camera_id": int(getattr(hit_event, "camera_id", self.camera_index)),
            },
        )

        observation = SimpleNamespace(
            camera_index=int(self.camera_index),
            frame_ok=True,
            detector_ready=self.detector is not None,
            reference_available=True,
            candidate_count=1,
            impact_count=1,
            scored_count=1,
            best_image_point=image_point,
            best_topdown_point=topdown_point,
            best_label=str(getattr(hit_event, "label", "")),
            best_score=int(getattr(hit_event, "score", 0)),
            best_ring=str(getattr(hit_event, "ring", "")),
            best_segment=getattr(hit_event, "segment", None),
            best_multiplier=int(getattr(hit_event, "multiplier", 0)),
            best_combined_confidence=confidence,
            best_impact_confidence=confidence,
            best_candidate_confidence=confidence,
            estimates=[estimate],
            metadata={
                "source": "vision_service_hit_event",
            },
            debug={
                "observation_mode": "confirmed_hit_event",
            },
            raw_result=None,
        )
        return observation

    def _clear_runtime_detection_cache(self) -> None:
        self._last_detection_result = None
        self._last_observation = None
        self._last_result_status = None
        self._last_confirmed_hit_event = None
        self._latched_fusion_observation = None

    def save_reference_frame(self) -> None:
        if self._last_raw_frame is None:
            QMessageBox.warning(self, "Referenz", "Noch kein Kameraframe verfügbar.")
            return

        with self._vision_lock:
            self.vision_service.set_reference_frame(
                self.camera_index,
                self._last_raw_frame,
            )

        if self.detector is None:
            self.vision_status_label.setText("Vision: Referenz gespeichert, aber kein Detector gesetzt")
        else:
            self.vision_status_label.setText("Vision: Referenz gespeichert")

        self._last_observation = None
        self._last_detection_result = None
        self._last_confirmed_hit_event = None
        self._latched_fusion_observation = None
        self.hit_label.setText("Treffer: -")

    def arm_detection(self) -> None:
        with self._vision_lock:
            state = self.vision_service.get_state(
                self.camera_index
            )

        if self.detector is None:
            QMessageBox.warning(self, "Erkennung", "Kein SingleCamDetector gesetzt.")
            return

        if state.reference_frame is None:
            QMessageBox.warning(self, "Erkennung", "Bitte zuerst ein leeres Board speichern.")
            return

        with self._vision_lock:
            self.vision_service.arm(
                self.camera_index
            )
        self.vision_status_label.setText("Vision: armed")

    def disarm_detection(self) -> None:
        with self._vision_lock:
            self.vision_service.disarm(
                self.camera_index
            )

    def handle_raw_frame(self, frame_bgr) -> None:
        """
        Wird vom Kamera-Thread mit neuen Frames versorgt.

        WICHTIG:
        Hier findet KEINE YOLO-/Vision-Auswertung mehr statt.
        Wir speichern nur den aktuellsten Frame.

        Die eigentliche Inference läuft über VisionInferenceWorker.
        """
        self._last_raw_frame = frame_bgr

        # Vorschau sofort aktualisieren.
        # Dadurch bleibt die Kamera flüssig, auch wenn YOLO arbeitet.
        if self.detector is None:
            self._last_image = self._bgr_to_qimage(frame_bgr)
        else:
            self._last_image = self._render_detection_overlay_to_qimage(
                frame_bgr
            )

        self._render_last_image()

        # Nur den NEUESTEN Frame für die KI behalten.
        # Kein 30-FPS-Queue-Aufbau.
        self._pending_inference_frame = frame_bgr.copy()

    def _dispatch_latest_frame_to_inference(self) -> None:
        """
        Übergibt maximal einen aktuellen Frame an den Vision-Thread.

        Wenn YOLO noch arbeitet:
        - nichts tun
        - alten Frame nicht aufstauen

        Beim nächsten Timer-Tick wird automatisch der dann neueste
        Kameraframe verwendet.
        """
        if self.detector is None:
            return

        if self._inference_busy:
            return

        if self._pending_inference_frame is None:
            return

        frame = self._pending_inference_frame
        self._pending_inference_frame = None

        self._inference_busy = True

        self.inference_frame_requested.emit(frame)

    @pyqtSlot()
    def _handle_inference_finished(self) -> None:
        self._inference_busy = False

    @pyqtSlot(str)
    def _handle_vision_error(self, error_text: str) -> None:
        self._inference_busy = False

        self.vision_status_label.setText(
            f"Vision: Fehler – {error_text}"
        )

    @pyqtSlot(object)
    def _handle_vision_result(self, result) -> None:
        """
        Läuft wieder im GUI-Thread.

        Hier wird NUR das bereits berechnete Vision-Ergebnis
        in UI/Fusion übernommen.
        """
        self._last_result_status = result.status
        self._last_detection_result = result.detection_result

        if result.hit_event is not None:
            self._last_confirmed_hit_event = result.hit_event

        if result.hit_event is not None:
            hit = result.hit_event

        with self._vision_lock:
            state = self.vision_service.get_state(
                self.camera_index
            )

        # --------------------------------------------------------------
        # Fusion-Latch
        # --------------------------------------------------------------
        if (
            result.status == STATUS_HIT_DETECTED
            and result.hit_event is not None
        ):
            latched = self._build_fusion_observation_from_hit_event(
                result.hit_event
            )

            self._latched_fusion_observation = latched
            self._last_observation = latched

        elif result.status in {
            STATUS_WAITING_FOR_CLEAR,
            STATUS_COOLDOWN,
            STATUS_NO_HIT,
        }:
            self._last_observation = (
                self._latched_fusion_observation
            )

        elif result.status in {
            STATUS_READY,
            STATUS_BOARD_NOT_REFERENCED,
            STATUS_DISARMED,
        }:
            self._latched_fusion_observation = None
            self._last_observation = None

            if result.status != STATUS_READY:
                self._last_confirmed_hit_event = None

        elif result.status == STATUS_ERROR:
            self._last_observation = (
                self._latched_fusion_observation
            )

        else:
            self._last_observation = (
                self._latched_fusion_observation
            )

        if callable(self._fusion_update_callback):
            self._fusion_update_callback()

        # --------------------------------------------------------------
        # Overlay aktualisieren
        # --------------------------------------------------------------
        if self._last_raw_frame is not None:
            self._last_image = (
                self._render_detection_overlay_to_qimage(
                    self._last_raw_frame
                )
            )
            self._render_last_image()

        # --------------------------------------------------------------
        # UI-Status
        # --------------------------------------------------------------
        if (
            result.status == STATUS_HIT_DETECTED
            and result.hit_event is not None
        ):
            self.vision_status_label.setText(
                f"Vision: Treffer erkannt "
                f"({result.hit_event.label})"
            )

            self.hit_label.setText(
                self._format_hit_text(result.hit_event)
            )

        elif result.status == STATUS_WAITING_FOR_CLEAR:
            ratio = getattr(
                result,
                "board_changed_ratio",
                None,
            )

            if ratio is None:
                ratio_text = "-"
            else:
                ratio_text = f"{float(ratio):.5f}"

            # ----------------------------------------------------------
            # Temporärer Debug:
            # Nur Board-Clear-Ratio anzeigen
            # ----------------------------------------------------------

            self.vision_status_label.setText(
                f"Vision: warte auf freies Board | "
                f"Diff-Ratio: {ratio_text}"
            )

            self.hit_label.setText(
                self._format_hit_text(
                    self._last_confirmed_hit_event
                )
            )

        elif result.status == STATUS_READY:
            self.vision_status_label.setText(
                "Vision: bereit"
            )

            self._last_confirmed_hit_event = None
            self.hit_label.setText("Treffer: -")

        elif result.status == STATUS_NO_HIT:
            self.vision_status_label.setText(
                "Vision: kein bestätigter Treffer"
            )

            self.hit_label.setText(
                self._format_hit_text(
                    self._last_confirmed_hit_event
                )
            )

        elif result.status == STATUS_BOARD_NOT_REFERENCED:
            self.vision_status_label.setText(
                "Vision: keine Referenz"
            )

            self._last_confirmed_hit_event = None
            self.hit_label.setText("Treffer: -")

        elif result.status == STATUS_DISARMED:
            if getattr(
                state,
                "reference_frame",
                None,
            ) is not None:
                self.vision_status_label.setText(
                    "Vision: Referenz vorhanden, "
                    "Erkennung deaktiviert"
                )
            else:
                self.vision_status_label.setText(
                    "Vision: deaktiviert"
                )

            self._last_confirmed_hit_event = None
            self.hit_label.setText("Treffer: -")

        elif result.status == STATUS_COOLDOWN:
            self.vision_status_label.setText(
                "Vision: cooldown"
            )

            self.hit_label.setText(
                self._format_hit_text(
                    self._last_confirmed_hit_event
                )
            )

        elif result.status == STATUS_ERROR:
            self.vision_status_label.setText(
                f"Vision: Fehler – {result.message}"
            )

            self.hit_label.setText(
                self._format_hit_text(
                    self._last_confirmed_hit_event
                )
            )

    def stop_worker(self) -> bool:
        """
        Stoppt den Kamera-Worker sicher.

        Der Worker wird erst vergessen, wenn sein Thread
        tatsächlich beendet wurde.
        """

        if self.worker is not None:
            worker = self.worker

            stopped = worker.stop(
                timeout_ms=5000
            )

            if not stopped:
                self.set_status(
                    "Fehler: Kamera-Thread konnte nicht beendet werden"
                )

                print(
                    f"[CameraCard] STOP FEHLGESCHLAGEN "
                    f"logical_camera={self.camera_index} "
                    f"physical_device={worker.device_id}"
                )

                return False

            self.worker = None

        self._inference_busy = False
        self._pending_inference_frame = None

        self._last_raw_frame = None
        self._last_detection_result = None
        self._last_observation = None
        self._last_confirmed_hit_event = None
        self._latched_fusion_observation = None
        self._last_image = None

        self.clear_preview(
            "Keine Vorschau"
        )

        self.set_status(
            "gestoppt"
        )

        self.vision_status_label.setText(
            "Vision: gestoppt"
        )

        self.hit_label.setText(
            "Treffer: -"
        )

        if callable(
            self._fusion_update_callback
        ):
            self._fusion_update_callback()

        return True

    def start_worker(self, config: Dict) -> None:
        if not self.stop_worker():
            print(
                f"[CameraCard] START ABGEBROCHEN "
                f"logical_camera={self.camera_index}: "
                f"alter Worker läuft noch."
            )
            return

        self._pending_inference_frame = None
        self._inference_busy = False

        self.hit_label.setText("Treffer: -")
        self.vision_status_label.setText("Vision: starte Kamera ...")
        self._last_confirmed_hit_event = None
        self._latched_fusion_observation = None
        self._last_observation = None
        self._last_detection_result = None

        if not config.get("enabled", True):
            self.clear_preview("Deaktiviert")
            self.set_status("deaktiviert")
            return

        device_id = config.get("device_id", -1)
        if device_id < 0:
            self.clear_preview("Keine Kamera ausgewählt")
            self.set_status("keine Kamera gewählt")
            return

        print(
            f"[CameraCard] START "
            f"logical_camera={self.camera_index} "
            f"physical_device={device_id}"
        )

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

        # Diese Fusion-Konfiguration ist toleranter, damit zwei real passende Kameras
        # trotz leichter Topdown-Abweichung noch gemeinsam fusioniert werden.
        # Kamera 2 darf dabei als Ausreißer herausfallen.
        self.fusion_engine = MultiCamFusionEngine(
            MultiCamFusionConfig(
                max_estimates_per_camera=1,

                min_cameras_for_fusion=2,

                allow_single_camera_fallback=True,

                confidence_floor=0.05,

                spatial_outlier_distance_px=45.0,

                single_camera_min_confidence=0.08,
            )
        )

        self.fused_result_label = QLabel("Fused Hit: -")
        self.fused_result_label.setWordWrap(True)
        self.fused_result_label.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #ffd27f;"
        )

        self._fused_hit_locked = False
        self._last_fused_result = None
        self._primary_camera_index_for_scoring = 0
        self._use_primary_camera_fallback = True

        self.card_1 = CameraCard(
            "Kamera 1",
            camera_index=0,
            detector=self.detectors[0],
            fusion_update_callback=self.update_fused_result,
        )
        self.card_2 = CameraCard(
            "Kamera 2",
            camera_index=1,
            detector=self.detectors[1],
            fusion_update_callback=self.update_fused_result,
        )
        self.card_3 = CameraCard(
            "Kamera 3",
            camera_index=2,
            detector=self.detectors[2],
            fusion_update_callback=self.update_fused_result,
        )
        self.cards = [self.card_1, self.card_2, self.card_3]

        #kameras ausblenden
        if SINGLE_CAMERA_TEST_MODE:
        # Kamera 2 und 3 für die aktuellen KI-Tests vollständig deaktivieren.
            self.card_2.enabled_check.setChecked(False)
            self.card_3.enabled_check.setChecked(False)

            self.card_2.setVisible(False)
            self.card_3.setVisible(False)


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

    def update_camera_config(self, new_config: Dict) -> None:
        """
        Übernimmt die zentrale Kamera-Konfiguration aus MainWindow.
        Startet dabei keine Kamera.
        """
        self.config_data = deepcopy(new_config)
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

        if SINGLE_CAMERA_TEST_MODE:
            # Kamera 1 bekommt die komplette verfügbare Breite.
            cards_layout.addWidget(self.card_1, 1)
        else:
            cards_layout.addWidget(self.card_1, 1)
            cards_layout.addWidget(self.card_2, 1)
            cards_layout.addWidget(self.card_3, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(self.fused_result_label)
        main_layout.addLayout(buttons_layout)
        main_layout.addLayout(cards_layout, 1)

    def _camera_status_allows_fusion_input(self, card) -> bool:
        status = getattr(card, "_last_result_status", None)
        return status in {
            STATUS_HIT_DETECTED,
            STATUS_WAITING_FOR_CLEAR,
            STATUS_COOLDOWN,
            STATUS_NO_HIT,
        }

    def _observation_is_usable_for_fusion(self, observation) -> bool:
        if observation is None:
            return False

        if getattr(observation, "best_topdown_point", None) is None:
            return False

        best_label = getattr(observation, "best_label", None)
        if best_label is not None and str(best_label).upper() == "MISS":
            return False

        return True

    # Diese Methode lässt für die Fusion nur den Segment-Mehrheitsblock durch.
    # Beispiel:
    # - K2 = S19
    # - K3 = S17
    # => kein gemeinsamer Segment-Konsens, also kein finaler Fused-Hit.
    
    def set_available_cameras(self, cameras: List[Dict[str, int]]) -> None:
        """
        Befüllt die Kamera-Auswahl für alle drei Slots.

        Entscheidend:
        Die gespeicherte device_id aus config_data ist die Quelle.
        Nicht der zufällige vorherige Combobox-Zustand.
        """
        self.available_cameras = list(cameras)

        camera_configs = self.config_data.get(
            "cameras",
            [],
        )

        for idx, card in enumerate(self.cards):
            desired_device_id = -1

            if idx < len(camera_configs):
                desired_device_id = int(
                    camera_configs[idx].get(
                        "device_id",
                        -1,
                    )
                )

            card.device_combo.blockSignals(True)

            card.device_combo.clear()
            card.device_combo.addItem(
                "Keine Kamera",
                -1,
            )

            for cam in cameras:
                card.device_combo.addItem(
                    f"{cam['name']} (Index {cam['index']})",
                    int(cam["index"]),
                )

            combo_index = card.device_combo.findData(
                desired_device_id
            )

            if combo_index >= 0:
                card.device_combo.setCurrentIndex(
                    combo_index
                )
            else:
                card.device_combo.setCurrentIndex(0)

            card.device_combo.blockSignals(False)

            print(
                f"[CamerasPage] slot={idx} "
                f"configured_device={desired_device_id} "
                f"combo_device={card.device_combo.currentData()}"
            )

    def load_config_into_ui(self) -> None:
        """
        Lädt ausschließlich die gespeicherte Konfiguration in die UI.

        Hier werden KEINE Kameras gestartet.
        Gestartet wird ausschließlich über:
        'Vorschau starten / aktualisieren'
        """
        cameras_config = self.config_data.get(
            "cameras",
            [],
        )

        for idx, card in enumerate(self.cards):
            if idx >= len(cameras_config):
                continue

            config = cameras_config[idx]

            card.set_config(config)

            print(
                f"[CAMERA CONFIG LOAD] "
                f"slot={idx} "
                f"device={config.get('device_id', -1)} "
                f"enabled={config.get('enabled', True)}"
            )

    def set_detectors(self, detectors: List[Optional[SingleCamDetector]]) -> None:
        self.detectors = list(detectors)
        while len(self.detectors) < 3:
            self.detectors.append(None)

        for idx, card in enumerate(self.cards):
            card.set_detector(self.detectors[idx])

    def _reset_fused_hit_state(self) -> None:
        self._fused_hit_locked = False
        self._last_fused_result = None
        self.fused_result_label.setText("Fused Hit: -")

    def _camera_is_active_for_fusion(self, card) -> bool:
        return bool(card.enabled_check.isChecked())

    def _get_primary_camera_observation(self):
        if not self._use_primary_camera_fallback:
            return None, None

        idx = int(self._primary_camera_index_for_scoring)
        if idx < 0 or idx >= len(self.cards):
            return None, None

        card = self.cards[idx]
        if not self._camera_is_active_for_fusion(card):
            return None, None

        observation = getattr(card, "_latched_fusion_observation", None)
        if observation is None:
            return None, None

        if not self._camera_status_allows_fusion_input(card):
            return None, None

        if not self._observation_is_usable_for_fusion(observation):
            return None, None

        return idx, observation

    def _show_primary_camera_fallback(self, camera_index: int, observation) -> None:
        label = getattr(observation, "best_label", "-")
        score = getattr(observation, "best_score", "-")
        segment = getattr(observation, "best_segment", "-")
        ring = getattr(observation, "best_ring", "-")

        confidence = getattr(observation, "best_combined_confidence", 0.0)
        confidence = 0.0 if confidence is None else float(confidence)

        camera_debug = self._build_camera_hit_debug_text()

        self._last_fused_result = None
        self._fused_hit_locked = True

        self.fused_result_label.setText(
            f"Fused Hit: {label} | "
            f"Score: {score} | "
            f"Segment: {segment} | "
            f"Ring: {ring} | "
            f"Kameras: {camera_index + 1} | "
            f"Conf: {confidence:.2f} | "
            f"Modus: Primärkamera-Fallback\n"
            f"{camera_debug}"
        )

    def _all_active_cameras_are_clear(self) -> bool:
        """
        Prüft, ob der vorherige Fused-Hit freigegeben werden darf.

        WICHTIG:
        Eine Kamera ohne eigenen gelatchten Treffer darf mit STATUS_NO_HIT
        die Freigabe nicht blockieren.

        Beispiel:
            K1 vorher T19 -> danach READY
            K2 hatte nie einen TIP -> STATUS_NO_HIT
            K3 vorher T19 -> danach READY

        Dann ist der alte T19-Wurf beendet und die Fusion muss für den
        nächsten Dart wieder freigegeben werden.
        """

        saw_active_camera = False

        for idx, card in enumerate(self.cards):
            if not self._camera_is_active_for_fusion(card):
                continue

            saw_active_camera = True

            status = getattr(
                card,
                "_last_result_status",
                None,
            )

            latched_observation = getattr(
                card,
                "_latched_fusion_observation",
                None,
            )

            # --------------------------------------------------------
            # Noch gar kein Status:
            # noch nicht bereit zum globalen Clear.
            # --------------------------------------------------------
            if status is None:
                return False

            # --------------------------------------------------------
            # Eindeutig freie Zustände
            # --------------------------------------------------------
            if status in {
                STATUS_READY,
                STATUS_BOARD_NOT_REFERENCED,
                STATUS_DISARMED,
            }:
                continue

            # --------------------------------------------------------
            # WICHTIG:
            #
            # STATUS_NO_HIT ist ebenfalls neutral/frei,
            # ABER nur wenn diese Kamera keinen alten gelatchten
            # Treffer mehr besitzt.
            #
            # Dadurch blockiert eine Kamera, die den Dart überhaupt
            # nicht gesehen hat, den nächsten Wurf nicht.
            # --------------------------------------------------------
            if (
                status == STATUS_NO_HIT
                and latched_observation is None
            ):
                continue

            # --------------------------------------------------------
            # Alles andere bedeutet:
            # - alter Treffer noch vorhanden
            # - waiting_for_clear
            # - cooldown
            # usw.
            # --------------------------------------------------------
            return False

        print(
            "[FUSION CLEAR] "
            "alle aktiven Kameras für neuen Wurf freigegeben"
        )

        return saw_active_camera

    def _any_camera_has_latched_hit(self) -> bool:
            for card in self.cards:
                if not self._camera_is_active_for_fusion(card):
                    continue
                if getattr(card, "_latched_fusion_observation", None) is not None:
                    return True
            return False

    # Diese Hilfsmethode formatiert einen vorläufigen Einzelkamera-Kandidaten.
    # Er wird angezeigt, aber noch nicht als finaler Fused-Hit gelockt.
    def _format_single_camera_candidate(self, camera_index: int, observation) -> str:
        label = getattr(observation, "best_label", "-")
        score = getattr(observation, "best_score", "-")
        segment = getattr(observation, "best_segment", "-")
        ring = getattr(observation, "best_ring", "-")

        confidence = getattr(observation, "best_combined_confidence", 0.0)
        confidence = 0.0 if confidence is None else float(confidence)

        return (
            f"Fused Hit: warte auf 2. Kamera | "
            f"Kandidat: {label} | "
            f"Score: {score} | "
            f"Segment: {segment} | "
            f"Ring: {ring} | "
            f"Kamera: {camera_index + 1} | "
            f"Conf: {confidence:.2f}"
        )

    # Diese Methode baut einen kompakten Text mit den aktuell gelatchten Kamera-Treffern.
    # So sieht man direkt in der UI, welche Kamera welchen Treffer in die Fusion einspeist.
    def _build_camera_hit_debug_text(self) -> str:
        parts = []

        for idx, card in enumerate(self.cards):
            obs = getattr(card, "_latched_fusion_observation", None)

            if obs is None:
                parts.append(f"K{idx + 1}: -")
                continue

            label = getattr(obs, "best_label", "-")
            score = getattr(obs, "best_score", "-")
            conf = getattr(obs, "best_combined_confidence", 0.0)
            conf = 0.0 if conf is None else float(conf)

            parts.append(f"K{idx + 1}: {label} ({score}) @{conf:.2f}")

        return " | ".join(parts)

    # Diese Methode verarbeitet die gelatchten Kamera-Treffer für die Mehrkamera-Fusion.
    # Final gelockt wird nur noch, wenn mindestens 2 Kameras wirklich zum Fusion-Ergebnis beitragen.
    def update_fused_result(self) -> None:
        # --------------------------------------------------------------
        # 1. Falls ein alter Fused-Hit noch gelockt ist:
        #    erst freigeben, wenn der vorige Wurf wirklich beendet ist.
        # --------------------------------------------------------------
        if self._fused_hit_locked:
            if self._all_active_cameras_are_clear():
                self._reset_fused_hit_state()
            else:
                return

        # --------------------------------------------------------------
        # 2. Gibt es überhaupt aktuell gelatchte Treffer?
        # --------------------------------------------------------------
        if not self._any_camera_has_latched_hit():
            if self._last_fused_result is None:
                self.fused_result_label.setText(
                    "Fused Hit: -"
                )
            return

        detectors_by_camera = {}
        observations_by_camera = {}

        # --------------------------------------------------------------
        # 3. Nur verwertbare Kamera-Beobachtungen einsammeln
        # --------------------------------------------------------------
        for idx, card in enumerate(self.cards):
            observation = getattr(
                card,
                "_latched_fusion_observation",
                None,
            )

            if card.detector is None:
                continue

            if observation is None:
                continue

            if not self._camera_status_allows_fusion_input(
                card
            ):
                continue

            if not self._observation_is_usable_for_fusion(
                observation
            ):
                continue

            detectors_by_camera[idx] = card.detector
            observations_by_camera[idx] = observation


        # --------------------------------------------------------------
        # 5. Keine brauchbare Observation
        # --------------------------------------------------------------
        if (
            not detectors_by_camera
            or not observations_by_camera
        ):
            self.fused_result_label.setText(
                "Fused Hit: -"
            )
            return

        # --------------------------------------------------------------
        # 6. Nur eine Kamera:
        #    anzeigen, aber NICHT final locken.
        #
        #    Kein Segment-Filter mehr.
        # --------------------------------------------------------------
        if len(observations_by_camera) == 1:
            cam_idx, observation = next(
                iter(observations_by_camera.items())
            )

            self._last_fused_result = None
            self._fused_hit_locked = False

            camera_debug = (
                self._build_camera_hit_debug_text()
            )

            self.fused_result_label.setText(
                f"{self._format_single_camera_candidate(cam_idx, observation)}\n"
                f"{camera_debug}"
            )
            return

        # --------------------------------------------------------------
        # 7. Ab zwei Kameras entscheidet ausschließlich
        #    MultiCamFusionEngine.
        #
        #    KEIN vorgeschalteter Segment-Konsens mehr.
        #
        #    Dadurch funktionieren auch:
        #
        #    DBULL + DBULL
        #    SBULL + SBULL
        #
        #    obwohl segment=None ist.
        # --------------------------------------------------------------
        try:
            fused = self.fusion_engine.fuse(
                observations_by_camera=(
                    observations_by_camera
                ),
                detectors_by_camera=(
                    detectors_by_camera
                ),
            )

        except Exception as exc:
            self._last_fused_result = None
            self._fused_hit_locked = False

            self.fused_result_label.setText(
                f"Fused Hit: Fehler – {exc}"
            )
            return

        # --------------------------------------------------------------
        # 8. Keine gültige Mehrheitsentscheidung
        # --------------------------------------------------------------
        if fused is None:
            self._last_fused_result = None
            self._fused_hit_locked = False

            camera_debug = (
                self._build_camera_hit_debug_text()
            )

            self.fused_result_label.setText(
                "Fused Hit: warte auf übereinstimmende Kameras ...\n"
                f"{camera_debug}"
            )
            return

        # --------------------------------------------------------------
        # 9. Welche Kameras wurden tatsächlich verwendet?
        # --------------------------------------------------------------
        used_camera_indices = sorted(
            {
                int(obs.camera_index)
                for obs in fused.observations_used
            }
        )

        used_camera_count = len(
            used_camera_indices
        )

        # --------------------------------------------------------------
        # 10. Sicherheitsnetz:
        #     Falls die Fusion intern doch nur eine Kamera benutzt,
        #     nicht final locken.
        # --------------------------------------------------------------
        if used_camera_count < 2:
            if used_camera_indices:
                cam_idx = used_camera_indices[0]
            else:
                cam_idx = next(
                    iter(
                        observations_by_camera.keys()
                    )
                )

            observation = (
                observations_by_camera.get(
                    cam_idx
                )
            )

            self._last_fused_result = None
            self._fused_hit_locked = False

            if observation is not None:
                camera_debug = (
                    self._build_camera_hit_debug_text()
                )

                self.fused_result_label.setText(
                    f"{self._format_single_camera_candidate(cam_idx, observation)}\n"
                    f"{camera_debug}"
                )
            else:
                self.fused_result_label.setText(
                    "Fused Hit: warte auf 2. Kamera ..."
                )

            return

        # --------------------------------------------------------------
        # 11. Final bestätigter Mehrkamera-Treffer
        # --------------------------------------------------------------
        cam_list = [
            cam_idx + 1
            for cam_idx in used_camera_indices
        ]

        cam_text = (
            ", ".join(
                str(cam)
                for cam in cam_list
            )
            if cam_list
            else "-"
        )

        self._last_fused_result = fused
        self._fused_hit_locked = True

        camera_debug = (
            self._build_camera_hit_debug_text()
        )

        self.fused_result_label.setText(
            f"Fused Hit: {fused.label} | "
            f"Score: {fused.score} | "
            f"Segment: {fused.segment} | "
            f"Ring: {fused.ring} | "
            f"Kameras: {cam_text} | "
            f"Conf: {fused.confidence:.2f}\n"
            f"{camera_debug}"
        )

        # --------------------------------------------------------------
        # 12. Terminal-Debug für finalen Treffer
        # --------------------------------------------------------------
        print(
            "[FUSION RESULT]",
            f"label={fused.label}",
            f"score={fused.score}",
            f"ring={fused.ring}",
            f"segment={fused.segment}",
            f"cameras={used_camera_indices}",
            f"confidence={fused.confidence:.3f}",
        )

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
        """
        Startet die drei Kameras eindeutig und genau einmal.

        Ablauf:
        1. aktuelle UI-Auswahl lesen
        2. Doppelbelegung prüfen
        3. Config aktualisieren
        4. alle alten Worker stoppen
        5. jeden Kamera-Slot genau einmal starten
        """

        # ------------------------------------------------------------
        # 1. UI-Konfiguration einsammeln
        # ------------------------------------------------------------
        new_config = self.collect_ui_config()

        cameras_config = new_config.get(
            "cameras",
            [],
        )

        if len(cameras_config) < 3:
            QMessageBox.warning(
                self,
                "Kamera-Konfiguration",
                "Es müssen drei Kamera-Slots konfiguriert sein.",
            )
            return

        # ------------------------------------------------------------
        # 2. Aktive Geräte prüfen
        # ------------------------------------------------------------
        used_device_ids: set[int] = set()

        for idx, config in enumerate(cameras_config):
            enabled = bool(
                config.get(
                    "enabled",
                    True,
                )
            )

            if not enabled:
                continue

            device_id = int(
                config.get(
                    "device_id",
                    -1,
                )
            )

            if device_id < 0:
                QMessageBox.warning(
                    self,
                    "Kamera fehlt",
                    (
                        f"Für Kamera {idx + 1} wurde "
                        f"keine physische Kamera ausgewählt."
                    ),
                )
                return

            if device_id in used_device_ids:
                QMessageBox.warning(
                    self,
                    "Kamera doppelt ausgewählt",
                    (
                        f"Die physische Kamera {device_id} "
                        f"wurde mehrfach vergeben."
                    ),
                )
                return

            used_device_ids.add(
                device_id
            )

        # ------------------------------------------------------------
        # 3. Konfiguration übernehmen
        # ------------------------------------------------------------
        self.config_data = new_config

        # ------------------------------------------------------------
        # 4. Fusion zurücksetzen
        # ------------------------------------------------------------
        self._fused_hit_locked = False
        self._last_fused_result = None

        self.fused_result_label.setText(
            "Fused Hit: warte auf Kameradaten ..."
        )

        # ------------------------------------------------------------
        # 5. ALLE alten Worker zuerst stoppen
        # ------------------------------------------------------------
        for card in self.cards:
            card.stop_worker()

        # ------------------------------------------------------------
        # 6. Mapping ausgeben
        # ------------------------------------------------------------
        for idx in range(3):
            config = self.config_data["cameras"][idx]

            print(
                f"[CAMERA MAP] "
                f"slot={idx} "
                f"card_index={self.cards[idx].camera_index} "
                f"-> device={config.get('device_id')} "
                f"enabled={config.get('enabled')}"
            )

        # ------------------------------------------------------------
        # 7. Jeder Slot wird EXPLIZIT genau einmal gestartet.
        # ------------------------------------------------------------
        print("[CAMERA START] Slot 0")
        self.card_1.start_worker(
            self.config_data["cameras"][0]
        )

        print("[CAMERA START] Slot 1")
        self.card_2.start_worker(
            self.config_data["cameras"][1]
        )

        print("[CAMERA START] Slot 2")
        self.card_3.start_worker(
            self.config_data["cameras"][2]
        )

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
        self._reset_fused_hit_state()

    def shutdown(self) -> None:
        """
        Beendet die Kameraseite vollständig.

        Wichtig beim Programmende:
        - CameraWorker stoppen
        - Vision-Inference-Timer stoppen
        - Vision-QThreads sauber beenden
        """

        self._reset_fused_hit_state()

        for card in self.cards:
            card.shutdown()