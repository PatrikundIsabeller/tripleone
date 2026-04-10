# app/pages/calibration_page.py
# Triple One - Kalibrierungsseite
#
# Diese Version verwendet 3 Kalibrierkarten nebeneinander – analog zur Kameraseite.
# Jede Karte kann:
# - eine verfügbare Kamera auswählen
# - Livebild anzeigen
# - 4 Kalibrierpunkte setzen
# - Testpunkt prüfen
# - leeres Board speichern
# - Einzeldart-Detector scharf schalten
# - Detektion zurücksetzen
#
# WICHTIG:
# - Keine eigene Homography-Logik
# - Keine eigene Ring-/Sektorlogik
# - Keine eigene Scorelogik
#
# Alles Geometrische läuft über:
# - vision.calibration_geometry.py
# Alles Visuelle im Preview läuft über:
# - app/widgets/calibration_preview.py

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QComboBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.calibration_preview import CalibrationPreview
from vision.calibration_geometry import (
    build_pipeline_points,
    calculate_hit_from_image_point,
    compute_bull_from_manual_points,
    generate_ring_polylines_image,
)
from vision.camera_manager import CameraWorker
from vision.dart_detector import DartDetector


class CalibrationCard(QFrame):
    """
    Eine einzelne Kalibrierkarte für genau einen Kamera-Slot.
    """

    def __init__(self, title: str, slot_index: int, parent=None):
        super().__init__(parent)

        self.slot_index = int(slot_index)
        self.worker: Optional[CameraWorker] = None
        self.camera_config: Dict = {}
        self.available_cameras: List[Dict[str, int]] = []
        self._manual_points: List[Dict[str, int]] = []

        self.detector = DartDetector()
        self.last_frame_bgr: Optional[np.ndarray] = None

        self.setObjectName("CalibrationCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
            QComboBox, QDoubleSpinBox {
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

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.device_combo = QComboBox()

        self.device_info_label = QLabel("Gerät: keine Kamera gewählt")
        self.device_info_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self.help_label = QLabel(
            "4 feste Punkte setzen:\n"
            "- P1 = 20|1\n"
            "- P2 = 6|10\n"
            "- P3 = 3|19\n"
            "- P4 = 11|14\n"
            "- Bull wird automatisch berechnet\n"
            "- Rechtsklick = Präzisions-Testpunkt\n"
            "- Enter bestätigt Testpunkt\n"
            "- Tasten 1/2/3/4 wählen P1/P2/P3/P4\n"
            "- Pfeiltasten = 1 px | Shift = 5 px"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setStyleSheet("font-size: 12px; color: #d8d8d8;")

        self.preview = CalibrationPreview()
        self.preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.10, 1.00)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setDecimals(2)
        self.alpha_spin.setValue(0.35)

        self.show_numbers_check = QCheckBox("Zahlen anzeigen")
        self.show_numbers_check.setChecked(True)

        self.show_sector_lines_check = QCheckBox("Sektorlinien anzeigen")
        self.show_sector_lines_check.setChecked(True)

        self.show_filled_segments_check = QCheckBox("Feldfarben anzeigen")
        self.show_filled_segments_check.setChecked(True)

        self.reset_points_button = QPushButton("4 Punkte zurücksetzen")
        self.save_empty_board_button = QPushButton("Leeres Board speichern")
        self.arm_button = QPushButton("Einzeldart scharf")
        self.reset_detection_button = QPushButton("Detektion zurücksetzen")

        self.point_info_label = QLabel("Punkte: -")
        self.point_info_label.setWordWrap(True)
        self.point_info_label.setStyleSheet("font-size: 12px; color: #8ee6ff; font-weight: 600;")

        self.test_result_label = QLabel("Testpunkt: noch keiner gesetzt")
        self.test_result_label.setWordWrap(True)
        self.test_result_label.setStyleSheet("font-size: 12px; color: #8effc9; font-weight: 700;")

        self.auto_result_label = QLabel("Auto-Dart: noch keiner")
        self.auto_result_label.setWordWrap(True)
        self.auto_result_label.setStyleSheet("font-size: 12px; color: #ffd27f; font-weight: 700;")

        self.debug_label = QLabel("Debug: -")
        self.debug_label.setWordWrap(True)
        self.debug_label.setStyleSheet("font-size: 11px; color: #d0d0d0;")

        self.detector_status_label = QLabel("Detector: keine Referenz")
        self.detector_status_label.setWordWrap(True)
        self.detector_status_label.setStyleSheet("font-size: 12px; color: #ffdb8a; font-weight: 700;")

        self.status_label = QLabel("Status: nicht gestartet")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self._build_ui()
        self._connect_signals()
        self._push_overlay_to_preview()
        self._update_point_info_label()
        self._update_debug_label()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        device_row = QHBoxLayout()
        device_row.setSpacing(8)
        device_row.addWidget(QLabel("Kamera:"))
        device_row.addWidget(self.device_combo, 1)

        row_overlay = QHBoxLayout()
        row_overlay.setSpacing(8)
        row_overlay.addWidget(QLabel("Alpha:"))
        row_overlay.addWidget(self.alpha_spin)
        row_overlay.addStretch()
        row_overlay.addWidget(self.reset_points_button)

        row_checks = QHBoxLayout()
        row_checks.setSpacing(8)
        row_checks.addWidget(self.show_numbers_check)
        row_checks.addWidget(self.show_filled_segments_check)
        row_checks.addStretch()
        row_checks.addWidget(self.show_sector_lines_check)

        row_detector = QHBoxLayout()
        row_detector.setSpacing(8)
        row_detector.addWidget(self.save_empty_board_button)
        row_detector.addWidget(self.arm_button)
        row_detector.addWidget(self.reset_detection_button)

        layout.addWidget(self.title_label)
        layout.addLayout(device_row)
        layout.addWidget(self.device_info_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.preview, 1)
        layout.addLayout(row_overlay)
        layout.addLayout(row_checks)
        layout.addLayout(row_detector)
        layout.addWidget(self.point_info_label)
        layout.addWidget(self.test_result_label)
        layout.addWidget(self.auto_result_label)
        layout.addWidget(self.debug_label)
        layout.addWidget(self.detector_status_label)
        layout.addWidget(self.status_label)

    def _connect_signals(self) -> None:
        self.alpha_spin.valueChanged.connect(self._push_overlay_to_preview)
        self.show_numbers_check.toggled.connect(self._push_overlay_to_preview)
        self.show_sector_lines_check.toggled.connect(self._push_overlay_to_preview)
        self.show_filled_segments_check.toggled.connect(self._push_overlay_to_preview)
        self.preview.points_changed.connect(self._handle_points_changed)
        self.preview.test_point_selected.connect(self._handle_test_point_selected)

        self.reset_points_button.clicked.connect(self.reset_points)
        self.save_empty_board_button.clicked.connect(self.save_empty_board)
        self.arm_button.clicked.connect(self.arm_detector)
        self.reset_detection_button.clicked.connect(self.reset_detection)

    # ------------------------------------------------------------
    # Kameraauswahl / Runtime
    # ------------------------------------------------------------

    def set_available_cameras(self, cameras: List[Dict[str, int]]) -> None:
        self.available_cameras = list(cameras)
        previous_device_id = self.device_combo.currentData()

        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("Keine Kamera", -1)

        for cam in cameras:
            self.device_combo.addItem(f"{cam['name']} (Index {cam['index']})", cam["index"])

        found_index = self.device_combo.findData(previous_device_id)
        if found_index >= 0:
            self.device_combo.setCurrentIndex(found_index)

        self.device_combo.blockSignals(False)

    def reset_runtime_state(self) -> None:
        self.stop_worker()
        self.last_frame_bgr = None
        self.preview.clear_frame()
        self.preview.clear_test_point()
        self.detector.reset_detection()
        self.test_result_label.setText("Testpunkt: noch keiner gesetzt")
        self.auto_result_label.setText("Auto-Dart: noch keiner")
        self.detector_status_label.setText("Detector: keine Referenz")
        self.status_label.setText("Status: gestoppt")
        self._update_debug_label()

    # ------------------------------------------------------------
    # Daten / Overlay
    # ------------------------------------------------------------

    def _default_manual_points(self) -> List[Dict[str, int]]:
        frame_width = self._get_frame_width()
        frame_height = self._get_frame_height()
        return [
            {"x_px": int(frame_width * 0.60), "y_px": int(frame_height * 0.28)},
            {"x_px": int(frame_width * 0.70), "y_px": int(frame_height * 0.72)},
            {"x_px": int(frame_width * 0.26), "y_px": int(frame_height * 0.67)},
            {"x_px": int(frame_width * 0.37), "y_px": int(frame_height * 0.30)},
        ]

    def _get_frame_width(self) -> int:
        return max(320, int(self.camera_config.get("width", 1280)))

    def _get_frame_height(self) -> int:
        return max(240, int(self.camera_config.get("height", 720)))

    def _pipeline_points(self) -> List[Dict[str, int]]:
        manual = self._manual_points if self._manual_points else self._default_manual_points()
        return build_pipeline_points(manual)

    def _push_overlay_to_preview(self) -> None:
        self.preview.set_overlay_config(self.get_calibration_config())

    def _handle_points_changed(self, points: list) -> None:
        self._manual_points = deepcopy(points[:4])
        self._update_point_info_label()

    def _handle_test_point_selected(self, x_px: int, y_px: int) -> None:
        result = calculate_hit_from_image_point(x_px, y_px, self.get_calibration_config())
        if result is None:
            self.test_result_label.setText(
                f"Testpunkt: Fehler bei Rückprojektion | Bild=({x_px}, {y_px})"
            )
            return

        self.preview.set_test_point(x_px, y_px)
        self.test_result_label.setText(
            f"Testpunkt: {result.label} = {result.score} | "
            f"Ring: {result.ring_name} | "
            f"r={result.radius:.3f} | "
            f"Bild=({result.image_x_px}, {result.image_y_px})"
        )

    def _update_point_info_label(self) -> None:
        manual = deepcopy(self._manual_points if self._manual_points else self._default_manual_points())
        bull = compute_bull_from_manual_points(manual)

        labels = ["P1(20|1)", "P2(6|10)", "P3(3|19)", "P4(11|14)"]
        parts = []

        for i, point in enumerate(manual[:4]):
            parts.append(f"{labels[i]}=({int(point['x_px'])},{int(point['y_px'])})")

        parts.append(f"C(auto)=({bull['x_px']},{bull['y_px']})")
        self.point_info_label.setText("Punkte: " + " | ".join(parts))

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
            f"ref={dbg.has_reference} | "
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

    def set_camera_config(self, camera_config: Dict) -> None:
        self.camera_config = dict(camera_config or {})

        desired_device_id = int(self.camera_config.get("device_id", -1))
        combo_index = self.device_combo.findData(desired_device_id)
        if combo_index >= 0:
            self.device_combo.setCurrentIndex(combo_index)
        else:
            self.device_combo.setCurrentIndex(0)

        enabled = bool(self.camera_config.get("enabled", True))
        device_id = int(self.camera_config.get("device_id", -1))

        if not enabled:
            self.device_info_label.setText("Gerät: deaktiviert")
        elif device_id < 0:
            self.device_info_label.setText("Gerät: keine Kamera gewählt")
        else:
            self.device_info_label.setText(
                f"Gerät: Index {device_id} – "
                f"{self.camera_config.get('width', 1280)}x{self.camera_config.get('height', 720)} @ "
                f"{self.camera_config.get('fps', 30)} FPS"
            )

        self.status_label.setText("Status: Kamera-Konfiguration geladen")

    def set_calibration_config(self, calibration_config: Dict) -> None:
        self.alpha_spin.setValue(float(calibration_config.get("overlay_alpha", 0.35)))
        self.show_numbers_check.setChecked(bool(calibration_config.get("show_numbers", True)))
        self.show_sector_lines_check.setChecked(bool(calibration_config.get("show_sector_lines", True)))
        self.show_filled_segments_check.setChecked(bool(calibration_config.get("show_filled_segments", True)))

        raw_points = deepcopy(calibration_config.get("points", self._default_manual_points()))
        self._manual_points = raw_points[:4]

        self.preview.set_overlay_config(self.get_calibration_config())
        self.preview.clear_test_point()
        self.detector.reset_detection()
        self.test_result_label.setText("Testpunkt: noch keiner gesetzt")
        self.auto_result_label.setText("Auto-Dart: noch keiner")
        self.detector_status_label.setText("Detector: keine Referenz")
        self._update_point_info_label()
        self._update_debug_label()

    def get_calibration_config(self) -> Dict:
        return {
            "name": f"Kamera {self.slot_index + 1}",
            "frame_width": self._get_frame_width(),
            "frame_height": self._get_frame_height(),
            "overlay_alpha": float(self.alpha_spin.value()),
            "show_numbers": bool(self.show_numbers_check.isChecked()),
            "show_sector_lines": bool(self.show_sector_lines_check.isChecked()),
            "show_filled_segments": bool(self.show_filled_segments_check.isChecked()),
            "points": self._pipeline_points(),
        }

    # ------------------------------------------------------------
    # Aktionen
    # ------------------------------------------------------------

    def reset_points(self) -> None:
        self._manual_points = self._default_manual_points()
        self.preview.set_overlay_config(self.get_calibration_config())
        self.preview.clear_test_point()
        self.detector.reset_detection()
        self.test_result_label.setText("Testpunkt: noch keiner gesetzt")
        self.auto_result_label.setText("Auto-Dart: noch keiner")
        self.detector_status_label.setText("Detector: keine Referenz")
        self._update_point_info_label()
        self._update_debug_label()

    def _qimage_to_bgr(self, image: QImage) -> np.ndarray:
        converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
        width = converted.width()
        height = converted.height()

        ptr = converted.bits()
        ptr.setsize(height * width * 4)

        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

    def _build_board_mask(self, frame_shape) -> Optional[np.ndarray]:
        """
        Baut eine Maske für den eigentlichen Dartboard-Bereich.
        Grundlage ist der äußerste Ring des Kalibrierungs-Overlays.
        """
        calibration = self.get_calibration_config()
        polylines = generate_ring_polylines_image(calibration, degree_step=3)

        if not polylines:
            return None

        outer_ring = polylines[0]
        if outer_ring is None or len(outer_ring) < 8:
            return None

        if len(frame_shape) == 3:
            height, width = frame_shape[:2]
        else:
            height, width = frame_shape

        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.round(np.asarray(outer_ring, dtype=np.float32)).astype(np.int32)
        pts = pts.reshape(-1, 1, 2)

        cv2.fillPoly(mask, [pts], 255)

        # Kleine Reserve, damit Draht / Spitze am Rand nicht abgeschnitten werden
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _apply_board_mask_to_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Blendet alles außerhalb des Boards aus.
        """
        mask = self._build_board_mask(frame_bgr.shape)
        if mask is None:
            return frame_bgr.copy()

        return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)

    def _build_board_mask(self, frame_shape: tuple[int, int, int] | tuple[int, int]) -> Optional[np.ndarray]:
        """
        Baut eine Binärmaske für den eigentlichen Dartboard-Bereich.

        Grundlage:
        - äußerster Ring aus generate_ring_polylines_image(...)
        - daraus gefülltes Polygon
        - leicht dilatiert, damit Draht / Barrel / Spitze am Rand nicht abgeschnitten werden
        """
        calibration = self.get_calibration_config()
        polylines = generate_ring_polylines_image(calibration, degree_step=3)

        if not polylines:
            return None

        outer_ring = polylines[0]
        if outer_ring is None or len(outer_ring) < 8:
            return None

        if len(frame_shape) == 3:
            height, width = frame_shape[:2]
        else:
            height, width = frame_shape

        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.round(np.asarray(outer_ring, dtype=np.float32)).astype(np.int32)
        pts = pts.reshape(-1, 1, 2)

        cv2.fillPoly(mask, [pts], 255)

        # Leichte Sicherheitsreserve rund um das Board
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def _apply_board_mask_to_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Blendet alles außerhalb des Dartboards aus, damit der Detector
        nur auf dem Board arbeitet.
        """
        mask = self._build_board_mask(frame_bgr.shape)
        if mask is None:
            return frame_bgr.copy()

        masked = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
        return masked

    def update_preview(self, image: QImage) -> None:
        self.preview.set_frame(image)

        try:
            self.last_frame_bgr = self._qimage_to_bgr(image)
        except Exception:
            self.last_frame_bgr = None
            return

        if self.last_frame_bgr is None:
            return

        detector_frame = self._apply_board_mask_to_frame(self.last_frame_bgr)

        detection = self.detector.process_frame(
            detector_frame,
            self.get_calibration_config(),
        )
        self._update_debug_label()

        if detection is not None:
            # Auto-Dart direkt im Bild markieren:
            # Wir nutzen denselben sichtbaren Punkt wie beim manuellen Testpunkt,
            # damit sofort erkennbar ist, wo der Detector wirklich misst.
            self.preview.set_test_point(int(detection.x_px), int(detection.y_px))

            self.auto_result_label.setText(
                f"Auto-Dart: {detection.score_label} = {detection.score_value} | "
                f"Ring: {detection.ring_name} | "
                f"Bild=({detection.x_px}, {detection.y_px})"
            )
            self.detector_status_label.setText(
                "Detector: Dart erkannt, bitte zurücksetzen oder leeres Board neu speichern"
            )

    def set_status(self, text: str) -> None:
        self.status_label.setText(f"Status: {text}")
        self.preview.set_status_text(text)

    def save_empty_board(self) -> None:
        if self.last_frame_bgr is None:
            self.detector_status_label.setText("Detector: kein Livebild für Referenz vorhanden")
            return

        detector_frame = self._apply_board_mask_to_frame(self.last_frame_bgr)

        ok = self.detector.set_reference_frame(
            detector_frame,
            self.get_calibration_config(),
        )
        if ok:
            self.preview.clear_test_point()
            self.auto_result_label.setText("Auto-Dart: noch keiner")
            self.detector_status_label.setText("Detector: leeres Board gespeichert")
        else:
            self.detector_status_label.setText("Detector: Referenz konnte nicht gespeichert werden")

        self._update_debug_label()

    def arm_detector(self) -> None:
        ok = self.detector.arm()
        if ok:
            self.preview.clear_test_point()
            self.auto_result_label.setText("Auto-Dart: warte auf Einzeldart")
            self.detector_status_label.setText("Detector: scharf für genau einen Dart")
        else:
            self.detector_status_label.setText("Detector: zuerst leeres Board speichern")

        self._update_debug_label()

    def reset_detection(self) -> None:
        self.detector.reset_detection()
        self.preview.clear_test_point()
        self.auto_result_label.setText("Auto-Dart: noch keiner")

        if self.detector.reference_topdown_gray is not None:
            self.detector_status_label.setText("Detector: zurückgesetzt, Referenz bleibt erhalten")
        else:
            self.detector_status_label.setText("Detector: zurückgesetzt, keine Referenz")

        self._update_debug_label()

    # ------------------------------------------------------------
    # Kamera
    # ------------------------------------------------------------

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        self.preview.clear_frame()
        self.last_frame_bgr = None
        self.set_status("gestoppt")

    def start_worker(self) -> None:
        self.stop_worker()

        device_id = int(self.device_combo.currentData())
        self.camera_config["device_id"] = device_id

        enabled = bool(self.camera_config.get("enabled", True))

        if not enabled:
            self.preview.clear_frame()
            self.set_status("deaktiviert")
            return

        if device_id < 0:
            self.preview.clear_frame()
            self.set_status("keine Kamera gewählt")
            return

        print(
            "[CalibrationCard] start_worker -> "
            f"slot={self.slot_index}, "
            f"device_id={device_id}, "
            f"enabled={enabled}, "
            f"width={self.camera_config.get('width', 1280)}, "
            f"height={self.camera_config.get('height', 720)}, "
            f"fps={self.camera_config.get('fps', 30)}"
        )

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
    """
    Mehrkamera-Kalibrierungsseite analog zur Kameraseite.
    """

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
        self.available_cameras: List[Dict[str, int]] = []

        self.title_label = QLabel("TripleOne – Kalibrierung")
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold;")

        self.info_label = QLabel(
            "Jede Karte kalibriert genau einen Kamera-Slot.\n"
            "Wähle pro Karte eine verfügbare Kamera, starte das Livebild und setze die 4 Punkte."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 13px; color: #cccccc;")

        self.card_1 = CalibrationCard("Kalibrierung – Kamera 1", slot_index=0)
        self.card_2 = CalibrationCard("Kalibrierung – Kamera 2", slot_index=1)
        self.card_3 = CalibrationCard("Kalibrierung – Kamera 3", slot_index=2)
        self.cards = [self.card_1, self.card_2, self.card_3]

        self.start_button = QPushButton("Livebilder starten / aktualisieren")
        self.stop_button = QPushButton("Alle Kameras stoppen")
        self.save_button = QPushButton("Kalibrierung speichern")

        self.start_button.clicked.connect(self.apply_preview)
        self.stop_button.clicked.connect(self.stop_all_cameras)
        self.save_button.clicked.connect(self.save_settings)

        self.global_info_label = QLabel("Hinweis: Jede Karte arbeitet unabhängig.")
        self.global_info_label.setWordWrap(True)
        self.global_info_label.setStyleSheet("font-size: 12px; color: #8effc9; font-weight: 700;")

        self._build_ui()
        self._load_all_data()

    # ------------------------------------------------------------
    # UI
    # ------------------------------------------------------------

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

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        top_row.addWidget(self.start_button)
        top_row.addWidget(self.stop_button)
        top_row.addWidget(self.save_button)
        top_row.addStretch()

        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(14)
        cards_layout.addWidget(self.card_1, 1)
        cards_layout.addWidget(self.card_2, 1)
        cards_layout.addWidget(self.card_3, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(self.title_label)
        layout.addWidget(self.info_label)
        layout.addLayout(top_row)
        layout.addWidget(self.global_info_label)
        layout.addLayout(cards_layout, 1)

    # ------------------------------------------------------------
    # Daten / Config
    # ------------------------------------------------------------

    def set_available_cameras(self, cameras: List[Dict[str, int]]) -> None:
        self.available_cameras = list(cameras)
        for card in self.cards:
            card.set_available_cameras(cameras)

    def _load_all_data(self) -> None:
        cam_list = self.camera_config.get("cameras", [])
        cal_list = self.calibration_config.get("cameras", [])

        for idx, card in enumerate(self.cards):
            if idx < len(cam_list):
                card.set_camera_config(cam_list[idx])
            else:
                card.set_camera_config({})

            if idx < len(cal_list):
                card.set_calibration_config(cal_list[idx])
            else:
                card.set_calibration_config({})

    def update_camera_config(self, new_camera_config: Dict) -> None:
        self.camera_config = deepcopy(new_camera_config)
        self._load_all_data()

    def collect_calibration_config(self) -> Dict:
        result = deepcopy(self.calibration_config)
        cameras = result.get("cameras", [])

        while len(cameras) < len(self.cards):
            cameras.append({"name": f"Kamera {len(cameras) + 1}"})

        for idx, card in enumerate(self.cards):
            cameras[idx]["name"] = f"Kamera {idx + 1}"
            cameras[idx].update(card.get_calibration_config())

        result["cameras"] = cameras
        return result

    # ------------------------------------------------------------
    # Aktionen
    # ------------------------------------------------------------

    def apply_preview(self) -> None:
        for idx, card in enumerate(self.cards):
            if idx < len(self.camera_config.get("cameras", [])):
                # Beim Start die aktuelle UI-Auswahl als device_id übernehmen
                selected_device_id = int(card.device_combo.currentData())
                self.camera_config["cameras"][idx]["device_id"] = selected_device_id
                card.set_camera_config(self.camera_config["cameras"][idx])

            card.start_worker()

    def stop_all_cameras(self) -> None:
        for card in self.cards:
            card.stop_worker()

    def save_settings(self) -> None:
        self.calibration_config = self.collect_calibration_config()
        self.save_callback(self.calibration_config)

        QMessageBox.information(
            self,
            "Gespeichert",
            "Die Kalibrierung aller Kamera-Slots wurde gespeichert.",
        )