# app/pages/calibration_page.py
# Triple One - Schritt 3:
# Calibration Page = Orchestrierung + UI
#
# Diese Datei darf:
# - Kamera starten/stoppen
# - Preview versorgen
# - Testpunkt auswerten
# - Referenzbild speichern
# - Detector scharf schalten
# - Kalibrierung speichern
#
# Diese Datei darf NICHT:
# - eigene Homography rechnen
# - eigene Ring-/Winkelgeometrie rechnen
# - eigene Scorelogik erfinden
#
# Alles Geometrische läuft über:
# - vision/calibration_geometry.py
# Alles Visuelle im Preview läuft über:
# - app/widgets/calibration_preview.py

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.widgets.calibration_preview import CalibrationPreview
from vision.calibration_geometry import (
    build_pipeline_points,
    calculate_hit_from_image_point,
    compute_bull_from_manual_points,
)
from vision.camera_manager import CameraWorker
from vision.dart_detector import DartDetector


class SingleCamCalibrationCard(QFrame):
    """
    UI + Steuerung für genau eine aktive Kamera.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        self.setObjectName("SingleCamCalibrationCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setStyleSheet("""
            QFrame#SingleCamCalibrationCard {
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
        self._manual_points: List[Dict[str, int]] = []

        self.detector = DartDetector()
        self.last_frame_bgr: Optional[np.ndarray] = None

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        self.device_info_label = QLabel("Gerät: keine Kamera gewählt")
        self.device_info_label.setStyleSheet("font-size: 12px; color: #bbbbbb;")

        self.help_label = QLabel(
            "Kalibrierung mit 4 festen Punkten:\n"
            "- P1 = 20|1\n"
            "- P2 = 6|10\n"
            "- P3 = 3|19\n"
            "- P4 = 11|14\n"
            "- Bull wird automatisch berechnet\n"
            "- Rechtsklick = Präzisions-Testpunkt\n"
            "- Enter bestätigt den Testpunkt"
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

        self.reset_points_button = QPushButton("4 Punkte zurücksetzen")
        self.save_empty_board_button = QPushButton("Leeres Board speichern")
        self.arm_button = QPushButton("Einzeldart scharf")
        self.reset_detection_button = QPushButton("Detektion zurücksetzen")

        self.reset_points_button.clicked.connect(self.reset_points)
        self.save_empty_board_button.clicked.connect(self.save_empty_board)
        self.arm_button.clicked.connect(self.arm_detector)
        self.reset_detection_button.clicked.connect(self.reset_detection)

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
    # UI-Aufbau
    # ------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        layout.addWidget(self.title_label)
        layout.addWidget(self.device_info_label)
        layout.addWidget(self.help_label)
        layout.addWidget(self.preview, 1)

        row_overlay = QHBoxLayout()
        row_overlay.setSpacing(8)
        row_overlay.addWidget(QLabel("Alpha:"))
        row_overlay.addWidget(self.alpha_spin)
        row_overlay.addStretch()
        row_overlay.addWidget(self.reset_points_button)

        row_checks = QHBoxLayout()
        row_checks.setSpacing(8)
        row_checks.addWidget(self.show_numbers_check)
        row_checks.addStretch()
        row_checks.addWidget(self.show_sector_lines_check)

        row_detector = QHBoxLayout()
        row_detector.setSpacing(8)
        row_detector.addWidget(self.save_empty_board_button)
        row_detector.addWidget(self.arm_button)
        row_detector.addWidget(self.reset_detection_button)
        row_detector.addStretch()

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
        self.preview.points_changed.connect(self._handle_points_changed)
        self.preview.test_point_selected.connect(self._handle_test_point_selected)

    # ------------------------------------------------------------
    # Daten / Konfig
    # ------------------------------------------------------------

    def _default_manual_points(self) -> List[Dict[str, int]]:
        frame_width = self._get_frame_width()
        frame_height = self._get_frame_height()

        return [
            {"x_px": int(frame_width * 0.60), "y_px": int(frame_height * 0.28)},  # P1 = 20|1
            {"x_px": int(frame_width * 0.70), "y_px": int(frame_height * 0.72)},  # P2 = 6|10
            {"x_px": int(frame_width * 0.26), "y_px": int(frame_height * 0.67)},  # P3 = 3|19
            {"x_px": int(frame_width * 0.37), "y_px": int(frame_height * 0.30)},  # P4 = 11|14
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

    def reset_runtime_state(self) -> None:
        """
        Setzt nur den flüchtigen Laufzeitzustand der Kalibrierkarte zurück,
        ohne gespeicherte Kalibrierpunkte zu zerstören.
        """
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

    def set_camera_config(self, camera_config: Dict) -> None:
        self.camera_config = dict(camera_config or {})

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

        raw_points = deepcopy(calibration_config.get("points", self._default_manual_points()))
        self._manual_points = raw_points[:4]

        self.preview.set_overlay_config(self.get_calibration_config())
        self._update_point_info_label()

    def get_calibration_config(self) -> Dict:
        return {
            "frame_width": self._get_frame_width(),
            "frame_height": self._get_frame_height(),
            "overlay_alpha": float(self.alpha_spin.value()),
            "show_numbers": bool(self.show_numbers_check.isChecked()),
            "show_sector_lines": bool(self.show_sector_lines_check.isChecked()),
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

        ok = self.detector.set_reference_frame(self.last_frame_bgr, self.get_calibration_config())
        if ok:
            self.auto_result_label.setText("Auto-Dart: noch keiner")
            self.detector_status_label.setText("Detector: leeres Board gespeichert")
        else:
            self.detector_status_label.setText("Detector: Referenz konnte nicht gespeichert werden")

        self._update_debug_label()

    def arm_detector(self) -> None:
        ok = self.detector.arm()
        if ok:
            self.auto_result_label.setText("Auto-Dart: warte auf Einzeldart")
            self.detector_status_label.setText("Detector: scharf für genau einen Dart")
        else:
            self.detector_status_label.setText("Detector: zuerst leeres Board speichern")

        self._update_debug_label()

    def reset_detection(self) -> None:
        self.detector.reset_detection()
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

        print(
            "[CalibrationCard] start_worker -> "
            f"device_id={device_id}, "
            f"enabled={enabled}, "
            f"width={self.camera_config.get('width', 1280)}, "
            f"height={self.camera_config.get('height', 720)}, "
            f"fps={self.camera_config.get('fps', 30)}"
        )

        print(
            "[CalibrationCard] start_worker -> "
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
    Seite für die Kalibrierung.
    Verwendet aktuell bewusst nur die erste aktive Kamera.
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

        self.title_label = QLabel("TripleOne – Schritt 3 / Calibration Page")
        self.title_label.setStyleSheet("font-size: 28px; font-weight: bold;")

        self.info_label = QLabel(
            "Diese Seite orchestriert nur noch Kamera, Preview, Testpunkt und Detector.\n"
            "Geometrie und Overlay kommen vollständig aus calibration_geometry.py."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 13px; color: #cccccc;")

        self.selected_camera_index = 0

        self.camera_select_label = QLabel("Zu kalibrierende Kamera:")
        self.camera_select_label.setStyleSheet("font-size: 13px; color: #dddddd; font-weight: 600;")

        self.camera_select_combo = QComboBox()
        self.camera_select_combo.addItems(["Kamera 1", "Kamera 2", "Kamera 3"])
        self.camera_select_combo.currentIndexChanged.connect(self._on_selected_camera_changed)

        self.card = SingleCamCalibrationCard("Kamera – Präzisionskalibrierung")

        self.start_button = QPushButton("Livebild starten / aktualisieren")
        self.stop_button = QPushButton("Kamera stoppen")
        self.save_button = QPushButton("Kalibrierung speichern")

        self.start_button.clicked.connect(self.apply_preview)
        self.stop_button.clicked.connect(self.stop_camera)
        self.save_button.clicked.connect(self.save_settings)

        self.global_info_label = QLabel(
            "Hinweis: Hier wird immer genau eine ausgewählte Kamera kalibriert."
        )
        self.global_info_label.setWordWrap(True)
        self.global_info_label.setStyleSheet("font-size: 12px; color: #8effc9; font-weight: 700;")

        self._build_ui()
        self.camera_select_combo.setCurrentIndex(0)
        self._load_single_camera_data()

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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        layout.addWidget(self.title_label)
        layout.addWidget(self.info_label)
        select_row = QHBoxLayout()
        select_row.setSpacing(10)
        select_row.addWidget(self.camera_select_label)
        select_row.addWidget(self.camera_select_combo)
        select_row.addStretch()

        layout.addLayout(top_row)
        layout.addLayout(select_row)
        layout.addWidget(self.global_info_label)
        layout.addWidget(self.card, 1)
    # ------------------------------------------------------------
    # Kamera / Daten
    # ------------------------------------------------------------

    def _get_selected_camera_index(self) -> int:
        return int(self.selected_camera_index)

    def _load_single_camera_data(self) -> None:
        cam_list = self.camera_config.get("cameras", [])
        cal_list = self.calibration_config.get("cameras", [])

        idx = self._get_selected_camera_index()

        selected_camera_cfg = cam_list[idx] if idx < len(cam_list) else {}
        selected_cal_cfg = cal_list[idx] if idx < len(cal_list) else {}

        self.card.set_camera_config(selected_camera_cfg)
        self.card.set_calibration_config(selected_cal_cfg)

        self.card.preview.clear_test_point()
        self.card.test_result_label.setText("Testpunkt: noch keiner gesetzt")
        self.card.auto_result_label.setText("Auto-Dart: noch keiner")
        self.card.detector_status_label.setText("Detector: keine Referenz")
        self.card._update_point_info_label()
        self.card._update_debug_label()

    def _on_selected_camera_changed(self, index: int) -> None:
        self.selected_camera_index = int(index)
        print(f"[CalibrationPage] selected_camera_index = {self.selected_camera_index}")

        # Alten Zustand wirklich beenden
        self.card.reset_runtime_state()

        # Neue Kamera-/Kalibrierdaten laden
        self._load_single_camera_data()
        print(f"[CalibrationPage] loaded camera_config = {self.card.camera_config}")

        self.global_info_label.setText(
            f"Hinweis: Aktuell ausgewählt ist Kamera {self.selected_camera_index + 1}. "
            f"Bitte 'Livebild starten / aktualisieren' drücken."
        )

    def update_camera_config(self, new_camera_config: Dict) -> None:
        self.camera_config = deepcopy(new_camera_config)
        self._load_single_camera_data()

    def collect_calibration_config(self) -> Dict:
        result = deepcopy(self.calibration_config)
        cameras = result.get("cameras", [])

        idx = self._get_selected_camera_index()

        while len(cameras) <= idx:
            cameras.append({"name": f"Kamera {len(cameras) + 1}"})

        cameras[idx]["name"] = f"Kamera {idx + 1}"
        cameras[idx].update(self.card.get_calibration_config())
        result["cameras"] = cameras
        return result

    # ------------------------------------------------------------
    # Aktionen
    # ------------------------------------------------------------

    def apply_preview(self) -> None:
        self.card.start_worker()

    def stop_camera(self) -> None:
        self.card.stop_worker()

    def stop_all_cameras(self) -> None:
        self.stop_camera()

    def save_settings(self) -> None:
        self.calibration_config = self.collect_calibration_config()
        self.save_callback(self.calibration_config)

        QMessageBox.information(
            self,
            "Gespeichert",
            "Die Kalibrierung der aktiven Kamera wurde gespeichert.",
        )