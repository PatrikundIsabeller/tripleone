# app/main_window.py
# Diese Datei enthält das Hauptfenster der Anwendung:
# - linke Navigation
# - Seitenverwaltung
# - Laden/Speichern der Konfiguration
# - Kameraerkennung
# - Weitergabe der Daten an Dashboard, Kameraseite und Kalibrierungsseite

from __future__ import annotations

from copy import deepcopy

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QLabel
)

from config.settings import load_config, save_config
from config.calibration_settings import load_calibration, save_calibration
from vision.camera_manager import probe_available_cameras
from app.pages.dashboard_page import DashboardPage
from app.pages.cameras_page import CamerasPage
from app.pages.calibration_page import CalibrationPage
from vision.single_cam_detector import SingleCamDetector
from vision.score_mapper import build_score_mapper
from vision.calibration_storage import CalibrationStorage


class MainWindow(QMainWindow):
    """
    Hauptfenster der Anwendung.
    Verwaltet Navigation, Seiten und zentrale Konfiguration.
    """

    PAGE_DASHBOARD = 0
    PAGE_CAMERAS = 1
    PAGE_CALIBRATION = 2

    def __init__(self):
        super().__init__()

        self.config_data = load_config()
        self.calibration_data = load_calibration()
        self.available_cameras = []

        # ------------------------------------------------------------
        # Zentrale persistente Kamera-/Kalibrierungsquelle
        # ------------------------------------------------------------
        self.calibration_storage = CalibrationStorage()

        stored_records = self.calibration_storage.load_all_records()

        if stored_records:
            (
                self.config_data,
                self.calibration_data,
            ) = self.calibration_storage.apply_records_to_app_configs(
                records=stored_records,
                base_camera_config=self.config_data,
                base_calibration_config=self.calibration_data,
            )

            print(
                "[MainWindow] calibration_store übernommen:",
                [
                    (
                        record.camera_index,
                        record.device_id,
                        record.enabled,
                    )
                    for record in stored_records
                ],
            )

        # Detectoren ERST NACH dem Laden des zentralen Stores bauen
        self.camera_detectors = [None, None, None]
        self._rebuild_camera_detectors()

        self.setWindowTitle(self.config_data.get("app", {}).get("title", "TripleOne"))
        self.resize(1600, 900)

        self.sidebar = QListWidget()
        self.sidebar.setFixedWidth(220)
        self.sidebar.setStyleSheet("""
            QListWidget {
                background-color: #161616;
                color: #f2f2f2;
                border: none;
                padding: 8px;
            }
            QListWidget::item {
                padding: 14px 12px;
                border-radius: 8px;
                margin-bottom: 6px;
            }
            QListWidget::item:selected {
                background-color: #2d6cdf;
                color: white;
            }
        """)

        self.header_label = QLabel("TripleOne")
        self.header_label.setStyleSheet("font-size: 24px; font-weight: bold; color: white;")

        self.dashboard_page = DashboardPage(refresh_callback=self.refresh_available_cameras)

        self.cameras_page = CamerasPage(
            config_data=deepcopy(self.config_data),
            save_callback=self.handle_save_config,
            refresh_cameras_callback=self.refresh_available_cameras,
            detectors=self.camera_detectors,
        )

        self.calibration_page = CalibrationPage(
            camera_config=deepcopy(self.config_data),
            calibration_config=deepcopy(self.calibration_data),
            save_callback=self.handle_save_calibration
        )

        self.stack = QStackedWidget()
        self.stack.addWidget(self.dashboard_page)
        self.stack.addWidget(self.cameras_page)
        self.stack.addWidget(self.calibration_page)

        self._build_ui()
        self._apply_dark_style()
        self._populate_sidebar()

        self.sidebar.currentRowChanged.connect(self.change_page)
        self.sidebar.setCurrentRow(self.PAGE_DASHBOARD)

        self.refresh_available_cameras()

    def _rebuild_camera_detectors(self) -> None:
        """
        Baut pro Kamera einen SingleCamDetector aus:
        - config_data
        - calibration_data

        Wenn für eine Kamera noch keine saubere Kalibrierung vorhanden ist,
        bleibt der Detector für diese Kamera None.
        """
        detectors = []

        records = self.calibration_storage.build_records_from_app_configs(
            camera_config=self.config_data,
            calibration_config=self.calibration_data,
        )

        for idx in range(3):
            detector = None

            if idx < len(records):
                record = records[idx]

                try:
                    if record.manual_points and len(record.manual_points) >= 4:
                        score_mapper = build_score_mapper(calibration_record=record)
                        detector = SingleCamDetector(score_mapper=score_mapper)
                except Exception as exc:
                    print(f"[WARN] Detector für Kamera {idx} konnte nicht gebaut werden: {exc}")
                    detector = None

            detectors.append(detector)

        while len(detectors) < 3:
            detectors.append(None)

        self.camera_detectors = detectors[:3]

    def _build_ui(self) -> None:
        """Erstellt das Hauptlayout des Fensters."""
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(14, 14, 14, 14)
        sidebar_layout.setSpacing(16)
        sidebar_layout.addWidget(self.header_label)
        sidebar_layout.addWidget(self.sidebar)
        sidebar_layout.addStretch()

        sidebar_widget = QWidget()
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_widget.setFixedWidth(240)
        sidebar_widget.setStyleSheet("background-color: #111111;")

        central_layout = QHBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(sidebar_widget)
        central_layout.addWidget(self.stack)

        central_widget = QWidget()
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)

    def _populate_sidebar(self) -> None:
        """Füllt die linke Navigation mit den verfügbaren Seiten."""
        self.sidebar.clear()

        for page_name in ["Dashboard", "Kameras", "Kalibrierung"]:
            item = QListWidgetItem(page_name)
            self.sidebar.addItem(item)

    def _apply_dark_style(self) -> None:
        """Setzt das Grunddesign der App auf ein dunkles Layout."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #181818;
                color: #f2f2f2;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 14px;
            }
        """)

    def _stop_all_page_cameras(self) -> None:
        """Stoppt alle Kamera-Threads auf allen Seiten, die Kameras verwenden."""
        self.cameras_page.stop_all_cameras()
        self.calibration_page.stop_all_cameras()

    def _start_current_page_if_needed(self) -> None:
        """
        Kameras werden aktuell NICHT automatisch gestartet.

        Start ausschließlich manuell über:
        'Vorschau starten / aktualisieren'.

        Dadurch verhindern wir versteckte Mehrfachstarts beim
        Seitenwechsel oder Kamera-Refresh.
        """
        return

    def change_page(self, index: int) -> None:
        """Wechselt die sichtbare Seite und startet/stopppt Kameras passend dazu."""
        if 0 <= index < self.stack.count():
            self._stop_all_page_cameras()
            self.stack.setCurrentIndex(index)
            self._start_current_page_if_needed()

    def refresh_available_cameras(self) -> None:
        """
        Sucht alle verfügbaren Kameras neu und synchronisiert
        Dashboard, Kameraseite und Kalibrierungsseite mit derselben
        zentralen Kamera-Konfiguration.
        """

        max_scan = self.config_data.get(
            "app",
            {},
        ).get(
            "max_camera_scan",
            10,
        )

        # ------------------------------------------------------------
        # 1. Physische Kameras neu erkennen
        # ------------------------------------------------------------
        self.available_cameras = probe_available_cameras(
            max_devices=max_scan
        )

        print(
            "[MainWindow] available_cameras =",
            self.available_cameras,
        )

        # ------------------------------------------------------------
        # 2. Dashboard aktualisieren
        # ------------------------------------------------------------
        self.dashboard_page.update_camera_count(
            len(self.available_cameras)
        )

        # ------------------------------------------------------------
        # 3. BEIDE Seiten zuerst mit derselben zentralen Config
        #    synchronisieren
        # ------------------------------------------------------------
        self.cameras_page.update_camera_config(
            deepcopy(self.config_data)
        )

        self.calibration_page.update_camera_config(
            deepcopy(self.config_data)
        )

        # ------------------------------------------------------------
        # 4. Danach verfügbare physische Kameras in die Comboboxen laden
        #
        # Jetzt kennen die Seiten bereits die gespeicherten device_ids
        # 0 / 1 / 2 und können die richtige Auswahl setzen.
        # ------------------------------------------------------------
        self.cameras_page.set_available_cameras(
            self.available_cameras
        )

        self.calibration_page.set_available_cameras(
            self.available_cameras
        )

        # ------------------------------------------------------------
        # 5. Alte Kamera-Worker sicher stoppen
        # ------------------------------------------------------------
        self._stop_all_page_cameras()

        # ------------------------------------------------------------
        # 6. Nur die aktuell sichtbare Seite ggf. automatisch starten
        # ------------------------------------------------------------
        self._start_current_page_if_needed()
    
    def handle_save_config(self, new_config: dict) -> None:
        """Speichert die Phase-1-Kamerakonfiguration zentral."""
        self.config_data = deepcopy(new_config)
        save_config(self.config_data)

        # Kalibrierungsseite muss die neuen Kameradaten direkt übernehmen
        self.calibration_page.update_camera_config(self.config_data)

        # Detectoren neu bauen und an Kameraseite weiterreichen
        self._rebuild_camera_detectors()
        self.cameras_page.set_detectors(self.camera_detectors)

    def handle_save_calibration(self, new_calibration: dict) -> None:
        """Speichert die Kalibrierungsdaten zentral."""
        self.calibration_data = deepcopy(new_calibration)
        save_calibration(self.calibration_data)

        # Detectoren mit neuer Kalibrierung neu bauen
        self._rebuild_camera_detectors()
        self.cameras_page.set_detectors(self.camera_detectors)

    def closeEvent(self, event) -> None:
        """
        Beendet beim Schließen alle Kamera- und Vision-Threads sauber.
        """

        # Kameraseite:
        # CameraWorker + VisionInferenceWorker/QThread vollständig beenden.
        self.cameras_page.shutdown()

        # Kalibrierungsseite besitzt ebenfalls Kamera-Worker.
        self.calibration_page.stop_all_cameras()

        super().closeEvent(event)