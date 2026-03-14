# app/pages/dashboard_page.py
# Diese Seite zeigt den Grundstatus der App:
# - Titel
# - kurze Beschreibung
# - Anzahl verfügbarer Kameras
# - Button zum Aktualisieren der Kameraerkennung

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFrame


class DashboardPage(QWidget):
    """
    Einfache Startseite der Anwendung.
    Diese Seite zeigt Basisinformationen und den erkannten Kamerastatus.
    """

    def __init__(self, refresh_callback, parent=None):
        super().__init__(parent)
        self.refresh_callback = refresh_callback

        self.title_label = QLabel("TripleOne – Dashboard")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.title_label.setStyleSheet("font-size: 26px; font-weight: bold;")

        self.info_label = QLabel(
            "Phase 1 Grundstruktur:\n"
            "- Kameraverwaltung\n"
            "- Live-Vorschau\n"
            "- lokale Konfiguration\n"
            "- robuster Auto-Start\n\n"
            "Kalibrierung und Dart-Erkennung folgen in Phase 2."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 14px;")

        self.camera_status_label = QLabel("Verfügbare Kameras: wird geladen ...")
        self.camera_status_label.setStyleSheet("font-size: 15px; font-weight: 600;")

        self.refresh_button = QPushButton("Kameras neu erkennen")
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.clicked.connect(self.refresh_callback)

        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1f1f1f;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 12px;
            }
        """)

        card_layout = QVBoxLayout(card)
        card_layout.addWidget(self.info_label)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.camera_status_label)
        card_layout.addSpacing(10)
        card_layout.addWidget(self.refresh_button)
        card_layout.addStretch()

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        main_layout.addWidget(self.title_label)
        main_layout.addWidget(card)
        main_layout.addStretch()

    def update_camera_count(self, count: int) -> None:
        """Aktualisiert die Anzeige der aktuell erkannten Kameras."""
        self.camera_status_label.setText(f"Verfügbare Kameras: {count}")
