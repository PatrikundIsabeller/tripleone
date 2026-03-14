# main.py
# Einstiegspunkt der Anwendung.
# Diese Datei startet die PyQt6-App und öffnet das Hauptfenster.

import sys
from PyQt6.QtWidgets import QApplication
from app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("TripleOne - Phase 1")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 
