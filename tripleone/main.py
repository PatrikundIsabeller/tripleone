# main.py

from __future__ import annotations

import sys


# --------------------------------------------------------------
# WICHTIG WINDOWS / PYQT / PYTORCH
# --------------------------------------------------------------
# PyTorch MUSS vor PyQt geladen werden.
#
# Bei neueren PyTorch-Versionen kann unter Windows sonst beim
# späteren Import innerhalb der PyQt-App folgendes auftreten:
#
# WinError 1114
# Error loading ... torch\lib\c10.dll
#
# Deshalb laden wir torch + Ultralytics bewusst VOR QApplication
# und VOR app.main_window.
# --------------------------------------------------------------
try:
    import torch

    print(
        f"[BOOT] PyTorch geladen: {torch.__version__} | "
        f"CUDA: {torch.cuda.is_available()}"
    )

except Exception as exc:
    print(
        f"[BOOT] FEHLER beim Laden von PyTorch: "
        f"{type(exc).__name__}: {exc}"
    )
    raise


try:
    from ultralytics import YOLO

    print("[BOOT] Ultralytics YOLO geladen")

except Exception as exc:
    print(
        f"[BOOT] FEHLER beim Laden von Ultralytics: "
        f"{type(exc).__name__}: {exc}"
    )
    raise


# --------------------------------------------------------------
# ERST JETZT PyQt / TripleOne laden
# --------------------------------------------------------------
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()