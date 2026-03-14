# vision/camera_manager.py
# Diese Datei enthält die komplette Kameralogik:
# - verfügbare Kameras suchen
# - Kamera-Thread pro Vorschau
# - Frames lesen
# - Rotation / Spiegelung anwenden
# - Bild als QImage an die Oberfläche senden

from __future__ import annotations

import platform
import time
from typing import Optional, List, Dict

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


def get_capture_backend():
    """
    Wählt ein sinnvolles OpenCV-Capture-Backend aus.
    Auf Windows verwenden wir CAP_DSHOW, sonst Standard.
    """
    if platform.system().lower() == "windows":
        return cv2.CAP_DSHOW
    return cv2.CAP_ANY


def probe_available_cameras(max_devices: int = 10) -> List[Dict[str, int]]:
    """
    Sucht nach verfügbaren Kameras, indem die Geräte-Indizes geprüft werden.
    """
    available = []
    backend = get_capture_backend()

    for index in range(max_devices):
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                continue

            success, frame = cap.read()
            if success and frame is not None:
                available.append({
                    "index": index,
                    "name": f"Kamera {index}"
                })
        finally:
            if cap is not None:
                cap.release()

    return available


class CameraWorker(QThread):
    """
    Thread für eine einzelne Kamera-Vorschau.
    Sendet QImages an die UI, damit die Oberfläche flüssig bleibt.
    """

    frame_ready = pyqtSignal(QImage)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        device_id: int,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        rotation: int = 0,
        flip: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.rotation = rotation
        self.flip = flip

        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None

    def stop(self) -> None:
        """Beendet den Kamera-Thread sauber."""
        self._running = False
        self.wait(2000)

    def _apply_transformations(self, frame):
        """Wendet Rotation und Spiegelung auf das Kamerabild an."""
        if self.rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif self.rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.flip:
            frame = cv2.flip(frame, 1)

        return frame

    def _open_camera(self) -> bool:
        """Öffnet die Kamera mit den gewünschten Parametern."""
        backend = get_capture_backend()
        self._cap = cv2.VideoCapture(self.device_id, backend)

        if not self._cap.isOpened():
            self.status_changed.emit(
                f"Fehler: Kamera {self.device_id} konnte nicht geöffnet werden."
            )
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or self.fps

        self.status_changed.emit(
            f"Aktiv – Gerät {self.device_id} – {actual_width}x{actual_height} @ {actual_fps} FPS"
        )
        return True

    def run(self) -> None:
        """Startet die Frame-Schleife der Kamera."""
        if self.device_id < 0:
            self.status_changed.emit("Keine Kamera ausgewählt.")
            return

        if not self._open_camera():
            return

        self._running = True
        target_delay = 1.0 / max(1, self.fps)

        try:
            while self._running and self._cap is not None:
                loop_start = time.time()

                success, frame = self._cap.read()
                if not success or frame is None:
                    self.status_changed.emit(
                        f"Warnung: Kein Frame von Kamera {self.device_id}."
                    )
                    time.sleep(0.05)
                    continue

                frame = self._apply_transformations(frame)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w

                qt_image = QImage(
                    rgb_frame.data,
                    w,
                    h,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                ).copy()

                self.frame_ready.emit(qt_image)

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, target_delay - elapsed)
                time.sleep(sleep_time)
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            self.status_changed.emit(f"Gestoppt – Gerät {self.device_id}")