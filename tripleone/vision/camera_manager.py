# vision/camera_manager.py
# Diese Datei enthält die komplette Kameralogik:
# - verfügbare Kameras suchen
# - Kamera-Thread pro Vorschau
# - Frames lesen
# - Rotation / Spiegelung anwenden
# - Bild als QImage an die Oberfläche senden
# - rohes BGR-Frame für Vision/Detektion senden
# - robustes Öffnen mit Backend-Fallback

from __future__ import annotations

import platform
import time
from typing import Optional, List, Dict

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage


def get_preferred_capture_backends() -> List[int]:
    """
    Liefert eine sinnvolle Reihenfolge von Capture-Backends.

    Windows:
    - zuerst DirectShow
    - dann Media Foundation
    - dann OpenCV Default

    Andere Systeme:
    - Standardbackend
    """
    system = platform.system().lower()

    if system == "windows":
        backends: List[int] = [cv2.CAP_DSHOW]

        # Nicht jede OpenCV-Build hat CAP_MSMF.
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)

        backends.append(cv2.CAP_ANY)
        return backends

    return [cv2.CAP_ANY]


def _backend_name(backend: int) -> str:
    if backend == cv2.CAP_DSHOW:
        return "CAP_DSHOW"
    if hasattr(cv2, "CAP_MSMF") and backend == cv2.CAP_MSMF:
        return "CAP_MSMF"
    if backend == cv2.CAP_ANY:
        return "CAP_ANY"
    return str(backend)


def _try_open_camera_once(
    device_id: int,
    backend: int,
    width: int,
    height: int,
    fps: int,
) -> Optional[cv2.VideoCapture]:
    """
    Öffnet eine Kamera testweise genau einmal mit einem Backend.
    Gibt ein geöffnetes Capture zurück oder None.
    """
    cap = None
    try:
        cap = cv2.VideoCapture(device_id, backend)

        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            return None

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        success, frame = cap.read()
        if not success or frame is None:
            cap.release()
            return None

        return cap
    except Exception:
        if cap is not None:
            cap.release()
        return None


def probe_available_cameras(max_devices: int = 10) -> List[Dict[str, int]]:
    """
    Sucht nach verfügbaren Kameras, indem Geräte-Indizes geprüft werden.
    Nutzt mehrere Backends als Fallback.
    """
    available = []
    backends = get_preferred_capture_backends()

    for index in range(max_devices):
        found = False

        for backend in backends:
            cap = _try_open_camera_once(
                device_id=index,
                backend=backend,
                width=640,
                height=480,
                fps=15,
            )
            if cap is not None:
                cap.release()
                available.append({
                    "index": index,
                    "name": f"Kamera {index}",
                })
                found = True
                break

        if not found:
            continue

    return available


class CameraWorker(QThread):
    """
    Thread für eine einzelne Kamera-Vorschau.
    Sendet QImages an die UI, damit die Oberfläche flüssig bleibt.
    """

    frame_ready = pyqtSignal(QImage)
    raw_frame_ready = pyqtSignal(object)  # sendet np.ndarray (BGR)
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
        self._active_backend: Optional[int] = None

    def stop(self) -> None:
        """Beendet den Kamera-Thread sauber."""
        self._running = False
        self.wait(2000)

    def _apply_transformations(self, frame: np.ndarray) -> np.ndarray:
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
        """
        Öffnet die Kamera robust mit Backend-Fallback.
        """
        backends = get_preferred_capture_backends()

        for backend in backends:
            cap = _try_open_camera_once(
                device_id=self.device_id,
                backend=backend,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )

            if cap is None:
                continue

            self._cap = cap
            self._active_backend = backend

            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS)) or self.fps

            self.status_changed.emit(
                f"Aktiv – Gerät {self.device_id} – "
                f"{actual_width}x{actual_height} @ {actual_fps} FPS "
                f"({_backend_name(backend)})"
            )
            return True

        self.status_changed.emit(
            f"Fehler: Kamera {self.device_id} konnte mit keinem Backend geöffnet werden."
        )
        return False

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

                # Rohes BGR-Frame für Vision / Detektion
                self.raw_frame_ready.emit(frame.copy())

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

                # Nur EINMAL emitten
                self.frame_ready.emit(qt_image)

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, target_delay - elapsed)
                time.sleep(sleep_time)
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None

            self.status_changed.emit(
                f"Gestoppt – Gerät {self.device_id}"
                + (
                    f" ({_backend_name(self._active_backend)})"
                    if self._active_backend is not None
                    else ""
                )
            )