# vision/dart_detector.py
# Kompatibilitäts-Bridge für die bestehende CalibrationPage.
#
# Der alte Detector liegt unter vision/_legacy/dart_detector.py.
# Der aktive Livepfad läuft über vision.single_cam_detector.

from vision._legacy.dart_detector import (
    DartDetector,
    DartDetectionResult,
    DartDebugSnapshot,
)

__all__ = [
    "DartDetector",
    "DartDetectionResult",
    "DartDebugSnapshot",
]