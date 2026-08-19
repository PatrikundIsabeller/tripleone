from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))p

import argparse
import cv2

from vision.dart_keypoint_detector import DartKeypointDetector, DartKeypointDetectorConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/dart_tip_pose.pt")
    parser.add_argument("--conf", type=float, default=0.04)
    args = parser.parse_args()

    detector = DartKeypointDetector(
        DartKeypointDetectorConfig(
            model_path=args.model,
            confidence_threshold=args.conf,
            require_local_change=False,
        )
    )

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Kamera {args.camera} konnte nicht geöffnet werden.")
        return 2

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            try:
                result = detector.detect(frame)
                overlay = result.render_debug_overlay(frame)
            except Exception as exc:
                overlay = frame.copy()
                cv2.putText(overlay, str(exc), (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Triple One - Keypoint Test", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
