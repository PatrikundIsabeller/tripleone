from __future__ import annotations

import math
import threading

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PointF = tuple[float, float]
BBox = tuple[int, int, int, int]

_YOLO_IMPORT_LOCK = threading.Lock()
_YOLO_CLASS = None


def _get_yolo_class():
    """
    Lädt Ultralytics YOLO genau einmal pro Prozess.

    Wichtig bei Triple One:
    Mehrere Kamera-Worker dürfen den schweren torch/ultralytics-Import
    nicht gleichzeitig starten.
    """
    global _YOLO_CLASS

    if _YOLO_CLASS is not None:
        return _YOLO_CLASS

    with _YOLO_IMPORT_LOCK:
        if _YOLO_CLASS is not None:
            return _YOLO_CLASS

        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                f"Ultralytics/PyTorch konnte nicht geladen werden: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _YOLO_CLASS = YOLO
        return _YOLO_CLASS


@dataclass(slots=True)
class DartKeypointDetectorConfig:
    model_path: str = "models/dart_tip_pose.pt"
    confidence_threshold: float = 0.04
    iou_threshold: float = 0.50
    max_detections: int = 6
    image_size: int = 960
    device: Optional[str] = None
    tip_keypoint_index: int = 0
    min_keypoint_confidence: float = 0.01
    enforce_roi: bool = True
    use_board_crop: bool = True
    board_crop_margin_px: int = 120
    require_local_change: bool = False
    local_change_patch_radius_px: int = 16
    local_change_mean_absdiff_threshold: float = 5.0

    # --------------------------------------------------------------
    # Lokale Präzisierung der von YOLO geschätzten Dartspitze
    # --------------------------------------------------------------
    tip_refinement_enabled: bool = True

    # Suchradius um den YOLO-TIP.
    tip_refinement_radius_px: int = 32

    # Mindestdifferenz zum Leerboard für einen veränderten Pixel.
    tip_refinement_diff_threshold: int = 18

    # Wie weit der korrigierte TIP maximal vom YOLO-TIP weg sein darf.
    tip_refinement_max_shift_px: float = 26.0

    # Breite des Suchkorridors entlang der Dartachse.
    tip_refinement_axis_width_px: float = 14.0

    # Mindestens so viele veränderte Pixel müssen vorhanden sein.
    tip_refinement_min_pixels: int = 6

    keep_debug_images: bool = True

    # --------------------------------------------------------------
    # Zeitliche Stabilisierung der Dartspitze
    # --------------------------------------------------------------
    temporal_tracking_enabled: bool = True

    # Wie viele letzte Frames betrachtet werden.
    temporal_window_size: int = 6

    # Mindestens so viele passende TIP-Punkte müssen im Fenster liegen.
    temporal_min_stable_points: int = 4

    # Maximale Abweichung eines neuen TIP-Punkts vom aktuellen Cluster.
    temporal_max_distance_px: float = 12.0

    # Kurze YOLO-Aussetzer dürfen den stabilen Punkt nicht sofort löschen.
    temporal_hold_missing_frames: int = 12



@dataclass(slots=True)
class DartKeypointDetection:
    detection_index: int
    tip_point: PointF
    confidence: float
    keypoint_confidence: float
    box_confidence: float
    bbox: BBox
    class_id: Optional[int] = None
    class_name: Optional[str] = None
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def image_point(self) -> PointF:
        return self.tip_point

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_index": self.detection_index,
            "tip_point": self.tip_point,
            "confidence": self.confidence,
            "keypoint_confidence": self.keypoint_confidence,
            "box_confidence": self.box_confidence,
            "bbox": self.bbox,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "debug": self.debug,
        }

@dataclass(slots=True)
class DartKeypointDetectionResult:
    detections: list[DartKeypointDetection]
    metadata: dict[str, Any] = field(default_factory=dict)
    debug_images: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def best_detection(self) -> Optional[DartKeypointDetection]:
        return self.detections[0] if self.detections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "detections": [item.to_dict() for item in self.detections],
        }

    def render_debug_overlay(self, frame: np.ndarray) -> np.ndarray:
        canvas = _ensure_bgr(frame)
        for item in self.detections:
            x, y, w, h = item.bbox
            tx = int(round(item.tip_point[0]))
            ty = int(round(item.tip_point[1]))
            if w > 0 and h > 0:
                cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 255, 255), 1)
            cv2.circle(canvas, (tx, ty), 7, (0, 0, 255), 2)
            cv2.circle(canvas, (tx, ty), 2, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"TIP conf={item.confidence:.2f}",
                (tx + 8, max(18, ty - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas


class TemporalTipTracker:
    """
    Stabilisiert die Dartspitze über mehrere aufeinanderfolgende Frames.

    Erst mehrere räumlich ähnliche KI-Erkennungen ergeben einen
    stabilen Dart-TIP.
    """

    def __init__(
        self,
        *,
        window_size: int = 6,
        min_stable_points: int = 3,
        max_distance_px: float = 18.0,
        hold_missing_frames: int = 3,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.min_stable_points = max(1, int(min_stable_points))
        self.max_distance_px = max(1.0, float(max_distance_px))
        self.hold_missing_frames = max(0, int(hold_missing_frames))

        self._points: list[PointF] = []
        self._last_stable_point: Optional[PointF] = None
        self._missing_frames: int = 0

    def reset(self) -> None:
        self._points.clear()
        self._last_stable_point = None
        self._missing_frames = 0

    @property
    def last_stable_point(self) -> Optional[PointF]:
        return self._last_stable_point

    def update(
        self,
        point: Optional[PointF],
    ) -> Optional[PointF]:

        # ----------------------------------------------------------
        # 1. In diesem Frame wurde KEIN TIP erkannt
        # ----------------------------------------------------------
        if point is None:
            self._missing_frames += 1

            # Kurze YOLO-Aussetzer überbrücken.
            #
            # Dadurch blinkt der Punkt nicht sofort,
            # wenn YOLO für wenige Frames nichts erkennt.
            if (
                self._last_stable_point is not None
                and self._missing_frames <= self.hold_missing_frames
            ):
                return self._last_stable_point

            # ------------------------------------------------------
            # Zu viele Frames ohne TIP:
            # Dart gilt als verschwunden.
            #
            # Track vollständig zurücksetzen.
            # ------------------------------------------------------
            if self._missing_frames > self.hold_missing_frames:
                self._points.clear()
                self._last_stable_point = None
                self._missing_frames = 0

            return None

        # ----------------------------------------------------------
        # 2. Ein aktueller TIP wurde erkannt
        # ----------------------------------------------------------
        self._missing_frames = 0

        clean_point = (
            float(point[0]),
            float(point[1]),
        )

        # ----------------------------------------------------------
        # 3. Falls bereits ein stabiler TIP existiert:
        #
        # Nicht sofort auf jede neue YOLO-Position springen.
        # Kleine Bewegungen werden geglättet.
        # ----------------------------------------------------------
        if self._last_stable_point is not None:
            distance = _point_distance(
                clean_point,
                self._last_stable_point,
            )

            # ------------------------------------------------------
            # Kleine Abweichung:
            # sehr sanft nachführen.
            # ------------------------------------------------------
            if distance <= self.max_distance_px:
                alpha = 0.05

                smoothed_point = (
                    (1.0 - alpha) * self._last_stable_point[0]
                    + alpha * clean_point[0],

                    (1.0 - alpha) * self._last_stable_point[1]
                    + alpha * clean_point[1],
                )

                self._last_stable_point = smoothed_point

                return self._last_stable_point

            # ------------------------------------------------------
            # Großer Sprung:
            # nicht sofort übernehmen.
            #
            # Könnte eine falsche Detection oder ein neuer Dart sein.
            # Den bestehenden Lock vorerst halten.
            # ------------------------------------------------------
            return self._last_stable_point

        # ----------------------------------------------------------
        # 4. Noch kein stabiler TIP:
        # aktuellen Punkt in zeitlichen Puffer aufnehmen.
        # ----------------------------------------------------------
        self._points.append(clean_point)

        # Nur die letzten N Punkte behalten
        if len(self._points) > self.window_size:
            self._points = self._points[-self.window_size:]

        # Noch nicht genug Beobachtungen
        if len(self._points) < self.min_stable_points:
            return None

        # ----------------------------------------------------------
        # 5. Größten räumlichen Cluster suchen
        # ----------------------------------------------------------
        best_cluster: list[PointF] = []

        for anchor in self._points:
            cluster: list[PointF] = []

            for candidate in self._points:
                distance = _point_distance(
                    anchor,
                    candidate,
                )

                if distance <= self.max_distance_px:
                    cluster.append(candidate)

            if len(cluster) > len(best_cluster):
                best_cluster = cluster

        # Noch nicht genügend zusammenpassende Punkte
        if len(best_cluster) < self.min_stable_points:
            return None

        # ----------------------------------------------------------
        # 6. Stabilen TIP aus dem Cluster berechnen
        #
        # Median ist robust gegen einzelne Ausreißer.
        # ----------------------------------------------------------
        xs = np.asarray(
            [p[0] for p in best_cluster],
            dtype=np.float32,
        )

        ys = np.asarray(
            [p[1] for p in best_cluster],
            dtype=np.float32,
        )

        stable_point = (
            float(np.median(xs)),
            float(np.median(ys)),
        )

        # ----------------------------------------------------------
        # 7. Stabilen TIP locken
        # ----------------------------------------------------------
        self._last_stable_point = stable_point

        return stable_point


class DartKeypointDetector:
    """YOLO-Pose adapter. Liefert nur Dartspitzen im Kamerabild."""

    def __init__(
        self,
        config: Optional[DartKeypointDetectorConfig] = None,
    ) -> None:
        self.config = config or DartKeypointDetectorConfig()

        self._model: Any = None

        # Zeitliche Stabilisierung
        self._tip_tracker = TemporalTipTracker(
            window_size=self.config.temporal_window_size,
            min_stable_points=self.config.temporal_min_stable_points,
            max_distance_px=self.config.temporal_max_distance_px,
            hold_missing_frames=self.config.temporal_hold_missing_frames,
        )
        
    def reset_tracking(self) -> None:
        """
        Gibt einen gelockten Dart-TIP wieder frei.

        Wird aufgerufen:
        - bei neuer Leerboard-Referenz
        - sobald VisionService erkennt, dass das Board wieder frei ist
        """
        self._tip_tracker.reset()

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        model_path = Path(self.config.model_path)

        if not model_path.exists():
            raise RuntimeError(
                f"Dart-Keypoint-Modell fehlt: {model_path.resolve()}"
            )

        YOLO = _get_yolo_class()

        try:
            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Dart-Keypoint-Modell konnte nicht geladen werden: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return self._model

    def detect(
        self,
        frame: np.ndarray,
        *,
        reference_frame: Optional[np.ndarray] = None,
        board_mask: Optional[np.ndarray] = None,
        board_polygon: Optional[
            np.ndarray
            | list[tuple[int, int]]
            | list[tuple[float, float]]
        ] = None,
    ) -> DartKeypointDetectionResult:

        # --------------------------------------------------------------
        # Eingaben prüfen
        # --------------------------------------------------------------
        _validate_frame(frame, "frame")

        if reference_frame is not None:
            _validate_frame(reference_frame, "reference_frame")

            if frame.shape[:2] != reference_frame.shape[:2]:
                raise ValueError(
                    "frame und reference_frame müssen dieselbe Größe haben."
                )

        # --------------------------------------------------------------
        # Board-ROI erzeugen
        # --------------------------------------------------------------
        roi_mask = _prepare_roi_mask(
            frame.shape[:2],
            board_mask,
            board_polygon,
        )

        model = self._ensure_model()

        # --------------------------------------------------------------
        # Standard:
        # komplettes Kamerabild
        # --------------------------------------------------------------
        inference_frame = frame
        inference_reference_frame = reference_frame
        inference_roi_mask = roi_mask

        crop_x = 0
        crop_y = 0
        crop_w = int(frame.shape[1])
        crop_h = int(frame.shape[0])

        board_crop_used = False

        # --------------------------------------------------------------
        # Board-Crop
        #
        # Nur der Bereich um das Dartboard geht an YOLO.
        # Dadurch wird der Dart im Modellinput größer dargestellt.
        # --------------------------------------------------------------
        if (
            self.config.use_board_crop
            and roi_mask is not None
        ):
            crop_rect = _compute_board_crop_rect(
                roi_mask=roi_mask,
                image_shape=frame.shape[:2],
                margin_px=self.config.board_crop_margin_px,
            )

            if crop_rect is not None:
                crop_x, crop_y, crop_w, crop_h = crop_rect

                x0 = crop_x
                y0 = crop_y
                x1 = crop_x + crop_w
                y1 = crop_y + crop_h

                inference_frame = frame[
                    y0:y1,
                    x0:x1,
                ].copy()

                inference_roi_mask = roi_mask[
                    y0:y1,
                    x0:x1,
                ].copy()

                if reference_frame is not None:
                    inference_reference_frame = reference_frame[
                        y0:y1,
                        x0:x1,
                    ].copy()

                board_crop_used = True

        # --------------------------------------------------------------
        # YOLO-Inference
        # --------------------------------------------------------------
        kwargs: dict[str, Any] = {
            "source": inference_frame,
            "conf": float(self.config.confidence_threshold),
            "iou": float(self.config.iou_threshold),
            "imgsz": int(self.config.image_size),
            "max_det": int(self.config.max_detections),
            "verbose": False,
        }

        if self.config.device is not None:
            kwargs["device"] = self.config.device

        results = model.predict(**kwargs)

        if not results:
            return DartKeypointDetectionResult(
                detections=[],
                metadata={
                    "backend": "ultralytics_pose",
                    "board_crop_used": board_crop_used,
                },
            )

        result = results[0]

        # --------------------------------------------------------------
        # Debug:
        # Hat YOLO im Crop überhaupt etwas erkannt?
        # --------------------------------------------------------------
        keypoints_obj = getattr(result, "keypoints", None)
        boxes_obj = getattr(result, "boxes", None)

        model_keypoint_count = 0
        model_box_count = 0

        if keypoints_obj is not None:
            kp_xy = getattr(keypoints_obj, "xy", None)

            if kp_xy is not None:
                try:
                    model_keypoint_count = int(kp_xy.shape[0])
                except Exception:
                    model_keypoint_count = 0

        if boxes_obj is not None:
            box_xyxy = getattr(boxes_obj, "xyxy", None)

            if box_xyxy is not None:
                try:
                    model_box_count = int(box_xyxy.shape[0])
                except Exception:
                    model_box_count = 0

        # --------------------------------------------------------------
        # Rohe YOLO-Erkennungen
        #
        # Achtung:
        # _extract() arbeitet innerhalb des Crop-Bildes.
        # --------------------------------------------------------------
        detections = self._extract(
            result,
            inference_frame,
            inference_reference_frame,
            inference_roi_mask,
        )

        

        # --------------------------------------------------------------
        # Crop-Koordinaten wieder auf das originale 1280x720-Bild
        # zurückrechnen.
        #
        # ScoreMapper und Tracker arbeiten weiterhin ausschließlich
        # mit Original-Kamerakoordinaten.
        # --------------------------------------------------------------
        if board_crop_used:
            for detection in detections:
                local_tip_x = float(detection.tip_point[0])
                local_tip_y = float(detection.tip_point[1])

                detection.tip_point = (
                    local_tip_x + float(crop_x),
                    local_tip_y + float(crop_y),
                )

                bbox_x, bbox_y, bbox_w, bbox_h = detection.bbox

                detection.bbox = (
                    int(bbox_x + crop_x),
                    int(bbox_y + crop_y),
                    int(bbox_w),
                    int(bbox_h),
                )

                detection.debug = dict(
                    detection.debug or {}
                )

                detection.debug.update(
                    {
                        "board_crop_used": True,
                        "crop_rect": (
                            crop_x,
                            crop_y,
                            crop_w,
                            crop_h,
                        ),
                        "tip_point_in_crop": (
                            local_tip_x,
                            local_tip_y,
                        ),
                        "tip_point_in_full_frame": (
                            detection.tip_point
                        ),
                    }
                )

        # --------------------------------------------------------------
        # Höchste Confidence zuerst
        # --------------------------------------------------------------
        detections.sort(
            key=lambda d: d.confidence,
            reverse=True,
        )


        # --------------------------------------------------------------
        # Zeitliche TIP-Stabilisierung
        # --------------------------------------------------------------
        if self.config.temporal_tracking_enabled:

            raw_best = None

            if detections:
                last_stable = (
                    self._tip_tracker.last_stable_point
                )

                # ------------------------------------------------------
                # Noch kein stabiler Dart:
                # beste Modell-Confidence verwenden.
                # ------------------------------------------------------
                if last_stable is None:
                    raw_best = detections[0]

                else:
                    # --------------------------------------------------
                    # Bereits gelockter Track:
                    # Detection wählen, die am nächsten zum bisherigen
                    # stabilen TIP liegt.
                    # --------------------------------------------------
                    raw_best = min(
                        detections,
                        key=lambda detection: _point_distance(
                            detection.tip_point,
                            last_stable,
                        ),
                    )

                    distance_to_last = _point_distance(
                        raw_best.tip_point,
                        last_stable,
                    )

                    # Größerer Sprung = nicht derselbe Dart
                    if distance_to_last > 35.0:
                        raw_best = None

            raw_tip: Optional[PointF] = None

            if raw_best is not None:
                raw_tip = raw_best.tip_point

            # Tracker aktualisieren
            stable_tip = self._tip_tracker.update(
                raw_tip
            )


            # ----------------------------------------------------------
            # Noch kein stabiler TIP
            # ----------------------------------------------------------
            if stable_tip is None:
                detections = []

            else:

                # ------------------------------------------------------
                # Aktueller Frame enthält eine echte YOLO-Erkennung
                # ------------------------------------------------------
                if raw_best is not None:

                    stable_detection = DartKeypointDetection(
                        detection_index=raw_best.detection_index,
                        tip_point=stable_tip,

                        confidence=float(
                            raw_best.confidence
                        ),

                        keypoint_confidence=float(
                            raw_best.keypoint_confidence
                        ),

                        box_confidence=float(
                            raw_best.box_confidence
                        ),

                        bbox=raw_best.bbox,

                        class_id=raw_best.class_id,
                        class_name=raw_best.class_name,

                        debug={
                            **dict(raw_best.debug or {}),
                            "raw_tip_point": (
                                raw_best.tip_point
                            ),
                            "stable_tip_point": (
                                stable_tip
                            ),
                            "temporal_tracking": True,
                            "held_during_missing_frame": False,
                        },
                    )

                # ------------------------------------------------------
                # YOLO hat gerade kurz keine passende Detection,
                # aber der Dart ist bereits gelockt.
                # ------------------------------------------------------
                else:

                    stable_detection = DartKeypointDetection(
                        detection_index=-1,
                        tip_point=stable_tip,

                        # Bereits gelockter Dart.
                        confidence=1.00,

                        keypoint_confidence=0.0,
                        box_confidence=0.0,

                        bbox=(0, 0, 0, 0),

                        class_id=None,
                        class_name=None,

                        debug={
                            "raw_tip_point": None,
                            "stable_tip_point": (
                                stable_tip
                            ),
                            "temporal_tracking": True,
                            "held_during_missing_frame": True,
                            "confidence_source": "locked_tip",
                        },
                    )

                # Ab jetzt nur noch einen stabilisierten TIP weitergeben
                detections = [
                    stable_detection
                ]

        # --------------------------------------------------------------
        # Ergebnis
        # --------------------------------------------------------------
        output = DartKeypointDetectionResult(
            detections=detections,
            metadata={
                "backend": "ultralytics_pose",
                "model_path": str(
                    Path(self.config.model_path)
                ),

                "accepted_count": len(detections),

                "roi_used": (
                    roi_mask is not None
                ),

                # Board-Crop-Debug
                "board_crop_used": board_crop_used,

                "crop_x": crop_x,
                "crop_y": crop_y,
                "crop_width": crop_w,
                "crop_height": crop_h,

                "full_frame_width": int(
                    frame.shape[1]
                ),
                "full_frame_height": int(
                    frame.shape[0]
                ),

                "inference_width": int(
                    inference_frame.shape[1]
                ),
                "inference_height": int(
                    inference_frame.shape[0]
                ),
            },
        )

        # --------------------------------------------------------------
        # Debug-Bilder
        # --------------------------------------------------------------
        if self.config.keep_debug_images:
            output.debug_images[
                "keypoint_overlay"
            ] = output.render_debug_overlay(
                frame
            )

            if roi_mask is not None:
                output.debug_images[
                    "keypoint_roi_mask"
                ] = roi_mask.copy()

            # Zusätzlich den tatsächlichen YOLO-Crop speichern.
            if board_crop_used:
                output.debug_images[
                    "keypoint_inference_crop"
                ] = inference_frame.copy()

        return output

    def _extract(
        self,
        result: Any,
        frame: np.ndarray,
        reference_frame: Optional[np.ndarray],
        roi_mask: Optional[np.ndarray],
    ) -> list[DartKeypointDetection]:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or getattr(keypoints, "xy", None) is None:
            return []

        xy = _to_numpy(keypoints.xy)
        if xy.ndim == 2 and xy.shape[-1] == 2:
            xy = xy[:, None, :]
        if xy.ndim != 3 or xy.shape[-1] != 2:
            return []

        kp_conf = None
        if getattr(keypoints, "conf", None) is not None:
            kp_conf = _to_numpy(keypoints.conf)

        boxes = getattr(result, "boxes", None)
        box_xyxy = None
        box_conf = None
        box_cls = None
        if boxes is not None:
            if getattr(boxes, "xyxy", None) is not None:
                box_xyxy = _to_numpy(boxes.xyxy)
            if getattr(boxes, "conf", None) is not None:
                box_conf = _to_numpy(boxes.conf).reshape(-1)
            if getattr(boxes, "cls", None) is not None:
                box_cls = _to_numpy(boxes.cls).reshape(-1)

        names = getattr(result, "names", {}) or {}
        tip_idx = max(0, int(self.config.tip_keypoint_index))
        detections: list[DartKeypointDetection] = []

        for i in range(xy.shape[0]):
            if tip_idx >= xy.shape[1]:
                continue
            x = float(xy[i, tip_idx, 0])
            y = float(xy[i, tip_idx, 1])
            if x <= 0.0 and y <= 0.0:
                continue
            if not (0.0 <= x < frame.shape[1] and 0.0 <= y < frame.shape[0]):
                continue
            # ----------------------------------------------------------
            # YOLO Bounding Box bereits hier bestimmen.
            # Wird für die lokale TIP-Richtung benötigt.
            # ----------------------------------------------------------
            bbox: BBox = (0, 0, 0, 0)

            if box_xyxy is not None and i < len(box_xyxy):
                x1_box, y1_box, x2_box, y2_box = [
                    float(v)
                    for v in box_xyxy[i][:4]
                ]

                bbox = (
                    int(round(x1_box)),
                    int(round(y1_box)),
                    max(
                        0,
                        int(round(x2_box - x1_box)),
                    ),
                    max(
                        0,
                        int(round(y2_box - y1_box)),
                    ),
                )

            # ----------------------------------------------------------
            # Lokale TIP-Verfeinerung
            # ----------------------------------------------------------
            raw_yolo_tip = (
                float(x),
                float(y),
            )

            if (
                self.config.tip_refinement_enabled
                and reference_frame is not None
            ):
                refined_tip = _refine_tip_from_reference(
                    frame=frame,
                    reference_frame=reference_frame,
                    predicted_tip=raw_yolo_tip,
                    bbox=bbox,
                    radius_px=self.config.tip_refinement_radius_px,
                    diff_threshold=self.config.tip_refinement_diff_threshold,
                    max_shift_px=self.config.tip_refinement_max_shift_px,
                    axis_width_px=self.config.tip_refinement_axis_width_px,
                    min_pixels=self.config.tip_refinement_min_pixels,
                )

                x = float(
                    refined_tip[0]
                )

                y = float(
                    refined_tip[1]
                )

            if self.config.enforce_roi and roi_mask is not None:
                if not _inside_mask((x, y), roi_mask):
                    continue

            kpc = 0.0
            if kp_conf is not None and kp_conf.size > 0:
                try:
                    kpc = float(kp_conf[i] if kp_conf.ndim == 1 else kp_conf[i, tip_idx])
                except Exception:
                    kpc = 0.0
            if kpc > 0.0 and kpc < float(self.config.min_keypoint_confidence):
                continue

            bc = 0.0
            if box_conf is not None and i < len(box_conf):
                bc = float(box_conf[i])
            confidence = kpc if kpc > 0.0 else bc
            if confidence <= 0.0:
                confidence = float(self.config.confidence_threshold)

            change_score = None
            if reference_frame is not None and self.config.require_local_change:
                change_score = _local_change_score(
                    current_frame=frame,
                    reference_frame=reference_frame,
                    point=(x, y),
                    radius=max(2, int(self.config.local_change_patch_radius_px)),
                )
                if change_score < float(self.config.local_change_mean_absdiff_threshold):
                    continue

            class_id = None
            class_name = None
            if box_cls is not None and i < len(box_cls):
                class_id = int(round(float(box_cls[i])))
                if isinstance(names, dict):
                    class_name = str(names.get(class_id, class_id))
                elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                    class_name = str(names[class_id])

            detections.append(
                DartKeypointDetection(
                    detection_index=i,
                    tip_point=(x, y),
                    confidence=float(max(0.0, min(1.0, confidence))),
                    keypoint_confidence=float(max(0.0, kpc)),
                    box_confidence=float(max(0.0, bc)),
                    bbox=bbox,
                    class_id=class_id,
                    class_name=class_name,
                    debug={
                        "local_change_score": change_score,
                        "raw_yolo_tip": raw_yolo_tip,
                        "refined_tip": (
                            float(x),
                            float(y),
                        ),
                        "tip_refinement_enabled": bool(
                            self.config.tip_refinement_enabled
                        ),
                    },
                )
            )

        return detections


def _validate_frame(frame: np.ndarray, name: str) -> None:
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError(f"{name} ist ungültig.")


def _ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return frame.copy()
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"Nicht unterstützte Bildform: {frame.shape}")


def _to_numpy(value: Any) -> np.ndarray:
    cur = value
    for method_name in ("detach", "cpu"):
        method = getattr(cur, method_name, None)
        if callable(method):
            cur = method()
    method = getattr(cur, "numpy", None)
    if callable(method):
        return np.asarray(method())
    return np.asarray(cur)


def _prepare_roi_mask(
    image_shape: tuple[int, int],
    board_mask: Optional[np.ndarray],
    board_polygon: Optional[np.ndarray | list[tuple[int, int]] | list[tuple[float, float]]],
) -> Optional[np.ndarray]:
    if board_mask is not None and board_polygon is not None:
        raise ValueError("Nur board_mask oder board_polygon übergeben.")
    h, w = image_shape
    if board_mask is not None:
        if board_mask.shape[:2] != (h, w):
            raise ValueError("board_mask hat falsche Größe.")
        mask = board_mask
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return np.where(mask > 0, 255, 0).astype(np.uint8)
    if board_polygon is not None:
        pts = np.asarray(board_polygon, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            raise ValueError("board_polygon muss Form (N,2) haben.")
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
        return mask
    return None


def _inside_mask(point: PointF, mask: np.ndarray) -> bool:
    x = int(round(point[0]))
    y = int(round(point[1]))
    return 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and bool(mask[y, x] > 0)


def _local_change_score(
    *,
    current_frame: np.ndarray,
    reference_frame: np.ndarray,
    point: PointF,
    radius: int,
) -> float:
    cur = _ensure_bgr(current_frame)
    ref = _ensure_bgr(reference_frame)
    x = int(round(point[0]))
    y = int(round(point[1]))
    r = max(2, int(radius))
    x0, x1 = max(0, x-r), min(cur.shape[1], x+r+1)
    y0, y1 = max(0, y-r), min(cur.shape[0], y+r+1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    cg = cv2.cvtColor(cur[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(ref[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.absdiff(cg, rg)))

def _point_distance(
    a: PointF,
    b: PointF,
) -> float:
    """
    Euklidischer Abstand zwischen zwei Bildpunkten.
    """
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])

    return float(
        math.sqrt(
            dx * dx + dy * dy
        )
    )

def _refine_tip_from_reference(
    *,
    frame: np.ndarray,
    reference_frame: Optional[np.ndarray],
    predicted_tip: PointF,
    bbox: BBox,
    radius_px: int,
    diff_threshold: int,
    max_shift_px: float,
    axis_width_px: float,
    min_pixels: int,
) -> PointF:
    """
    Verfeinert einen von YOLO geschätzten Dart-TIP anhand der
    Bilddifferenz zum leeren Board.

    YOLO liefert nur den ungefähren TIP.
    Die lokale Bilddifferenz sucht anschließend den tatsächlichen
    äußersten Dartpunkt entlang der Dartachse.

    Falls keine sichere Verfeinerung möglich ist, wird unverändert
    predicted_tip zurückgegeben.
    """

    if reference_frame is None:
        return predicted_tip

    if (
        frame is None
        or reference_frame is None
        or frame.shape[:2] != reference_frame.shape[:2]
    ):
        return predicted_tip

    height, width = frame.shape[:2]

    tip_x = float(predicted_tip[0])
    tip_y = float(predicted_tip[1])

    radius = max(
        6,
        int(radius_px),
    )

    # ----------------------------------------------------------
    # Lokale ROI um den YOLO-TIP
    # ----------------------------------------------------------
    x1 = max(
        0,
        int(round(tip_x)) - radius,
    )

    y1 = max(
        0,
        int(round(tip_y)) - radius,
    )

    x2 = min(
        width,
        int(round(tip_x)) + radius + 1,
    )

    y2 = min(
        height,
        int(round(tip_y)) + radius + 1,
    )

    if x2 <= x1 or y2 <= y1:
        return predicted_tip

    current_crop = frame[
        y1:y2,
        x1:x2,
    ]

    reference_crop = reference_frame[
        y1:y2,
        x1:x2,
    ]

    if (
        current_crop.size == 0
        or reference_crop.size == 0
    ):
        return predicted_tip

    # ----------------------------------------------------------
    # Graubilder erzeugen
    # ----------------------------------------------------------
    if current_crop.ndim == 3:
        current_gray = cv2.cvtColor(
            current_crop,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        current_gray = current_crop

    if reference_crop.ndim == 3:
        reference_gray = cv2.cvtColor(
            reference_crop,
            cv2.COLOR_BGR2GRAY,
        )
    else:
        reference_gray = reference_crop

    # ----------------------------------------------------------
    # Differenz zum leeren Board
    # ----------------------------------------------------------
    diff = cv2.absdiff(
        current_gray,
        reference_gray,
    )

    diff = cv2.GaussianBlur(
        diff,
        (3, 3),
        0,
    )

    _, changed_mask = cv2.threshold(
        diff,
        int(diff_threshold),
        255,
        cv2.THRESH_BINARY,
    )

    # Kleine Einzelpixel entfernen.
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )

    changed_mask = cv2.morphologyEx(
        changed_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    ys, xs = np.where(
        changed_mask > 0
    )

    if len(xs) < int(min_pixels):
        return predicted_tip

    # Lokale Pixel -> Bildkoordinaten
    candidate_x = (
        xs.astype(np.float32)
        + float(x1)
    )

    candidate_y = (
        ys.astype(np.float32)
        + float(y1)
    )

    # ----------------------------------------------------------
    # Dartachse bestimmen.
    #
    # Richtung:
    # Bounding-Box-Zentrum -> YOLO-TIP
    #
    # Damit funktioniert das unabhängig davon, von welcher Seite
    # die Kamera auf das Board schaut.
    # ----------------------------------------------------------
    bx, by, bw, bh = bbox

    if bw > 0 and bh > 0:
        center_x = float(
            bx + bw / 2.0
        )

        center_y = float(
            by + bh / 2.0
        )

    else:
        # Ohne brauchbare Bounding Box keine sichere Richtung.
        return predicted_tip

    direction_x = (
        tip_x - center_x
    )

    direction_y = (
        tip_y - center_y
    )

    direction_length = math.hypot(
        direction_x,
        direction_y,
    )

    if direction_length < 2.0:
        return predicted_tip

    direction_x /= direction_length
    direction_y /= direction_length

    # Senkrechte Achse zur Dartachse
    normal_x = -direction_y
    normal_y = direction_x

    # ----------------------------------------------------------
    # Kandidaten relativ zum ursprünglichen YOLO-TIP
    # ----------------------------------------------------------
    rel_x = candidate_x - tip_x
    rel_y = candidate_y - tip_y

    # Position entlang der Dartachse.
    axial = (
        rel_x * direction_x
        + rel_y * direction_y
    )

    # Abstand seitlich zur Dartachse.
    perpendicular = np.abs(
        rel_x * normal_x
        + rel_y * normal_y
    )

    # Euklidischer Abstand zum ursprünglichen TIP.
    distance = np.sqrt(
        rel_x * rel_x
        + rel_y * rel_y
    )

    valid = (
        (distance <= float(max_shift_px))
        & (
            perpendicular
            <= float(axis_width_px)
        )
    )

    if not np.any(valid):
        return predicted_tip

    valid_indices = np.where(
        valid
    )[0]

    # ----------------------------------------------------------
    # Äußersten veränderten Punkt in Dart-Richtung suchen.
    #
    # Größerer axial-Wert =
    # weiter vom Dartkörper Richtung Spitze.
    # ----------------------------------------------------------
    best_index = valid_indices[
        int(
            np.argmax(
                axial[valid_indices]
            )
        )
    ]

    refined_x = float(
        candidate_x[best_index]
    )

    refined_y = float(
        candidate_y[best_index]
    )

    # Sicherheitsprüfung
    shift = math.hypot(
        refined_x - tip_x,
        refined_y - tip_y,
    )

    if shift > float(max_shift_px):
        return predicted_tip

    return (
        refined_x,
        refined_y,
    )

def _compute_board_crop_rect(
    roi_mask: Optional[np.ndarray],
    image_shape: tuple[int, int],
    margin_px: int,
) -> Optional[tuple[int, int, int, int]]:
    """
    Ermittelt aus der Board-ROI ein rechteckiges Crop-Fenster.

    Rückgabe:
        (x, y, width, height)

    Der zusätzliche Rand verhindert, dass Darts am äußeren
    Double-Ring zu knapp abgeschnitten werden.
    """
    if roi_mask is None:
        return None

    nonzero = cv2.findNonZero(roi_mask)

    if nonzero is None:
        return None

    x, y, w, h = cv2.boundingRect(nonzero)

    image_h, image_w = image_shape
    margin = max(0, int(margin_px))

    x0 = max(0, x - margin)
    y0 = max(0, y - margin)

    x1 = min(image_w, x + w + margin)
    y1 = min(image_h, y + h + margin)

    crop_w = x1 - x0
    crop_h = y1 - y0

    if crop_w <= 0 or crop_h <= 0:
        return None

    return (
        int(x0),
        int(y0),
        int(crop_w),
        int(crop_h),
    )

__all__ = [
    "PointF",
    "BBox",
    "DartKeypointDetectorConfig",
    "DartKeypointDetection",
    "DartKeypointDetectionResult",
    "DartKeypointDetector",
]
