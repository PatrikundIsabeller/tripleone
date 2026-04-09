# vision/vision_service.py
# Zweck:
# Diese Datei kapselt den echten Produktfluss für die Single-Camera-Erkennung.
#
# Verantwortlichkeiten:
# - Referenzframe pro Kamera speichern
# - Erkennungszustand pro Kamera verwalten
# - Live-Frames gegen Referenz prüfen
# - Treffer nur einmal pro Boardzustand auslösen
# - Cooldown und "Board wieder frei" behandeln
# - strukturierte Status-/Event-Rückgaben für UI oder Spielengine liefern
#
# WICHTIG:
# Diese Datei enthält bewusst KEINE:
# - eigene Homography
# - eigene Ring-/Sektorlogik
# - eigene Dartgeometrie
# - eigene Scoreberechnung
#
# Das alles bleibt in:
# - vision.single_cam_detector
# - vision.impact_estimator
# - vision.score_mapper

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

try:
    from .single_cam_detector import SingleCamDetector
except ImportError:  # pragma: no cover
    from vision.single_cam_detector import SingleCamDetector  # type: ignore


# -----------------------------------------------------------------------------
# Typen / Konstanten
# -----------------------------------------------------------------------------
PointF = tuple[float, float]

STATUS_BOARD_NOT_REFERENCED = "board_not_referenced"
STATUS_READY = "ready"
STATUS_NO_HIT = "no_hit"
STATUS_HIT_DETECTED = "hit_detected"
STATUS_WAITING_FOR_CLEAR = "waiting_for_board_clear"
STATUS_COOLDOWN = "cooldown"
STATUS_DISARMED = "disarmed"
STATUS_ERROR = "error"


# -----------------------------------------------------------------------------
# Konfiguration
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class VisionServiceConfig:
    """
    Produktlogik-Konfiguration für den VisionService.
    """

    # Soll nach dem Speichern der Referenz automatisch "armed" werden?
    auto_arm_on_reference_save: bool = True

    # Soll nach einem Treffer auf "Board wieder frei" gewartet werden?
    require_board_clear_after_hit: bool = True

    # Mindestabstand zwischen zwei bestätigten Treffern derselben Kamera
    min_seconds_between_hits: float = 0.80

    # ------------------------------------------------------------------
    # NEU: Confirming-Phase für echte Live-Treffer
    # ------------------------------------------------------------------
    # Ein Hit wird erst ausgelöst, wenn mehrere aufeinanderfolgende Frames
    # einen zusammenpassenden Treffer liefern.
    confirm_hit_required_consecutive_frames: int = 2

    # Mindest-Combined-Confidence eines Kandidaten, bevor er überhaupt
    # in die Confirming-Phase darf.
    min_hit_confidence: float = 0.05

    # Wie ähnlich müssen zwei aufeinanderfolgende Kandidaten sein,
    # damit sie als derselbe Wurf gelten?
    confirm_same_label_required: bool = False
    confirm_max_topdown_distance_px: float = 24.0
    confirm_max_image_distance_px: float = 40.0

    # Nach dieser Zeit ohne passenden Folgeframe wird ein Pending-Hit verworfen.
    pending_hit_max_age_seconds: float = 0.75

    # Für die Freigabe des Boards:
    # Frame wird mit Referenz verglichen, Differenzbild geschwellt, Anteil
    # veränderter Pixel im ROI wird gemessen.
    clear_board_diff_threshold: int = 18
    clear_board_changed_ratio_threshold: float = 0.0045

    # Kleine Glättung für die Board-clear-Prüfung
    clear_board_blur_kernel_size: int = 5

    # Optionaler Sicherheitszähler:
    # Erst nach N aufeinanderfolgenden "board clear"-Frames gilt das Board
    # wieder als frei.
    clear_board_required_consecutive_frames: int = 2

    # Soll die Board-Maske für die Board-clear-Prüfung verwendet werden?
    use_board_mask_for_clear_check: bool = True

    # Debugbilder aus Detector-Ergebnissen in der Rückgabe erhalten
    keep_debug_images: bool = True

    # Falls kein Timestamp übergeben wird, time.monotonic() verwenden
    use_monotonic_time_when_missing: bool = True


# -----------------------------------------------------------------------------
# Events / Rückgabeobjekte
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class VisionHitEvent:
    """
    Treffer-Event für UI / Spielengine.
    """

    camera_id: int
    timestamp: float
    label: str
    score: int
    ring: str
    segment: Optional[int]
    multiplier: int
    image_point: Optional[PointF]
    topdown_point: Optional[PointF]
    confidence: Optional[float]
    detection_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "label": self.label,
            "score": self.score,
            "ring": self.ring,
            "segment": self.segment,
            "multiplier": self.multiplier,
            "image_point": self.image_point,
            "topdown_point": self.topdown_point,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class VisionServiceResult:
    """
    Ergebnis eines verarbeiteten Liveframes.
    """

    camera_id: int
    timestamp: float
    status: str
    message: str
    hit_event: Optional[VisionHitEvent] = None
    detection_result: Any = None
    board_is_clear: Optional[bool] = None
    board_changed_ratio: Optional[float] = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "status": self.status,
            "message": self.message,
            "hit_event": None if self.hit_event is None else self.hit_event.to_dict(),
            "board_is_clear": self.board_is_clear,
            "board_changed_ratio": self.board_changed_ratio,
            "debug": self.debug,
        }


# -----------------------------------------------------------------------------
# Interner Kamerastatus
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class CameraVisionState:
    camera_id: int
    armed: bool = False
    awaiting_clear_board: bool = False
    reference_frame: Optional[np.ndarray] = None
    board_mask: Optional[np.ndarray] = None
    last_hit_event: Optional[VisionHitEvent] = None
    last_detection_result: Any = None
    last_detection_timestamp: Optional[float] = None
    last_status: str = STATUS_BOARD_NOT_REFERENCED
    clear_board_consecutive_ok: int = 0

    # --------------------------------------------------------------
    # NEU: Pending-/Confirming-State
    # --------------------------------------------------------------
    pending_hit_event: Optional[VisionHitEvent] = None
    pending_detection_result: Any = None
    pending_started_timestamp: Optional[float] = None
    pending_last_timestamp: Optional[float] = None
    pending_confirmation_count: int = 0


# -----------------------------------------------------------------------------
# Hauptservice
# -----------------------------------------------------------------------------
class VisionService:
    """
    Produktfluss-Service für Single-Camera-Erkennung.

    Typischer Ablauf:
    1. set_reference_frame(camera_id, frame)
    2. arm(camera_id)
    3. process_frame(camera_id, live_frame)
    4. bei Treffer -> waiting_for_board_clear
    5. sobald Board frei -> wieder ready
    """

    def __init__(
        self,
        *,
        config: Optional[VisionServiceConfig] = None,
        default_detector: Optional[SingleCamDetector] = None,
        detectors_by_camera: Optional[dict[int, SingleCamDetector]] = None,
    ) -> None:
        self.config = config or VisionServiceConfig()
        self._default_detector = default_detector
        self._detectors_by_camera: dict[int, SingleCamDetector] = dict(detectors_by_camera or {})
        self._states: dict[int, CameraVisionState] = {}

    # -------------------------------------------------------------------------
    # Öffentliche Verwaltung
    # -------------------------------------------------------------------------
    def register_detector(self, camera_id: int, detector: SingleCamDetector) -> None:
        self._detectors_by_camera[int(camera_id)] = detector
        self._ensure_state(camera_id)

    def set_default_detector(self, detector: SingleCamDetector) -> None:
        self._default_detector = detector

    def get_state(self, camera_id: int) -> CameraVisionState:
        return self._ensure_state(camera_id)

    def arm(self, camera_id: int) -> None:
        state = self._ensure_state(camera_id)
        state.armed = True

    def disarm(self, camera_id: int) -> None:
        state = self._ensure_state(camera_id)
        state.armed = False

    def reset_runtime_state(self, camera_id: int) -> None:
        state = self._ensure_state(camera_id)
        state.awaiting_clear_board = False
        state.last_hit_event = None
        state.last_detection_result = None
        state.last_detection_timestamp = None
        state.last_status = STATUS_BOARD_NOT_REFERENCED if state.reference_frame is None else STATUS_READY
        state.clear_board_consecutive_ok = 0
        self._reset_pending_hit(state)

    def set_board_mask(self, camera_id: int, board_mask: Optional[np.ndarray]) -> None:
        state = self._ensure_state(camera_id)
        if board_mask is None:
            state.board_mask = None
            return

        self._validate_mask(board_mask, name="board_mask")
        state.board_mask = self._normalize_mask(board_mask)

    def set_reference_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
    ) -> None:
        state = self._ensure_state(camera_id)
        self._validate_frame(frame, name="reference_frame")

        state.reference_frame = self._copy_frame(frame)
        state.awaiting_clear_board = False
        state.last_hit_event = None
        state.last_detection_result = None
        state.last_detection_timestamp = None
        state.clear_board_consecutive_ok = 0
        state.last_status = STATUS_READY
        self._reset_pending_hit(state)

        if self.config.auto_arm_on_reference_save:
            state.armed = True

    def clear_reference_frame(self, camera_id: int) -> None:
        state = self._ensure_state(camera_id)
        state.reference_frame = None
        state.awaiting_clear_board = False
        state.last_hit_event = None
        state.last_detection_result = None
        state.last_detection_timestamp = None
        state.clear_board_consecutive_ok = 0
        state.last_status = STATUS_BOARD_NOT_REFERENCED
        self._reset_pending_hit(state)

    def has_reference_frame(self, camera_id: int) -> bool:
        state = self._ensure_state(camera_id)
        return state.reference_frame is not None

    # -------------------------------------------------------------------------
    # Hauptlogik
    # -------------------------------------------------------------------------
    def process_frame(
        self,
        camera_id: int,
        frame: np.ndarray,
        *,
        timestamp: Optional[float] = None,
    ) -> VisionServiceResult:
        """
        Verarbeitet genau einen Liveframe.

        Produktlogik:
        - keine Referenz -> board_not_referenced
        - disarmed -> disarmed
        - awaiting_clear_board -> nur Clear-Check
        - sonst Detector aufrufen
        - neuer Treffer erst nach Confirming-Phase
        """
        camera_id = int(camera_id)
        state = self._ensure_state(camera_id)
        detector = self._get_detector(camera_id)
        ts = self._resolve_timestamp(timestamp)

        try:
            self._validate_frame(frame, name="frame")
        except Exception as exc:
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_ERROR,
                message=f"Ungültiges Frame: {exc}",
            )

        if state.reference_frame is None:
            state.last_status = STATUS_BOARD_NOT_REFERENCED
            self._reset_pending_hit(state)
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_BOARD_NOT_REFERENCED,
                message="Kein Referenzbild gesetzt.",
            )

        if not state.armed:
            state.last_status = STATUS_DISARMED
            self._reset_pending_hit(state)
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_DISARMED,
                message="Erkennung ist deaktiviert.",
            )

        if detector is None:
            state.last_status = STATUS_ERROR
            self._reset_pending_hit(state)
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_ERROR,
                message="Kein SingleCamDetector für diese Kamera registriert.",
            )

        # Sicherheitscheck Größe
        if frame.shape[:2] != state.reference_frame.shape[:2]:
            self._reset_pending_hit(state)
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_ERROR,
                message=(
                    "Frame und Referenzbild haben unterschiedliche Größe. "
                    f"frame={frame.shape[:2]} vs reference={state.reference_frame.shape[:2]}"
                ),
            )

        # Pending verwerfen, wenn zu alt
        self._expire_pending_hit_if_needed(state=state, timestamp=ts)

        # -------------------------------------------------------------
        # 1) Falls auf Board-clear gewartet wird: nur Clear-Logik
        # -------------------------------------------------------------
        if state.awaiting_clear_board and self.config.require_board_clear_after_hit:
            board_changed_ratio = self._compute_board_changed_ratio(
                current_frame=frame,
                reference_frame=state.reference_frame,
                board_mask=state.board_mask if self.config.use_board_mask_for_clear_check else None,
            )
            board_is_clear = (
                board_changed_ratio <= float(self.config.clear_board_changed_ratio_threshold)
            )

            if board_is_clear:
                state.clear_board_consecutive_ok += 1
            else:
                state.clear_board_consecutive_ok = 0

            if state.clear_board_consecutive_ok >= int(self.config.clear_board_required_consecutive_frames):
                state.awaiting_clear_board = False
                state.clear_board_consecutive_ok = 0
                state.last_status = STATUS_READY
                self._reset_pending_hit(state)
                return VisionServiceResult(
                    camera_id=camera_id,
                    timestamp=ts,
                    status=STATUS_READY,
                    message="Board ist wieder frei. Erkennung bereit.",
                    board_is_clear=True,
                    board_changed_ratio=board_changed_ratio,
                    debug={
                        "awaiting_clear_board": False,
                    },
                )

            state.last_status = STATUS_WAITING_FOR_CLEAR
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_WAITING_FOR_CLEAR,
                message="Warte darauf, dass das Board wieder frei ist.",
                hit_event=state.last_hit_event,
                detection_result=state.last_detection_result,
                board_is_clear=board_is_clear,
                board_changed_ratio=board_changed_ratio,
                debug={
                    "clear_board_consecutive_ok": state.clear_board_consecutive_ok,
                    "required_consecutive_frames": int(self.config.clear_board_required_consecutive_frames),
                    "last_hit_label": None if state.last_hit_event is None else state.last_hit_event.label,
                },
            )

        # -------------------------------------------------------------
        # 2) Cooldown prüfen
        # -------------------------------------------------------------
        if state.last_detection_timestamp is not None:
            delta = ts - float(state.last_detection_timestamp)
            if delta < float(self.config.min_seconds_between_hits):
                state.last_status = STATUS_COOLDOWN
                return VisionServiceResult(
                    camera_id=camera_id,
                    timestamp=ts,
                    status=STATUS_COOLDOWN,
                    message=f"Cooldown aktiv ({delta:.3f}s).",
                    hit_event=state.last_hit_event,
                    detection_result=state.last_detection_result,
                    debug={
                        "seconds_since_last_hit": delta,
                        "min_seconds_between_hits": float(self.config.min_seconds_between_hits),
                    },
                )

        # -------------------------------------------------------------
        # 3) Detector laufen lassen
        # -------------------------------------------------------------
        try:
            detection_result = detector.detect(
                frame=frame,
                reference_frame=state.reference_frame,
                board_mask=state.board_mask,
                board_polygon=None,
            )
        except Exception as exc:
            state.last_status = STATUS_ERROR
            self._reset_pending_hit(state)
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_ERROR,
                message=f"Detector-Fehler: {exc}",
            )

        state.last_detection_result = detection_result

        candidate_hit_event = self._build_candidate_hit_event(
            camera_id=camera_id,
            timestamp=ts,
            detection_result=detection_result,
        )

        # -------------------------------------------------------------
        # 4) Kein plausibler Hit -> Pending ggf. verwerfen
        # -------------------------------------------------------------
        if candidate_hit_event is None:
            self._reset_pending_hit(state)
            state.last_status = STATUS_NO_HIT
            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_NO_HIT,
                message="Kein gültiger Treffer erkannt.",
                detection_result=detection_result,
                debug=self._merge_debug(
                    self._build_debug_payload(detection_result),
                    {
                        "pending_confirmation_count": 0,
                        "confirm_required_frames": int(self.config.confirm_hit_required_consecutive_frames),
                        "candidate_hit_valid": False,
                    },
                ),
            )

        # -------------------------------------------------------------
        # 5) Pending-Phase / Confirming
        # -------------------------------------------------------------
        if state.pending_hit_event is None:
            state.pending_hit_event = candidate_hit_event
            state.pending_detection_result = detection_result
            state.pending_started_timestamp = ts
            state.pending_last_timestamp = ts
            state.pending_confirmation_count = 1
            state.last_status = STATUS_NO_HIT

            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_NO_HIT,
                message=(
                    f"Treffer-Kandidat erkannt ({candidate_hit_event.label}), "
                    "warte auf Bestätigung."
                ),
                detection_result=detection_result,
                debug=self._merge_debug(
                    self._build_debug_payload(detection_result),
                    {
                        "pending_confirmation_count": state.pending_confirmation_count,
                        "confirm_required_frames": int(self.config.confirm_hit_required_consecutive_frames),
                        "candidate_hit_valid": True,
                        "candidate_hit_label": candidate_hit_event.label,
                        "candidate_hit_confidence": candidate_hit_event.confidence,
                    },
                ),
            )

        # Prüfen, ob neuer Kandidat zum bestehenden Pending-Hit passt
        if self._hit_events_match(state.pending_hit_event, candidate_hit_event):
            state.pending_hit_event = self._select_better_hit_event(
                old_event=state.pending_hit_event,
                new_event=candidate_hit_event,
            )
            state.pending_detection_result = detection_result
            state.pending_last_timestamp = ts
            state.pending_confirmation_count += 1

            required = max(1, int(self.config.confirm_hit_required_consecutive_frames))
            if state.pending_confirmation_count < required:
                state.last_status = STATUS_NO_HIT
                return VisionServiceResult(
                    camera_id=camera_id,
                    timestamp=ts,
                    status=STATUS_NO_HIT,
                    message=(
                        f"Treffer-Kandidat bestätigt "
                        f"({state.pending_confirmation_count}/{required})."
                    ),
                    detection_result=detection_result,
                    debug=self._merge_debug(
                        self._build_debug_payload(detection_result),
                        {
                            "pending_confirmation_count": state.pending_confirmation_count,
                            "confirm_required_frames": required,
                            "candidate_hit_valid": True,
                            "candidate_hit_label": state.pending_hit_event.label,
                            "candidate_hit_confidence": state.pending_hit_event.confidence,
                        },
                    ),
                )

            # ---------------------------------------------------------
            # 6) Treffer jetzt final bestätigen
            # ---------------------------------------------------------
            confirmed_hit = state.pending_hit_event
            state.last_hit_event = confirmed_hit
            state.last_detection_result = state.pending_detection_result
            state.last_detection_timestamp = ts
            state.last_status = STATUS_HIT_DETECTED
            self._reset_pending_hit(state)

            if self.config.require_board_clear_after_hit:
                state.awaiting_clear_board = True
                state.clear_board_consecutive_ok = 0

            return VisionServiceResult(
                camera_id=camera_id,
                timestamp=ts,
                status=STATUS_HIT_DETECTED,
                message=f"Treffer erkannt: {confirmed_hit.label}",
                hit_event=confirmed_hit,
                detection_result=state.last_detection_result,
                debug=self._merge_debug(
                    self._build_debug_payload(state.last_detection_result),
                    {
                        "confirmed_after_frames": required,
                        "candidate_hit_valid": True,
                    },
                ),
            )

        # -------------------------------------------------------------
        # 7) Neuer Kandidat passt NICHT zum Pending-Hit:
        #    Pending ersetzen und Bestätigung neu starten
        # -------------------------------------------------------------
        state.pending_hit_event = candidate_hit_event
        state.pending_detection_result = detection_result
        state.pending_started_timestamp = ts
        state.pending_last_timestamp = ts
        state.pending_confirmation_count = 1
        state.last_status = STATUS_NO_HIT

        return VisionServiceResult(
            camera_id=camera_id,
            timestamp=ts,
            status=STATUS_NO_HIT,
            message=(
                f"Neuer Treffer-Kandidat erkannt ({candidate_hit_event.label}), "
                "Bestätigung neu gestartet."
            ),
            detection_result=detection_result,
            debug=self._merge_debug(
                self._build_debug_payload(detection_result),
                {
                    "pending_confirmation_count": state.pending_confirmation_count,
                    "confirm_required_frames": int(self.config.confirm_hit_required_consecutive_frames),
                    "candidate_hit_valid": True,
                    "candidate_hit_label": candidate_hit_event.label,
                    "candidate_hit_confidence": candidate_hit_event.confidence,
                    "pending_restarted": True,
                },
            ),
        )

    # -------------------------------------------------------------------------
    # Interne Hilfslogik
    # -------------------------------------------------------------------------
    def _ensure_state(self, camera_id: int) -> CameraVisionState:
        camera_id = int(camera_id)
        if camera_id not in self._states:
            self._states[camera_id] = CameraVisionState(camera_id=camera_id)
        return self._states[camera_id]

    def _get_detector(self, camera_id: int) -> Optional[SingleCamDetector]:
        camera_id = int(camera_id)
        if camera_id in self._detectors_by_camera:
            return self._detectors_by_camera[camera_id]
        return self._default_detector

    def _resolve_timestamp(self, timestamp: Optional[float]) -> float:
        if timestamp is not None:
            return float(timestamp)
        if self.config.use_monotonic_time_when_missing:
            return float(time.monotonic())
        return float(time.time())

    def _copy_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()

    def _reset_pending_hit(self, state: CameraVisionState) -> None:
        state.pending_hit_event = None
        state.pending_detection_result = None
        state.pending_started_timestamp = None
        state.pending_last_timestamp = None
        state.pending_confirmation_count = 0

    def _expire_pending_hit_if_needed(self, *, state: CameraVisionState, timestamp: float) -> None:
        if state.pending_hit_event is None:
            return

        reference_ts = state.pending_last_timestamp
        if reference_ts is None:
            reference_ts = state.pending_started_timestamp

        if reference_ts is None:
            self._reset_pending_hit(state)
            return

        age = float(timestamp) - float(reference_ts)
        if age > float(self.config.pending_hit_max_age_seconds):
            self._reset_pending_hit(state)

    def _compute_board_changed_ratio(
        self,
        *,
        current_frame: np.ndarray,
        reference_frame: np.ndarray,
        board_mask: Optional[np.ndarray],
    ) -> float:
        """
        Schätzt, wie stark sich das aktuelle Board vom Referenzboard unterscheidet.

        Rückgabe:
        - Anteil geänderter Pixel im ROI
        - kleine Werte => Board ist eher wieder frei
        """
        current_gray = cv2.cvtColor(self._ensure_bgr(current_frame), cv2.COLOR_BGR2GRAY)
        reference_gray = cv2.cvtColor(self._ensure_bgr(reference_frame), cv2.COLOR_BGR2GRAY)

        ksize = self._ensure_odd_kernel(self.config.clear_board_blur_kernel_size)
        if ksize > 1:
            current_gray = cv2.GaussianBlur(current_gray, (ksize, ksize), 0)
            reference_gray = cv2.GaussianBlur(reference_gray, (ksize, ksize), 0)

        diff = cv2.absdiff(reference_gray, current_gray)
        _, binary = cv2.threshold(
            diff,
            int(self.config.clear_board_diff_threshold),
            255,
            cv2.THRESH_BINARY,
        )

        if board_mask is not None:
            mask = self._normalize_mask(board_mask)
            binary = cv2.bitwise_and(binary, binary, mask=mask)
            roi_pixels = int(np.count_nonzero(mask))
        else:
            roi_pixels = int(binary.shape[0] * binary.shape[1])

        changed_pixels = int(np.count_nonzero(binary))
        if roi_pixels <= 0:
            return 1.0

        return float(changed_pixels / roi_pixels)

    def _extract_best_hit(self, detection_result: Any) -> Any:
        if detection_result is None:
            return None

        best_hit = getattr(detection_result, "best_hit", None)
        if best_hit is not None:
            return best_hit

        best_estimate = self._extract_best_estimate(detection_result)
        if best_estimate is not None:
            return getattr(best_estimate, "scored_hit", None)

        return None

    def _extract_best_estimate(self, detection_result: Any) -> Any:
        if detection_result is None:
            return None

        best_estimate = getattr(detection_result, "best_estimate", None)
        if best_estimate is not None:
            return best_estimate

        scored_estimates = getattr(detection_result, "scored_estimates", None)
        if isinstance(scored_estimates, list) and scored_estimates:
            return scored_estimates[0]

        return None

    def _extract_topdown_point(self, best_hit: Any) -> Optional[PointF]:
        if best_hit is None:
            return None

        point = getattr(best_hit, "topdown_point", None)
        coerced = self._coerce_point(point)
        if coerced is not None:
            return coerced

        raw_hit = getattr(best_hit, "raw_hit", None)
        if raw_hit is not None:
            # Objektstil
            tx = getattr(raw_hit, "topdown_x_px", None)
            ty = getattr(raw_hit, "topdown_y_px", None)
            if tx is not None and ty is not None:
                return float(tx), float(ty)

            # Dictstil
            if isinstance(raw_hit, dict):
                if "topdown_x_px" in raw_hit and "topdown_y_px" in raw_hit:
                    return float(raw_hit["topdown_x_px"]), float(raw_hit["topdown_y_px"])

        return None

    def _extract_confidence(self, best_estimate: Any) -> Optional[float]:
        if best_estimate is None:
            return None

        if hasattr(best_estimate, "combined_confidence"):
            value = getattr(best_estimate, "combined_confidence", None)
            if value is not None:
                return float(value)

        if hasattr(best_estimate, "impact_confidence"):
            value = getattr(best_estimate, "impact_confidence", None)
            if value is not None:
                return float(value)

        return None

    def _is_miss(self, best_hit: Any) -> bool:
        if best_hit is None:
            return True

        is_miss = getattr(best_hit, "is_miss", None)
        if is_miss is not None:
            return bool(is_miss)

        ring = getattr(best_hit, "ring", None)
        if ring is not None and str(ring).upper() == "MISS":
            return True

        label = getattr(best_hit, "label", None)
        if label is not None and str(label).upper() == "MISS":
            return True

        score = getattr(best_hit, "score", None)
        if score is not None and int(score) == 0:
            return True

        return False

    def _build_candidate_hit_event(
        self,
        *,
        camera_id: int,
        timestamp: float,
        detection_result: Any,
    ) -> Optional[VisionHitEvent]:
        best_hit = self._extract_best_hit(detection_result)
        best_estimate = self._extract_best_estimate(detection_result)

        if best_hit is None or best_estimate is None:
            return None

        if self._is_miss(best_hit):
            return None

        confidence = self._extract_confidence(best_estimate)
        if confidence is None:
            confidence = 0.0

        if float(confidence) < float(self.config.min_hit_confidence):
            return None

        return VisionHitEvent(
            camera_id=int(camera_id),
            timestamp=float(timestamp),
            label=str(getattr(best_hit, "label", "")),
            score=int(getattr(best_hit, "score", 0)),
            ring=str(getattr(best_hit, "ring", "")),
            segment=self._safe_int(getattr(best_hit, "segment", None)),
            multiplier=int(getattr(best_hit, "multiplier", 0)),
            image_point=self._coerce_point(getattr(best_estimate, "image_point", None)),
            topdown_point=self._extract_topdown_point(best_hit),
            confidence=float(confidence),
            detection_result=detection_result,
        )

    def _hit_events_match(
        self,
        old_event: VisionHitEvent,
        new_event: VisionHitEvent,
    ) -> bool:
        if self.config.confirm_same_label_required:
            if str(old_event.label).upper() != str(new_event.label).upper():
                return False

        topdown_dist = self._point_distance(old_event.topdown_point, new_event.topdown_point)
        image_dist = self._point_distance(old_event.image_point, new_event.image_point)

        if topdown_dist is not None:
            return topdown_dist <= float(self.config.confirm_max_topdown_distance_px)

        if image_dist is not None:
            return image_dist <= float(self.config.confirm_max_image_distance_px)

        # Wenn gar keine Geometrie vergleichbar ist, lieber nicht matchen.
        return False

    def _select_better_hit_event(
        self,
        *,
        old_event: VisionHitEvent,
        new_event: VisionHitEvent,
    ) -> VisionHitEvent:
        old_conf = 0.0 if old_event.confidence is None else float(old_event.confidence)
        new_conf = 0.0 if new_event.confidence is None else float(new_event.confidence)

        if new_conf >= old_conf:
            return new_event
        return old_event

    def _point_distance(
        self,
        a: Optional[PointF],
        b: Optional[PointF],
    ) -> Optional[float]:
        if a is None or b is None:
            return None

        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _build_debug_payload(self, detection_result: Any) -> dict[str, Any]:
        if detection_result is None:
            return {}

        metadata = copy.deepcopy(getattr(detection_result, "metadata", {}) or {})
        payload = {
            "metadata": metadata,
        }

        if self.config.keep_debug_images:
            debug_images = getattr(detection_result, "debug_images", None)
            if isinstance(debug_images, dict):
                payload["debug_image_keys"] = sorted(debug_images.keys())

        best_estimate = self._extract_best_estimate(detection_result)
        if best_estimate is not None:
            payload["best_candidate_id"] = getattr(best_estimate, "candidate_id", None)
            payload["best_image_point"] = self._coerce_point(getattr(best_estimate, "image_point", None))
            payload["best_combined_confidence"] = self._extract_confidence(best_estimate)

        best_hit = self._extract_best_hit(detection_result)
        if best_hit is not None:
            payload["best_label"] = getattr(best_hit, "label", None)
            payload["best_score"] = getattr(best_hit, "score", None)
            payload["best_ring"] = getattr(best_hit, "ring", None)
            payload["best_segment"] = getattr(best_hit, "segment", None)
            payload["best_topdown_point"] = self._extract_topdown_point(best_hit)

        return payload

    def _merge_debug(self, base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base or {})
        merged.update(extra or {})
        return merged

    # -------------------------------------------------------------------------
    # Kleine Utils
    # -------------------------------------------------------------------------
    def _validate_frame(self, frame: np.ndarray, *, name: str) -> None:
        if frame is None:
            raise ValueError(f"{name} must not be None.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.ndim not in (2, 3):
            raise ValueError(f"{name} must have ndim 2 or 3, got {frame.ndim}.")
        if frame.size == 0:
            raise ValueError(f"{name} must not be empty.")

    def _validate_mask(self, mask: np.ndarray, *, name: str) -> None:
        if mask is None:
            raise ValueError(f"{name} must not be None.")
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray, got {type(mask)!r}.")
        if mask.ndim != 2:
            raise ValueError(f"{name} must be a single-channel mask.")

    def _ensure_bgr(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame.copy()
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"Unsupported frame shape for BGR conversion: {frame.shape}")

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        return np.where(mask > 0, 255, 0).astype(np.uint8)

    def _ensure_odd_kernel(self, value: int) -> int:
        value = max(1, int(value))
        if value % 2 == 0:
            value += 1
        return value

    def _coerce_point(self, value: Any) -> Optional[PointF]:
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            arr = value.astype(float).reshape(-1)
            if arr.size >= 2:
                return float(arr[0]), float(arr[1])

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except Exception:
                return None

        if isinstance(value, dict):
            if "x" in value and "y" in value:
                try:
                    return float(value["x"]), float(value["y"])
                except Exception:
                    return None
            if "x_px" in value and "y_px" in value:
                try:
                    return float(value["x_px"]), float(value["y_px"])
                except Exception:
                    return None

        if hasattr(value, "x") and hasattr(value, "y"):
            try:
                return float(value.x), float(value.y)
            except Exception:
                return None

        if hasattr(value, "x_px") and hasattr(value, "y_px"):
            try:
                return float(value.x_px), float(value.y_px)
            except Exception:
                return None

        return None

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None


__all__ = [
    "STATUS_BOARD_NOT_REFERENCED",
    "STATUS_READY",
    "STATUS_NO_HIT",
    "STATUS_HIT_DETECTED",
    "STATUS_WAITING_FOR_CLEAR",
    "STATUS_COOLDOWN",
    "STATUS_DISARMED",
    "STATUS_ERROR",
    "VisionServiceConfig",
    "VisionHitEvent",
    "VisionServiceResult",
    "CameraVisionState",
    "VisionService",
]