# vision/dart_event_manager.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from vision.dart_detector import DartDetectionResult


@dataclass
class DartEventCandidate:
    camera_index: int
    detection: DartDetectionResult


@dataclass
class ClosedDartEvent:
    event_index: int
    candidates: List[DartEventCandidate]


class DartEventManager:
    def __init__(self) -> None:
        self.is_armed: bool = False
        self.current_event_index: int = 0

        # Etwas längeres Sammelfenster für genau EINEN Dart
        self.event_window_seconds: float = 0.70

        # Nach einem Dart klar blockieren, damit nicht gleich Schatten/Nachschwingen
        # als nächster Dart gezählt werden
        self.post_event_cooldown_seconds: float = 1.60

        self.max_events_per_round: int = 3

        self._event_open: bool = False
        self._event_started_at: float = 0.0
        self._cooldown_until: float = 0.0
        self._current_candidates: Dict[int, DartDetectionResult] = {}

        self.closed_events: List[ClosedDartEvent] = []

    def reset_round(self) -> None:
        self.is_armed = False
        self.current_event_index = 0
        self._event_open = False
        self._event_started_at = 0.0
        self._cooldown_until = 0.0
        self._current_candidates = {}
        self.closed_events = []

    def arm(self) -> None:
        self.reset_round()
        self.is_armed = True

    def _can_open_new_event(self) -> bool:
        if not self.is_armed:
            return False

        if len(self.closed_events) >= self.max_events_per_round:
            return False

        now = time.monotonic()
        if now < self._cooldown_until:
            return False

        return True

    def _open_event_if_needed(self) -> bool:
        if self._event_open:
            return True

        if not self._can_open_new_event():
            return False

        self._event_open = True
        self._event_started_at = time.monotonic()
        self._current_candidates = {}
        return True

    def add_candidate(self, camera_index: int, detection: DartDetectionResult) -> bool:
        if not self.is_armed:
            return False

        if len(self.closed_events) >= self.max_events_per_round:
            return False

        opened = self._open_event_if_needed()
        if not opened:
            return False

        # Pro Kamera nur EIN Kandidat pro Dart-Event
        if camera_index in self._current_candidates:
            return False

        self._current_candidates[camera_index] = detection
        return True

    def poll_closed_event(self) -> Optional[ClosedDartEvent]:
        if not self.is_armed:
            return None

        if not self._event_open:
            return None

        now = time.monotonic()
        if (now - self._event_started_at) < self.event_window_seconds:
            return None

        candidates = [
            DartEventCandidate(camera_index=cam_idx, detection=det)
            for cam_idx, det in sorted(self._current_candidates.items(), key=lambda item: item[0])
        ]

        closed = ClosedDartEvent(
            event_index=self.current_event_index + 1,
            candidates=candidates,
        )

        self.closed_events.append(closed)
        self.current_event_index += 1

        self._event_open = False
        self._event_started_at = 0.0
        self._current_candidates = {}

        # WICHTIG: Nach jedem Dart deutlicher Cooldown
        self._cooldown_until = time.monotonic() + self.post_event_cooldown_seconds

        if len(self.closed_events) >= self.max_events_per_round:
            self.is_armed = False

        return closed

    def current_status_text(self) -> str:
        if not self.is_armed:
            return "Event-Manager: nicht scharf"

        now = time.monotonic()

        if self._event_open:
            return (
                f"Event-Manager: Event {self.current_event_index + 1} offen | "
                f"Kandidaten: {len(self._current_candidates)}"
            )

        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            return (
                f"Event-Manager: Cooldown aktiv ({remaining:.2f}s) | "
                f"abgeschlossene Events: {len(self.closed_events)}/{self.max_events_per_round}"
            )

        return (
            f"Event-Manager: scharf | "
            f"abgeschlossene Events: {len(self.closed_events)}/{self.max_events_per_round}"
        )
