# vision/calibration_storage.py
# Triple One - Schritt 4:
# Zentrale Speicherung/Ladung der Kalibrierung.
#
# Ziel:
# - pro Kamera sauber speichern/laden
# - nur 4 manuelle Marker als Quelle behandeln
# - Bull automatisch aus den 4 Punkten ergänzen
# - Referenzbildpfade verwalten
# - robust gegen fehlende/alte JSON-Strukturen sein
#
# Wichtig:
# Diese Datei rechnet keine eigene Geometrie.
# Bull und Pipeline-Punkte kommen aus calibration_geometry.py.

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from vision.calibration_geometry import build_pipeline_points


# ------------------------------------------------------------
# Datenklassen
# ------------------------------------------------------------

@dataclass
class CameraCalibrationRecord:
    """
    Persistentes Kalibrierprofil für genau eine Kamera.
    Gespeichert werden primär die 4 manuellen Marker.
    """
    camera_index: int
    name: str
    enabled: bool
    device_id: int
    width: int
    height: int
    fps: int
    rotation: int
    flip: bool
    overlay_alpha: float
    show_numbers: bool
    show_sector_lines: bool
    manual_points: List[Dict[str, int]]
    empty_board_reference_path: str = ""

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "camera_index": int(self.camera_index),
            "name": str(self.name),
            "enabled": bool(self.enabled),
            "device_id": int(self.device_id),
            "width": int(self.width),
            "height": int(self.height),
            "fps": int(self.fps),
            "rotation": int(self.rotation),
            "flip": bool(self.flip),
            "overlay_alpha": float(self.overlay_alpha),
            "show_numbers": bool(self.show_numbers),
            "show_sector_lines": bool(self.show_sector_lines),
            "manual_points": deepcopy(self.manual_points[:4]),
            "empty_board_reference_path": str(self.empty_board_reference_path or ""),
        }


# ------------------------------------------------------------
# Hauptklasse
# ------------------------------------------------------------

class CalibrationStorage:
    """
    Verwaltet:
    - Laden/Speichern einer Gesamtdatei
    - Export/Import pro Kamera
    - Normalisierung alter Datenstrukturen
    """

    def __init__(
        self,
        base_dir: str | Path = "data",
        calibration_filename: str = "calibration_store.json",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.calibration_dir = self.base_dir / "calibration"
        self.reference_dir = self.base_dir / "references"
        self.log_dir = self.base_dir / "logs"

        self.calibration_dir.mkdir(parents=True, exist_ok=True)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.calibration_dir / calibration_filename

    # --------------------------------------------------------
    # Defaults / Normalisierung
    # --------------------------------------------------------

    def _default_manual_points(self, width: int, height: int) -> List[Dict[str, int]]:
        """
        Fallback-Startwerte für die 4 Marker.
        """
        width = max(320, int(width))
        height = max(240, int(height))

        return [
            {"x_px": int(width * 0.60), "y_px": int(height * 0.28)},  # P1 = 20|1
            {"x_px": int(width * 0.70), "y_px": int(height * 0.72)},  # P2 = 6|10
            {"x_px": int(width * 0.26), "y_px": int(height * 0.67)},  # P3 = 3|19
            {"x_px": int(width * 0.37), "y_px": int(height * 0.30)},  # P4 = 11|14
        ]

    def _normalize_manual_points(
        self,
        points: Any,
        width: int,
        height: int,
    ) -> List[Dict[str, int]]:
        """
        Nimmt beliebige Punktlisten und reduziert sie robust auf 4 manuelle Marker.
        Alte Formate mit 5 Punkten werden akzeptiert, aber nur die ersten 4 übernommen.
        """
        if not isinstance(points, list) or len(points) < 4:
            return self._default_manual_points(width, height)

        normalized: List[Dict[str, int]] = []
        for item in points[:4]:
            if not isinstance(item, dict):
                return self._default_manual_points(width, height)

            x_px = int(item.get("x_px", 0))
            y_px = int(item.get("y_px", 0))

            x_px = max(0, min(int(width) - 1, x_px))
            y_px = max(0, min(int(height) - 1, y_px))

            normalized.append({"x_px": x_px, "y_px": y_px})

        if len(normalized) != 4:
            return self._default_manual_points(width, height)

        return normalized

    def _camera_reference_filename(self, camera_index: int) -> str:
        return f"camera_{int(camera_index) + 1}_empty_board.png"

    def _default_camera_record(
        self,
        camera_index: int,
        camera_config: Optional[Dict[str, Any]] = None,
        calibration_config: Optional[Dict[str, Any]] = None,
    ) -> CameraCalibrationRecord:
        """
        Baut einen Default-Datensatz für eine Kamera aus vorhandenem Config-Kontext.
        """
        camera_config = dict(camera_config or {})
        calibration_config = dict(calibration_config or {})

        width = int(camera_config.get("width", calibration_config.get("frame_width", 1280)))
        height = int(camera_config.get("height", calibration_config.get("frame_height", 720)))

        manual_points = self._normalize_manual_points(
            calibration_config.get("points", []),
            width=width,
            height=height,
        )

        reference_path = str(
            self.reference_dir / self._camera_reference_filename(camera_index)
        )

        return CameraCalibrationRecord(
            camera_index=int(camera_index),
            name=str(calibration_config.get("name", f"Kamera {camera_index + 1}")),
            enabled=bool(camera_config.get("enabled", True)),
            device_id=int(camera_config.get("device_id", -1)),
            width=width,
            height=height,
            fps=int(camera_config.get("fps", 30)),
            rotation=int(camera_config.get("rotation", 0)),
            flip=bool(camera_config.get("flip", False)),
            overlay_alpha=float(calibration_config.get("overlay_alpha", 0.35)),
            show_numbers=bool(calibration_config.get("show_numbers", True)),
            show_sector_lines=bool(calibration_config.get("show_sector_lines", True)),
            manual_points=manual_points,
            empty_board_reference_path=str(
                calibration_config.get("empty_board_reference_path", reference_path)
            ),
        )

    def _record_from_storage_dict(
        self,
        data: Dict[str, Any],
        fallback_camera_index: int,
    ) -> CameraCalibrationRecord:
        """
        Baut einen Record aus gespeicherter JSON-Struktur.
        """
        width = int(data.get("width", 1280))
        height = int(data.get("height", 720))

        manual_points = self._normalize_manual_points(
            data.get("manual_points", data.get("points", [])),
            width=width,
            height=height,
        )

        reference_path = str(
            data.get(
                "empty_board_reference_path",
                self.reference_dir / self._camera_reference_filename(fallback_camera_index),
            )
        )

        return CameraCalibrationRecord(
            camera_index=int(data.get("camera_index", fallback_camera_index)),
            name=str(data.get("name", f"Kamera {fallback_camera_index + 1}")),
            enabled=bool(data.get("enabled", True)),
            device_id=int(data.get("device_id", -1)),
            width=width,
            height=height,
            fps=int(data.get("fps", 30)),
            rotation=int(data.get("rotation", 0)),
            flip=bool(data.get("flip", False)),
            overlay_alpha=float(data.get("overlay_alpha", 0.35)),
            show_numbers=bool(data.get("show_numbers", True)),
            show_sector_lines=bool(data.get("show_sector_lines", True)),
            manual_points=manual_points,
            empty_board_reference_path=reference_path,
        )

    # --------------------------------------------------------
    # Record -> App-Config
    # --------------------------------------------------------

    def record_to_camera_config(self, record: CameraCalibrationRecord) -> Dict[str, Any]:
        """
        Für camera_config/cameras[n]
        """
        return {
            "enabled": bool(record.enabled),
            "device_id": int(record.device_id),
            "width": int(record.width),
            "height": int(record.height),
            "fps": int(record.fps),
            "rotation": int(record.rotation),
            "flip": bool(record.flip),
        }

    def record_to_calibration_config(self, record: CameraCalibrationRecord) -> Dict[str, Any]:
        """
        Für calibration_config/cameras[n]
        Enthält points im kompatiblen 5-Punkte-Format.
        """
        return {
            "name": str(record.name),
            "frame_width": int(record.width),
            "frame_height": int(record.height),
            "overlay_alpha": float(record.overlay_alpha),
            "show_numbers": bool(record.show_numbers),
            "show_sector_lines": bool(record.show_sector_lines),
            "points": build_pipeline_points(record.manual_points),
            "empty_board_reference_path": str(record.empty_board_reference_path),
        }

    def record_to_runtime_bundle(self, record: CameraCalibrationRecord) -> Dict[str, Any]:
        """
        Praktisches Bundle für Laufzeitnutzung.
        """
        return {
            "camera": self.record_to_camera_config(record),
            "calibration": self.record_to_calibration_config(record),
            "manual_points": deepcopy(record.manual_points),
        }

    # --------------------------------------------------------
    # Store-Datei lesen / schreiben
    # --------------------------------------------------------

    def load_store(self) -> Dict[str, Any]:
        if not self.store_path.exists():
            return {"version": 1, "cameras": []}

        try:
            raw = json.loads(self.store_path.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "cameras": []}

        if not isinstance(raw, dict):
            return {"version": 1, "cameras": []}

        cameras = raw.get("cameras", [])
        if not isinstance(cameras, list):
            cameras = []

        return {
            "version": int(raw.get("version", 1)),
            "cameras": cameras,
        }

    def save_store(self, store_data: Dict[str, Any]) -> None:
        normalized = {
            "version": int(store_data.get("version", 1)),
            "cameras": store_data.get("cameras", []),
        }
        self.store_path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # --------------------------------------------------------
    # Records laden / speichern
    # --------------------------------------------------------

    def load_all_records(self) -> List[CameraCalibrationRecord]:
        store = self.load_store()
        raw_cameras = store.get("cameras", [])
        records: List[CameraCalibrationRecord] = []

        for idx, item in enumerate(raw_cameras):
            if not isinstance(item, dict):
                continue
            records.append(self._record_from_storage_dict(item, idx))

        return records

    def save_all_records(self, records: List[CameraCalibrationRecord]) -> None:
        payload = {
            "version": 1,
            "cameras": [record.to_storage_dict() for record in records],
        }
        self.save_store(payload)

    def load_record(self, camera_index: int) -> Optional[CameraCalibrationRecord]:
        records = self.load_all_records()
        for record in records:
            if int(record.camera_index) == int(camera_index):
                return record
        return None

    def save_record(self, record: CameraCalibrationRecord) -> None:
        records = self.load_all_records()
        updated = False

        for idx, existing in enumerate(records):
            if int(existing.camera_index) == int(record.camera_index):
                records[idx] = record
                updated = True
                break

        if not updated:
            records.append(record)

        records.sort(key=lambda r: int(r.camera_index))
        self.save_all_records(records)

    # --------------------------------------------------------
    # Synchronisierung mit bestehender App-Struktur
    # --------------------------------------------------------

    def build_records_from_app_configs(
        self,
        camera_config: Dict[str, Any],
        calibration_config: Dict[str, Any],
    ) -> List[CameraCalibrationRecord]:
        """
        Baut Records aus bestehender App-Struktur:
        camera_config["cameras"]
        calibration_config["cameras"]
        """
        cam_list = camera_config.get("cameras", [])
        cal_list = calibration_config.get("cameras", [])

        max_len = max(len(cam_list), len(cal_list), 1)
        records: List[CameraCalibrationRecord] = []

        for idx in range(max_len):
            cam_cfg = cam_list[idx] if idx < len(cam_list) and isinstance(cam_list[idx], dict) else {}
            cal_cfg = cal_list[idx] if idx < len(cal_list) and isinstance(cal_list[idx], dict) else {}

            record = self._default_camera_record(
                camera_index=idx,
                camera_config=cam_cfg,
                calibration_config=cal_cfg,
            )
            records.append(record)

        return records

    def apply_records_to_app_configs(
        self,
        records: List[CameraCalibrationRecord],
        base_camera_config: Optional[Dict[str, Any]] = None,
        base_calibration_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Schreibt Records zurück in die App-Strukturen.
        """
        camera_config = deepcopy(base_camera_config or {})
        calibration_config = deepcopy(base_calibration_config or {})

        camera_config["cameras"] = []
        calibration_config["cameras"] = []

        for record in sorted(records, key=lambda r: int(r.camera_index)):
            camera_config["cameras"].append(self.record_to_camera_config(record))
            calibration_config["cameras"].append(self.record_to_calibration_config(record))

        return camera_config, calibration_config

    def sync_from_app_configs(
        self,
        camera_config: Dict[str, Any],
        calibration_config: Dict[str, Any],
    ) -> List[CameraCalibrationRecord]:
        """
        Liest die aktuellen App-Daten und speichert sie direkt in die Store-Datei.
        """
        records = self.build_records_from_app_configs(camera_config, calibration_config)
        self.save_all_records(records)
        return records

    def load_into_app_configs(
        self,
        base_camera_config: Optional[Dict[str, Any]] = None,
        base_calibration_config: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Lädt die Store-Datei und baut daraus App-Configs.
        Falls keine Store-Datei existiert, werden die Basisdaten zurückgegeben.
        """
        records = self.load_all_records()
        if not records:
            return deepcopy(base_camera_config or {}), deepcopy(base_calibration_config or {})

        return self.apply_records_to_app_configs(
            records=records,
            base_camera_config=base_camera_config,
            base_calibration_config=base_calibration_config,
        )

    # --------------------------------------------------------
    # Referenzbildpfade
    # --------------------------------------------------------

    def get_reference_image_path(self, camera_index: int) -> Path:
        """
        Standardpfad für das leere Boardbild einer Kamera.
        """
        return self.reference_dir / self._camera_reference_filename(camera_index)

    def set_reference_image_path(self, camera_index: int, image_path: str | Path) -> None:
        """
        Aktualisiert nur den Referenzbildpfad eines bestehenden Records.
        """
        record = self.load_record(camera_index)
        if record is None:
            record = self._default_camera_record(camera_index)

        record.empty_board_reference_path = str(image_path)
        self.save_record(record)

    # --------------------------------------------------------
    # Komfortmethoden
    # --------------------------------------------------------

    def build_record_from_runtime(
        self,
        camera_index: int,
        camera_config: Dict[str, Any],
        calibration_config: Dict[str, Any],
    ) -> CameraCalibrationRecord:
        """
        Baut direkt aus einer aktiven Laufzeitkamera + Kalibrierung einen Record.
        """
        width = int(camera_config.get("width", calibration_config.get("frame_width", 1280)))
        height = int(camera_config.get("height", calibration_config.get("frame_height", 720)))

        raw_points = calibration_config.get("points", [])
        manual_points = self._normalize_manual_points(raw_points, width=width, height=height)

        reference_path = str(
            calibration_config.get(
                "empty_board_reference_path",
                self.get_reference_image_path(camera_index),
            )
        )

        return CameraCalibrationRecord(
            camera_index=int(camera_index),
            name=str(calibration_config.get("name", f"Kamera {camera_index + 1}")),
            enabled=bool(camera_config.get("enabled", True)),
            device_id=int(camera_config.get("device_id", -1)),
            width=width,
            height=height,
            fps=int(camera_config.get("fps", 30)),
            rotation=int(camera_config.get("rotation", 0)),
            flip=bool(camera_config.get("flip", False)),
            overlay_alpha=float(calibration_config.get("overlay_alpha", 0.35)),
            show_numbers=bool(calibration_config.get("show_numbers", True)),
            show_sector_lines=bool(calibration_config.get("show_sector_lines", True)),
            manual_points=manual_points,
            empty_board_reference_path=reference_path,
        )

    def update_single_camera_from_runtime(
        self,
        camera_index: int,
        camera_config: Dict[str, Any],
        calibration_config: Dict[str, Any],
    ) -> CameraCalibrationRecord:
        """
        Praktische Methode für 'jetzt genau diese Kamera speichern'.
        """
        record = self.build_record_from_runtime(
            camera_index=camera_index,
            camera_config=camera_config,
            calibration_config=calibration_config,
        )
        self.save_record(record)
        return record