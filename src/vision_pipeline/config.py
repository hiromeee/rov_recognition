from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class CameraConfig:
    fov_x_deg: float
    fov_y_deg: float
    altitude_m: float
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    marker_diameter_m: float = 0.07
    marker_spacing_m: float = 0.5
    # undistort 用キャリブレーションファイルパス（省略時は補正なし）
    calibration_file: Optional[str] = None


@dataclass
class SourceConfig:
    kind: str
    width: int
    height: int
    fps: int
    frames_limit: Optional[int] = None
    video_path: Optional[str] = None
    camera_index: int = 0


@dataclass
class HsvConfig:
    lower: Tuple[int, int, int]
    upper: Tuple[int, int, int]
    morphology_kernel: int
    min_contour_area: int
    min_diameter_px: float = 0.0
    min_circularity: float = 0.0
    border_margin: int = 0


@dataclass
class RuntimeConfig:
    display: bool
    log_interval_frames: int
    aggregation_strategy: str = "average"
    outlier_sigma: float = 0.0
    outlier_min_samples: int = 3


@dataclass
class PipelineConfig:
    source: SourceConfig
    runtime: RuntimeConfig
    hsv: Optional[HsvConfig] = None
    camera: Optional[CameraConfig] = None


def _as_tuple3(value: Any) -> Tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("Expected a list with 3 elements")
    return int(value[0]), int(value[1]), int(value[2])


def load_config(config_path: str) -> PipelineConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as fp:
        raw: Dict[str, Any] = yaml.safe_load(fp)

    source_raw = raw["source"]
    runtime_raw = raw["runtime"]

    source = SourceConfig(
        kind=str(source_raw["kind"]),
        width=int(source_raw["width"]),
        height=int(source_raw["height"]),
        fps=int(source_raw["fps"]),
        frames_limit=(
            int(source_raw["frames_limit"])
            if source_raw.get("frames_limit") is not None
            else None
        ),
        video_path=source_raw.get("video_path"),
        camera_index=int(source_raw.get("camera_index", 0)),
    )

    runtime = RuntimeConfig(
        display=bool(runtime_raw.get("display", False)),
        log_interval_frames=int(runtime_raw.get("log_interval_frames", 30)),
        aggregation_strategy=str(runtime_raw.get("aggregation_strategy", "average")),
        outlier_sigma=float(runtime_raw.get("outlier_sigma", 0.0)),
        outlier_min_samples=int(runtime_raw.get("outlier_min_samples", 3)),
    )

    hsv: Optional[HsvConfig] = None
    if "hsv" in raw:
        hsv_raw = raw["hsv"]
        hsv = HsvConfig(
            lower=_as_tuple3(hsv_raw["lower"]),
            upper=_as_tuple3(hsv_raw["upper"]),
            morphology_kernel=int(hsv_raw["morphology_kernel"]),
            min_contour_area=int(hsv_raw["min_contour_area"]),
            min_diameter_px=float(hsv_raw.get("min_diameter_px", 0.0)),
            min_circularity=float(hsv_raw.get("min_circularity", 0.0)),
            border_margin=int(hsv_raw.get("border_margin", 0)),
        )

    camera: Optional[CameraConfig] = None
    if "camera" in raw:
        cam_raw = raw["camera"]
        camera = CameraConfig(
            fov_x_deg=float(cam_raw.get("fov_x_deg", 80.0)),
            fov_y_deg=float(cam_raw.get("fov_y_deg", 60.0)),
            altitude_m=float(cam_raw.get("altitude_m", 1.0)),
            pitch_deg=float(cam_raw.get("pitch_deg", 0.0)),
            roll_deg=float(cam_raw.get("roll_deg", 0.0)),
            marker_diameter_m=float(cam_raw.get("marker_diameter_m", 0.07)),
            marker_spacing_m=float(cam_raw.get("marker_spacing_m", 0.5)),
            calibration_file=cam_raw.get("calibration_file"),
        )

    return PipelineConfig(
        source=source, runtime=runtime, hsv=hsv, camera=camera
    )
