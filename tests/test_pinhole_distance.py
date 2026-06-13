import math

import cv2  # type: ignore[import-not-found]
import numpy as np

from vision_pipeline.config import (
    CameraConfig,
    HsvConfig,
    PipelineConfig,
    RuntimeConfig,
    SourceConfig,
)
from vision_pipeline.pipeline import VisionPipeline
from vision_pipeline.sources.mock_source import MockFrameSource


def test_pinhole_distance_from_marker_size() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    radius = 20
    center = (160, 140)
    cv2.circle(frame, center, radius, (0, 255, 0), thickness=-1)

    config = PipelineConfig(
        source=SourceConfig(kind="mock", width=320, height=240, fps=30),
        runtime=RuntimeConfig(display=False, log_interval_frames=0),
        hsv=HsvConfig(
            lower=(40, 80, 80),
            upper=(85, 255, 255),
            morphology_kernel=3,
            min_contour_area=50,
        ),
        camera=CameraConfig(
            fov_x_deg=90.0,
            fov_y_deg=90.0,
            altitude_m=1.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            marker_diameter_m=0.07,
        ),
    )
    source = MockFrameSource(width=320, height=240, fps=30, frames_limit=1)
    pipeline = VisionPipeline(config=config, source=source)

    detections, _, _ = pipeline.process_frame(frame)
    assert detections

    d = detections[0]
    assert d.distance_y_m is not None
    # center=(160,140) は frame_cy=120 より下 → center_offset_y > 0 → distance_y_m も正
    assert d.distance_y_m > 0.0

    f_y = (240 / 2.0) / math.tan(math.radians(90.0) / 2.0)
    depth_m = f_y * 0.07 / d.diameter_px
    expected_y = (d.center_offset_y / f_y) * depth_m

    assert abs(d.distance_y_m - expected_y) < 1e-3
