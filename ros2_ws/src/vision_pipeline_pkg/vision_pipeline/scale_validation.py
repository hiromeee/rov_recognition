from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from vision_pipeline.detectors.hsv_contour_detector import Detection


@dataclass
class ScaleCheck:
    expected_spacing_m: float
    median_spacing_m: float
    scale_ratio: float
    rms_error_m: float
    num_markers: int
    num_pairs: int


def compute_scale_check(
    detections: List[Detection],
    expected_spacing_m: float,
) -> Optional[ScaleCheck]:
    if expected_spacing_m <= 0:
        return None

    points = [
        [d.distance_x_m, d.distance_y_m, d.distance_z_m]
        for d in detections
        if d.distance_x_m is not None
        and d.distance_y_m is not None
        and d.distance_z_m is not None
    ]
    if len(points) < 2:
        return None

    p = np.array(points, dtype=np.float64)
    p0 = p - p.mean(axis=0, keepdims=True)

    _, _, vh = np.linalg.svd(p0, full_matrices=False)
    axis = vh[0]

    t = p0 @ axis
    t_sorted = np.sort(t)
    spacings = np.diff(t_sorted)
    if spacings.size == 0:
        return None

    median_spacing = float(np.median(spacings))
    errors = spacings - expected_spacing_m
    rms_error = float(np.sqrt(np.mean(errors ** 2)))
    scale_ratio = median_spacing / expected_spacing_m

    return ScaleCheck(
        expected_spacing_m=expected_spacing_m,
        median_spacing_m=median_spacing,
        scale_ratio=scale_ratio,
        rms_error_m=rms_error,
        num_markers=len(points),
        num_pairs=int(spacings.size),
    )
