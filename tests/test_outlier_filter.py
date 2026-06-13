from vision_pipeline.detectors.hsv_contour_detector import Detection
from vision_pipeline.pipeline import _filter_depth_outliers


def _det(z: float) -> Detection:
    return Detection(
        bbox=(0, 0, 0, 0),
        area=1.0,
        diameter_px=10.0,
        center_offset_x=0.0,
        center_offset_y=0.0,
        norm_offset_x=0.0,
        norm_offset_y=0.0,
        distance_x_m=0.0,
        distance_y_m=0.0,
        distance_z_m=z,
    )


def test_filter_depth_outliers_mad() -> None:
    detections = [_det(2.0), _det(2.1), _det(10.0)]
    filtered = _filter_depth_outliers(detections, outlier_sigma=3.0, min_samples=3)
    assert len(filtered) == 2
    assert all(d.distance_z_m is not None and d.distance_z_m < 3.0 for d in filtered)
