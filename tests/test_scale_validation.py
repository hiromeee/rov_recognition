from vision_pipeline.detectors.hsv_contour_detector import Detection
from vision_pipeline.scale_validation import compute_scale_check


def test_scale_validation_median_spacing() -> None:
    detections = []
    for i in range(4):
        detections.append(
            Detection(
                bbox=(0, 0, 0, 0),
                area=100.0,
                diameter_px=20.0,
                distance_x_m=i * 0.5,
                distance_y_m=0.0,
                distance_z_m=2.0,
            )
        )

    check = compute_scale_check(detections, expected_spacing_m=0.5)
    assert check is not None
    assert abs(check.median_spacing_m - 0.5) < 1e-6
    assert abs(check.scale_ratio - 1.0) < 1e-6
