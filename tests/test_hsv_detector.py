import math

import cv2  # type: ignore[import-not-found]
import numpy as np

from vision_pipeline.detectors.hsv_contour_detector import HsvContourDetector

_DETECTOR = HsvContourDetector(
    lower_hsv=(40, 80, 80),
    upper_hsv=(85, 255, 255),
    morphology_kernel=3,
    min_contour_area=50,
)


def test_hsv_detector_detects_green_circle() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 20, (0, 255, 0), thickness=-1)

    detections = _DETECTOR.detect(frame)
    assert len(detections) >= 1
    assert detections[0].area > 50


def test_center_offset_centered_marker() -> None:
    """画面中央にマーカーを配置したとき offset ≈ 0 になる。"""
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 20, (0, 255, 0), thickness=-1)

    detections = _DETECTOR.detect(frame)
    assert len(detections) >= 1
    d = detections[0]
    assert abs(d.center_offset_x) < 5.0
    assert abs(d.center_offset_y) < 5.0
    assert abs(d.norm_offset_x) < 0.2
    assert abs(d.norm_offset_y) < 0.2


def test_center_offset_off_center_marker() -> None:
    """画面右下にマーカーを配置したとき offset の符号が正しいことを確認。
    座標系: ボール基準（カメラ中央がボールより下→マイナス、上→プラス）
    """
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    radius = 20
    # 中央(160,120)より右下: (220, 160)
    cv2.circle(frame, (220, 160), radius, (0, 255, 0), thickness=-1)

    detections = _DETECTOR.detect(frame)
    assert len(detections) >= 1
    d = detections[0]
    assert d.center_offset_x > 0, "右側マーカーは center_offset_x > 0 のはず"
    assert d.center_offset_y > 0, "下側マーカーは center_offset_y > 0 のはず（カメラ中央がボールより上）"

    expected_diameter = 2.0 * math.sqrt(d.area / math.pi)
    assert abs(d.norm_offset_y - d.center_offset_y / expected_diameter) < 1e-6


def test_center_offset_sign_convention() -> None:
    """符号規約の確認（ボール基準座標系）。
    - ボールが画面上半分 → カメラ中央がボールより下 → center_offset_y < 0（マイナス）
    - ボールが画面下半分 → カメラ中央がボールより上 → center_offset_y > 0（プラス）
    """
    frame_h, frame_w = 240, 320

    # ボールを画面上半分（y=60）に配置: カメラ中央がボールより下 → マイナスになるはず
    frame_upper = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.circle(frame_upper, (160, 60), 20, (0, 255, 0), thickness=-1)
    dets_upper = _DETECTOR.detect(frame_upper)
    assert len(dets_upper) >= 1
    assert dets_upper[0].center_offset_y < 0, (
        f"上半分ボール: center_offset_y={dets_upper[0].center_offset_y:.1f} はマイナスになるはず"
    )

    # ボールを画面下半分（y=180）に配置: カメラ中央がボールより上 → プラスになるはず
    frame_lower = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.circle(frame_lower, (160, 180), 20, (0, 255, 0), thickness=-1)
    dets_lower = _DETECTOR.detect(frame_lower)
    assert len(dets_lower) >= 1
    assert dets_lower[0].center_offset_y > 0, (
        f"下半分ボール: center_offset_y={dets_lower[0].center_offset_y:.1f} はプラスになるはず"
    )


def test_min_diameter_px_filter() -> None:
    detector = HsvContourDetector(
        lower_hsv=(40, 80, 80),
        upper_hsv=(85, 255, 255),
        morphology_kernel=3,
        min_contour_area=10,
        min_diameter_px=50,
    )
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(frame, (160, 120), 10, (0, 255, 0), thickness=-1)

    detections = detector.detect(frame)
    assert len(detections) == 0
