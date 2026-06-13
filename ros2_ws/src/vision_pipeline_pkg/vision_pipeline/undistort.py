"""カメラ歪み補正モジュール。

キャリブレーション YAML を読み込み、undistort マップおよびカメラ行列を提供する。
VisionPipeline.__init__ で初期化し、process_frame の冒頭で apply() を呼ぶ。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2  # type: ignore[import-not-found]
import numpy as np
import yaml


@dataclass
class CalibrationData:
    """キャリブレーションファイルから読み込んだパラメータ。"""
    fx: float
    fy: float
    cx: float
    cy: float
    undistort_map: Optional["UndistortMap"]


class UndistortMap:
    """事前計算済みの undistort リマップ。"""

    def __init__(self, map1: np.ndarray, map2: np.ndarray) -> None:
        self._map1 = map1
        self._map2 = map2

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)


def load_calibration(
    calibration_file: str,
    image_width: int,
    image_height: int,
) -> Optional[CalibrationData]:
    """calibration_file からカメラ行列と undistort マップを読み込む。

    ファイルが存在しない場合は None を返す。
    歪み係数がすべてゼロの場合は undistort_map=None のまま fx/fy のみ返す。
    """
    path = Path(calibration_file)
    if not path.exists():
        print(f"[WARN] calibration_file not found: {path} — キャリブレーション無効")
        return None

    with path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    cm = raw.get("camera_matrix", {})
    dc = raw.get("dist_coeffs", {})

    fx = float(cm.get("fx", image_width / 2))
    fy = float(cm.get("fy", image_height / 2))
    cx = float(cm.get("cx", image_width / 2))
    cy = float(cm.get("cy", image_height / 2))

    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.array(
        [
            float(dc.get("k1", 0)),
            float(dc.get("k2", 0)),
            float(dc.get("p1", 0)),
            float(dc.get("p2", 0)),
            float(dc.get("k3", 0)),
        ],
        dtype=np.float64,
    )

    undistort_map: Optional[UndistortMap] = None
    if not np.all(dist_coeffs == 0):
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            (image_width, image_height),
            alpha=0,
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist_coeffs,
            np.eye(3),  # rectification transform = identity (no stereo rectification)
            new_camera_matrix,
            (image_width, image_height),
            cv2.CV_16SC2,
        )
        undistort_map = UndistortMap(map1, map2)
        # 補正後の主点・焦点距離を更新
        fx = float(new_camera_matrix[0, 0])
        fy = float(new_camera_matrix[1, 1])
        cx = float(new_camera_matrix[0, 2])
        cy = float(new_camera_matrix[1, 2])

    return CalibrationData(fx=fx, fy=fy, cx=cx, cy=cy, undistort_map=undistort_map)


def load_undistort_map(
    calibration_file: str,
    image_width: int,
    image_height: int,
) -> Optional[UndistortMap]:
    """後方互換のラッパー。undistort マップのみ返す。"""
    result = load_calibration(calibration_file, image_width, image_height)
    return result.undistort_map if result is not None else None
