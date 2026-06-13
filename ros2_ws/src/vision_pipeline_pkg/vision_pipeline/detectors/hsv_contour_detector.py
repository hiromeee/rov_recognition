from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2  # type: ignore[import-not-found]
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    area: float
    circularity: float = 0.0
    diameter_px: float = 0.0
    # 画面中央からのピクセルずれ（右/下が正）
    center_offset_x: float = 0.0
    center_offset_y: float = 0.0
    # マーカー直径で正規化したずれ（採点基準に対応）
    norm_offset_x: float = 0.0
    norm_offset_y: float = 0.0
    # 機体重心（カメラ直下）からの物理的距離（m）
    distance_x_m: Optional[float] = None
    distance_y_m: Optional[float] = None
    distance_z_m: Optional[float] = None



class HsvContourDetector:
    def __init__(
        self,
        lower_hsv: Tuple[int, int, int],
        upper_hsv: Tuple[int, int, int],
        morphology_kernel: int,
        min_contour_area: int,
        min_diameter_px: float = 0.0,
        min_circularity: float = 0.0,
        max_aspect_ratio: float = 10.0,
        border_margin: int = 0,
    ) -> None:
        self.lower = np.array(lower_hsv, dtype=np.uint8)
        self.upper = np.array(upper_hsv, dtype=np.uint8)
        self.kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
        self.min_contour_area = float(min_contour_area)
        self.min_diameter_px = float(min_diameter_px)
        self.min_circularity = min_circularity
        self.max_aspect_ratio = max_aspect_ratio
        self.border_margin = border_margin

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_cx = frame_w / 2.0
        frame_cy = frame_h / 2.0

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        detections: List[Detection] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4 * math.pi * area / (perimeter * perimeter)
                if perimeter > 0
                else 0.0
            )
            if circularity < self.min_circularity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            m = self.border_margin
            if m > 0 and (x <= m or y <= m or x + w >= frame_w - m or y + h >= frame_h - m):
                continue
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > self.max_aspect_ratio:
                continue

            # バウンディングボックス中心ではなく輪郭の重心（moments）を使用する。
            # バウンディングボックス中心はボール+紐が連結した場合にパイプ側へずれる。
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0

            offset_x = cx - frame_cx
            # ボール基準座標系: カメラ中央がボールより下→マイナス、上→プラス（画像 y 軸と同方向）
            offset_y = cy - frame_cy
            # 真円近似による直径（マーカー直径で正規化: 採点基準に対応）
            diameter = 2.0 * math.sqrt(area / math.pi)
            if diameter < self.min_diameter_px:
                continue
            norm_x = offset_x / diameter if diameter > 0 else 0.0
            norm_y = offset_y / diameter if diameter > 0 else 0.0

            detections.append(Detection(
                bbox=(x, y, w, h),
                area=area,
                circularity=circularity,
                diameter_px=diameter,
                center_offset_x=offset_x,
                center_offset_y=offset_y,
                norm_offset_x=norm_x,
                norm_offset_y=norm_y,
            ))

        return detections

    def annotate(
        self,
        frame_bgr: np.ndarray,
        detections: List[Detection],
    ) -> np.ndarray:
        output = frame_bgr.copy()
        fh, fw = output.shape[:2]
        fcx, fcy = fw // 2, fh // 2

        # 画面中心十字線
        cv2.line(output, (fcx, 0), (fcx, fh), (0, 200, 0), 1, cv2.LINE_AA)
        cv2.line(output, (0, fcy), (fw, fcy), (0, 200, 0), 1, cv2.LINE_AA)

        for det in detections:
            x, y, w, h = det.bbox
            # 描画にも重心（center_offset）を使用してバウンディングボックス中心との差を可視化
            cx = int(fcx + det.center_offset_x)
            cy = int(fcy + det.center_offset_y)

            # Bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

            # Center crosshair（重心）
            cv2.drawMarker(
                output, (cx, cy), (0, 0, 255),
                cv2.MARKER_CROSS, markerSize=16, thickness=2,
            )

            # 垂直ずれ線（ボール重心 → 水平中心線への垂直線）
            cv2.line(output, (cx, fcy), (cx, cy), (255, 100, 0), 2, cv2.LINE_AA)

            # 座標 + 真円度 + 垂直ずれラベル
            label = f"circ={det.circularity:.2f} dy={det.center_offset_y:+.0f}px norm_y={det.norm_offset_y:+.2f}"
            label_y = y - 6 if y > 20 else y + h + 16
            cv2.putText(
                output, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                output, label, (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return output
