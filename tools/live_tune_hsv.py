#!/usr/bin/env python3
"""ライブHSVチューナー。

ROS2 カメラトピックを購読し、OpenCV ウィンドウにライブ表示する。
SPACE でフレーム凍結後、左クリックでマーカー・右クリックで背景をサンプリング、
'A' で提案HSV範囲を自動算出、トラックバーで微調整、'S' でYAML保存。

実行例:
    source /opt/ros/jazzy/setup.bash
    python3 tools/live_tune_hsv.py --topic /camera_driver/camera_image \
        --config configs/camera_hsv_baseline.yaml
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
from vision_pipeline.detectors.hsv_contour_detector import HsvContourDetector  # noqa: E402

SAMPLE_HALF = 3  # クリック中心 ±N ピクセルでHSVサンプリング


@dataclass
class TunerState:
    marker_samples: List[Tuple[int, int, int]] = field(default_factory=list)
    bg_samples: List[Tuple[int, int, int]] = field(default_factory=list)
    marker_clicks: List[Tuple[int, int]] = field(default_factory=list)
    bg_clicks: List[Tuple[int, int]] = field(default_factory=list)
    frozen_frame: Optional[np.ndarray] = None
    frozen_count: int = -1


class LiveTuneHsvNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__('live_tune_hsv')
        self._bridge = CvBridge()
        self._latest: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._frame_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Image, topic, self._on_image, qos)
        self.get_logger().info(f'Subscribed to {topic} (BEST_EFFORT)')

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return
        with self._lock:
            self._latest = frame
            self._frame_count += 1

    def get_latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._latest is None:
                return None, self._frame_count
            return self._latest.copy(), self._frame_count


def sample_patch(frame: np.ndarray, x: int, y: int) -> List[Tuple[int, int, int]]:
    h, w = frame.shape[:2]
    x0, x1 = max(0, x - SAMPLE_HALF), min(w, x + SAMPLE_HALF + 1)
    y0, y1 = max(0, y - SAMPLE_HALF), min(h, y + SAMPLE_HALF + 1)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return [tuple(int(c) for c in px) for px in hsv.reshape(-1, 3)]


def on_mouse(event: int, x: int, y: int, flags: int, state: TunerState) -> None:
    if state.frozen_frame is None:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        state.marker_clicks.append((x, y))
        state.marker_samples.extend(sample_patch(state.frozen_frame, x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        state.bg_clicks.append((x, y))
        state.bg_samples.extend(sample_patch(state.frozen_frame, x, y))


def compute_proposed_hsv(
    state: TunerState,
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    if not state.marker_samples:
        return None
    arr = np.array(state.marker_samples)
    h_lo = max(0, int(np.percentile(arr[:, 0], 5)) - 5)
    h_hi = min(179, int(np.percentile(arr[:, 0], 95)) + 5)
    s_lo = max(0, int(np.percentile(arr[:, 1], 5)) - 10)
    v_lo = max(0, int(np.percentile(arr[:, 2], 5)) - 10)

    if state.bg_samples:
        bg = np.array(state.bg_samples)
        marker_h_med = int(np.median(arr[:, 0]))
        bg_h_med = int(np.median(bg[:, 0]))
        # 背景H中央値がマーカーH中央値より十分高い/低ければ、H範囲を背景から離す
        if bg_h_med > marker_h_med + 10:
            h_hi = min(h_hi, int(np.percentile(bg[:, 0], 5)) - 2)
        elif bg_h_med < marker_h_med - 10:
            h_lo = max(h_lo, int(np.percentile(bg[:, 0], 95)) + 2)
        # マーカーが背景より明るければ V_lo を引き上げて背景を除外
        marker_v_p5 = int(np.percentile(arr[:, 2], 5))
        bg_v_p95 = int(np.percentile(bg[:, 2], 95))
        if marker_v_p5 > bg_v_p95:
            v_lo = max(v_lo, (marker_v_p5 + bg_v_p95) // 2)

    h_lo = max(0, min(179, h_lo))
    h_hi = max(h_lo, min(179, h_hi))
    return ((h_lo, s_lo, v_lo), (h_hi, 255, 255))


def make_detector(values: Dict[str, int]) -> HsvContourDetector:
    return HsvContourDetector(
        lower_hsv=(values['H_lo'], values['S_lo'], values['V_lo']),
        upper_hsv=(values['H_hi'], values['S_hi'], values['V_hi']),
        morphology_kernel=max(1, values['morph']),
        min_contour_area=values['min_area'],
        min_diameter_px=10.0,
        min_circularity=values['min_circ'] / 100.0,
    )


def render_display(
    frame: np.ndarray,
    state: TunerState,
    is_live: bool,
    rx_count: int,
    detector: HsvContourDetector,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, detector.lower, detector.upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, detector.kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, detector.kernel)
    detections = detector.detect(frame)

    out = frame.copy()
    # マスク領域を緑に半透明オーバーレイ
    tint = np.zeros_like(out)
    tint[mask > 0] = (0, 200, 0)
    out = cv2.addWeighted(out, 0.6, tint, 0.4, 0)
    # 元のフレーム外のオーバーレイを抑制（マスク==0 の領域は元映像のまま）
    out[mask == 0] = frame[mask == 0]

    for det in detections:
        x, y, w, h = det.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label = f'circ={det.circularity:.2f} d={det.diameter_px:.0f}'
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    for cx, cy in state.marker_clicks:
        cv2.circle(out, (cx, cy), 5, (0, 255, 0), -1)
        cv2.circle(out, (cx, cy), 5, (0, 0, 0), 1)
    for cx, cy in state.bg_clicks:
        cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(out, (cx, cy), 5, (0, 0, 0), 1)

    status = 'LIVE' if is_live else 'FROZEN'
    color = (0, 255, 0) if is_live else (0, 0, 255)
    lines = [
        f'{status} | rx={rx_count} | det={len(detections)}'
        f' | marker={len(state.marker_clicks)} bg={len(state.bg_clicks)}',
        'SPACE=freeze  L/R-click=sample marker/bg  A=auto-range  C=clear  S=save  ESC=quit',
    ]
    y_text = 25
    for line in lines:
        cv2.putText(out, line, (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y_text += 22

    return out


def save_hsv_to_yaml(config_path: Path, values: Dict[str, int]) -> None:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    text = config_path.read_text(encoding='utf-8')

    new_block = (
        "hsv:\n"
        f"  lower: [{values['H_lo']}, {values['S_lo']}, {values['V_lo']}]\n"
        f"  upper: [{values['H_hi']}, {values['S_hi']}, {values['V_hi']}]\n"
        f"  morphology_kernel: {max(1, values['morph'])}\n"
        f"  min_contour_area: {values['min_area']}\n"
        f"  min_diameter_px: 10\n"
        f"  min_circularity: {values['min_circ'] / 100.0:.2f}\n"
    )

    pattern = re.compile(r'^hsv:\n(?:[ \t].*\n?|\n)*', re.MULTILINE)
    if pattern.search(text):
        new_text = pattern.sub(new_block + '\n', text, count=1)
    else:
        new_text = text.rstrip() + '\n\n' + new_block
    config_path.write_text(new_text, encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Live HSV tuner')
    parser.add_argument('--topic', default='/camera_driver/camera_image')
    parser.add_argument(
        '--config',
        default=str(REPO_ROOT / 'configs' / 'camera_hsv_baseline.yaml'),
        help='S キー押下時の保存先 YAML',
    )
    args = parser.parse_args()

    rclpy.init()
    node = LiveTuneHsvNode(args.topic)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    state = TunerState()
    window = 'live_tune_hsv'
    controls = 'controls'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(controls, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(controls, 420, 320)

    def _nop(_):
        pass

    cv2.createTrackbar('H_lo', controls, 30, 179, _nop)
    cv2.createTrackbar('H_hi', controls, 85, 179, _nop)
    cv2.createTrackbar('S_lo', controls, 0, 255, _nop)
    cv2.createTrackbar('S_hi', controls, 255, 255, _nop)
    cv2.createTrackbar('V_lo', controls, 140, 255, _nop)
    cv2.createTrackbar('V_hi', controls, 255, 255, _nop)
    cv2.createTrackbar('morph', controls, 3, 9, _nop)
    cv2.createTrackbar('min_area', controls, 100, 1000, _nop)
    cv2.createTrackbar('min_circ', controls, 25, 100, _nop)

    cv2.setMouseCallback(window, on_mouse, state)
    config_path = Path(args.config)

    try:
        while rclpy.ok():
            values = {
                'H_lo': cv2.getTrackbarPos('H_lo', controls),
                'H_hi': cv2.getTrackbarPos('H_hi', controls),
                'S_lo': cv2.getTrackbarPos('S_lo', controls),
                'S_hi': cv2.getTrackbarPos('S_hi', controls),
                'V_lo': cv2.getTrackbarPos('V_lo', controls),
                'V_hi': cv2.getTrackbarPos('V_hi', controls),
                'morph': cv2.getTrackbarPos('morph', controls),
                'min_area': cv2.getTrackbarPos('min_area', controls),
                'min_circ': cv2.getTrackbarPos('min_circ', controls),
            }
            detector = make_detector(values)

            if state.frozen_frame is None:
                frame, count = node.get_latest()
                is_live = True
            else:
                frame = state.frozen_frame
                count = state.frozen_count
                is_live = False

            if frame is not None:
                cv2.imshow(window, render_display(frame, state, is_live, count, detector))
            else:
                placeholder = np.zeros((240, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, 'Waiting for frames...', (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
                cv2.imshow(window, placeholder)

            key = cv2.waitKey(33) & 0xFF
            if key == 27:
                break
            if key == ord(' '):
                if state.frozen_frame is None:
                    fr, cnt = node.get_latest()
                    if fr is not None:
                        state.frozen_frame = fr
                        state.frozen_count = cnt
                        node.get_logger().info(f'Frozen at rx_frame #{cnt}')
                else:
                    state.frozen_frame = None
                    node.get_logger().info('Resumed live view')
            elif key == ord('c'):
                state.marker_samples.clear()
                state.bg_samples.clear()
                state.marker_clicks.clear()
                state.bg_clicks.clear()
                node.get_logger().info('Cleared all samples')
            elif key == ord('a'):
                proposed = compute_proposed_hsv(state)
                if proposed is None:
                    node.get_logger().warn('No marker samples; left-click markers first')
                else:
                    lo, hi = proposed
                    cv2.setTrackbarPos('H_lo', controls, lo[0])
                    cv2.setTrackbarPos('S_lo', controls, lo[1])
                    cv2.setTrackbarPos('V_lo', controls, lo[2])
                    cv2.setTrackbarPos('H_hi', controls, hi[0])
                    cv2.setTrackbarPos('S_hi', controls, hi[1])
                    cv2.setTrackbarPos('V_hi', controls, hi[2])
                    node.get_logger().info(f'Auto-set HSV: lower={lo} upper={hi}')
            elif key == ord('s'):
                try:
                    save_hsv_to_yaml(config_path, values)
                    node.get_logger().info(f'Saved HSV block to {config_path}')
                except Exception as exc:
                    node.get_logger().error(f'Save failed: {exc}')
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
