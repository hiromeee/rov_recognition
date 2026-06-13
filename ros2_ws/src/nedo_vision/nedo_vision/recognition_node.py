from __future__ import annotations

import threading
from dataclasses import replace
from typing import Optional

import cv2  # type: ignore[import-not-found]
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from bluehacks_rov_interfaces.msg import RovPoseAcc
from sensor_msgs.msg import Image
from std_msgs.msg import Float32

from vision_pipeline.config import CameraConfig, load_config
from vision_pipeline.pipeline import VisionPipeline, draw_hud
from vision_pipeline.sources.mock_source import MockFrameSource

from nedo_vision.imu_utils import quaternion_to_pitch_roll


class RecognitionNode(Node):
    def __init__(self) -> None:
        super().__init__('recognition_node')

        self.declare_parameter('config_file', '')
        self.declare_parameter('imu_topic', '/mavlink_communicator/current_pose_acc')
        self.declare_parameter('camera_topic', '/camera_driver/camera_image')

        config_file: str = (
            self.get_parameter('config_file').get_parameter_value().string_value
        )
        imu_topic: str = (
            self.get_parameter('imu_topic').get_parameter_value().string_value
        )
        camera_topic: str = (
            self.get_parameter('camera_topic').get_parameter_value().string_value
        )

        if not config_file:
            raise RuntimeError("ROS2 parameter 'config_file' is required but not set")

        config = load_config(config_file)
        # VisionPipeline の初期化に source が必要だが、ROS2 ノードでは
        # process_frame() を直接呼ぶため MockFrameSource をダミーとして渡す
        dummy_source = MockFrameSource(
            width=config.source.width,
            height=config.source.height,
            fps=config.source.fps,
            frames_limit=None,
        )
        self._pipeline = VisionPipeline(config, dummy_source)
        self._config = config

        self._latest_pitch_deg: float = 0.0
        self._latest_roll_deg: float = 0.0
        self._imu_lock = threading.Lock()

        self._prev_norm_offset_x: Optional[float] = None
        self._frame_count: int = 0
        self._recent_latencies: list[float] = []
        self._display: bool = config.runtime.display

        self._latest_annotated: Optional[np.ndarray] = None
        self._annotated_lock = threading.Lock()

        self._bridge = CvBridge()

        cb_camera = MutuallyExclusiveCallbackGroup()
        cb_imu = MutuallyExclusiveCallbackGroup()

        self._pub = self.create_publisher(Float32, '~/difference', 10)
        self._debug_pub = self.create_publisher(Image, '~/debug_image', 1)

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            Image, camera_topic, self._image_callback, best_effort_qos, callback_group=cb_camera
        )
        self.create_subscription(
            RovPoseAcc, imu_topic, self._imu_callback, best_effort_qos, callback_group=cb_imu
        )

        self.get_logger().info(
            f'RecognitionNode started | config={config_file}'
            f' | camera={camera_topic} | imu={imu_topic}'
        )

    def _imu_callback(self, msg: RovPoseAcc) -> None:
        pitch_deg, roll_deg = quaternion_to_pitch_roll(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        with self._imu_lock:
            self._latest_pitch_deg = pitch_deg
            self._latest_roll_deg = roll_deg

    def _image_callback(self, msg: Image) -> None:
        try:
            frame: np.ndarray = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge conversion failed: {exc}')
            return

        camera_override: Optional[CameraConfig] = None
        if self._config.camera is not None:
            with self._imu_lock:
                pitch_deg = self._latest_pitch_deg
                roll_deg = self._latest_roll_deg
            camera_override = replace(
                self._config.camera,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
            )

        detections, aggregated, latency_ms = self._pipeline.process_frame(
            frame, camera_override
        )

        self._frame_count += 1
        self._recent_latencies.append(latency_ms)
        if len(self._recent_latencies) > 30:
            self._recent_latencies.pop(0)

        # --- アノテーション描画 ---
        annotated = self._pipeline.detector.annotate(frame, detections)
        if aggregated is not None:
            ax, ay, aw, ah = aggregated.bbox
            acx, acy = int(ax + aw / 2), int(ay + ah / 2)
            cv2.circle(annotated, (acx, acy), 12, (0, 255, 0), 3)
            fh, fw = annotated.shape[:2]
            cv2.line(annotated, (fw // 2, fh // 2), (acx, acy), (0, 255, 0), 2)

        recent_p50 = sorted(self._recent_latencies)[len(self._recent_latencies) // 2]
        vertical_offset_text = None
        if aggregated is not None:
            dy_px = aggregated.center_offset_y
            if aggregated.distance_y_m is not None:
                vertical_offset_text = (
                    f"Vertical: {aggregated.distance_y_m:+.2f} m ({dy_px:+.0f}px)"
                )
            else:
                vertical_offset_text = f"Vertical: {dy_px:+.0f} px"
        annotated = draw_hud(
            annotated, self._frame_count, len(detections),
            latency_ms, recent_p50, vertical_offset_text,
        )

        # debug_image トピックに publish（rqt_image_view で確認可能）
        debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        debug_msg.header = msg.header
        self._debug_pub.publish(debug_msg)

        # ローカルウィンドウ用に最新フレームを保存（表示はメインスレッドで行う）
        if self._display:
            with self._annotated_lock:
                self._latest_annotated = annotated

        # --- ズレ計算・publish ---
        if aggregated is None:
            self._prev_norm_offset_x = None
            return

        curr_norm_x = aggregated.norm_offset_x
        prev = self._prev_norm_offset_x
        self._prev_norm_offset_x = curr_norm_x

        # 符号反転 = マーカーが画面鉛直中心線を通過した瞬間
        if prev is not None and prev * curr_norm_x < 0.0:
            value = (
                aggregated.distance_y_m
                if aggregated.distance_y_m is not None
                else float(aggregated.center_offset_y)
            )
            out = Float32()
            out.data = float(value)
            self._pub.publish(out)
            self.get_logger().info(f'difference published: {value:+.4f} m')


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RecognitionNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    # executor.spin() をサブスレッドで動かし、メインスレッドを GUI ループに使う。
    # OpenCV の imshow/waitKey はメインスレッドからしか安定して呼べないため。
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if node._display:
            cv2.namedWindow('recognition_node', cv2.WINDOW_NORMAL)
            while rclpy.ok():
                with node._annotated_lock:
                    frame = node._latest_annotated
                if frame is not None:
                    cv2.imshow('recognition_node', frame)
                key = cv2.waitKey(33) & 0xFF  # ~30 fps でポーリング
                if key == ord('q'):
                    break
        else:
            spin_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
