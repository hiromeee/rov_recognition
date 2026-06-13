from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2  # type: ignore[import-not-found]
import numpy as np

from vision_pipeline.config import PipelineConfig, CameraConfig
from vision_pipeline.detectors.hsv_contour_detector import Detection, HsvContourDetector
from vision_pipeline.metrics import FrameMetrics, elapsed_ms, start_timer
from vision_pipeline.scale_validation import compute_scale_check
from vision_pipeline.sources.base import FrameSource
from vision_pipeline.undistort import CalibrationData, UndistortMap, load_calibration


def aggregate_detections(detections: List[Detection], strategy: str) -> Optional[Detection]:
    """複数認識されたターゲットを1つの出力に集約する。
    
    ROS2ノードでの単一float出力に向けて、複数のターゲットが見つかった場合の
    代表値を決定する。
    """
    if not detections:
        return None
    if len(detections) == 1:
        return detections[0]
        
    if strategy == "nearest_center":
        # 採点基準: 画面を左右2等分する縦線に最も近いマーカーを選択
        return min(detections, key=lambda d: abs(d.center_offset_x))
    elif strategy == "leftmost":
        return min(detections, key=lambda d: d.bbox[0])
    elif strategy == "rightmost":
        return max(detections, key=lambda d: d.bbox[0])
    elif strategy == "largest":
        return max(detections, key=lambda d: d.area)
    else:  # "average"
        avg_dx = sum(d.center_offset_x for d in detections) / len(detections)
        avg_dy = sum(d.center_offset_y for d in detections) / len(detections)
        avg_norm_x = sum(d.norm_offset_x for d in detections) / len(detections)
        avg_norm_y = sum(d.norm_offset_y for d in detections) / len(detections)
        
        avg_bbox = (
            int(sum(d.bbox[0] for d in detections) / len(detections)),
            int(sum(d.bbox[1] for d in detections) / len(detections)),
            int(sum(d.bbox[2] for d in detections) / len(detections)),
            int(sum(d.bbox[3] for d in detections) / len(detections)),
        )
        avg_area = sum(d.area for d in detections) / len(detections)
        avg_diameter = sum(d.diameter_px for d in detections) / len(detections)
        
        avg_dx_m = None
        avg_dy_m = None
        avg_dz_m = None
        if all(d.distance_x_m is not None for d in detections):
            avg_dx_m = sum(d.distance_x_m for d in detections) / len(detections)  # type: ignore
            avg_dy_m = sum(d.distance_y_m for d in detections) / len(detections)  # type: ignore
            avg_dz_m = sum(d.distance_z_m for d in detections) / len(detections)  # type: ignore
        
        return Detection(
            bbox=avg_bbox,
            area=avg_area,
            diameter_px=avg_diameter,
            center_offset_x=avg_dx,
            center_offset_y=avg_dy,
            norm_offset_x=avg_norm_x,
            norm_offset_y=avg_norm_y,
            distance_x_m=avg_dx_m,
            distance_y_m=avg_dy_m,
            distance_z_m=avg_dz_m,
        )


def _filter_depth_outliers(
    detections: List[Detection],
    outlier_sigma: float,
    min_samples: int,
) -> List[Detection]:
    if outlier_sigma <= 0:
        return detections

    valid = [d for d in detections if d.distance_z_m is not None]
    if len(valid) < max(min_samples, 2):
        return detections

    values = np.array([d.distance_z_m for d in valid], dtype=np.float64)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad <= 1e-6:
        return detections

    threshold = outlier_sigma * 1.4826 * mad
    filtered = [
        d
        for d in valid
        if d.distance_z_m is not None and abs(d.distance_z_m - median) <= threshold
    ]
    return filtered if filtered else detections


def draw_hud(
    frame: np.ndarray,
    frame_num: int,
    n_detections: int,
    latency_ms: float,
    recent_p50_ms: float,
    vertical_offset_text: str | None = None,
) -> np.ndarray:
    """検出結果のテレメトリをフレーム左上に重畳する。"""
    out = frame.copy()
    lines = [
        f"Frame   : {frame_num}",
        f"Detect  : {n_detections}",
        f"Latency : {latency_ms:.1f} ms",
        f"P50     : {recent_p50_ms:.1f} ms",
    ]
    if vertical_offset_text:
        lines.append(vertical_offset_text)
    for i, text in enumerate(lines):
        y = 22 + i * 24
        cv2.putText(
            out, text, (9, y + 1),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            out, text, (9, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


@dataclass
class PipelineResult:
    metrics: FrameMetrics
    total_detections: int


@dataclass
class OutputSettings:
    annotated_dir: Path | None = None
    annotated_video: Path | None = None
    save_interval: int = 1
    video_fps: int = 30


class VisionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        source: FrameSource,
        output: OutputSettings | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.output = output

        if config.hsv is None:
            raise ValueError("config requires a [hsv] section")
        self.detector = HsvContourDetector(
            lower_hsv=config.hsv.lower,
            upper_hsv=config.hsv.upper,
            morphology_kernel=config.hsv.morphology_kernel,
            min_contour_area=config.hsv.min_contour_area,
            min_diameter_px=config.hsv.min_diameter_px,
            min_circularity=config.hsv.min_circularity,
            border_margin=config.hsv.border_margin,
        )

        self._calibration: CalibrationData | None = None
        if config.camera is not None and config.camera.calibration_file is not None:
            self._calibration = load_calibration(
                config.camera.calibration_file,
                image_width=config.source.width,
                image_height=config.source.height,
            )

    def process_frame(
        self,
        frame: np.ndarray,
        camera_override: Optional[CameraConfig] = None,
    ) -> Tuple[List[Detection], Optional[Detection], float]:
        """単一フレームに対する検出と物理距離算出、集約を行う"""
        from vision_pipeline.metrics import elapsed_ms, start_timer
        if self._calibration is not None and self._calibration.undistort_map is not None:
            frame = self._calibration.undistort_map.apply(frame)
        t0 = start_timer()
        detections = self.detector.detect(frame)
        detect_ms = elapsed_ms(t0)

        cam_config = camera_override if camera_override is not None else self.config.camera
        if cam_config is not None and detections:
            h, w = frame.shape[:2]
            # キャリブファイルがある場合は実測 fx/fy を優先、なければ FOV から計算
            # IMU 姿勢上書き（camera_override）と焦点距離は独立 — override 時もキャリブ値を使う
            if self._calibration is not None:
                f_x = self._calibration.fx
                f_y = self._calibration.fy
            else:
                fov_x_rad = math.radians(cam_config.fov_x_deg)
                fov_y_rad = math.radians(cam_config.fov_y_deg)
                f_x = (w / 2.0) / math.tan(fov_x_rad / 2.0) if fov_x_rad > 0 else 1.0
                f_y = (h / 2.0) / math.tan(fov_y_rad / 2.0) if fov_y_rad > 0 else 1.0
            
            pitch_rad = math.radians(cam_config.pitch_deg)
            roll_rad = math.radians(cam_config.roll_deg)
            
            # 回転行列の計算（カメラ座標系を基準とした簡易的な姿勢補正）
            # imu_utils: roll=x軸回り, pitch=y軸回り に対応させる
            cr, sr = math.cos(roll_rad), math.sin(roll_rad)
            cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
            
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
            R = Ry @ Rx

            marker_diameter_m = float(cam_config.marker_diameter_m)

            for d in detections:
                # center_offset_y は画像座標系（下向き正）。カメラ座標系の y 軸と同方向。
                ray_c = np.array([d.center_offset_x / f_x, d.center_offset_y / f_y, 1.0])
                diameter_px = d.diameter_px if d.diameter_px > 0 else 2.0 * math.sqrt(d.area / math.pi)

                if marker_diameter_m > 0 and diameter_px > 1e-3:
                    # マーカー直径から深度を推定（ピンホールモデル）
                    depth_m = (f_y * marker_diameter_m) / diameter_px
                    p_cam = ray_c * depth_m
                    p_world = R @ p_cam
                    d.distance_x_m = float(p_world[0])
                    d.distance_y_m = float(p_world[1])
                    d.distance_z_m = float(p_world[2])
                else:
                    ray_w = R @ ray_c
                    if ray_w[2] > 0.001:
                        scale = cam_config.altitude_m / ray_w[2]
                        p_world = ray_w * scale
                        d.distance_x_m = float(p_world[0])
                        d.distance_y_m = float(p_world[1])
                        d.distance_z_m = float(p_world[2])
                    else:
                        d.distance_x_m = None
                        d.distance_y_m = None
                        d.distance_z_m = None

        detections = _filter_depth_outliers(
            detections,
            outlier_sigma=self.config.runtime.outlier_sigma,
            min_samples=self.config.runtime.outlier_min_samples,
        )

        aggregated_target = aggregate_detections(
            detections, self.config.runtime.aggregation_strategy
        )
        return detections, aggregated_target, detect_ms

    def run(self) -> PipelineResult:
        latencies_ms: List[float] = []
        frames = 0
        total_detections = 0
        started_at = start_timer()

        annotated_dir: Path | None = None
        output_video: Path | None = None
        save_interval = 1
        video_fps = self.config.source.fps
        if self.output is not None:
            annotated_dir = self.output.annotated_dir
            output_video = self.output.annotated_video
            save_interval = max(int(self.output.save_interval), 1)
            if self.output.video_fps > 0:
                video_fps = int(self.output.video_fps)
            if annotated_dir is not None:
                annotated_dir.mkdir(parents=True, exist_ok=True)
            if output_video is not None:
                output_video.parent.mkdir(parents=True, exist_ok=True)

        writer = None

        self.source.open()
        try:
            while True:
                frame = self.source.read()
                if frame is None:
                    break

                frame_index = frames + 1
                detections, aggregated_target, detect_ms = self.process_frame(frame)
                total_detections += len(detections)

                needs_annotated = (
                    self.config.runtime.display
                    or annotated_dir is not None
                    or output_video is not None
                )
                annotated = None
                if needs_annotated:
                    annotated = self.detector.annotate(frame, detections)
                    # 集約された最終ターゲット（ROS2出力用）を緑色で強調描画
                    if aggregated_target is not None:
                        x, y, w, h = aggregated_target.bbox
                        cx, cy = int(x + w / 2), int(y + h / 2)
                        cv2.circle(annotated, (cx, cy), 12, (0, 255, 0), 3)
                        fh, fw = annotated.shape[:2]
                        fcx, fcy = fw // 2, fh // 2
                        cv2.line(annotated, (fcx, fcy), (cx, cy), (0, 255, 0), 2)
                        label = f"Agg: {self.config.runtime.aggregation_strategy}"
                        cv2.putText(annotated, label, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                        cv2.putText(annotated, label, (cx + 15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # HUD（数値情報）を表示・保存フレーム共通で焼き込む
                    recent = latencies_ms[-30:] if latencies_ms else [detect_ms]
                    recent_p50 = sorted(recent)[len(recent) // 2]
                    vertical_offset_text = None
                    if aggregated_target is not None:
                        dy_px = aggregated_target.center_offset_y
                        if aggregated_target.distance_y_m is not None:
                            vertical_offset_text = (
                                f"Vertical offset: {aggregated_target.distance_y_m:+.2f} m"
                                f" ({dy_px:+.0f}px)"
                            )
                        else:
                            vertical_offset_text = f"Vertical offset: {dy_px:+.0f} px"
                    annotated = draw_hud(
                        annotated,
                        frame_index,
                        len(detections),
                        detect_ms,
                        recent_p50,
                        vertical_offset_text,
                    )

                if self.config.runtime.display and annotated is not None:
                    cv2.imshow("vision_pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if annotated_dir is not None and annotated is not None:
                    if (frame_index - 1) % save_interval == 0:
                        filename = f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(annotated_dir / filename), annotated)

                if output_video is not None and annotated is not None:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(output_video), fourcc, video_fps, (w, h)
                        )
                        if not writer.isOpened():
                            print(
                                f"[WARN] Failed to open video writer: {output_video}"
                            )
                            writer.release()
                            writer = None
                            output_video = None
                    if writer is not None:
                        writer.write(annotated)

                latencies_ms.append(detect_ms)
                frames += 1

                should_log = (
                    self.config.runtime.log_interval_frames > 0
                    and frames % self.config.runtime.log_interval_frames == 0
                )
                if should_log:
                    current_p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
                    # 採点基準: 各マーカーの正規化ずれ（垂直方向）と物理距離を出力
                    offset_info = " ".join(
                        f"[dx={d.center_offset_x:+.1f} dy={d.center_offset_y:+.1f} norm_y={d.norm_offset_y:+.3f}"
                        + (f" dist_y={d.distance_y_m:+.2f}m" if d.distance_y_m is not None else "")
                        + "]"
                        for d in detections
                    ) or "-"
                    
                    agg_info = ""
                    if aggregated_target is not None:
                        dist_info = f" dist_y={aggregated_target.distance_y_m:+.2f}m" if aggregated_target.distance_y_m is not None else ""
                        agg_info = f" | Agg({self.config.runtime.aggregation_strategy}): dx={aggregated_target.center_offset_x:+.1f} norm_y={aggregated_target.norm_offset_y:+.3f}{dist_info}"
                    
                    scale_info = ""
                    cam_config = self.config.camera
                    if cam_config is not None:
                        scale = compute_scale_check(detections, cam_config.marker_spacing_m)
                        if scale is not None:
                            scale_info = (
                                f" | scale: median={scale.median_spacing_m:.2f}m"
                                f" ratio={scale.scale_ratio:.2f}"
                                f" rms={scale.rms_error_m:.2f} n={scale.num_pairs}"
                            )
                        
                    print(
                        f"frames={frames} detections={total_detections} "
                        f"latency_p50_ms={current_p50:.2f} offsets={offset_info}{agg_info}{scale_info}"
                    )
        finally:
            self.source.close()
            if writer is not None:
                writer.release()
            if self.config.runtime.display:
                cv2.destroyAllWindows()

        ended_at = start_timer()
        metrics = FrameMetrics(
            latencies_ms=latencies_ms,
            frames_processed=frames,
            started_at=started_at,
            ended_at=ended_at,
        )
        return PipelineResult(
            metrics=metrics,
            total_detections=total_detections,
        )
