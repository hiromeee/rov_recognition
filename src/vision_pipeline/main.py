from __future__ import annotations

import argparse
from pathlib import Path

from vision_pipeline.config import PipelineConfig, load_config
from vision_pipeline.pipeline import OutputSettings, VisionPipeline
from vision_pipeline.sources.base import FrameSource
from vision_pipeline.sources.camera_source import CameraSource
from vision_pipeline.sources.mock_source import MockFrameSource
from vision_pipeline.sources.video_file_source import VideoFileSource


def build_source(config: PipelineConfig) -> FrameSource:
    src = config.source
    if src.kind == "mock":
        return MockFrameSource(
            width=src.width,
            height=src.height,
            fps=src.fps,
            frames_limit=src.frames_limit,
        )

    if src.kind == "video":
        if not src.video_path:
            raise ValueError(
                "source.video_path is required when source.kind=video"
            )
        return VideoFileSource(
            video_path=src.video_path,
            frames_limit=src.frames_limit,
        )

    if src.kind == "camera":
        return CameraSource(
            camera_index=src.camera_index,
            width=src.width,
            height=src.height,
            fps=src.fps,
            frames_limit=src.frames_limit,
        )

    raise ValueError(f"Unsupported source.kind: {src.kind}")


def _compress_video(src: Path, frames: int, fps: float, target_mb: float = 9.8) -> None:
    """ffmpeg で target_mb 未満になるようビットレートを計算して再エンコードする。"""
    import os
    import subprocess

    duration_s = frames / fps if fps > 0 else 1.0
    bitrate_kbps = min(int(target_mb * 8 * 1024 / duration_s), 4000)
    dst = src.with_stem(src.stem + "_compressed")

    print(f"圧縮中 → {dst}  (目標 {target_mb} MB / {bitrate_kbps} kbps / {duration_s:.1f} s)")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-b:v", f"{bitrate_kbps}k",
                "-maxrate", f"{int(bitrate_kbps * 1.5)}k",
                "-bufsize", f"{bitrate_kbps * 3}k",
                "-preset", "slow",
                # テキスト・補助線の輪郭を優先する設定
                "-tune", "animation",
                # エッジ（補助線・数値）をシャープに保つ
                "-vf", "unsharp=lx=3:ly=3:la=1.2:cx=3:cy=3:ca=0.0",
                "-an",
                str(dst),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        size_mb = os.path.getsize(dst) / (1024 * 1024)
        print(f"保存完了: {dst}  ({size_mb:.1f} MB)")
    except FileNotFoundError:
        print("[WARN] ffmpeg が見つかりません。https://ffmpeg.org/ からインストールしてください")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg 圧縮に失敗しました: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realtime HSV baseline pipeline"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--video", help="Path to video file (overrides config source)"
    )
    parser.add_argument(
        "--display", action="store_true", help="Show annotated display window"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save annotated frames to disk",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video (mp4) to disk",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1,
        help="Save every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for annotated frames/video",
    )
    parser.add_argument(
        "--run-id",
        help="Experiment ID (docs/experiments/<run-id>/results)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.video:
        config.source.kind = "video"
        config.source.video_path = args.video
    if args.display:
        config.runtime.display = True

    source = build_source(config)

    output = None
    if args.save_frames or args.save_video:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        elif args.run_id:
            output_dir = Path("docs") / "experiments" / args.run_id / "results"
        else:
            raise ValueError(
                "--save-frames/--save-video requires --output-dir or --run-id"
            )

        annotated_dir = output_dir / "annotated" if args.save_frames else None
        annotated_video = (
            output_dir / "annotated.mp4" if args.save_video else None
        )
        output = OutputSettings(
            annotated_dir=annotated_dir,
            annotated_video=annotated_video,
            save_interval=args.save_interval,
            video_fps=config.source.fps,
        )

    result = VisionPipeline(config=config, source=source, output=output).run()
    summary = result.metrics.summary()
    print("=== Summary ===")
    print(f"frames={int(summary['frames'])}")
    print(f"total_detections={result.total_detections}")
    print(f"latency_p50_ms={summary['latency_p50_ms']:.2f}")
    print(f"latency_p95_ms={summary['latency_p95_ms']:.2f}")
    print(f"effective_fps={summary['effective_fps']:.2f}")
    if output is not None:
        if output.annotated_dir is not None:
            print(f"annotated_frames_dir={output.annotated_dir}")
        if output.annotated_video is not None:
            print(f"annotated_video={output.annotated_video}")
            _compress_video(output.annotated_video, result.metrics.frames_processed, config.source.fps)

if __name__ == "__main__":
    main()
