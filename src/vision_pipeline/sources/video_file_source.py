from __future__ import annotations

from typing import Optional

import cv2  # type: ignore[import-not-found]
import numpy as np

from vision_pipeline.sources.base import FrameSource


class VideoFileSource(FrameSource):
    def __init__(
        self,
        video_path: str,
        frames_limit: Optional[int] = None,
    ) -> None:
        self.video_path = video_path
        self.frames_limit = frames_limit
        self._frame_idx = 0
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.video_path)
        cap = self._cap
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")
        self._frame_idx = 0

    def read(self) -> Optional[np.ndarray]:
        if self._cap is None:
            raise RuntimeError("VideoFileSource is not opened")

        if (
            self.frames_limit is not None
            and self._frame_idx >= self.frames_limit
        ):
            return None

        ok, frame = self._cap.read()
        if not ok:
            return None

        self._frame_idx += 1
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
