from __future__ import annotations

from typing import Optional

import cv2  # type: ignore[import-not-found]
import numpy as np

from vision_pipeline.sources.base import FrameSource


class MockFrameSource(FrameSource):
    """Synthetic source for testing without real data."""

    def __init__(
        self,
        width: int,
        height: int,
        fps: int,
        frames_limit: Optional[int],
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.frames_limit = frames_limit
        self._frame_idx = 0

    def open(self) -> None:
        self._frame_idx = 0

    def read(self) -> Optional[np.ndarray]:
        if (
            self.frames_limit is not None
            and self._frame_idx >= self.frames_limit
        ):
            return None

        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Green circle in BGR should match mock HSV thresholds.
        center_x = int((self._frame_idx * 8) % max(self.width - 40, 1)) + 20
        center_y = self.height // 2
        cv2.circle(frame, (center_x, center_y), 18, (0, 255, 0), thickness=-1)

        self._frame_idx += 1
        return frame

    def close(self) -> None:
        return
