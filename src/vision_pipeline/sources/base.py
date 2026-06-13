from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class FrameSource(ABC):
    """Common interface for all frame input sources."""

    @abstractmethod
    def open(self) -> None:
        """Prepare underlying resources."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return next frame in BGR format, or None when stream ends."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""
