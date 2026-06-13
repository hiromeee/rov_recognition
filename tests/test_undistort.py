"""undistort モジュールのテスト。"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from vision_pipeline.undistort import UndistortMap, load_undistort_map


def _write_calib(path: Path, k1: float, k2: float = 0.0) -> None:
    data = {
        "camera_matrix": {"fx": 400.0, "fy": 400.0, "cx": 160.0, "cy": 120.0},
        "dist_coeffs": {"k1": k1, "k2": k2, "p1": 0.0, "p2": 0.0, "k3": 0.0},
        "image_size": {"width": 320, "height": 240},
    }
    with path.open("w", encoding="utf-8") as fp:
        yaml.dump(data, fp)


def test_load_undistort_map_all_zero_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "calib.yaml"
        _write_calib(p, k1=0.0)
        result = load_undistort_map(str(p), image_width=320, image_height=240)
    assert result is None


def test_load_undistort_map_nonzero_returns_map() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "calib.yaml"
        _write_calib(p, k1=-0.3)
        result = load_undistort_map(str(p), image_width=320, image_height=240)
    assert isinstance(result, UndistortMap)


def test_load_undistort_map_missing_file_returns_none(capsys) -> None:
    result = load_undistort_map("/nonexistent/calib.yaml", image_width=320, image_height=240)
    assert result is None
    captured = capsys.readouterr()
    assert "WARN" in captured.out


def test_apply_preserves_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "calib.yaml"
        _write_calib(p, k1=-0.3)
        umap = load_undistort_map(str(p), image_width=320, image_height=240)
    assert umap is not None
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = umap.apply(frame)
    assert result.shape == frame.shape


def test_apply_modifies_distorted_image() -> None:
    """歪みありのマップを適用すると入力と異なる画像が出力される。"""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "calib.yaml"
        _write_calib(p, k1=-0.5)
        umap = load_undistort_map(str(p), image_width=320, image_height=240)
    assert umap is not None
    rng = np.random.default_rng(42)
    frame = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    result = umap.apply(frame)
    assert not np.array_equal(result, frame)
