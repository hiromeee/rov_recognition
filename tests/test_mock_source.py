from vision_pipeline.sources.mock_source import MockFrameSource


def test_mock_source_returns_limited_frames() -> None:
    source = MockFrameSource(width=320, height=240, fps=30, frames_limit=5)
    source.open()
    try:
        frames = []
        while True:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)

        assert len(frames) == 5
        assert frames[0].shape == (240, 320, 3)
    finally:
        source.close()
