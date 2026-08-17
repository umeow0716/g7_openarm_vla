from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

RGBFrame = npt.NDArray[np.uint8]

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080


class CameraInput(ABC):
    """Minimal RGB camera contract used by the HoMMI observation sampler."""

    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT

    @abstractmethod
    def read(self) -> RGBFrame:
        """Return one RGB uint8 frame with shape [1080, 1920, 3]."""

    def close(self) -> None:
        """Release camera resources. Stateless cameras may keep the default no-op."""


class BlackCamera(CameraInput):
    """Simulation camera that always returns the same immutable 1080p black RGB frame."""

    def __init__(self) -> None:
        self._frame = np.zeros(
            (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8,
        )
        self._frame.setflags(write=False)

    def read(self) -> RGBFrame:
        return self._frame


class RealSenseCamera(CameraInput):
    """Intel RealSense RGB-only camera configured for 1920x1080 RGB."""

    def __init__(
        self,
        *,
        fps: int = 30,
        timeout_ms: int = 1000,
        warmup_frames: int = 5,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")
        if warmup_frames < 0:
            raise ValueError(f"warmup_frames must be >= 0, got {warmup_frames}")

        self._timeout_ms = int(timeout_ms)
        self._pipeline = rs.pipeline()
        stream_config = rs.config()
        stream_config.enable_stream(
            rs.stream.color,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            rs.format.rgb8,
            int(fps),
        )
        self._pipeline.start(stream_config)
        self._closed = False

        for _ in range(warmup_frames):
            self._pipeline.wait_for_frames(self._timeout_ms)

    def read(self) -> RGBFrame:
        if self._closed:
            raise RuntimeError("RealSenseCamera is closed")

        frames = self._pipeline.wait_for_frames(self._timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("RealSense frame set did not contain a color frame")

        # Copy because the SDK owns the frame memory after this call returns.
        frame = np.asanyarray(color.get_data()).copy()
        expected = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)
        if frame.shape != expected:
            raise RuntimeError(
                f"RealSense RGB frame has shape {frame.shape}, expected {expected}"
            )
        if frame.dtype != np.uint8:
            raise RuntimeError(
                f"RealSense RGB frame has dtype {frame.dtype}, expected uint8"
            )
        return frame

    def close(self) -> None:
        if self._closed:
            return
        self._pipeline.stop()
        self._closed = True
