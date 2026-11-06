from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pyrealsense2 as rs

RGBFrame = npt.NDArray[np.uint8]
DepthFrame = npt.NDArray[np.float32]

COLOR_WIDTH = 1920
COLOR_HEIGHT = 1080
DEPTH_WIDTH = 848
DEPTH_HEIGHT = 480


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Pinhole intrinsics in pixels for an image of ``width`` x ``height``."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_rs(cls, intr: rs.intrinsics) -> PinholeIntrinsics:
        # D435i color/depth streams report (inverse) Brown-Conrady models whose
        # coefficients are zero on factory calibration, so distortion is ignored.
        return cls(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """One captured frame set.

    ``depth`` is metric depth in meters (0 = invalid) in the depth sensor frame;
    ``depth_to_color`` maps depth-frame points into the color optical frame.
    """

    timestamp: float
    color: RGBFrame
    color_intrinsics: PinholeIntrinsics
    depth: DepthFrame | None = None
    depth_intrinsics: PinholeIntrinsics | None = None
    depth_to_color: npt.NDArray[np.float64] | None = None


class CameraInput(ABC):
    """Latest-frame camera contract used by the HoMMI observation sampler."""

    @abstractmethod
    def latest(self) -> CameraFrame:
        """Return the newest frame set (blocking only until the first frame exists)."""

    def close(self) -> None:  # noqa: B027 - optional hook, default no-op
        """Release camera resources. Stateless cameras may keep the default no-op."""


# A plausible D435i 1920x1080 color calibration; only used by BlackCamera so the
# simulation exercises the same resampling path as the real cameras.
_SIM_COLOR_INTRINSICS = PinholeIntrinsics(COLOR_WIDTH, COLOR_HEIGHT, 1380.0, 1380.0, 960.0, 540.0)
_SIM_DEPTH_INTRINSICS = PinholeIntrinsics(DEPTH_WIDTH, DEPTH_HEIGHT, 425.0, 425.0, 424.0, 240.0)


class BlackCamera(CameraInput):
    """Simulation camera: black color frame and (optionally) all-invalid depth."""

    def __init__(self, *, with_depth: bool = False) -> None:
        color = np.zeros((COLOR_HEIGHT, COLOR_WIDTH, 3), dtype=np.uint8)
        color.setflags(write=False)
        depth = None
        if with_depth:
            depth = np.zeros((DEPTH_HEIGHT, DEPTH_WIDTH), dtype=np.float32)
            depth.setflags(write=False)
        self._color = color
        self._depth = depth
        self._with_depth = with_depth

    def latest(self) -> CameraFrame:
        return CameraFrame(
            timestamp=time.monotonic(),
            color=self._color,
            color_intrinsics=_SIM_COLOR_INTRINSICS,
            depth=self._depth,
            depth_intrinsics=_SIM_DEPTH_INTRINSICS if self._with_depth else None,
            depth_to_color=np.eye(4) if self._with_depth else None,
        )


class RealSenseCamera(CameraInput):
    """One Intel RealSense D435i selected by serial number.

    A background thread keeps only the newest frame set, so reading three cameras
    from the 20 Hz observation loop never blocks on a 30 Hz device.
    """

    def __init__(
        self,
        serial: str,
        *,
        name: str,
        with_depth: bool,
        fps: int = 30,
        timeout_ms: int = 1000,
        warmup_frames: int = 5,
    ) -> None:
        if not serial:
            raise ValueError(
                f"RealSense serial for {name!r} camera is empty; set it in config.toml"
            )
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if timeout_ms <= 0:
            raise ValueError(f"timeout_ms must be positive, got {timeout_ms}")

        self.name = name
        self.serial = serial
        self._timeout_ms = int(timeout_ms)
        self._with_depth = with_depth

        stream_config = rs.config()
        stream_config.enable_device(serial)
        stream_config.enable_stream(
            rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.rgb8, int(fps)
        )
        if with_depth:
            stream_config.enable_stream(
                rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, int(fps)
            )

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(stream_config)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._color_intrinsics = PinholeIntrinsics.from_rs(color_profile.get_intrinsics())

        self._depth_intrinsics: PinholeIntrinsics | None = None
        self._depth_to_color: npt.NDArray[np.float64] | None = None
        self._depth_scale = 0.0
        if with_depth:
            depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            self._depth_intrinsics = PinholeIntrinsics.from_rs(depth_profile.get_intrinsics())
            extr = depth_profile.get_extrinsics_to(color_profile)
            transform = np.eye(4)
            # librealsense stores the rotation column-major.
            transform[:3, :3] = np.asarray(extr.rotation, dtype=np.float64).reshape(3, 3).T
            transform[:3, 3] = np.asarray(extr.translation, dtype=np.float64)
            self._depth_to_color = transform
            self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())

        for _ in range(max(0, warmup_frames)):
            self._pipeline.wait_for_frames(self._timeout_ms)

        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._frame: CameraFrame | None = None
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop, name=f"realsense-{name}", daemon=True
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                frames = self._pipeline.wait_for_frames(self._timeout_ms)
                timestamp = time.monotonic()
                color = frames.get_color_frame()
                if not color:
                    raise RuntimeError(f"RealSense {self.name} frame set has no color frame")
                # Copy because the SDK owns the frame memory after this call returns.
                color_np = np.asanyarray(color.get_data()).copy()
                if color_np.shape != (COLOR_HEIGHT, COLOR_WIDTH, 3) or color_np.dtype != np.uint8:
                    raise RuntimeError(
                        f"RealSense {self.name} color frame has shape {color_np.shape} "
                        f"dtype {color_np.dtype}"
                    )
                depth_np = None
                if self._with_depth:
                    depth = frames.get_depth_frame()
                    if not depth:
                        raise RuntimeError(f"RealSense {self.name} frame set has no depth frame")
                    depth_np = (
                        np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
                    )
                frame = CameraFrame(
                    timestamp=timestamp,
                    color=color_np,
                    color_intrinsics=self._color_intrinsics,
                    depth=depth_np,
                    depth_intrinsics=self._depth_intrinsics,
                    depth_to_color=self._depth_to_color,
                )
                with self._ready:
                    self._frame = frame
                    self._ready.notify_all()
        except BaseException as exc:
            with self._ready:
                self._error = exc
                self._ready.notify_all()

    def latest(self) -> CameraFrame:
        deadline = time.monotonic() + self._timeout_ms / 1000.0
        with self._ready:
            while self._frame is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"RealSense {self.name} produced no frame")
                self._ready.wait(remaining)
            if self._error is not None:
                raise RuntimeError(f"RealSense {self.name} capture failed") from self._error
            assert self._frame is not None
            return self._frame

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._pipeline.stop()
