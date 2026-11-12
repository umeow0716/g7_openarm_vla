"""ZED 2 head stereo camera: rectified left RGB and Fast-FoundationStereo depth.

Rectification and disparity/depth rules are ported from ``~/Downloads/camera``
(``calib.py``, ``pointmap.py``); keep them in step.
"""

from __future__ import annotations

import configparser
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch

from .camera import CameraFrame, CameraInput, PinholeIntrinsics

ZED_MODE_BY_WIDTH = {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}
CAPTURE_SIZE = (1280, 720)  # HD720, per eye
# Official Fast-FoundationStereo 20-30-48 single ONNX (valid_iters 8, max_disp 192).
STEREO_SIZE = (960, 576)
FFS_ONNX = (
    Path(__file__).resolve().parents[2] / "model" / "ffs" / "20_30_48_iters_8_res_576x960.onnx"
)
FFS_ENGINE = FFS_ONNX.with_suffix(".engine")
# The single ONNX has ImageNet normalization stripped; inputs are normalized in 0-255 scale.
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


@dataclass(frozen=True, slots=True)
class ZedCalibration:
    """Raw (unrectified) stereo calibration, OpenCV convention: x_right = R @ x_left + T."""

    image_size: tuple[int, int]
    K1: npt.NDArray[np.float64]
    D1: npt.NDArray[np.float64]
    K2: npt.NDArray[np.float64]
    D2: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]
    T: npt.NDArray[np.float64]

    @property
    def baseline(self) -> float:
        return float(np.linalg.norm(self.T))


def load_zed_conf(path: str | Path, image_size: tuple[int, int]) -> ZedCalibration:
    """Parse a Stereolabs factory calibration file for the given eye resolution.

    Download the file for your camera serial from https://calib.stereolabs.com/?SN=<serial>.
    """
    mode = ZED_MODE_BY_WIDTH.get(image_size[0])
    if mode is None:
        raise ValueError(f"no ZED 2 calibration mode for eye width {image_size[0]}")
    cp = configparser.ConfigParser()
    if not cp.read(path):
        raise FileNotFoundError(f"ZED calibration file not found: {path}")

    def camera(section: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        s = cp[section]
        K = np.array(
            [
                [float(s["fx"]), 0.0, float(s["cx"])],
                [0.0, float(s["fy"]), float(s["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        D = np.array([float(s.get(k, 0)) for k in ("k1", "k2", "p1", "p2", "k3")], dtype=np.float64)
        return K, D

    K1, D1 = camera(f"LEFT_CAM_{mode}")
    K2, D2 = camera(f"RIGHT_CAM_{mode}")
    st = cp["STEREO"]
    rvec = np.array(
        [float(st[f"RX_{mode}"]), float(st[f"CV_{mode}"]), float(st[f"RZ_{mode}"])],
        dtype=np.float64,
    )
    # Stereolabs: rotation is a Rodrigues vector (RX, CV, RZ); Baseline is positive mm and
    # the right camera sits at -X in the left camera frame.
    R, _ = cv2.Rodrigues(rvec)
    T = (
        np.array(
            [[-float(st["Baseline"])], [float(st.get("TY", 0))], [float(st.get("TZ", 0))]],
            dtype=np.float64,
        )
        / 1000.0
    )
    return ZedCalibration(image_size, K1, D1, K2, D2, R, T)


def _scale_K(K: npt.NDArray[np.float64], sx: float, sy: float) -> npt.NDArray[np.float64]:
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


class Rectifier:
    """Maps raw eyes at the calibration size onto the working size, rectified."""

    def __init__(self, calib: ZedCalibration, out_size: tuple[int, int]) -> None:
        W0, H0 = calib.image_size
        W1, H1 = out_size
        K1 = _scale_K(calib.K1, W1 / W0, H1 / H0)
        K2 = _scale_K(calib.K2, W1 / W0, H1 / H0)
        R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
            K1,
            calib.D1,
            K2,
            calib.D2,
            (W1, H1),
            calib.R,
            calib.T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0,
            newImageSize=(W1, H1),
        )
        self.size = (W1, H1)
        self.map1 = cv2.initUndistortRectifyMap(K1, calib.D1, R1, P1, (W1, H1), cv2.CV_16SC2)
        self.map2 = cv2.initUndistortRectifyMap(K2, calib.D2, R2, P2, (W1, H1), cv2.CV_16SC2)
        self.K = P1[:3, :3].copy()
        self.R1 = R1  # raw left -> rectified left
        self.baseline = float(-P2[0, 3] / P2[0, 0])
        if self.baseline <= 0.0:
            raise RuntimeError("rectified baseline is not positive; check the eye order")

    def __call__(
        self, left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8]
    ) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        if (left.shape[1], left.shape[0]) != self.size:
            left = cv2.resize(left, self.size, interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, self.size, interpolation=cv2.INTER_AREA)
        return (
            cv2.remap(left, *self.map1, cv2.INTER_LINEAR),
            cv2.remap(right, *self.map2, cv2.INTER_LINEAR),
        )


class FfsEngine:
    """Fast-FoundationStereo TensorRT engine: rectified RGB pair -> disparity (pixels)."""

    def __init__(self, engine_path: Path = FFS_ENGINE, device: str = "cuda:0") -> None:
        import tensorrt as trt

        if not engine_path.is_file():
            raise FileNotFoundError(
                f"no Fast-FoundationStereo engine at {engine_path}; build it on this machine with "
                "`uv run python scripts/build_ffs_engine.py`"
            )
        self.device = torch.device(device)
        logger = trt.Logger(trt.Logger.WARNING)
        self._engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(
                f"cannot load {engine_path} with TensorRT {trt.__version__}; rebuild it with "
                "`uv run python scripts/build_ffs_engine.py`"
            )
        self._context = self._engine.create_execution_context()
        for name in ("left_image", "right_image", "disparity"):
            if self._engine.get_tensor_dtype(name) != trt.DataType.FLOAT:
                raise RuntimeError(f"engine tensor {name} is not float32")
        shape = tuple(self._engine.get_tensor_shape("left_image"))
        if shape != (1, 3, STEREO_SIZE[1], STEREO_SIZE[0]):
            raise RuntimeError(
                f"engine input is {shape}, expected 1x3x{STEREO_SIZE[1]}x{STEREO_SIZE[0]}"
            )
        self._mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        self._disparity = torch.empty((1, 1, STEREO_SIZE[1], STEREO_SIZE[0]), device=self.device)
        self._stream = torch.cuda.Stream(device=self.device)

    def _input(self, image: npt.NDArray[np.uint8]) -> torch.Tensor:
        x = torch.from_numpy(image).to(self.device).permute(2, 0, 1)[None].float()
        return ((x - self._mean) / self._std).contiguous()

    @torch.inference_mode()
    def __call__(
        self, left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        with torch.cuda.stream(self._stream):
            tensors = {
                "left_image": self._input(left),
                "right_image": self._input(right),
                "disparity": self._disparity,
            }
            for name, tensor in tensors.items():
                self._context.set_tensor_address(name, tensor.data_ptr())
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("Fast-FoundationStereo TensorRT inference failed")
            disparity = self._disparity[0, 0].clamp(min=0).cpu().numpy()
        return disparity

    def warmup(self, n: int = 3) -> None:
        x = np.zeros((STEREO_SIZE[1], STEREO_SIZE[0], 3), dtype=np.uint8)
        for _ in range(n):
            self(x, x)


def rectification_rotation(calibration: str | Path) -> npt.NDArray[np.float64]:
    """Stereo ``R1`` (raw left -> rectified left) that ``ZedStereoCamera`` applies."""
    return Rectifier(load_zed_conf(calibration, CAPTURE_SIZE), STEREO_SIZE).R1


def remove_invisible(disp: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Invalidate left pixels whose match falls outside the right image (u - d < 0)."""
    u = np.arange(disp.shape[1], dtype=np.float32)[None, :]
    out = disp.copy()
    out[u - disp < 0] = 0
    return out


def disparity_to_depth(
    disp: npt.NDArray[np.float32], fx: float, baseline: float, zfar: float = 100.0
) -> npt.NDArray[np.float32]:
    """Z = fx * B / d. Non-finite disparity or Z beyond zfar becomes 0 (invalid)."""
    valid = np.isfinite(disp) & (disp > fx * baseline / zfar)
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid] = fx * baseline / disp[valid]
    return depth


class ZedStereoCamera(CameraInput):
    """ZED 2 head camera: latest rectified left frame plus its FoundationStereo depth.

    The capture thread keeps only the newest result, like ``RealSenseCamera``. The
    stereo model sets the frame rate, so the observation loop may see the same head
    frame twice.
    """

    def __init__(
        self,
        *,
        video_device: int | str,
        calibration: str | Path,
        timeout_ms: int = 1000,
        name: str = "head",
        fps: int = 30,
        cuda_device: str = "cuda:0",
    ) -> None:
        self.name = name
        self._timeout_s = timeout_ms / 1000.0
        eye_w, eye_h = CAPTURE_SIZE
        self._rect = Rectifier(load_zed_conf(calibration, CAPTURE_SIZE), STEREO_SIZE)
        self._ffs = FfsEngine(device=cuda_device)
        self._ffs.warmup()
        self._intrinsics = PinholeIntrinsics(
            STEREO_SIZE[0],
            STEREO_SIZE[1],
            float(self._rect.K[0, 0]),
            float(self._rect.K[1, 1]),
            float(self._rect.K[0, 2]),
            float(self._rect.K[1, 2]),
        )

        self._cap = cv2.VideoCapture(
            video_device, cv2.CAP_V4L2 if isinstance(video_device, int) else cv2.CAP_ANY
        )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * eye_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eye_h)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open ZED 2 stereo device {video_device!r}")
        ok, frame = self._cap.read()
        if not ok or frame.shape[:2] != (eye_h, 2 * eye_w):
            self._cap.release()
            got = None if not ok else frame.shape[:2]
            raise RuntimeError(
                f"ZED 2 side-by-side stream is {got}, expected {(eye_h, 2 * eye_w)}; "
                "check zed_device"
            )

        self._lock = threading.Condition()
        self._frame: CameraFrame | None = None
        self._raw: tuple[npt.NDArray[np.uint8], float] | None = None
        self._raw_seq = 0
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._threads = (
            threading.Thread(target=self._grab_loop, name=f"zed-{name}-grab", daemon=True),
            threading.Thread(target=self._process_loop, name=f"zed-{name}-ffs", daemon=True),
        )
        for thread in self._threads:
            thread.start()

    def _process(self, sbs_bgr: npt.NDArray[np.uint8], timestamp: float) -> CameraFrame:
        half = sbs_bgr.shape[1] // 2
        left, right = self._rect(sbs_bgr[:, :half], sbs_bgr[:, half:])
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
        disp = remove_invisible(self._ffs(left, right))
        depth = disparity_to_depth(disp, self._intrinsics.fx, self._rect.baseline)
        return CameraFrame(
            timestamp=timestamp,
            color=left,
            color_intrinsics=self._intrinsics,
            depth=depth,
            depth_intrinsics=self._intrinsics,
            depth_to_color=np.eye(4),
        )

    def _fail(self, exc: BaseException) -> None:
        with self._lock:
            if self._error is None:
                self._error = exc
            self._lock.notify_all()

    def _grab_loop(self) -> None:
        # Drain the stream at camera rate so FoundationStereo always gets the newest
        # frame with its real capture time, not one that waited in the V4L2 queue.
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                timestamp = time.monotonic()
                if not ok:
                    raise RuntimeError(f"ZED {self.name} frame read failed")
                with self._lock:
                    self._raw = (frame, timestamp)
                    self._raw_seq += 1
                    self._lock.notify_all()
        except BaseException as exc:
            self._fail(exc)

    def _process_loop(self) -> None:
        last_seq = 0
        try:
            while not self._stop.is_set():
                with self._lock:
                    while self._raw_seq == last_seq and self._error is None:
                        if self._stop.is_set():
                            return
                        self._lock.wait(0.1)
                    if self._error is not None:
                        return
                    assert self._raw is not None
                    frame, timestamp = self._raw
                    last_seq = self._raw_seq
                result = self._process(frame, timestamp)
                with self._lock:
                    self._frame = result
                    self._lock.notify_all()
        except BaseException as exc:
            self._fail(exc)

    def latest(self) -> CameraFrame:
        deadline = time.monotonic() + self._timeout_s
        with self._lock:
            while self._frame is None and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(f"ZED {self.name} produced no frame")
                self._lock.wait(remaining)
            if self._error is not None:
                raise RuntimeError(f"ZED {self.name} capture failed") from self._error
            assert self._frame is not None
            return self._frame

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._cap.release()
