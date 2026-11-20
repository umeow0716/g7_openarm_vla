"""ZED 2 head stereo camera: rectified left RGB and Fast-FoundationStereo depth.

Rectification, disparity/depth rules and USB discovery are ported from
``~/Downloads/camera`` (``calib.py``, ``pointmap.py``, ``zed_uvc.py``); keep them in step.
"""

from __future__ import annotations

import configparser
import ctypes
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.synchronize import Event, Lock
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from .camera import CameraFrame, CameraInput, PinholeIntrinsics
from .color import ColorCalibration, ColorCorrector
from .gpu_turns import GpuTurns
from .trt import TrtEngine

# The head camera always streams HD720, so only the HD sections of the factory file are used.
CAPTURE_SIZE = (1280, 720)  # per eye
# Stereolabs factory calibration of the head camera, from https://calib.stereolabs.com/.
ZED_CALIBRATION = Path(__file__).resolve().parent / "calibration" / "zed2.conf"
ZED2_USB_ID = ("2b03", "f780")
# Official Fast-FoundationStereo 20-30-48 single ONNX (valid_iters 8, max_disp 192).
STEREO_SIZE = (960, 576)
FFS_ONNX = (
    Path(__file__).resolve().parents[2] / "model" / "ffs" / "20_30_48_iters_8_res_576x960.onnx"
)
FFS_ENGINE = FFS_ONNX.with_suffix(".engine")
# The single ONNX has ImageNet normalization stripped; inputs are normalized in 0-255 scale.
IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)
# The camera process loads the engine and warms it up before its first frame.
STARTUP_TIMEOUT_S = 60.0


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


def _sysfs_text(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def find_zed2(sysfs: Path = Path("/sys")) -> str:
    """Return the capture node (``/dev/videoN``) of the one attached ZED 2.

    The bundled calibration belongs to one camera, so anything other than exactly
    one ZED 2 is an error.
    """
    nodes = []
    for video in sorted((sysfs / "class" / "video4linux").glob("video*")):
        if _sysfs_text(video / "index") != "0":  # index 1 is the metadata node
            continue
        dev = video.resolve()
        while dev != dev.parent and not (dev / "idVendor").exists():
            dev = dev.parent
        if (_sysfs_text(dev / "idVendor"), _sysfs_text(dev / "idProduct")) == ZED2_USB_ID:
            nodes.append(f"/dev/{video.name}")
    if len(nodes) != 1:
        raise RuntimeError(f"expected exactly one ZED 2, found {len(nodes)}: {nodes}")
    return nodes[0]


def load_zed_conf(path: str | Path = ZED_CALIBRATION) -> ZedCalibration:
    """Parse the HD720 calibration from a Stereolabs factory calibration file."""
    mode = "HD"
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
    return ZedCalibration(CAPTURE_SIZE, K1, D1, K2, D2, R, T)


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
        self._engine = TrtEngine(
            engine_path,
            device=device,
            build_hint="uv run python scripts/build_ffs_engine.py",
        )
        shape = self._engine.input_shapes["left_image"]
        if shape != (1, 3, STEREO_SIZE[1], STEREO_SIZE[0]):
            raise RuntimeError(
                f"engine input is {shape}, expected 1x3x{STEREO_SIZE[1]}x{STEREO_SIZE[0]}"
            )
        self._mean = torch.tensor(IMAGENET_MEAN, device=self._engine.device).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD, device=self._engine.device).view(1, 3, 1, 1)

    def _input(self, image: npt.NDArray[np.uint8]) -> torch.Tensor:
        x = torch.from_numpy(image).to(self._engine.device).permute(2, 0, 1)[None].float()
        return (x - self._mean) / self._std

    @torch.inference_mode()
    def __call__(
        self, left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        out = self._engine(left_image=self._input(left), right_image=self._input(right))
        disparity: npt.NDArray[np.float32] = out["disparity"][0, 0].clamp(min=0).cpu().numpy()
        return disparity

    def warmup(self, n: int = 3) -> None:
        x = np.zeros((STEREO_SIZE[1], STEREO_SIZE[0], 3), dtype=np.uint8)
        for _ in range(n):
            self(x, x)


def rectification_rotation(
    calibration: str | Path = ZED_CALIBRATION,
) -> npt.NDArray[np.float64]:
    """Stereo ``R1`` (raw left -> rectified left) that ``ZedStereoCamera`` applies."""
    return Rectifier(load_zed_conf(calibration), STEREO_SIZE).R1


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


_FRAME_HEADER_BYTES = 16
_FRAME_BYTES = _FRAME_HEADER_BYTES + STEREO_SIZE[0] * STEREO_SIZE[1] * (3 + 4)


def _frame_views(buffer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sequence number, capture time, colour and depth of the frame shared by both processes."""
    w, h = STEREO_SIZE
    seq: np.ndarray = np.ndarray((1,), np.int64, buffer, 0)
    stamp: np.ndarray = np.ndarray((1,), np.float64, buffer, 8)
    color: np.ndarray = np.ndarray((h, w, 3), np.uint8, buffer, _FRAME_HEADER_BYTES)
    depth: np.ndarray = np.ndarray((h, w), np.float32, buffer, _FRAME_HEADER_BYTES + color.nbytes)
    return seq, stamp, color, depth


def _camera_process(
    buffer: Any,
    lock: Lock,
    stop: Event,
    conn: Connection,
    fps: int,
    cuda_device: str,
    parent_pid: int,
    gpu_turns: GpuTurns | None,
    color_calibration: ColorCalibration | None,
) -> None:
    """Capture, rectify and run FoundationStereo, publishing only the newest frame."""
    # Ctrl-C reaches the whole process group; the parent decides when this one stops.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    cap = None
    grabber = None
    try:
        seq, stamp, color, depth = _frame_views(buffer)
        eye_w, eye_h = CAPTURE_SIZE
        video_node = find_zed2()
        rect = Rectifier(load_zed_conf(), STEREO_SIZE)
        fx = float(rect.K[0, 0])
        ffs = FfsEngine(device=cuda_device)
        ffs.warmup()

        cap = cv2.VideoCapture(video_node, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * eye_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, eye_h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open ZED 2 stereo device {video_node}")
        correct = None
        # V4L2 controls persist after the device is closed, so always set auto WB.
        cap.set(cv2.CAP_PROP_AUTO_WB, float(color_calibration is None))
        if color_calibration is not None:
            # Fixed white balance: the correction LUT is only valid at this temperature.
            if not (
                cap.set(cv2.CAP_PROP_AUTO_WB, 0)
                and cap.set(cv2.CAP_PROP_WB_TEMPERATURE, color_calibration.white_balance)
            ):
                raise RuntimeError(f"cannot lock ZED 2 white balance on {video_node}")
            correct = ColorCorrector(color_calibration)
        ok, frame = cap.read()
        if not ok or frame.shape[:2] != (eye_h, 2 * eye_w):
            got = None if not ok else frame.shape[:2]
            raise RuntimeError(
                f"ZED 2 side-by-side stream on {video_node} is {got}, expected {(eye_h, 2 * eye_w)}"
            )

        raw_ready = threading.Condition()
        raw: list[Any] = [None, 0, None]  # frame, sequence, grab error

        def grab() -> None:
            # Drain the stream at camera rate so FoundationStereo always gets the newest
            # frame with its real capture time, not one that waited in the V4L2 queue.
            try:
                while not stop.is_set():
                    ok, frame = cap.read()
                    timestamp = time.monotonic()
                    if not ok:
                        raise RuntimeError("ZED 2 frame read failed")
                    with raw_ready:
                        raw[0] = (frame, timestamp)
                        raw[1] += 1
                        raw_ready.notify_all()
            except BaseException as exc:
                with raw_ready:
                    raw[2] = exc
                    raw_ready.notify_all()

        grabber = threading.Thread(target=grab, name="zed-grab", daemon=True)
        grabber.start()

        last_seq = 0
        # A parent that was killed cannot stop this process, so stop with it.
        while not stop.is_set() and os.getppid() == parent_pid:
            # Take the GPU turn before picking the frame, so the newest frame is the one
            # that gets processed rather than one that aged while the policy ran.
            with gpu_turns.turn("stereo") if gpu_turns is not None else nullcontext():
                with raw_ready:
                    if raw[1] == last_seq and raw[2] is None:
                        raw_ready.wait(0.1)
                    if raw[2] is not None:
                        raise raw[2]
                    if raw[1] == last_seq:
                        continue
                    (sbs_bgr, timestamp), last_seq = raw[0], raw[1]
                half = sbs_bgr.shape[1] // 2
                left, right = rect(sbs_bgr[:, :half], sbs_bgr[:, half:])
                left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
                right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
                disp = remove_invisible(ffs(left, right))
                metric = disparity_to_depth(disp, fx, rect.baseline)
                if correct is not None:
                    # Stereo matched the raw images; only the policy's RGB is corrected.
                    left = correct(left)
                # Publish before the turn ends, so the policy's turn sees this frame.
                with lock:
                    color[:] = left
                    depth[:] = metric
                    stamp[0] = timestamp
                    seq[0] += 1
                    first = seq[0] == 1
            if first:
                conn.send(
                    ("ready", tuple(float(v) for v in (*np.diag(rect.K)[:2], *rect.K[:2, 2])))
                )
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        stop.set()
        if grabber is not None:
            grabber.join(timeout=2.0)
        if cap is not None:
            cap.release()


class ZedStereoCamera(CameraInput):
    """ZED 2 head camera: latest rectified left frame plus its FoundationStereo depth.

    Capture and the stereo model run in their own process, so they never hold this
    process's interpreter while the policy is inferring. That process keeps only the
    newest result, like ``RealSenseCamera``; the stereo model sets the frame rate, so
    the observation loop may see the same head frame twice.
    """

    def __init__(
        self,
        *,
        name: str = "head",
        fps: int = 30,
        cuda_device: str = "cuda:0",
        gpu_turns: GpuTurns | None = None,
        color_calibration: ColorCalibration | None = None,
    ) -> None:
        self.name = name
        ctx = mp.get_context("spawn")
        self._buffer = ctx.RawArray(ctypes.c_uint8, _FRAME_BYTES)
        self._seq, self._stamp, self._color, self._depth = _frame_views(self._buffer)
        self._lock = ctx.Lock()
        self._stop = ctx.Event()
        self._conn, child_conn = ctx.Pipe(duplex=False)
        self._error: str | None = None
        self._frame: CameraFrame | None = None
        self._frame_seq = 0
        self._process = ctx.Process(
            target=_camera_process,
            args=(
                self._buffer,
                self._lock,
                self._stop,
                child_conn,
                fps,
                cuda_device,
                os.getpid(),
                gpu_turns,
                color_calibration,
            ),
            name=f"zed-{name}",
            daemon=True,
        )
        self._process.start()
        child_conn.close()

        try:
            if not self._conn.poll(STARTUP_TIMEOUT_S):
                raise TimeoutError(f"ZED {name} produced no frame in {STARTUP_TIMEOUT_S:.0f} s")
            self._raise_if_failed()
        except BaseException:
            self.close()
            raise
        w, h = STEREO_SIZE
        self._intrinsics = PinholeIntrinsics(w, h, *self._ready)

    def _raise_if_failed(self) -> None:
        if self._error is None:
            try:
                if self._conn.poll():
                    kind, payload = self._conn.recv()
                    if kind == "ready":
                        self._ready: tuple[float, float, float, float] = payload
                    else:
                        self._error = payload
            except EOFError:
                self._error = "the camera process closed its pipe"
            if self._error is None and not self._process.is_alive():
                self._error = f"the camera process exited with code {self._process.exitcode}"
        if self._error is not None:
            raise RuntimeError(f"ZED {self.name} capture failed:\n{self._error}")

    def latest(self) -> CameraFrame:
        self._raise_if_failed()
        with self._lock:
            seq = int(self._seq[0])
            if seq != self._frame_seq:
                self._frame = CameraFrame(
                    timestamp=float(self._stamp[0]),
                    color=self._color.copy(),
                    color_intrinsics=self._intrinsics,
                    depth=self._depth.copy(),
                    depth_intrinsics=self._intrinsics,
                    depth_to_color=np.eye(4),
                )
                self._frame_seq = seq
            assert self._frame is not None
            return self._frame

    def close(self) -> None:
        self._stop.set()
        self._process.join(timeout=3.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
