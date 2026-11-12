from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from unitree_sdk2py.idl.default import HGLowState_

from g7_openarm_idl import Odom
from g7_openarm_pinnzoo import PinnZooModel, build_lib_x, kinematics
from g7_openarm_utils.layout import Joint, KinBody, low_idx

from .geometry import pose7_wxyz_to_matrix, quaternion_wxyz_to_matrix
from .gripper import GripperWidthRange

if TYPE_CHECKING:
    from .config import HommiConfig

FloatArray = npt.NDArray[np.float32]

# Fixed HEAD_color_link -> HEAD_color_optical_frame rotation from the MJCF/URDF
# (quat wxyz = -0.5 0.5 -0.5 0.5): z forward, x right, y down. This is the frame
# HoMMI's head pose/pointmap use (iPhone ARKit camera pose * diag(1, -1, -1)).
COLOR_LINK_TO_OPTICAL = np.eye(4, dtype=np.float32)
COLOR_LINK_TO_OPTICAL[:3, :3] = quaternion_wxyz_to_matrix([-0.5, 0.5, -0.5, 0.5])


def head_camera_extrinsic(rectification: npt.ArrayLike | None = None) -> FloatArray:
    """``HEAD_color_link`` -> frame of the head pointmap.

    ``rectification`` is the stereo ``R1`` (raw left -> rectified left).
    """
    extrinsic = COLOR_LINK_TO_OPTICAL.astype(np.float64)
    if rectification is not None:
        rect = np.eye(4)
        rect[:3, :3] = np.asarray(rectification, dtype=np.float64).T
        extrinsic = extrinsic @ rect
    return extrinsic.astype(np.float32)


def head_extrinsic(cfg: HommiConfig) -> FloatArray:
    if cfg.head_camera != "zed2":
        return COLOR_LINK_TO_OPTICAL.copy()
    from .zed_camera import rectification_rotation

    return head_camera_extrinsic(rectification_rotation(cfg.zed_calibration))


class StateHistory:
    """Recent lowstate / odom messages keyed by their monotonic arrival time."""

    def __init__(self, horizon_s: float = 1.0) -> None:
        self._horizon_s = float(horizon_s)
        self._lock = threading.Lock()
        self._lowstate: deque[tuple[float, HGLowState_]] = deque()
        self._odom: deque[tuple[float, Odom]] = deque()

    def _add(self, buffer: deque, t: float, msg: object) -> None:
        with self._lock:
            buffer.append((t, msg))
            while buffer and t - buffer[0][0] > self._horizon_s:
                buffer.popleft()

    def add_lowstate(self, t: float, msg: HGLowState_) -> None:
        self._add(self._lowstate, t, msg)

    def add_odom(self, t: float, msg: Odom) -> None:
        self._add(self._odom, t, msg)

    @staticmethod
    def _nearest(buffer: deque, t: float) -> object:
        best = buffer[-1]
        for item in reversed(buffer):
            if abs(item[0] - t) <= abs(best[0] - t):
                best = item
            if item[0] < t:
                break
        return best[1]

    def latest(self) -> tuple[HGLowState_, Odom] | None:
        with self._lock:
            if not self._lowstate or not self._odom:
                return None
            return self._lowstate[-1][1], self._odom[-1][1]

    def at(self, t: float) -> tuple[HGLowState_, Odom] | None:
        with self._lock:
            if not self._lowstate or not self._odom:
                return None
            return self._nearest(self._lowstate, t), self._nearest(self._odom, t)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    """Robot state at one observation instant, in the world (odom) frame.

    ``left_tcp`` / ``right_tcp`` are the FK ``*_tcp`` frames, which must follow
    HoMMI's TCP convention (origin at the fingertip centre, +z along the approach
    direction, +y pointing away from the wrist camera; verify with
    ``scripts/check_hommi_frames.py``). ``head`` is the head color optical frame
    and gripper widths are in meters.
    """

    timestamp: float
    left_tcp: FloatArray
    right_tcp: FloatArray
    head: FloatArray
    left_gripper_width: float
    right_gripper_width: float

    def tcp(self, side: str) -> FloatArray:
        if side == "left":
            return self.left_tcp
        if side == "right":
            return self.right_tcp
        raise ValueError(f"unknown arm side {side!r}")

    def gripper_width(self, side: str) -> float:
        if side == "left":
            return self.left_gripper_width
        if side == "right":
            return self.right_gripper_width
        raise ValueError(f"unknown arm side {side!r}")


class RobotStateProjector:
    """Convert the latest lowstate/odom pair to HoMMI observation state via FK."""

    def __init__(
        self,
        gripper_ranges: dict[str, GripperWidthRange],
        head_extrinsic: npt.ArrayLike = COLOR_LINK_TO_OPTICAL,
    ) -> None:
        self._gripper = gripper_ranges
        self._head_extrinsic = np.asarray(head_extrinsic, dtype=np.float32)
        self._model = PinnZooModel()
        size = self._model.kinematics_body_size
        self._kin_slice = {
            KinBody.name_to_kin_body(name): slice(i * size, i * size + 7)
            for i, name in enumerate(self._model.kinematics_bodies)
        }

    def _pose(self, kin: npt.NDArray[np.float64], body: KinBody) -> FloatArray:
        # Kinematics bodies are [x, y, z, qw, qx, qy, qz, ...] (as used by the WBC).
        pose7 = kin[self._kin_slice[body]].astype(np.float32)
        return pose7_wxyz_to_matrix(pose7).astype(np.float32, copy=False)

    def snapshot(self, lowstate: HGLowState_, odom: Odom, *, timestamp: float) -> RobotSnapshot:
        kin = np.asarray(
            kinematics(self._model, build_lib_x(self._model, lowstate, odom)), dtype=np.float64
        )
        if kin.shape != (self._model.kinematics_size,):
            raise RuntimeError(
                f"PinnZoo kinematics returned shape {kin.shape}, "
                f"expected ({self._model.kinematics_size},)"
            )

        # Gripper motor state q is normalized openness (0=closed, 1=open).
        left_open = lowstate.motor_state[low_idx(Joint.L8)].q
        right_open = lowstate.motor_state[low_idx(Joint.R8)].q
        return RobotSnapshot(
            timestamp=float(timestamp),
            left_tcp=self._pose(kin, KinBody.L_TCP),
            right_tcp=self._pose(kin, KinBody.R_TCP),
            head=self._pose(kin, KinBody.HEAD) @ self._head_extrinsic,
            left_gripper_width=self._gripper["left"].to_width(left_open),
            right_gripper_width=self._gripper["right"].to_width(right_open),
        )
