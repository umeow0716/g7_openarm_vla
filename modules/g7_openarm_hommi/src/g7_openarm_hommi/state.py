from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from unitree_sdk2py.idl.default import HGLowState_

from g7_openarm_idl import Odom
from g7_openarm_pinnzoo import PinnZooModel, build_lib_x, kinematics
from g7_openarm_utils.layout import Joint, KinBody, low_idx

from .geometry import pose7_wxyz_to_matrix, quaternion_wxyz_to_matrix
from .gripper import GripperWidthRange

FloatArray = npt.NDArray[np.float32]

# Fixed HEAD_color_link -> HEAD_color_optical_frame rotation from the MJCF/URDF
# (quat wxyz = -0.5 0.5 -0.5 0.5): z forward, x right, y down. This is the frame
# HoMMI's head pose/pointmap use (iPhone ARKit camera pose * diag(1, -1, -1)).
COLOR_LINK_TO_OPTICAL = np.eye(4, dtype=np.float32)
COLOR_LINK_TO_OPTICAL[:3, :3] = quaternion_wxyz_to_matrix([-0.5, 0.5, -0.5, 0.5])


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

    def __init__(self, gripper_ranges: dict[str, GripperWidthRange]) -> None:
        self._gripper = gripper_ranges
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
            head=self._pose(kin, KinBody.HEAD) @ COLOR_LINK_TO_OPTICAL,
            left_gripper_width=self._gripper["left"].to_width(left_open),
            right_gripper_width=self._gripper["right"].to_width(right_open),
        )
