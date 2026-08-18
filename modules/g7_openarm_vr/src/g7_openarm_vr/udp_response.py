from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import quat_mul, quat_normalize

# The Quest application sends Unity world coordinates:
#   +X right, +Y up, +Z forward (left-handed).
# G7/MuJoCo uses the robot world convention:
#   +X forward, +Y left, +Z up (right-handed).
#
# Position therefore maps as:
#   [x_robot, y_robot, z_robot] = [z_unity, -x_unity, y_unity]
#
# Orientation is converted in two explicit steps:
#   1. Unity LH -> an RH frame with axes [X right, Y up, Z backward]
#   2. RH frame -> robot/MuJoCo frame [X forward, Y left, Z up]
VR_RH_TO_MUJOCO_QUAT: npt.NDArray[np.float64] = np.array(
    [-0.5, -0.5, 0.5, 0.5],
    dtype=np.float64,
)


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"Expected numeric VR field {key!r}, got {value!r}")
    return float(value)


@dataclass(frozen=True, slots=True)
class VRControllerPose:
    x: float
    y: float
    z: float
    qw: float
    qx: float
    qy: float
    qz: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> VRControllerPose:
        return cls(
            x=_number(payload, "x"),
            y=_number(payload, "y"),
            z=_number(payload, "z"),
            qw=_number(payload, "qw"),
            qx=_number(payload, "qx"),
            qy=_number(payload, "qy"),
            qz=_number(payload, "qz"),
        )

    def as_mujoco_pose(self) -> npt.NDArray[np.float64]:
        """Convert a Unity controller pose to the G7/MuJoCo world frame.

        Input position axes are Unity +X right, +Y up, +Z forward. Output axes
        are robot/MuJoCo +X forward, +Y left, +Z up. Quaternions use wxyz.
        This is the fixed coordinate-basis conversion; session-specific heading
        calibration is applied separately by the VR node as a world-Z yaw only.
        """
        position = np.array([self.z, -self.x, self.y], dtype=np.float64)

        # Unity LH quaternion (xyzw in the packet) -> RH wxyz quaternion.
        right_handed = np.array(
            [self.qw, -self.qx, -self.qy, self.qz],
            dtype=np.float64,
        )
        orientation = quat_normalize(quat_mul(VR_RH_TO_MUJOCO_QUAT, right_handed))
        return np.concatenate((position, orientation), dtype=np.float64)


def trigger_to_gripper_openness(trigger: float) -> float:
    """Convert raw VR trigger amount (0=released, 1=pressed) to openness."""
    return float(np.clip(1.0 - float(trigger), 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class VRUDPResponse:
    left_controller: VRControllerPose
    right_controller: VRControllerPose
    left_trigger: float  # 0=released/open, 1=pressed/closed
    right_trigger: float  # 0=released/open, 1=pressed/closed

    lsx: float
    lsy: float
    rsx: float
    rsy: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> VRUDPResponse:
        left = payload.get("lc")
        right = payload.get("rc")
        lt = payload.get("lt")
        rt = payload.get("rt")
        lxs = payload.get("lsx")
        lys = payload.get("lsy")
        rxs = payload.get("rsx")
        rys = payload.get("rsy")
        if (
            not isinstance(left, Mapping)
            or not isinstance(right, Mapping)
            or not isinstance(lt, (int, float))
            or not isinstance(rt, (int, float))
            or not isinstance(lxs, (int, float))
            or not isinstance(lys, (int, float))
            or not isinstance(rxs, (int, float))
            or not isinstance(rys, (int, float))
        ):
            raise ValueError(
                "VR packet must contain controller objects 'lc' and 'rc' and gripper values 'lt' and 'rt'"
            )
        return cls(
            left_controller=VRControllerPose.from_mapping(left),
            right_controller=VRControllerPose.from_mapping(right),
            left_trigger=float(lt),
            right_trigger=float(rt),
            lsx=float(lxs),
            lsy=float(lys),
            rsx=float(rxs),
            rsy=float(rys),
        )
