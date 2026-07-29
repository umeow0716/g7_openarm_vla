from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from g7_openarm_utils import quat_to_rotation_matrix

ArmSide = Literal["left", "right"]
FloatArray = npt.NDArray[np.float64]

_JOINT_AXES = (
    np.array([0.0, -1.0, 0.0], dtype=np.float64),
    np.array([1.0, 0.0, 0.0], dtype=np.float64),
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
    np.array([0.0, 1.0, 0.0], dtype=np.float64),
)

_SHOULDER_Z = 0.82196
_JOINT_3_OFFSET = np.array([0.0284, 0.0, -0.051], dtype=np.float64)
_JOINT_5_OFFSET = np.array([0.0, 0.0, -0.0955], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ArmSwivelKinematics:
    shoulder_position: FloatArray
    swivel_point_position: FloatArray
    jacobian: FloatArray


def _rotation_about_axis(axis: FloatArray, angle: float) -> FloatArray:
    x, y, z = axis
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (
        skew @ skew
    )


def _arm_geometry(side: ArmSide) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    if side == "left":
        side_sign = 1.0
        joint_4_y = -5.0e-5
    elif side == "right":
        side_sign = -1.0
        joint_4_y = 0.0
    else:
        raise ValueError(f"Unsupported arm side: {side}")

    shoulder_offset = np.array([0.0, side_sign * 0.0875, _SHOULDER_Z], dtype=np.float64)
    joint_2_offset = np.array([-0.0284, side_sign * 0.065, 0.0], dtype=np.float64)
    joint_4_offset = np.array([0.0, joint_4_y, -0.169], dtype=np.float64)
    outward_direction = np.array([0.0, side_sign, 0.0], dtype=np.float64)
    return shoulder_offset, joint_2_offset, joint_4_offset, outward_direction


def arm_swivel_kinematics(
    base_position: npt.ArrayLike,
    base_quaternion: npt.ArrayLike,
    shoulder_elbow_q: npt.ArrayLike,
    *,
    side: ArmSide,
) -> ArmSwivelKinematics:
    """Return a forearm swivel point and its Jacobian with respect to J1-J4.

    The point is the J5 joint origin. Unlike the J4 origin, it moves when J3 rotates
    a bent elbow, so it is a useful model-consistent proxy for the arm-angle degree
    of freedom.
    """
    base_pos = np.asarray(base_position, dtype=np.float64)
    q = np.asarray(shoulder_elbow_q, dtype=np.float64)

    if base_pos.shape != (3,):
        raise ValueError(f"base_position must have shape (3,), got {base_pos.shape}")
    if q.shape != (4,):
        raise ValueError(f"shoulder_elbow_q must have shape (4,), got {q.shape}")

    shoulder_offset, joint_2_offset, joint_4_offset, _ = _arm_geometry(side)

    rotation = quat_to_rotation_matrix(base_quaternion)
    position = base_pos + rotation @ shoulder_offset
    shoulder_position = position.copy()

    joint_origins: list[FloatArray] = []
    joint_axes_world: list[FloatArray] = []
    child_offsets = (
        joint_2_offset,
        _JOINT_3_OFFSET,
        joint_4_offset,
        _JOINT_5_OFFSET,
    )

    for axis, angle, child_offset in zip(_JOINT_AXES, q, child_offsets, strict=True):
        joint_origins.append(position.copy())
        joint_axes_world.append(rotation @ axis)
        rotation = rotation @ _rotation_about_axis(axis, float(angle))
        position = position + rotation @ child_offset

    swivel_point_position = position
    jacobian = np.column_stack(
        [
            np.cross(axis_world, swivel_point_position - joint_origin)
            for axis_world, joint_origin in zip(joint_axes_world, joint_origins, strict=True)
        ]
    )

    return ArmSwivelKinematics(
        shoulder_position=shoulder_position,
        swivel_point_position=swivel_point_position,
        jacobian=jacobian,
    )


def arm_swivel_direction_body(
    arm: ArmSwivelKinematics,
    tcp_position: npt.ArrayLike,
    base_quaternion: npt.ArrayLike,
) -> FloatArray | None:
    """Return the current arm-angle direction expressed in the base frame."""
    tcp = np.asarray(tcp_position, dtype=np.float64)
    if tcp.shape != (3,):
        raise ValueError(f"tcp_position must have shape (3,), got {tcp.shape}")

    shoulder_to_tcp = tcp - arm.shoulder_position
    tcp_distance = float(np.linalg.norm(shoulder_to_tcp))
    if tcp_distance < 1.0e-9:
        return None

    swivel_axis = shoulder_to_tcp / tcp_distance
    shoulder_to_point = arm.swivel_point_position - arm.shoulder_position
    radial = shoulder_to_point - float(shoulder_to_point @ swivel_axis) * swivel_axis
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm < 1.0e-9:
        return None

    base_rotation = quat_to_rotation_matrix(base_quaternion)
    return base_rotation.T @ (radial / radial_norm)


def preferred_swivel_point_position(
    arm: ArmSwivelKinematics,
    tcp_target_position: npt.ArrayLike,
    base_quaternion: npt.ArrayLike,
    *,
    side: ArmSide,
    max_swivel_step: float,
    preferred_direction_body: npt.ArrayLike | None = None,
) -> FloatArray:
    """Rotate the forearm toward a preferred body-frame arm-angle direction.

    The axial and radial distances around the shoulder-to-target line are preserved.
    Only a bounded arm-angle step is requested, which keeps the secondary task smooth.
    The arm's outward direction is used only when no valid reference is available.
    """
    if max_swivel_step < 0.0:
        raise ValueError(f"max_swivel_step must be non-negative, got {max_swivel_step}")

    target = np.asarray(tcp_target_position, dtype=np.float64)
    if target.shape != (3,):
        raise ValueError(f"tcp_target_position must have shape (3,), got {target.shape}")

    shoulder_to_target = target - arm.shoulder_position
    target_distance = float(np.linalg.norm(shoulder_to_target))
    if target_distance < 1.0e-9:
        return arm.swivel_point_position.copy()

    swivel_axis = shoulder_to_target / target_distance
    shoulder_to_point = arm.swivel_point_position - arm.shoulder_position
    axial_distance = float(shoulder_to_point @ swivel_axis)
    radial = shoulder_to_point - axial_distance * swivel_axis
    radial_distance = float(np.linalg.norm(radial))
    if radial_distance < 1.0e-9:
        return arm.swivel_point_position.copy()

    _, _, _, outward_body = _arm_geometry(side)
    if preferred_direction_body is None:
        preferred_body = outward_body
    else:
        preferred_body = np.asarray(preferred_direction_body, dtype=np.float64)
        if preferred_body.shape != (3,):
            raise ValueError(
                "preferred_direction_body must have shape (3,), "
                f"got {preferred_body.shape}"
            )
        preferred_body_norm = float(np.linalg.norm(preferred_body))
        if preferred_body_norm < 1.0e-9:
            preferred_body = outward_body
        else:
            preferred_body = preferred_body / preferred_body_norm

    base_rotation = quat_to_rotation_matrix(base_quaternion)
    preferred = base_rotation @ preferred_body
    preferred = preferred - float(preferred @ swivel_axis) * swivel_axis

    preferred_norm = float(np.linalg.norm(preferred))
    if preferred_norm < 1.0e-9:
        preferred = base_rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        preferred = preferred - float(preferred @ swivel_axis) * swivel_axis
        preferred_norm = float(np.linalg.norm(preferred))
    if preferred_norm < 1.0e-9:
        return arm.swivel_point_position.copy()

    radial_unit = radial / radial_distance
    preferred_unit = preferred / preferred_norm
    signed_sine = float(swivel_axis @ np.cross(radial_unit, preferred_unit))
    cosine = float(np.clip(radial_unit @ preferred_unit, -1.0, 1.0))
    swivel_error = float(np.arctan2(signed_sine, cosine))
    swivel_step = float(np.clip(swivel_error, -max_swivel_step, max_swivel_step))

    desired_radial = _rotation_about_axis(swivel_axis, swivel_step) @ radial
    return arm.shoulder_position + axial_distance * swivel_axis + desired_radial
