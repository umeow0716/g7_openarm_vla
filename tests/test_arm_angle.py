from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from g7_openarm_utils import quat_from_yaw
from g7_openarm_wbc.arm_angle import (
    ArmSide,
    arm_swivel_direction_body,
    arm_swivel_kinematics,
    preferred_swivel_point_position,
)


@pytest.mark.parametrize("side_value", ["left", "right"])
def test_arm_swivel_jacobian_matches_finite_difference(side_value: str) -> None:
    side = cast(ArmSide, side_value)
    base_position = np.array([0.2, -0.1, 0.16], dtype=np.float64)
    base_quaternion = quat_from_yaw(0.47)
    shoulder_elbow_q = np.array([0.35, -0.42, 0.61, -1.17], dtype=np.float64)

    arm = arm_swivel_kinematics(
        base_position,
        base_quaternion,
        shoulder_elbow_q,
        side=side,
    )

    epsilon = 1.0e-7
    finite_difference = np.empty((3, 4), dtype=np.float64)
    for index in range(4):
        delta = np.zeros(4, dtype=np.float64)
        delta[index] = epsilon
        plus = arm_swivel_kinematics(
            base_position,
            base_quaternion,
            shoulder_elbow_q + delta,
            side=side,
        ).swivel_point_position
        minus = arm_swivel_kinematics(
            base_position,
            base_quaternion,
            shoulder_elbow_q - delta,
            side=side,
        ).swivel_point_position
        finite_difference[:, index] = (plus - minus) / (2.0 * epsilon)

    np.testing.assert_allclose(arm.jacobian, finite_difference, rtol=1.0e-7, atol=1.0e-9)


@pytest.mark.parametrize("side_value", ["left", "right"])
def test_preferred_swivel_point_preserves_axis_and_radius(side_value: str) -> None:
    side = cast(ArmSide, side_value)
    base_position = np.array([0.0, 0.0, 0.16], dtype=np.float64)
    base_quaternion = quat_from_yaw(-0.31)
    shoulder_elbow_q = np.array([0.25, -0.55, 0.72, -1.1], dtype=np.float64)
    arm = arm_swivel_kinematics(
        base_position,
        base_quaternion,
        shoulder_elbow_q,
        side=side,
    )
    tcp_target = np.array([0.45, 0.12 if side == "left" else -0.12, 0.72])

    desired = preferred_swivel_point_position(
        arm,
        tcp_target,
        base_quaternion,
        side=side,
        max_swivel_step=np.deg2rad(15.0),
    )

    axis = tcp_target - arm.shoulder_position
    axis /= np.linalg.norm(axis)
    current = arm.swivel_point_position - arm.shoulder_position
    requested = desired - arm.shoulder_position

    assert requested @ axis == pytest.approx(current @ axis, abs=1.0e-12)
    current_radius = np.linalg.norm(current - (current @ axis) * axis)
    requested_radius = np.linalg.norm(requested - (requested @ axis) * axis)
    assert requested_radius == pytest.approx(current_radius, abs=1.0e-12)


def test_current_swivel_direction_reproduces_the_same_arm_angle() -> None:
    base_position = np.array([0.0, 0.0, 0.16], dtype=np.float64)
    base_quaternion = quat_from_yaw(0.28)
    arm = arm_swivel_kinematics(
        base_position,
        base_quaternion,
        np.array([0.4, -0.3, 0.7, -1.2], dtype=np.float64),
        side="left",
    )
    tcp_position = np.array([0.42, 0.18, 0.68], dtype=np.float64)
    direction_body = arm_swivel_direction_body(
        arm,
        tcp_position,
        base_quaternion,
    )
    assert direction_body is not None

    desired = preferred_swivel_point_position(
        arm,
        tcp_position,
        base_quaternion,
        side="left",
        max_swivel_step=np.deg2rad(15.0),
        preferred_direction_body=direction_body,
    )
    np.testing.assert_allclose(desired, arm.swivel_point_position, atol=1.0e-12)
