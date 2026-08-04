from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from g7_openarm_utils import (
    ARM_JOINT_NAMES,
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_MOTOR16_INDICES,
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
    position_limited_velocity_bounds,
)


def _urdf_arm_limits() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repository_root = Path(__file__).resolve().parents[1]
    urdf_path = repository_root / (
        "modules/g7_openarm_mujoco/src/g7_openarm_mujoco/model/urdf/g7_openarm.urdf"
    )
    root = ElementTree.parse(urdf_path).getroot()
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    lower = []
    upper = []
    velocity = []
    for name in ARM_JOINT_NAMES:
        limit = joints[name].find("limit")
        assert limit is not None
        lower.append(float(limit.attrib["lower"]))
        upper.append(float(limit.attrib["upper"]))
        velocity.append(float(limit.attrib["velocity"]))

    return (
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(velocity, dtype=np.float64),
    )


def test_arm_limit_constants_match_urdf() -> None:
    lower, upper, velocity = _urdf_arm_limits()

    np.testing.assert_allclose(ARM_POSITION_LOWER_RAD, lower, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ARM_POSITION_UPPER_RAD, upper, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(ARM_VELOCITY_LIMIT_RAD_S, velocity, rtol=0.0, atol=1e-12)


def test_both_j3_limits_are_plus_minus_pi_over_four() -> None:
    for j3_index in (2, 9):
        assert ARM_POSITION_LOWER_RAD[j3_index] == -np.pi / 4.0
        assert ARM_POSITION_UPPER_RAD[j3_index] == np.pi / 4.0


def test_project_arm_layout_indices_exclude_grippers() -> None:
    np.testing.assert_array_equal(
        ARM_LOWSTATE_MOTOR_INDICES,
        np.array([*range(8, 15), *range(16, 23)]),
    )
    np.testing.assert_array_equal(
        ARM_MOTOR16_INDICES,
        np.array([*range(0, 7), *range(8, 15)]),
    )


def test_velocity_bounds_stop_motion_outward_at_limits() -> None:
    vmax = np.ones(14, dtype=np.float64)

    lower_at_min, upper_at_min = position_limited_velocity_bounds(
        ARM_POSITION_LOWER_RAD,
        vmax,
        dt=0.01,
    )
    lower_at_max, upper_at_max = position_limited_velocity_bounds(
        ARM_POSITION_UPPER_RAD,
        vmax,
        dt=0.01,
    )

    np.testing.assert_array_equal(lower_at_min, np.zeros(14))
    np.testing.assert_array_equal(upper_at_min, vmax)
    np.testing.assert_array_equal(lower_at_max, -vmax)
    np.testing.assert_array_equal(upper_at_max, np.zeros(14))


def test_velocity_bounds_allow_only_recovery_when_feedback_is_outside() -> None:
    vmax = np.ones(14, dtype=np.float64)
    position = (ARM_POSITION_LOWER_RAD + ARM_POSITION_UPPER_RAD) / 2.0
    position[0] = ARM_POSITION_LOWER_RAD[0] - 0.1
    position[1] = ARM_POSITION_UPPER_RAD[1] + 0.1

    lower, upper = position_limited_velocity_bounds(position, vmax, dt=0.01)

    assert lower[0] == 0.0
    assert upper[0] == 1.0
    assert lower[1] == -1.0
    assert upper[1] == 0.0


def test_velocity_bounds_reduce_speed_near_position_limit() -> None:
    vmax = np.ones(14, dtype=np.float64)
    position = (ARM_POSITION_LOWER_RAD + ARM_POSITION_UPPER_RAD) / 2.0
    position[2] = ARM_POSITION_UPPER_RAD[2] - 0.002

    lower, upper = position_limited_velocity_bounds(position, vmax, dt=0.01)

    assert lower[2] == -1.0
    assert np.isclose(upper[2], 0.2)
