from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from g7_openarm_utils import (
    ARM_JOINT_NAMES,
    ARM_LOWSTATE_MOTOR_INDICES,
    ARM_MOTOR16_INDICES,
    ARM_MOTOR_NAMES,
    ARM_POSITION_LOWER_RAD,
    ARM_POSITION_UPPER_RAD,
    ARM_VELOCITY_LIMIT_RAD_S,
    arm_limit_arrays,
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


def test_arm_limit_constants_match_urdf_except_intentional_j3_software_envelope() -> None:
    urdf_lower, urdf_upper, urdf_velocity = _urdf_arm_limits()

    # Preserve the controller's existing asymmetric J3 software safety envelope.
    j3_by_motor = {
        "L_3": (-np.pi / 6.0, np.pi / 3.0),
        "R_3": (-np.pi / 3.0, np.pi / 6.0),
    }
    for index, motor_name in enumerate(ARM_MOTOR_NAMES):
        if motor_name in j3_by_motor:
            expected_lower, expected_upper = j3_by_motor[motor_name]
            assert ARM_POSITION_LOWER_RAD[index] == expected_lower
            assert ARM_POSITION_UPPER_RAD[index] == expected_upper
            assert expected_lower >= urdf_lower[index]
            assert expected_upper <= urdf_upper[index]
        else:
            assert np.isclose(ARM_POSITION_LOWER_RAD[index], urdf_lower[index], atol=1e-12)
            assert np.isclose(ARM_POSITION_UPPER_RAD[index], urdf_upper[index], atol=1e-12)

    np.testing.assert_allclose(
        ARM_VELOCITY_LIMIT_RAD_S, urdf_velocity, rtol=0.0, atol=1e-12
    )


def test_named_limit_lookup_is_order_independent() -> None:
    names = ("R_3", "L_1", "L_3")
    lower, upper, velocity = arm_limit_arrays(names)
    for output_index, motor_name in enumerate(names):
        canonical_index = ARM_MOTOR_NAMES.index(motor_name)
        assert lower[output_index] == ARM_POSITION_LOWER_RAD[canonical_index]
        assert upper[output_index] == ARM_POSITION_UPPER_RAD[canonical_index]
        assert velocity[output_index] == ARM_VELOCITY_LIMIT_RAD_S[canonical_index]


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
        motor_names=ARM_MOTOR_NAMES,
    )
    lower_at_max, upper_at_max = position_limited_velocity_bounds(
        ARM_POSITION_UPPER_RAD,
        vmax,
        dt=0.01,
        motor_names=ARM_MOTOR_NAMES,
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

    lower, upper = position_limited_velocity_bounds(
        position,
        vmax,
        dt=0.01,
        motor_names=ARM_MOTOR_NAMES,
    )

    assert lower[0] == 0.0
    assert upper[0] == 1.0
    assert lower[1] == -1.0
    assert upper[1] == 0.0


def test_velocity_bounds_reduce_speed_near_position_limit() -> None:
    vmax = np.ones(14, dtype=np.float64)
    position = (ARM_POSITION_LOWER_RAD + ARM_POSITION_UPPER_RAD) / 2.0
    position[2] = ARM_POSITION_UPPER_RAD[2] - 0.002

    lower, upper = position_limited_velocity_bounds(
        position,
        vmax,
        dt=0.01,
        motor_names=ARM_MOTOR_NAMES,
    )

    assert lower[2] == -1.0
    assert np.isclose(upper[2], 0.2)
