from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_config import GeneralConfig
from g7_openarm_lowlevel.initialization import (
    INITIAL_DURATION_S,
    ArmInitializer,
)
from g7_openarm_mujoco.actuation import (
    gripper_command_to_mujoco_position,
    gripper_command_velocity_to_mujoco_velocity,
    motor_actuation_enabled,
    mujoco_gripper_position_to_command,
    mujoco_gripper_velocity_to_command_velocity,
)
from g7_openarm_utils import (
    GRIPPER_COMMAND_RANGE,
    gripper_command_to_motor_position,
    gripper_command_velocity_to_motor_velocity,
    gripper_motor_position_to_command,
    gripper_motor_velocity_to_command_velocity,
    gripper_openness_to_command,
)


def _general(mode: str) -> GeneralConfig:
    return GeneralConfig.from_mapping(
        {
            "general": {
                "debugging": False,
                "control_mode": mode,
                "initial_pos": [0.435, 0.0, 0.0, -0.525, 0.0, 0.0, 0.613],
                "initial_gripper": 1.0,
            }
        }
    )


@pytest.mark.parametrize(
    ("left_enabled", "right_enabled", "active_slice", "inactive_slice"),
    [
        (True, False, slice(0, 8), slice(8, 16)),
        (False, True, slice(8, 16), slice(0, 8)),
    ],
)
def test_single_arm_initializer_only_moves_enabled_side(
    left_enabled: bool,
    right_enabled: bool,
    active_slice: slice,
    inactive_slice: slice,
) -> None:
    target = (0.435, 0.0, 0.0, -0.525, 0.0, 0.0, 0.613)
    start = np.arange(16, dtype=np.float64) * 0.1
    initializer = ArmInitializer(
        target,
        target_gripper=1.0,
        left_enabled=left_enabled,
        right_enabled=right_enabled,
    )
    initializer.start(start, now=1.0)

    q_end, dq_end, done = initializer.sample(now=1.0 + INITIAL_DURATION_S)

    expected_active = np.concatenate(
        [np.asarray(target), [gripper_openness_to_command(1.0)]]
    )
    np.testing.assert_allclose(q_end[active_slice], expected_active)
    np.testing.assert_allclose(q_end[inactive_slice], start[inactive_slice])
    np.testing.assert_array_equal(dq_end, np.zeros(16))
    assert done is True


@pytest.mark.parametrize(
    ("mode", "allowed"),
    [
        ("wbc", True),
        ("arm-only", True),
        ("base-only", False),
        ("left-arm", True),
        ("right-arm", True),
        ("left-arm-only", True),
        ("right-arm-only", True),
    ],
)
def test_lowlevel_initial_is_allowed_for_every_active_arm_mode(
    mode: str,
    allowed: bool,
) -> None:
    assert _general(mode).lowlevel_initial_allowed is allowed


def test_initial_gripper_is_normalized_openness() -> None:
    general = _general("wbc")
    assert general.initial_gripper == 1.0
    assert gripper_openness_to_command(general.initial_gripper) == 0.0
    assert gripper_openness_to_command(0.0) == GRIPPER_COMMAND_RANGE


def test_initial_gripper_defaults_to_fully_open() -> None:
    general = GeneralConfig.from_mapping(
        {"general": {"debugging": False, "control_mode": "wbc"}}
    )
    assert general.initial_gripper == 1.0


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("inf"), float("nan")])
def test_initial_gripper_rejects_invalid_values(invalid: float) -> None:
    with pytest.raises(ValueError):
        GeneralConfig.from_mapping(
            {
                "general": {
                    "debugging": False,
                    "control_mode": "wbc",
                    "initial_gripper": invalid,
                }
            }
        )


def test_hardware_gripper_mapping_round_trip() -> None:
    open_position = 0.0
    close_position = -1.2
    for command in (0.0, 0.1, GRIPPER_COMMAND_RANGE):
        position = gripper_command_to_motor_position(
            command,
            open_position=open_position,
            close_position=close_position,
        )
        restored = gripper_motor_position_to_command(
            position,
            open_position=open_position,
            close_position=close_position,
        )
        assert restored == pytest.approx(command)

    command_velocity = -0.03
    motor_velocity = gripper_command_velocity_to_motor_velocity(
        command_velocity,
        open_position=open_position,
        close_position=close_position,
    )
    restored_velocity = gripper_motor_velocity_to_command_velocity(
        motor_velocity,
        open_position=open_position,
        close_position=close_position,
    )
    assert restored_velocity == pytest.approx(command_velocity)


def test_mujoco_gripper_mapping_round_trip_and_open_direction() -> None:
    assert gripper_command_to_mujoco_position(0.0) == pytest.approx(0.045)
    assert gripper_command_to_mujoco_position(GRIPPER_COMMAND_RANGE) == pytest.approx(0.0)

    for command in (0.0, 0.2, GRIPPER_COMMAND_RANGE):
        position = gripper_command_to_mujoco_position(command)
        assert mujoco_gripper_position_to_command(position) == pytest.approx(command)

    command_velocity = -0.04
    mujoco_velocity = gripper_command_velocity_to_mujoco_velocity(command_velocity)
    assert mujoco_velocity > 0.0
    assert mujoco_gripper_velocity_to_command_velocity(mujoco_velocity) == pytest.approx(
        command_velocity
    )


@pytest.mark.parametrize(
    ("mode", "base", "left", "right"),
    [
        ("wbc", True, True, True),
        ("arm-only", True, True, True),
        ("base-only", True, False, False),
        ("left-arm", True, True, False),
        ("right-arm", True, False, True),
        ("left-arm-only", False, True, False),
        ("right-arm-only", False, False, True),
    ],
)
def test_mujoco_motor_actuation_matches_control_mode(
    mode: str,
    base: bool,
    left: bool,
    right: bool,
) -> None:
    general = _general(mode)
    assert motor_actuation_enabled(0, general) is base
    assert motor_actuation_enabled(8, general) is left
    assert motor_actuation_enabled(16, general) is right
