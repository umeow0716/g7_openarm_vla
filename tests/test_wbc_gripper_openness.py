from __future__ import annotations

import numpy as np
import pytest

from g7_openarm_utils import (
    LEFT_GRIPPER_MOTOR_NAME,
    RIGHT_GRIPPER_MOTOR_NAME,
    arm_command_index,
)
from g7_openarm_wbc.control_layout import ARM_CONTROL_SIZE, openarm_command_from_arm_control


def test_wbc_wire_carries_gripper_openness_without_rescaling_or_inversion() -> None:
    output = openarm_command_from_arm_control(
        np.zeros(ARM_CONTROL_SIZE, dtype=np.float64),
        left_gripper_openness=0.25,
        right_gripper_openness=0.75,
    )
    assert output[arm_command_index(LEFT_GRIPPER_MOTOR_NAME)] == pytest.approx(0.25)
    assert output[arm_command_index(RIGHT_GRIPPER_MOTOR_NAME)] == pytest.approx(0.75)


@pytest.mark.parametrize("invalid", [-0.01, 1.01, float("inf"), float("nan")])
def test_wbc_rejects_gripper_values_outside_openness_range(invalid: float) -> None:
    with pytest.raises(ValueError):
        openarm_command_from_arm_control(
            np.zeros(ARM_CONTROL_SIZE, dtype=np.float64),
            left_gripper_openness=invalid,
            right_gripper_openness=0.5,
        )
