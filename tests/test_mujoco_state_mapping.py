from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from g7_openarm_mujoco.state_mapping import write_floating_base_qpos, write_lowstate_qpos
from g7_openarm_utils import (
    FLOATING_BASE_CONFIG_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    gripper_command_to_model_position,
    motor_index,
)


class _PermutedLayout:
    def __init__(self) -> None:
        names = [
            *FLOATING_BASE_CONFIG_NAMES,
            *(
                joint_name
                for motor_name in MOTOR_NAMES
                for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
            ),
        ]
        self._qpos = {
            name: index
            for index, name in enumerate(reversed(tuple(dict.fromkeys(names))))
        }

    def qpos_index(self, name: str) -> int:
        return self._qpos[name]


def test_lowstate_to_mujoco_qpos_uses_joint_names_not_offsets() -> None:
    layout = _PermutedLayout()
    data = SimpleNamespace(qpos=np.full(len(layout._qpos), np.nan, dtype=np.float64))
    lowstate = SimpleNamespace(
        motor_state=[SimpleNamespace(q=float(index + 1)) for index in range(len(MOTOR_NAMES))]
    )

    write_lowstate_qpos(layout, data, lowstate)

    for motor_name in MOTOR_NAMES:
        expected = lowstate.motor_state[motor_index(motor_name)].q
        if motor_name in (LEFT_GRIPPER_MOTOR_NAME, RIGHT_GRIPPER_MOTOR_NAME):
            expected = gripper_command_to_model_position(expected)
        for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]:
            assert data.qpos[layout.qpos_index(joint_name)] == expected


def test_floating_base_qpos_uses_component_names_not_offsets() -> None:
    layout = _PermutedLayout()
    data = SimpleNamespace(qpos=np.full(len(layout._qpos), np.nan, dtype=np.float64))
    odom = SimpleNamespace(
        position=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        quaternion=SimpleNamespace(w=0.9, x=0.1, y=0.2, z=0.3),
    )

    write_floating_base_qpos(layout, data, odom, z=4.0)

    expected = {
        "x": 1.0,
        "y": 2.0,
        "z": 4.0,
        "q_w": 0.9,
        "q_x": 0.1,
        "q_y": 0.2,
        "q_z": 0.3,
    }
    for name, value in expected.items():
        assert data.qpos[layout.qpos_index(name)] == value
