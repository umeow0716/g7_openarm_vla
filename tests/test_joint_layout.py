from __future__ import annotations


from g7_openarm_utils import (
    AMR_COMMAND_NAMES,
    ARM_COMMAND_MOTOR_NAMES,
    ARM_MOTOR_NAMES,
    BASE_MOTOR_NAMES,
    LEFT_GRIPPER_MOTOR_NAME,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
    RIGHT_GRIPPER_MOTOR_NAME,
    amr_command_index,
    arm_command_index,
    motor_index,
)


def test_canonical_motor_names_are_unique() -> None:
    assert len(MOTOR_NAMES) == 24
    assert len(MOTOR_NAMES) == len(set(MOTOR_NAMES))
    assert len(ARM_COMMAND_MOTOR_NAMES) == 16
    assert len(ARM_MOTOR_NAMES) == 14


def test_unitree_wire_indices_are_resolved_from_motor_names() -> None:
    expected = {
        **{name: index for index, name in enumerate(BASE_MOTOR_NAMES)},
        **{
            name: 8 + index
            for index, name in enumerate(ARM_COMMAND_MOTOR_NAMES)
        },
    }
    for name, index in expected.items():
        assert motor_index(name) == index


def test_openarm_wire_indices_are_resolved_from_motor_names() -> None:
    for expected, name in enumerate(ARM_COMMAND_MOTOR_NAMES):
        assert arm_command_index(name) == expected


def test_amr_wire_indices_are_resolved_from_command_names() -> None:
    for expected, name in enumerate(AMR_COMMAND_NAMES):
        assert amr_command_index(name) == expected


def test_logical_gripper_maps_to_two_model_joints() -> None:
    assert MODEL_JOINTS_BY_MOTOR_NAME[LEFT_GRIPPER_MOTOR_NAME] == (
        "gripper_LL_joint",
        "gripper_LR_joint",
    )
    assert MODEL_JOINTS_BY_MOTOR_NAME[RIGHT_GRIPPER_MOTOR_NAME] == (
        "gripper_RL_joint",
        "gripper_RR_joint",
    )
    assert all(
        len(joint_names) == 1
        for motor_name, joint_names in MODEL_JOINTS_BY_MOTOR_NAME.items()
        if motor_name not in (LEFT_GRIPPER_MOTOR_NAME, RIGHT_GRIPPER_MOTOR_NAME)
    )
