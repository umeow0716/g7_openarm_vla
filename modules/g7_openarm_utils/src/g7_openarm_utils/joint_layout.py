from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

# Logical motor names define the first 24 slots of Unitree LowState_/LowCmd_
# used by this project. The Unitree message itself is index-only, so this is
# the single canonical bridge between a semantic motor name and that protocol
# index.
BASE_MOTOR_NAMES = (
    "AMR_FL",
    "AMR_FLW",
    "AMR_FR",
    "AMR_FRW",
    "AMR_RL",
    "AMR_RLW",
    "AMR_RR",
    "AMR_RRW",
)
BASE_STEER_MOTOR_NAMES = (
    "AMR_FL",
    "AMR_FR",
    "AMR_RL",
    "AMR_RR",
)
BASE_WHEEL_MOTOR_NAMES = (
    "AMR_FLW",
    "AMR_FRW",
    "AMR_RLW",
    "AMR_RRW",
)
LEFT_ARM_MOTOR_NAMES = tuple(f"L_{index}" for index in range(1, 8))
RIGHT_ARM_MOTOR_NAMES = tuple(f"R_{index}" for index in range(1, 8))
LEFT_GRIPPER_MOTOR_NAME = "gripper_L"
RIGHT_GRIPPER_MOTOR_NAME = "gripper_R"
LEFT_HARDWARE_MOTOR_NAMES = LEFT_ARM_MOTOR_NAMES + (LEFT_GRIPPER_MOTOR_NAME,)
RIGHT_HARDWARE_MOTOR_NAMES = RIGHT_ARM_MOTOR_NAMES + (RIGHT_GRIPPER_MOTOR_NAME,)
ARM_COMMAND_MOTOR_NAMES = LEFT_HARDWARE_MOTOR_NAMES + RIGHT_HARDWARE_MOTOR_NAMES
ARM_MOTOR_NAMES = LEFT_ARM_MOTOR_NAMES + RIGHT_ARM_MOTOR_NAMES
MOTOR_NAMES = BASE_MOTOR_NAMES + ARM_COMMAND_MOTOR_NAMES
UNITREE_HG_MOTOR_ARRAY_SIZE = 35
if len(MOTOR_NAMES) > UNITREE_HG_MOTOR_ARRAY_SIZE:
    raise RuntimeError(
        "G7 logical motor layout exceeds Unitree HG LowState/LowCmd motor array size"
    )

MOTOR_INDEX_BY_NAME = MappingProxyType(
    {name: index for index, name in enumerate(MOTOR_NAMES)}
)
ARM_COMMAND_INDEX_BY_NAME = MappingProxyType(
    {name: index for index, name in enumerate(ARM_COMMAND_MOTOR_NAMES)}
)

AMR_COMMAND_NAMES = ("base_vx", "base_vy", "base_wz")
AMR_COMMAND_INDEX_BY_NAME = MappingProxyType(
    {name: index for index, name in enumerate(AMR_COMMAND_NAMES)}
)

# One logical gripper motor drives two model joints. All other motors are
# one-to-one with the URDF/MJCF joint carrying the same stem.
MODEL_JOINTS_BY_MOTOR_NAME = MappingProxyType(
    {
        **{name: (f"{name}_joint",) for name in BASE_MOTOR_NAMES},
        **{name: (f"{name}_joint",) for name in LEFT_ARM_MOTOR_NAMES},
        LEFT_GRIPPER_MOTOR_NAME: ("gripper_LL_joint", "gripper_LR_joint"),
        **{name: (f"{name}_joint",) for name in RIGHT_ARM_MOTOR_NAMES},
        RIGHT_GRIPPER_MOTOR_NAME: ("gripper_RL_joint", "gripper_RR_joint"),
    }
)
PRIMARY_MODEL_JOINT_BY_MOTOR_NAME = MappingProxyType(
    {name: joints[0] for name, joints in MODEL_JOINTS_BY_MOTOR_NAME.items()}
)
MODEL_JOINT_TO_MOTOR_NAME = MappingProxyType(
    {
        joint_name: motor_name
        for motor_name, joint_names in MODEL_JOINTS_BY_MOTOR_NAME.items()
        for joint_name in joint_names
    }
)

LEFT_ARM_JOINT_NAMES = tuple(f"{name}_joint" for name in LEFT_ARM_MOTOR_NAMES)
RIGHT_ARM_JOINT_NAMES = tuple(f"{name}_joint" for name in RIGHT_ARM_MOTOR_NAMES)
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES
ACTUATED_MODEL_JOINT_NAMES = tuple(
    PRIMARY_MODEL_JOINT_BY_MOTOR_NAME[name]
    for name in BASE_MOTOR_NAMES + ARM_MOTOR_NAMES
)

FLOATING_BASE_JOINT_NAME = "floating_base_joint"
FLOATING_BASE_CONFIG_NAMES = ("x", "y", "z", "q_w", "q_x", "q_y", "q_z")
FLOATING_BASE_VELOCITY_NAMES = (
    "lin_v_x",
    "lin_v_y",
    "lin_v_z",
    "ang_v_x",
    "ang_v_y",
    "ang_v_z",
)


def _index(mapping: Mapping[str, int], name: str, *, kind: str) -> int:
    try:
        return mapping[name]
    except KeyError as exc:
        raise KeyError(f"unknown {kind} name {name!r}") from exc


def motor_index(name: str) -> int:
    return _index(MOTOR_INDEX_BY_NAME, name, kind="motor")


def motor_indices(names: tuple[str, ...] | list[str]) -> npt.NDArray[np.intp]:
    return np.asarray([motor_index(name) for name in names], dtype=np.intp)


def arm_command_index(name: str) -> int:
    return _index(ARM_COMMAND_INDEX_BY_NAME, name, kind="arm command motor")


def arm_command_indices(names: tuple[str, ...] | list[str]) -> npt.NDArray[np.intp]:
    return np.asarray([arm_command_index(name) for name in names], dtype=np.intp)


def amr_command_index(name: str) -> int:
    return _index(AMR_COMMAND_INDEX_BY_NAME, name, kind="AMR command")


def amr_command_values(command: Any) -> npt.NDArray[np.float64]:
    return np.asarray(
        [command.data[amr_command_index(name)] for name in AMR_COMMAND_NAMES],
        dtype=np.float64,
    )


def arm_command_values(
    command: Any,
    names: tuple[str, ...] | list[str] = ARM_COMMAND_MOTOR_NAMES,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        [command.data[arm_command_index(name)] for name in names],
        dtype=np.float64,
    )


def motor_state_values(
    lowstate: Any,
    names: tuple[str, ...] | list[str],
    field: str,
) -> npt.NDArray[np.float64]:
    """Read a named set of Unitree motor-state fields in the requested name order."""
    try:
        values = [getattr(lowstate.motor_state[motor_index(name)], field) for name in names]
    except AttributeError as exc:
        raise AttributeError(f"motor state has no field {field!r}") from exc
    return np.asarray(values, dtype=np.float64)


def motor_command(lowcmd: Any, name: str) -> Any:
    """Return the Unitree MotorCmd_ associated with a canonical logical motor name."""
    return lowcmd.motor_cmd[motor_index(name)]


# Backward-compatible, derived index arrays. New code should prefer names and
# resolve them at the boundary where an index-only protocol/API must be used.
ARM_LOWSTATE_MOTOR_INDICES = motor_indices(list(ARM_MOTOR_NAMES))
ARM_MOTOR16_INDICES = arm_command_indices(list(ARM_MOTOR_NAMES))
for _array in (ARM_LOWSTATE_MOTOR_INDICES, ARM_MOTOR16_INDICES):
    _array.setflags(write=False)
