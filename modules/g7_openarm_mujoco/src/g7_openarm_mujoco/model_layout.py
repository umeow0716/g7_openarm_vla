from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import mujoco

from g7_openarm_utils import (
    FLOATING_BASE_CONFIG_NAMES,
    FLOATING_BASE_JOINT_NAME,
    FLOATING_BASE_VELOCITY_NAMES,
    MODEL_JOINTS_BY_MOTOR_NAME,
    MOTOR_NAMES,
)

from .sensors import scalar_sensor_address


@dataclass(frozen=True)
class MuJoCoJointAddress:
    """Resolved MuJoCo addresses for one named one-DoF model joint."""

    name: str
    joint_id: int
    qpos_index: int
    qvel_index: int
    actuator_id: int


@dataclass(frozen=True)
class MuJoCoMotorAddress:
    """One logical hardware motor mapped to one or more MuJoCo joints."""

    name: str
    joints: tuple[MuJoCoJointAddress, ...]
    torque_sensor_address: int

    @property
    def primary_joint(self) -> MuJoCoJointAddress:
        return self.joints[0]


class MuJoCoModelLayout:
    """Fail-fast, name-resolved view of the G7 MuJoCo model.

    MuJoCo is free to lay qpos, qvel, and actuators out in different orders.
    This object resolves each domain independently once during initialization and
    leaves the control loop with precomputed integer addresses only.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model

        floating = self._joint_by_name(FLOATING_BASE_JOINT_NAME)
        if int(model.jnt_type[floating.id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(
                f"MuJoCo joint {FLOATING_BASE_JOINT_NAME!r} must be a free joint"
            )

        self.floating_joint_id = int(floating.id)
        self.floating_qpos_start = int(model.jnt_qposadr[floating.id])
        self.floating_qvel_start = int(model.jnt_dofadr[floating.id])
        self.floating_qpos_slice = slice(
            self.floating_qpos_start,
            self.floating_qpos_start + len(FLOATING_BASE_CONFIG_NAMES),
        )
        self.floating_qvel_slice = slice(
            self.floating_qvel_start,
            self.floating_qvel_start + len(FLOATING_BASE_VELOCITY_NAMES),
        )
        self._floating_qpos_by_name = MappingProxyType(
            {
                name: self.floating_qpos_start + offset
                for offset, name in enumerate(FLOATING_BASE_CONFIG_NAMES)
            }
        )
        self._floating_qvel_by_name = MappingProxyType(
            {
                name: self.floating_qvel_start + offset
                for offset, name in enumerate(FLOATING_BASE_VELOCITY_NAMES)
            }
        )

        joint_names = tuple(
            dict.fromkeys(
                joint_name
                for motor_name in MOTOR_NAMES
                for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
            )
        )
        resolved_joints = {
            joint_name: self._resolve_scalar_joint(joint_name) for joint_name in joint_names
        }
        self.joint_by_name: Mapping[str, MuJoCoJointAddress] = MappingProxyType(
            resolved_joints
        )

        motors = {
            motor_name: MuJoCoMotorAddress(
                name=motor_name,
                joints=tuple(
                    resolved_joints[joint_name]
                    for joint_name in MODEL_JOINTS_BY_MOTOR_NAME[motor_name]
                ),
                torque_sensor_address=scalar_sensor_address(
                    model, f"{motor_name}_torque"
                ),
            )
            for motor_name in MOTOR_NAMES
        }
        self.motor_by_name: Mapping[str, MuJoCoMotorAddress] = MappingProxyType(motors)

        actuator_ids = [
            joint.actuator_id for motor in motors.values() for joint in motor.joints
        ]
        if len(actuator_ids) != len(set(actuator_ids)):
            raise ValueError("MuJoCo actuator mapping is not one-to-one by joint name")

    def _joint_by_name(self, name: str):
        try:
            return self.model.joint(name)
        except KeyError as exc:
            raise KeyError(f"MuJoCo joint not found: {name}") from exc

    def _resolve_scalar_joint(self, name: str) -> MuJoCoJointAddress:
        joint = self._joint_by_name(name)
        joint_id = int(joint.id)
        joint_type = int(self.model.jnt_type[joint_id])
        if joint_type not in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            raise ValueError(
                f"MuJoCo joint {name!r} must be scalar hinge/slide, type={joint_type}"
            )

        actuator_ids = [
            actuator_id
            for actuator_id in range(self.model.nu)
            if int(self.model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(actuator_ids) != 1:
            raise ValueError(
                f"MuJoCo joint {name!r} must have exactly one actuator, "
                f"found {len(actuator_ids)}"
            )

        return MuJoCoJointAddress(
            name=name,
            joint_id=joint_id,
            qpos_index=int(self.model.jnt_qposadr[joint_id]),
            qvel_index=int(self.model.jnt_dofadr[joint_id]),
            actuator_id=actuator_ids[0],
        )

    def qpos_index(self, name: str) -> int:
        """Resolve either a floating-base config name or a scalar joint name."""
        if name in self._floating_qpos_by_name:
            return self._floating_qpos_by_name[name]
        try:
            return self.joint_by_name[name].qpos_index
        except KeyError as exc:
            raise KeyError(f"Unknown MuJoCo q name: {name}") from exc

    def qvel_index(self, name: str) -> int:
        """Resolve either a floating-base velocity name or a scalar joint name."""
        if name in self._floating_qvel_by_name:
            return self._floating_qvel_by_name[name]
        try:
            return self.joint_by_name[name].qvel_index
        except KeyError as exc:
            raise KeyError(f"Unknown MuJoCo v name: {name}") from exc

    def motor(self, name: str) -> MuJoCoMotorAddress:
        try:
            return self.motor_by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown logical motor name: {name}") from exc
