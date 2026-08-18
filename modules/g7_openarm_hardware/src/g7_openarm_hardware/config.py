from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig, parse_bool

_EXPECTED_MOTOR_IDS = set(range(1, 9))
_ALLOWED_DIRECTIONS = {-1.0, 1.0}


def _require_sequence_length(values: list[Any], *, field: str, length: int) -> None:
    if len(values) != length:
        raise ValueError(f"{field} must contain {length} values, got {len(values)}")


@dataclass(frozen=True, slots=True)
class HardwareConfig(BaseConfig):
    hz: float
    imu_hz: int
    base_can: str
    left_arm_can: str
    right_arm_can: str
    can_fd: bool
    base_ids: list[int]
    base_direction: list[float]
    left_arm_direction: list[float]
    right_arm_direction: list[float]
    # Calibrated physical motor positions in radians.
    left_gripper_open: float
    left_gripper_close: float
    right_gripper_open: float
    right_gripper_close: float
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"hardware.hz must be positive, got {self.hz}")

        if self.imu_hz <= 0:
            raise ValueError(f"hardware.imu_hz must be positive, got {self.imu_hz}")

        for field, value in (
            ("hardware.base_can", self.base_can),
            ("hardware.left_arm_can", self.left_arm_can),
            ("hardware.right_arm_can", self.right_arm_can),
        ):
            if not value:
                raise ValueError(f"{field} must not be empty")

        _require_sequence_length(self.base_ids, field="hardware.base_ids", length=8)
        if any(type(value) is not int for value in self.base_ids):
            raise ValueError(f"hardware.base_ids must contain only integers: {self.base_ids!r}")
        if set(self.base_ids) != _EXPECTED_MOTOR_IDS:
            raise ValueError(
                f"hardware.base_ids must be a permutation of motor IDs 1..8, got {self.base_ids!r}"
            )

        direction_fields = (
            ("hardware.base_direction", self.base_direction),
            ("hardware.left_arm_direction", self.left_arm_direction),
            ("hardware.right_arm_direction", self.right_arm_direction),
        )
        for field, values in direction_fields:
            _require_sequence_length(values, field=field, length=8)
            invalid = [value for value in values if value not in _ALLOWED_DIRECTIONS]
            if invalid:
                raise ValueError(f"{field} values must be -1.0 or 1.0, got {invalid!r}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    def active_can_interfaces(
        self,
        *,
        base_enabled: bool,
        arms_enabled: bool = True,
    ) -> tuple[str, ...]:
        interfaces: list[str] = []
        if base_enabled:
            interfaces.append(self.base_can)
        if arms_enabled:
            interfaces.extend((self.left_arm_can, self.right_arm_can))
        return tuple(interfaces)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> HardwareConfig:
        section = data.get("hardware")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [hardware] section")

        return cls(
            hz=float(section["hz"]),
            imu_hz=int(section["imu_hz"]),
            base_can=str(section["base_can"]).strip(),
            left_arm_can=str(section["left_arm_can"]).strip(),
            right_arm_can=str(section["right_arm_can"]).strip(),
            can_fd=parse_bool(section["can_fd"], field="hardware.can_fd"),
            base_ids=list(section["base_ids"]),
            base_direction=[float(value) for value in section["base_direction"]],
            left_arm_direction=[float(value) for value in section["left_arm_direction"]],
            right_arm_direction=[float(value) for value in section["right_arm_direction"]],
            left_gripper_open=float(section["left_gripper_open"]),
            left_gripper_close=float(section["left_gripper_close"]),
            right_gripper_open=float(section["right_gripper_open"]),
            right_gripper_close=float(section["right_gripper_close"]),
            dds=DDSConfig.from_mapping(data),
        )


config = HardwareConfig.load()
