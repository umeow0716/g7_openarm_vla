from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig


@dataclass(frozen=True, slots=True)
class DDSConfig:
    domain_id: int
    interface: str


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

    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"hardware.hz must be positive, got {self.hz}")

        if self.imu_hz <= 0:
            raise ValueError(f"hardware.imu_hz must be positive, got {self.imu_hz}")

        if not self.dds.interface:
            raise ValueError("dds.interface must not be empty")

        for i, val in enumerate(self.base_ids):
            if not isinstance(val, int):
                raise ValueError(f"hardware.base_ids[{i}] must be int: {val}")

        for i, val in enumerate(self.base_direction):
            if not isinstance(val, float):
                raise ValueError(f"hardware.base_direction[{i}] must be float: {val}")

        for i, val in enumerate(self.left_arm_direction):
            if not isinstance(val, float):
                raise ValueError(f"hardware.left_arm_direction[{i}] must be float: {val}")

        for i, val in enumerate(self.right_arm_direction):
            if not isinstance(val, float):
                raise ValueError(f"hardware.right_arm_direction[{i}] must be float: {val}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> HardwareConfig:
        section = data.get("hardware")
        dds_section = data.get("dds")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [hardware] section")

        if not isinstance(dds_section, Mapping):
            raise ValueError("Missing [dds] section")

        return cls(
            hz=float(section["hz"]),
            imu_hz=int(section["imu_hz"]),
            base_can=str(section["base_can"]),
            left_arm_can=str(section["left_arm_can"]),
            right_arm_can=str(section["right_arm_can"]),
            can_fd=bool(section["can_fd"]),
            base_ids=list(section["base_ids"]),
            base_direction=list(section["base_direction"]),
            left_arm_direction=list(section["left_arm_direction"]),
            right_arm_direction=list(section["right_arm_direction"]),
            dds=DDSConfig(
                domain_id=int(dds_section.get("domain_id", 0)),
                interface=str(dds_section.get("interface", "lo")),
            ),
        )


config = HardwareConfig.load()
