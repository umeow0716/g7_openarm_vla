from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from g7_openarm_config import BaseConfig


@dataclass(frozen=True, slots=True)
class DDSConfig:
    domain_id: int
    interface: str


@dataclass(frozen=True, slots=True)
class HardwareConfig(BaseConfig):
    hz: float
    base_can: str
    left_arm_can: str
    right_arm_can: str
    can_fd: bool

    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(
                f"hardware.hz must be positive, got {self.hz}"
            )

        if not self.dds.interface:
            raise ValueError(
                "dds.interface must not be empty"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "HardwareConfig":
        section = data.get("hardware")
        dds_section = data.get("dds")

        if not isinstance(section, Mapping):
            raise ValueError(
                "Missing [hardware] section"
            )

        if not isinstance(dds_section, Mapping):
            raise ValueError(
                "Missing [dds] section"
            )

        return cls(
            hz=float(section["hz"]),
            base_can=str(section["base_can"]),
            left_arm_can=str(section["left_arm_can"]),
            right_arm_can=str(section["right_arm_can"]),
            can_fd=bool(section["can_fd"]),
            dds=DDSConfig(
                domain_id=int(
                    dds_section.get("domain_id", 0)
                ),
                interface=str(
                    dds_section.get("interface", "lo")
                ),
            ),
        )


config = HardwareConfig.load()