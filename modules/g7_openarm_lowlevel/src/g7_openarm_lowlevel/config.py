from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


class ControlMode(StrEnum):
    WBC = "wbc"
    ARM_ONLY = "arm-only"

    @classmethod
    def parse(cls, value: Any) -> ControlMode:
        if isinstance(value, cls):
            return value

        if isinstance(value, str):
            normalized = value.strip().casefold()
            try:
                return cls(normalized)
            except ValueError:
                pass

        allowed = ", ".join(mode.value for mode in cls)
        raise ValueError(f"lowlevel.control_mode must be one of [{allowed}], got {value!r}")


@dataclass(frozen=True, slots=True)
class LowLevelConfig(BaseConfig):
    hz: float
    control_mode: ControlMode
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"lowlevel.hz must be positive, got {self.hz}")

        if not isinstance(self.control_mode, ControlMode):
            raise ValueError(f"Invalid lowlevel.control_mode: {self.control_mode!r}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @property
    def base_enabled(self) -> bool:
        return self.control_mode is ControlMode.WBC

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> LowLevelConfig:
        section = data.get("lowlevel")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [lowlevel] section")

        return cls(
            hz=float(section["hz"]),
            control_mode=ControlMode.parse(section.get("control_mode", ControlMode.WBC.value)),
            dds=DDSConfig.from_mapping(data),
        )


config = LowLevelConfig.load()
