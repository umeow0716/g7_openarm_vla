from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .base import BaseConfig
from .parsing import parse_bool


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
        raise ValueError(f"general.control_mode must be one of [{allowed}], got {value!r}")


@dataclass(frozen=True, slots=True)
class GeneralConfig(BaseConfig):
    debugging: bool
    control_mode: ControlMode

    def __post_init__(self) -> None:
        if type(self.debugging) is not bool:
            raise ValueError(f"general.debugging must be bool, got {self.debugging!r}")

        if not isinstance(self.control_mode, ControlMode):
            raise ValueError(f"Invalid general.control_mode: {self.control_mode!r}")

    @property
    def base_enabled(self) -> bool:
        return self.control_mode is ControlMode.WBC

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> GeneralConfig:
        section = data.get("general")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [general] section")

        return cls(
            debugging=parse_bool(
                section.get("debugging", False),
                field="general.debugging",
            ),
            control_mode=ControlMode.parse(section.get("control_mode", ControlMode.WBC.value)),
        )


config = GeneralConfig.load()
