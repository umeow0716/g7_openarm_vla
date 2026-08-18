from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .base import BaseConfig
from .parsing import parse_bool

DEFAULT_INITIAL_POS = (0.435, 0.0, 0.0, -0.525, 0.0, 0.0, 0.613)
DEFAULT_INITIAL_GRIPPER = 1.0


class ControlMode(StrEnum):
    WBC = "wbc"
    ARM_ONLY = "arm-only"
    BASE_ONLY = "base-only"
    LEFT_ARM = "left-arm"
    RIGHT_ARM = "right-arm"
    LEFT_ARM_ONLY = "left-arm-only"
    RIGHT_ARM_ONLY = "right-arm-only"

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
    initial_pos: tuple[float, ...]
    initial_gripper: float

    def __post_init__(self) -> None:
        if type(self.debugging) is not bool:
            raise ValueError(f"general.debugging must be bool, got {self.debugging!r}")

        if not isinstance(self.control_mode, ControlMode):
            raise ValueError(f"Invalid general.control_mode: {self.control_mode!r}")

        if len(self.initial_pos) != 7:
            raise ValueError(
                f"general.initial_pos must contain 7 joint positions, got {len(self.initial_pos)}"
            )
        if any(not math.isfinite(value) for value in self.initial_pos):
            raise ValueError("general.initial_pos must contain only finite values")

        if not math.isfinite(self.initial_gripper):
            raise ValueError("general.initial_gripper must be finite")
        if not 0.0 <= self.initial_gripper <= 1.0:
            raise ValueError(
                f"general.initial_gripper must be in [0, 1], got {self.initial_gripper}"
            )

    @property
    def base_enabled(self) -> bool:
        """Whether the WBC optimization may use base velocity DOFs."""
        return self.control_mode in (
            ControlMode.WBC,
            ControlMode.LEFT_ARM,
            ControlMode.RIGHT_ARM,
        )

    @property
    def base_actuation_enabled(self) -> bool:
        """Whether base CAN and low-level base motor output must be active."""
        return self.control_mode not in (
            ControlMode.ARM_ONLY,
            ControlMode.LEFT_ARM_ONLY,
            ControlMode.RIGHT_ARM_ONLY,
        )

    @property
    def left_arm_actuation_enabled(self) -> bool:
        """Whether commands may be sent to the left arm."""
        return self.control_mode not in (
            ControlMode.BASE_ONLY,
            ControlMode.RIGHT_ARM,
            ControlMode.RIGHT_ARM_ONLY,
        )

    @property
    def right_arm_actuation_enabled(self) -> bool:
        """Whether commands may be sent to the right arm."""
        return self.control_mode not in (
            ControlMode.BASE_ONLY,
            ControlMode.LEFT_ARM,
            ControlMode.LEFT_ARM_ONLY,
        )

    @property
    def arm_actuation_enabled(self) -> bool:
        """Whether any arm CAN/state handling is required."""
        return self.left_arm_actuation_enabled or self.right_arm_actuation_enabled

    @property
    def lowlevel_initial_allowed(self) -> bool:
        """Whether this control mode has at least one arm to initialize."""
        return self.arm_actuation_enabled

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> GeneralConfig:
        section = data.get("general")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [general] section")

        initial_pos_raw = section.get("initial_pos", DEFAULT_INITIAL_POS)
        if not isinstance(initial_pos_raw, (list, tuple)):
            raise ValueError("general.initial_pos must be an array of 7 joint positions")

        return cls(
            debugging=parse_bool(
                section.get("debugging", False),
                field="general.debugging",
            ),
            control_mode=ControlMode.parse(section.get("control_mode", ControlMode.WBC.value)),
            initial_pos=tuple(float(value) for value in initial_pos_raw),
            initial_gripper=float(section.get("initial_gripper", DEFAULT_INITIAL_GRIPPER)),
        )


config = GeneralConfig.load()
