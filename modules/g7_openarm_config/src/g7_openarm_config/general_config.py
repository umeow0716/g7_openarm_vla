from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from .base import BaseConfig
from .parsing import parse_bool

DEFAULT_INITIAL_POS = [0.435, 0.0, 0.0, -0.525, 0.0, 0.0, 0.613]
DEFAULT_INITIAL_GRIPPER = 1.0


@dataclass(frozen=True, slots=True)
class GeneralConfig(BaseConfig):
    debugging: bool
    control_mode: str
    initial_pos_raw: Sequence[float]
    initial_pos: dict[str, float]
    initial_gripper: float

    def __post_init__(self) -> None:
        if type(self.debugging) is not bool:
            raise ValueError(f"general.debugging must be bool, got {self.debugging!r}")

        if not isinstance(self.control_mode, str):
            raise ValueError(f"Invalid general.control_mode: {self.control_mode!r}")

        if len(self.initial_pos_raw) != 7:
            raise ValueError(
                f"general.initial_pos must contain 14 joint positions, got {len(self.initial_pos)}"
            )
        if any(not math.isfinite(value) for value in self.initial_pos.values()):
            raise ValueError("general.initial_pos must contain only finite values")

        if not math.isfinite(self.initial_gripper):
            raise ValueError("general.initial_gripper must be finite")
        if not 0.0 <= self.initial_gripper <= 1.0:
            raise ValueError(
                f"general.initial_gripper must be in [0, 1], got {self.initial_gripper}"
            )

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> GeneralConfig:
        section = data.get("general")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [general] section")

        initial_pos_raw = section.get("initial_pos", DEFAULT_INITIAL_POS)
        if not isinstance(initial_pos_raw, (list, tuple)) or len(initial_pos_raw) != 7:
            raise ValueError("general.initial_pos must be an array of 7 joint positions")

        return cls(
            debugging=parse_bool(
                section.get("debugging", False),
                field="general.debugging",
            ),
            control_mode=section.get("control_mode", "wbc"),
            initial_pos_raw=initial_pos_raw,
            initial_pos={
                "L1": initial_pos_raw[0],
                "L2": initial_pos_raw[1],
                "L3": initial_pos_raw[2],
                "L4": initial_pos_raw[3],
                "L5": initial_pos_raw[4],
                "L6": initial_pos_raw[5],
                "L7": initial_pos_raw[6],
                "R1": initial_pos_raw[0],
                "R2": initial_pos_raw[1],
                "R3": initial_pos_raw[2],
                "R4": initial_pos_raw[3],
                "R5": initial_pos_raw[4],
                "R6": initial_pos_raw[5],
                "R7": initial_pos_raw[6], 
            },
            initial_gripper=float(section.get("initial_gripper", DEFAULT_INITIAL_GRIPPER)),
        )


config = GeneralConfig.load()
