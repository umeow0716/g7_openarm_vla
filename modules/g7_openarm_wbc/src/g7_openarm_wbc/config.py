from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class WBCConfig(BaseConfig):
    hz: float
    dds: DDSConfig
    arm_swivel_weight: float
    arm_swivel_max_step_deg: float

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"wbc.hz must be positive, got {self.hz}")
        if self.arm_swivel_weight < 0.0:
            raise ValueError(
                "wbc.arm_swivel_weight must be non-negative, "
                f"got {self.arm_swivel_weight}"
            )
        if not 0.0 <= self.arm_swivel_max_step_deg <= 180.0:
            raise ValueError(
                "wbc.arm_swivel_max_step_deg must be within [0, 180], "
                f"got {self.arm_swivel_max_step_deg}"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> WBCConfig:
        section = data.get("wbc")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [wbc] section")

        return cls(
            hz=float(section["hz"]),
            dds=DDSConfig.from_mapping(data),
            arm_swivel_weight=float(section.get("arm_swivel_weight", 4.0)),
            arm_swivel_max_step_deg=float(section.get("arm_swivel_max_step_deg", 15.0)),
        )


config = WBCConfig.load()
