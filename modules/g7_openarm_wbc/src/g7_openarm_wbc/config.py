from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, radians
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class WBCConfig(BaseConfig):
    hz: float
    left_sew_target_deg: float
    right_sew_target_deg: float
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"wbc.hz must be positive, got {self.hz}")
        if not isfinite(self.left_sew_target_deg):
            raise ValueError(
                "wbc.left_sew_target_deg must be finite, "
                f"got {self.left_sew_target_deg}"
            )
        if not isfinite(self.right_sew_target_deg):
            raise ValueError(
                "wbc.right_sew_target_deg must be finite, "
                f"got {self.right_sew_target_deg}"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @property
    def left_sew_target_rad(self) -> float:
        return radians(self.left_sew_target_deg)

    @property
    def right_sew_target_rad(self) -> float:
        return radians(self.right_sew_target_deg)

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
            left_sew_target_deg=float(section["left_sew_target_deg"]),
            right_sew_target_deg=float(section["right_sew_target_deg"]),
            dds=DDSConfig.from_mapping(data),
        )


config = WBCConfig.load()
