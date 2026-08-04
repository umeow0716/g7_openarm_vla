from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, radians
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class WBCConfig(BaseConfig):
    hz: float
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"wbc.hz must be positive, got {self.hz}")

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
        )


config = WBCConfig.load()
