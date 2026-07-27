from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class MonitorConfig(BaseConfig):
    hz: float
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"monitor.hz must be positive, got {self.hz}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> MonitorConfig:
        section = data.get("monitor")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [monitor] section")

        return cls(
            hz=float(section["hz"]),
            dds=DDSConfig.from_mapping(data),
        )


config = MonitorConfig.load()
