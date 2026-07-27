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
class HommiInterfaceConfig(BaseConfig):
    hz: float

    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"lowlevel.hz must be positive, got {self.hz}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> HommiInterfaceConfig:
        hommi_section = data.get("hommi_interface")
        dds_section = data.get("dds")

        if not isinstance(hommi_section, Mapping):
            raise ValueError("Missing [hommi_interface] section")

        if not isinstance(dds_section, Mapping):
            raise ValueError("Missing [dds] section")

        return cls(
            hz=float(hommi_section["hz"]),
            dds=DDSConfig(
                domain_id=int(dds_section.get("domain_id", 0)),
                interface=str(dds_section.get("interface", "lo")),
            ),
        )


config = HommiInterfaceConfig.load()
