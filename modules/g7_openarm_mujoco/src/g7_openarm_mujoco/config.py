from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from g7_openarm_config import BaseConfig


@dataclass(frozen=True, slots=True)
class DDSConfig:
    domain_id: int
    interface: str


@dataclass(frozen=True, slots=True)
class MujocoConfig(BaseConfig):
    hz: float
    fps: float
    
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(
                f"mujoco.hz must be positive, got {self.hz}"
            )

        if self.fps <= 0.0:
            raise ValueError(
                f"mujoco.fps must be positive, got {self.hz}"
            )

        if not self.dds.interface:
            raise ValueError(
                "dds.interface must not be empty"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz
    
    @property
    def fps_interval(self) -> float:
        return 1.0 / self.fps

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> "MujocoConfig":
        section = data.get("mujoco")
        dds_section = data.get("dds")

        if not isinstance(section, Mapping):
            raise ValueError(
                "Missing [mujoco] section"
            )

        if not isinstance(dds_section, Mapping):
            raise ValueError(
                "Missing [dds] section"
            )

        return cls(
            hz=float(section["hz"]),
            fps=float(section["fps"]),
            dds=DDSConfig(
                domain_id=int(
                    dds_section.get("domain_id", 0)
                ),
                interface=str(
                    dds_section.get("interface", "lo")
                ),
            ),
        )


config = MujocoConfig.load()