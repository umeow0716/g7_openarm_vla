from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class MujocoConfig(BaseConfig):
    hz: float
    imu_hz: float
    eetarget_hz: float
    fps: float
    dds: DDSConfig

    def __post_init__(self) -> None:
        for field, value in (
            ("mujoco.hz", self.hz),
            ("mujoco.imu_hz", self.imu_hz),
            ("mujoco.eetarget_hz", self.eetarget_hz),
            ("mujoco.fps", self.fps),
        ):
            if value <= 0.0:
                raise ValueError(f"{field} must be positive, got {value}")

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @property
    def imu_interval(self) -> float:
        return 1.0 / self.imu_hz

    @property
    def eetarget_interval(self) -> float:
        return 1.0 / self.eetarget_hz

    @property
    def fps_interval(self) -> float:
        return 1.0 / self.fps

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> MujocoConfig:
        section = data.get("mujoco")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [mujoco] section")

        return cls(
            hz=float(section["hz"]),
            imu_hz=float(section["imu_hz"]),
            eetarget_hz=float(section["eetarget_hz"]),
            fps=float(section["fps"]),
            dds=DDSConfig.from_mapping(data),
        )


config = MujocoConfig.load()
