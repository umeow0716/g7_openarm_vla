from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from g7_openarm_config import BaseConfig, DDSConfig

ArmSide = Literal["left", "right"]


@dataclass(frozen=True, slots=True)
class HommiConfig(BaseConfig):
    hz: float
    arm: ArmSide
    device: str
    camera_fps: int
    camera_timeout_ms: int
    latency_compensation: bool
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"hommi.hz must be positive, got {self.hz}")
        if self.arm not in ("left", "right"):
            raise ValueError(f"hommi.arm must be 'left' or 'right', got {self.arm!r}")
        if not self.device:
            raise ValueError("hommi.device must not be empty")
        if self.camera_fps <= 0:
            raise ValueError(
                f"hommi.camera_fps must be positive, got {self.camera_fps}"
            )
        if self.camera_timeout_ms <= 0:
            raise ValueError(
                "hommi.camera_timeout_ms must be positive, "
                f"got {self.camera_timeout_ms}"
            )
        if type(self.latency_compensation) is not bool:
            raise ValueError(
                "hommi.latency_compensation must be bool, "
                f"got {self.latency_compensation!r}"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> HommiConfig:
        section = data.get("hommi")
        if not isinstance(section, Mapping):
            raise ValueError("Missing [hommi] section")

        latency_compensation = section.get("latency_compensation", True)
        if type(latency_compensation) is not bool:
            raise ValueError(
                "hommi.latency_compensation must be a TOML boolean"
            )

        arm = str(section.get("arm", "left")).strip().casefold()
        if arm not in ("left", "right"):
            raise ValueError(f"hommi.arm must be 'left' or 'right', got {arm!r}")

        return cls(
            hz=float(section.get("hz", 20.0)),
            arm=arm,  # type: ignore[arg-type]
            device=str(section.get("device", "cuda:0")),
            camera_fps=int(section.get("camera_fps", 30)),
            camera_timeout_ms=int(section.get("camera_timeout_ms", 1000)),
            latency_compensation=latency_compensation,
            dds=DDSConfig.from_mapping(data),
        )


config = HommiConfig.load()
