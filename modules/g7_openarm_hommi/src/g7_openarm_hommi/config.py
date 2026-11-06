from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig, DDSConfig


def _bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"hommi.{key} must be a TOML boolean, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class HommiConfig(BaseConfig):
    hz: float
    device: str
    camera_fps: int
    camera_timeout_ms: int
    head_camera_serial: str
    left_wrist_camera_serial: str
    right_wrist_camera_serial: str
    latency_compensation: bool
    publish_hz: float  # rt/target rate; follows [wbc].hz
    dds: DDSConfig

    def __post_init__(self) -> None:
        if self.hz <= 0.0:
            raise ValueError(f"hommi.hz must be positive, got {self.hz}")
        if self.publish_hz < self.hz:
            raise ValueError(f"wbc.hz ({self.publish_hz}) must be >= hommi.hz ({self.hz})")
        if not self.device:
            raise ValueError("hommi.device must not be empty")
        if self.camera_fps <= 0:
            raise ValueError(f"hommi.camera_fps must be positive, got {self.camera_fps}")
        if self.camera_timeout_ms <= 0:
            raise ValueError(
                f"hommi.camera_timeout_ms must be positive, got {self.camera_timeout_ms}"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @property
    def publish_interval(self) -> float:
        return 1.0 / self.publish_hz

    def camera_serials(self) -> dict[str, str]:
        """Return the real camera serials; an empty string means "not configured"."""
        serials = {
            "head": self.head_camera_serial,
            "left": self.left_wrist_camera_serial,
            "right": self.right_wrist_camera_serial,
        }
        for name, serial in serials.items():
            if not serial:
                key = "head_camera_serial" if name == "head" else f"{name}_wrist_camera_serial"
                raise ValueError(
                    f"hommi.{key} is not set; fill in the D435i serial "
                    "(see `rs-enumerate-devices -s`)"
                )
        if len(set(serials.values())) != len(serials):
            raise ValueError(f"hommi camera serials must be distinct, got {serials}")
        return serials

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> HommiConfig:
        section = data.get("hommi")
        if not isinstance(section, Mapping):
            raise ValueError("Missing [hommi] section")

        return cls(
            hz=float(section.get("hz", 20.0)),
            device=str(section.get("device", "cuda:0")),
            camera_fps=int(section.get("camera_fps", 30)),
            camera_timeout_ms=int(section.get("camera_timeout_ms", 1000)),
            head_camera_serial=str(section.get("head_camera_serial", "")),
            left_wrist_camera_serial=str(section.get("left_wrist_camera_serial", "")),
            right_wrist_camera_serial=str(section.get("right_wrist_camera_serial", "")),
            latency_compensation=_bool(section, "latency_compensation", True),
            publish_hz=float((data.get("wbc") or {}).get("hz", 100.0)),
            dds=DDSConfig.from_mapping(data),
        )


config = HommiConfig.load()
