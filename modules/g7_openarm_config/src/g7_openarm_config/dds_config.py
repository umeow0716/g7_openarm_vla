from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base import BaseConfig


@dataclass(frozen=True, slots=True)
class DDSConfig(BaseConfig):
    domain_id: int
    interface: str

    def __post_init__(self) -> None:
        if type(self.domain_id) is not int:
            raise ValueError(f"dds.domain_id must be int, got {self.domain_id!r}")

        if self.domain_id < 0:
            raise ValueError(f"dds.domain_id must be non-negative, got {self.domain_id}")

        if not isinstance(self.interface, str) or not self.interface.strip():
            raise ValueError("dds.interface must be a non-empty string")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> DDSConfig:
        section = data.get("dds")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [dds] section")

        domain_id = section.get("domain_id", 0)
        interface = section.get("interface", "lo")

        if type(domain_id) is not int:
            raise ValueError(f"dds.domain_id must be int, got {domain_id!r}")

        if not isinstance(interface, str):
            raise ValueError(f"dds.interface must be str, got {interface!r}")

        return cls(
            domain_id=domain_id,
            interface=interface.strip(),
        )


config = DDSConfig.load()
