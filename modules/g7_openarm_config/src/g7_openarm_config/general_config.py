from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from g7_openarm_config import BaseConfig


@dataclass(frozen=True, slots=True)
class GeneralConfig(BaseConfig):
    debugging: bool

    def __post_init__(self) -> None:
        if not isinstance(self.debugging, bool):
            raise ValueError(f"general.hz must be bool, got {self.debugging}")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> GeneralConfig:
        section = data.get("general")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [general] section")

        return cls(
            debugging=bool(section["debugging"]),
        )


config = GeneralConfig.load()
