from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .base import BaseConfig
from .parsing import parse_bool


@dataclass(frozen=True, slots=True)
class GeneralConfig(BaseConfig):
    debugging: bool

    def __post_init__(self) -> None:
        if type(self.debugging) is not bool:
            raise ValueError(f"general.debugging must be bool, got {self.debugging!r}")

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> GeneralConfig:
        section = data.get("general")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [general] section")

        return cls(
            debugging=parse_bool(
                section.get("debugging", False),
                field="general.debugging",
            ),
        )


config = GeneralConfig.load()
