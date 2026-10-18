
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

from g7_openarm_config import BaseConfig, DDSConfig


@dataclass(frozen=True, slots=True)
class LowLevelConfig(BaseConfig):
    hz: float
    natural_frequency_raw: Sequence[float]
    natural_frequency: dict[str, float]
    damping_ratio_raw: Sequence[float]
    damping_ratio: dict[str, float]
    dds: DDSConfig

    def __post_init__(self) -> None:
        if not math.isfinite(self.hz) or self.hz <= 0.0:
            raise ValueError(
                f"lowlevel.hz must be positive and finite, got {self.hz}"
            )

        for name, raw, values in (
            (
                "natural_frequency",
                self.natural_frequency_raw,
                self.natural_frequency,
            ),
            (
                "damping_ratio",
                self.damping_ratio_raw,
                self.damping_ratio,
            ),
        ):
            if len(raw) != 7:
                raise ValueError(
                    f"lowlevel.{name} must contain 7 joint values, got {len(raw)}"
                )

            if any(not math.isfinite(value) for value in values.values()):
                raise ValueError(
                    f"lowlevel.{name} must contain only finite values"
                )

        if any(value <= 0.0 for value in self.natural_frequency.values()):
            raise ValueError(
                "lowlevel.natural_frequency must contain positive values"
            )

        if any(value < 0.0 for value in self.damping_ratio.values()):
            raise ValueError(
                "lowlevel.damping_ratio must contain non-negative values"
            )

    @property
    def interval(self) -> float:
        return 1.0 / self.hz

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> LowLevelConfig:
        section = data.get("lowlevel")

        if not isinstance(section, Mapping):
            raise ValueError("Missing [lowlevel] section")

        natural_frequency_raw = section.get("natural_frequency")
        if (
            not isinstance(natural_frequency_raw, (list, tuple))
            or len(natural_frequency_raw) != 7
        ):
            raise ValueError(
                "lowlevel.natural_frequency must be an array of 7 joint values"
            )

        damping_ratio_raw = section.get("damping_ratio")
        if (
            not isinstance(damping_ratio_raw, (list, tuple))
            or len(damping_ratio_raw) != 7
        ):
            raise ValueError(
                "lowlevel.damping_ratio must be an array of 7 joint values"
            )

        return cls(
            hz=float(section["hz"]),
            natural_frequency_raw=natural_frequency_raw,
            natural_frequency={
                f"{side}{i + 1}": natural_frequency_raw[i]
                for side in ("L", "R")
                for i in range(7)
            },
            damping_ratio_raw=damping_ratio_raw,
            damping_ratio={
                f"{side}{i + 1}": damping_ratio_raw[i]
                for side in ("L", "R")
                for i in range(7)
            },
            dds=DDSConfig.from_mapping(data),
        )


config = LowLevelConfig.load()
