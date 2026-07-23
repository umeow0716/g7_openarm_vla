from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Self

from .loader import load_config_file


class BaseConfig(ABC):
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
    ) -> Self:
        data = load_config_file(path)
        return cls.from_mapping(data)

    @classmethod
    @abstractmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> Self:
        raise NotImplementedError