from __future__ import annotations

import os
import tomllib

from functools import lru_cache
from pathlib import Path
from typing import Any


CONFIG_FILENAME = "config.toml"


class ConfigError(RuntimeError):
    pass


def _search_upwards(start: Path) -> Path | None:
    current = start.resolve()

    if current.is_file():
        current = current.parent

    for directory in (current, *current.parents):
        candidate = directory / CONFIG_FILENAME

        if candidate.is_file():
            return candidate

    return None


def find_config_path(
    path: str | Path | None = None,
) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()

        if not resolved.is_file():
            raise ConfigError(
                f"Config file does not exist: {resolved}"
            )

        return resolved

    for start in (
        Path.cwd(),
        Path(__file__).resolve(),
    ):
        found = _search_upwards(start)

        if found is not None:
            return found

    raise ConfigError(
        f"Unable to locate {CONFIG_FILENAME}. "
    )


@lru_cache(maxsize=1)
def load_config_file(
    path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = find_config_path(path)

    with config_path.open("rb") as file:
        return tomllib.load(file)


def reload_config_file(
    path: str | Path | None = None,
) -> dict[str, Any]:
    load_config_file.cache_clear()
    return load_config_file(path)