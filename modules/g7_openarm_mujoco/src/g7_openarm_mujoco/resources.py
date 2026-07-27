from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path


@contextmanager
def model_directory() -> Iterator[Path]:
    resource = files("g7_openarm_mujoco").joinpath("model")

    if not resource.is_dir():
        raise FileNotFoundError(f"Packaged MuJoCo model directory not found: {resource}")

    with as_file(resource) as path:
        yield path
