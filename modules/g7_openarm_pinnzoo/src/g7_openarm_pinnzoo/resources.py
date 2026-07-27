from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

from .pinnzoo_utils import get_arch


@contextmanager
def native_library_path() -> Iterator[Path]:
    resource = files("g7_openarm_pinnzoo").joinpath(
        "lib",
        f"libg7_openarm_quat_{get_arch()}.so",
    )

    if not resource.is_file():
        raise FileNotFoundError(f"Packaged PinnZoo library not found: {resource}")

    with as_file(resource) as path:
        yield path
