from __future__ import annotations

from typing import Any

_TRUE_STRINGS = frozenset({"1", "true", "yes", "on"})
_FALSE_STRINGS = frozenset({"0", "false", "no", "off"})


def parse_bool(value: Any, *, field: str) -> bool:
    """Parse an explicit boolean value without Python's truthiness coercion."""
    if type(value) is bool:
        return value

    if isinstance(value, str):
        normalized = value.strip().casefold()

        if normalized in _TRUE_STRINGS:
            return True

        if normalized in _FALSE_STRINGS:
            return False

    raise ValueError(
        f"{field} must be a boolean or one of "
        f"{sorted(_TRUE_STRINGS | _FALSE_STRINGS)}, got {value!r}"
    )
