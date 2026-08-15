"""Duration parsing for time-to-live arguments.

Kept dependency-free and separate from :mod:`agentrecall.memory` so both the library
(``mem.add(..., ttl="30d")``) and the CLI (``agentrecall add --ttl 30d``) accept the same
spellings without either importing the other.
"""

from __future__ import annotations

import re
from datetime import timedelta

_DURATION = re.compile(
    r"""
    (?:(?P<weeks>\d+(?:\.\d+)?)\s*w)?
    (?:(?P<days>\d+(?:\.\d+)?)\s*d)?
    (?:(?P<hours>\d+(?:\.\d+)?)\s*h)?
    (?:(?P<minutes>\d+(?:\.\d+)?)\s*m(?!s))?
    (?:(?P<seconds>\d+(?:\.\d+)?)\s*s)?
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_duration(value: str | int | float | timedelta) -> timedelta:
    """Coerce ``value`` to a :class:`~datetime.timedelta`.

    Accepts a ``timedelta`` unchanged, a bare number as **seconds**, and compact strings
    combining ``w``/``d``/``h``/``m``/``s`` units — ``"30d"``, ``"12h"``, ``"1h30m"``,
    ``"90"`` (seconds). Raises :class:`ValueError` on unparseable input or a non-positive
    result, so a typo can't silently create a memory that expires immediately.
    """
    if isinstance(value, timedelta):
        delta = value
    elif isinstance(value, bool):
        # bool is an int subclass; a stray True would otherwise mean "1 second".
        raise ValueError(f"invalid duration: {value!r}")
    elif isinstance(value, (int, float)):
        delta = timedelta(seconds=float(value))
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("invalid duration: empty string")
        try:
            delta = timedelta(seconds=float(text))  # bare number = seconds
        except ValueError:
            match = _DURATION.fullmatch(text)
            if match is None or not any(match.groupdict().values()):
                raise ValueError(
                    f"invalid duration: {value!r} "
                    "(expected e.g. '30d', '12h', '1h30m', or a number of seconds)"
                ) from None
            delta = timedelta(
                **{unit: float(raw) for unit, raw in match.groupdict().items() if raw}
            )
    else:
        raise ValueError(f"invalid duration: {value!r}")

    if delta <= timedelta(0):
        raise ValueError(f"duration must be positive, got {value!r}")
    return delta


__all__ = ["parse_duration"]
