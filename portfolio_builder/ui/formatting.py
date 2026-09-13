"""Money formatting for text inputs shown as "$#,000"."""

from __future__ import annotations

import math
import re
from typing import Any

_SUFFIXES = {"k": 1_000, "m": 1_000_000}
_MONEY_RE = re.compile(r"^\$?\s*(\d+(?:\.\d*)?|\.\d+)\s*([kKmM])?$")


def parse_money(value: Any) -> float | str | None:
    """Parse "$50,000", "50000", "50k" or a number into dollars.

    Blank input returns ``None``. Anything unparseable is returned unchanged, so request validation can
    report a clear field error instead of silently guessing.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "").replace(" ", "")
    if not text:
        return None
    match = _MONEY_RE.match(text)
    if not match:
        return str(value)
    amount = float(match.group(1)) * _SUFFIXES.get((match.group(2) or "").lower(), 1)
    return amount if math.isfinite(amount) else str(value)


def format_money(value: Any) -> str:
    """Dollars as "$#,000" (whole dollars); blank stays blank and unparseable text is returned as typed."""
    amount = parse_money(value)
    if amount is None:
        return ""
    if isinstance(amount, str):
        return amount
    return f"${round(amount):,}"
