"""Strict string-only STAID normalization (Milestone 2K-G-I I-B).

Rules (docs/stage1_baseline_package_implementation_plan.md §5):
- input must already be str (loaders must read ID columns with dtype=str) —
  any numeric input is rejected with TypeError, never coerced;
- surrounding whitespace is stripped; the result must be ASCII decimal
  digits only;
- length < 8  -> left-pad with zeros to 8 (standard USGS gauge IDs);
- length 8, 9, or 15 -> preserved unchanged (the Stage 1 universe contains
  one 9-char and five 15-char coordinate-based station numbers);
- any other final length -> ValueError (fail loud on unknown formats).

No int()/float() is used anywhere: numeric round-trips are exactly how the
older 48-column artifact lost leading zeros. There is deliberately no
permissive "repair" helper in this module.
"""
from __future__ import annotations

import re

_DIGITS_ONLY = re.compile(r"[0-9]+")
_ALLOWED_UNPADDED_LENGTHS = frozenset({8, 9, 15})


def normalize_staid(value: str) -> str:
    """Normalize a USGS station ID string; see module docstring for rules.

    Raises TypeError for non-str input and ValueError for empty,
    non-digit, or unsupported-length input.
    """
    if not isinstance(value, str):
        raise TypeError(
            f"STAID must be a str (read ID columns with dtype=str); "
            f"got {type(value).__name__}"
        )
    staid = value.strip()
    if not staid:
        raise ValueError("STAID is empty or whitespace-only")
    if _DIGITS_ONLY.fullmatch(staid) is None:
        raise ValueError(
            f"STAID must contain ASCII decimal digits only, got {value!r}"
        )
    if len(staid) < 8:
        return staid.zfill(8)
    if len(staid) in _ALLOWED_UNPADDED_LENGTHS:
        return staid
    raise ValueError(
        f"unsupported STAID length {len(staid)} (allowed: <8 zero-padded to 8, "
        f"or exactly 8/9/15), got {value!r}"
    )