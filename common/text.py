from __future__ import annotations

from typing import List, Optional


def read_text_lines(path: str) -> List[str]:
    with open(path, "r") as f:
        return f.readlines()


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


