"""Project-wide Python startup customization.

When the `scripts/` directory is on `sys.path` (common in this repo), Python's
`site` module will automatically import `sitecustomize` if present.

We use this to make experimental DDD packages under `scripts/1. src/`
importable without requiring every entrypoint to manually patch sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _maybe_add_one_src() -> None:
    scripts_dir = Path(__file__).resolve().parent
    one_src = scripts_dir / "1. src"
    if one_src.exists() and one_src.is_dir():
        s = str(one_src)
        if s not in sys.path:
            sys.path.append(s)


_maybe_add_one_src()
