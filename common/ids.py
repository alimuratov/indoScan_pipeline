from __future__ import annotations

import uuid


def generate_prefixed_uuid(prefix: str) -> str:
    """Generate a random id string with a given prefix, e.g., "rd-<uuid>"."""
    return f"{prefix}-{uuid.uuid4()}"


