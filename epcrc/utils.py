from __future__ import annotations

import hashlib


def make_stable_seed(*parts: str) -> int:
    """Create a deterministic integer seed from text parts."""
    msg = "||".join(parts)
    digest = hashlib.sha256(msg.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)
