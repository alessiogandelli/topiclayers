from __future__ import annotations

import hashlib
import json


def cache_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True).encode())
    return h.hexdigest()[:16]
