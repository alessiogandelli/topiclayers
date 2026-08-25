"""Input adapter registry — maps format strings to loader functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List

from topiclayers.schema import Post


AdapterFn = Callable[[Path], List[Post]]

_registry: dict[str, AdapterFn] = {}


def register(format_name: str, fn: AdapterFn) -> None:
    _registry[format_name] = fn


def resolve_adapter(path: Path) -> tuple[str, AdapterFn]:
    """Auto-detect format from file extension and return (format_name, loader)."""
    ext = path.suffix.lower()
    if ext in (".jsonl", ".json", ".dat"):
        fmt = "twitter_jsonl"
    elif ext == ".csv":
        fmt = "csv_posts"
    elif ext == ".parquet":
        fmt = "parquet_posts"
    else:
        raise ValueError(
            f"Cannot auto-detect format for extension '{ext}'. "
            f"Supported: .jsonl, .json, .dat, .csv, .parquet. "
            f"Use explicit `format:` in config for other formats."
        )
    if fmt not in _registry:
        raise KeyError(
            f"No adapter registered for format '{fmt}'. "
            f"Available adapters: {list(_registry.keys())}. "
            f"Did you import topiclayers.adapters.{fmt.split('_')[0]}?"
        )
    return fmt, _registry[fmt]


class AdapterRegistry:
    """Explicit registry for manual adapter lookups."""
    pass
