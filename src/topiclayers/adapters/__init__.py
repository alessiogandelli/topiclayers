"""Input adapters: convert various formats into a list of Post dataclasses."""

from __future__ import annotations

from .registry import AdapterRegistry, resolve_adapter

__all__ = ["AdapterRegistry", "resolve_adapter"]
