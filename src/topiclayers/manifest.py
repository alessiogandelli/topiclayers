"""Run manifest: reproducibility metadata for each pipeline run."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class RunManifest:
    topiclayers_version: str
    git_hash: Optional[str]
    python_version: str
    library_versions: dict
    params: dict
    input_hash: str
    row_counts: dict
    dropped_edges: int
    topic_distribution: dict
    status: str
    timestamp: str

    @classmethod
    def create(cls, params, row_counts, topic_dist, dropped, status):
        import hashlib
        import subprocess
        import sys
        from datetime import datetime, timezone

        import topiclayers

        git_hash = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except Exception:
            pass

        library_versions = _gather_versions()

        params_serialized = json.dumps(params, sort_keys=True, default=str)
        input_hash = hashlib.sha256(params_serialized.encode()).hexdigest()[:16]

        return cls(
            topiclayers_version=topiclayers.__version__,
            git_hash=git_hash,
            python_version=sys.version.split()[0],
            library_versions=library_versions,
            params=params,
            input_hash=input_hash,
            row_counts=row_counts,
            dropped_edges=dropped,
            topic_distribution=topic_dist,
            status=status,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def write(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "RunManifest":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def _gather_versions() -> dict:
    deps = [
        "numpy", "pandas", "networkx", "scipy",
        "bertopic", "umap", "hdbscan", "sentence_transformers",
        "openai", "qdrant_client",
    ]
    versions: dict = {}
    for pkg in deps:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions
