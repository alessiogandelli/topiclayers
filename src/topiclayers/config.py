"""YAML run configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class InputConfig:
    format: str = "twitter_jsonl"
    posts: Path = Path(".")
    users: Optional[Path] = None


@dataclass
class TopicModelConfig:
    embedder: str = "all-MiniLM-L6-v2"
    min_topic_size: int = 50
    nr_topics: str = "auto"
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    exact_probabilities: bool = False
    repro_mode: str = "strict"
    # "pca" (seeded) is deterministic and ~20x faster than "spectral"
    # with equal or better clustering quality; see docs/UMAP_INIT_BENCHMARK.md
    umap_init: str = "pca"


@dataclass
class NetworkConfig:
    type: str = "multilayer_repost"
    project_ttn: bool = True


@dataclass
class RunConfig:
    input: InputConfig = field(default_factory=InputConfig)
    name: str = "default"
    topic_model: TopicModelConfig = field(default_factory=TopicModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output_dir: Path = Path("./out")

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        import yaml

        path = Path(path)
        with open(path) as f:
            raw = yaml.safe_load(f)

        input_raw = dict(raw.get("input", {}))
        # input paths are resolved relative to the config file's directory
        for key in ("posts", "users"):
            if input_raw.get(key) is not None:
                p = Path(input_raw[key])
                if not p.is_absolute():
                    p = (path.parent / p).resolve()
                input_raw[key] = p

        return cls(
            input=InputConfig(**input_raw),
            name=raw.get("name", "default"),
            topic_model=TopicModelConfig(**raw.get("topic_model", {})),
            network=NetworkConfig(**raw.get("network", {})),
            output_dir=Path(raw.get("output_dir", "./out")),
        )
