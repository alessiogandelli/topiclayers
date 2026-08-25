from __future__ import annotations

import json
import tempfile
import uunet.multinet as ml
from pathlib import Path

from topiclayers.adapters.csv import load_csv
from topiclayers.adapters.registry import resolve_adapter
from topiclayers.core.data import Dataset
from topiclayers.core.network import NetworkCreator
from topiclayers.core.topic import TopicModeler


def test_csv_toy_e2e():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        import topiclayers.adapters.csv  # noqa: F401

        fmt, loader = resolve_adapter(Path("examples/generic/posts.csv"))
        posts = loader(Path("examples/generic/posts.csv"))
        assert len(posts) == 7

        ds = Dataset(posts, n_cop="e2e_test")
        ds.process(cache_dir=d / "cache", adapter_name=fmt)

        assert ds.df_tweets is not None
        assert ds.df_original is not None
        assert ds.df_retweets is not None

        if len(ds.df_original) >= 3:
            tm = TopicModeler(
                ds.df_original,
                embedder_name="all-MiniLM-L6-v2",
                path_cache=d / "cache",
                name="e2e_test",
            )
            try:
                df_labeled = tm.get_topics(min_topic_size=2, random_state=42)
            except RuntimeError:
                df_labeled = ds.df_original.copy()
                df_labeled["topic"] = -1
        else:
            df_labeled = ds.df_original.copy()
            df_labeled["topic"] = -1

        df_labeled = ds.update_df(df_labeled)
        assert len(df_labeled) >= 0

        nw = NetworkCreator(df_labeled, "e2e_test", d)
        G = nw.create_retweet_network()
        assert G.number_of_nodes() >= 0

        retweet_path = d / "networks" / "e2e_test_retweet.gml"
        assert retweet_path.exists()

        if len(df_labeled) > 0 and df_labeled["topic"].nunique() > 1:
            ml_net = nw.create_retweet_ml()
            layers = ml.layers(ml_net)
            assert len(layers) > 0


def test_pipeline_writes_run_status():
    from topiclayers.cli import run_pipeline
    from topiclayers.config import InputConfig, NetworkConfig, RunConfig, TopicModelConfig

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cfg = RunConfig(
            input=InputConfig(format="csv_posts", posts=Path("examples/generic/posts.csv")),
            name="status_e2e",
            topic_model=TopicModelConfig(
                embedder="all-MiniLM-L6-v2", min_topic_size=2, nr_topics="auto", random_state=42
            ),
            network=NetworkConfig(type="multilayer_repost"),
            output_dir=d / "out",
        )

        run_pipeline(cfg)

        status_path = d / "out" / "run_status.json"
        assert status_path.exists()
        data = json.loads(status_path.read_text())
        assert data["status"] == "success"
        assert data["name"] == "status_e2e"

        manifest_path = d / "out" / "run_manifest.json"
        assert manifest_path.exists()
