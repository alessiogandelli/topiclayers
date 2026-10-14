from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from topiclayers.config import RunConfig


def _adapters_loaded() -> None:
    import topiclayers.adapters.csv
    import topiclayers.adapters.twitter


def run_pipeline(config: RunConfig) -> None:
    from topiclayers.adapters.registry import resolve_adapter
    from topiclayers.core.data import Dataset
    from topiclayers.manifest import RunManifest
    from topiclayers.core.topic import TopicModeler
    from topiclayers.status import RunStatus

    _adapters_loaded()

    input_path = Path(config.input.posts)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    status = RunStatus(config.output_dir, config.name)
    status.start()

    try:
        _run_pipeline_stages(config, input_path, status)
    except BaseException:
        status.finish("failed")
        raise


def _run_pipeline_stages(config: RunConfig, input_path: Path, status) -> None:
    from topiclayers.adapters.registry import resolve_adapter
    from topiclayers.core.data import Dataset
    from topiclayers.manifest import RunManifest
    from topiclayers.core.topic import TopicModeler

    status.stage("load")
    fmt, loader = resolve_adapter(input_path)
    status.log(f"Loading posts via '{fmt}' adapter from {input_path}")
    posts = loader(input_path)
    status.log(f"  {len(posts)} posts loaded")

    row_counts = {"raw_posts": len(posts)}
    tm = None

    cache_dir = config.cache_dir or (input_path.parent / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    status.log(f"Cache: {cache_dir.resolve()}")

    status.stage("data", total=len(posts))
    ds = Dataset(posts, n_cop=config.name, file_user=config.input.users)
    ds.process(cache_dir=cache_dir, adapter_name=fmt, reporter=status)
    row_counts["tweets"] = len(ds.df_tweets)
    row_counts["original"] = len(ds.df_original)
    row_counts["retweets"] = len(ds.df_retweets)
    status.log(f"  {len(ds.df_original)} original, {len(ds.df_retweets)} retweets, "
               f"{len(ds.df_quotes)} quotes, {len(ds.df_reply)} replies")

    if len(ds.df_original) == 0:
        status.log("  No original posts found — skipping topic modeling. All retweets will be assigned topic -1.")
        df_labeled = ds.df_original.copy()
        df_labeled["topic"] = -1
        df_labeled["topic_prob"] = None
        row_counts["labeled_originals"] = len(df_labeled)
        topic_dist = {"-1": len(df_labeled)}
    else:
        tm = TopicModeler(
            ds.df_original,
            embedder_name=config.topic_model.embedder,
            path_cache=cache_dir,
            name=config.name,
        )
        try:
            df_labeled = tm.get_topics(
                min_topic_size=config.topic_model.min_topic_size,
                nr_topics=config.topic_model.nr_topics,
                random_state=config.topic_model.random_state,
                reporter=status,
                exact_probabilities=config.topic_model.exact_probabilities,
                repro_mode=config.topic_model.repro_mode,
                umap_init=config.topic_model.umap_init,
                umap_n_neighbors=config.topic_model.umap_n_neighbors,
                umap_n_components=config.topic_model.umap_n_components,
                embedder_prompt=config.topic_model.embedder_prompt,
                embedder_normalize=config.topic_model.embedder_normalize,
                embedder_device=config.topic_model.embedder_device,
            )
        except RuntimeError as e:
            status.log(f"  Topic modeling failed: {e}")
            status.log("  Assigning topic -1 to all originals and continuing.")
            df_labeled = ds.df_original.copy()
            df_labeled["topic"] = -1
            df_labeled["topic_prob"] = None
        row_counts["labeled_originals"] = len(df_labeled)
        topic_counts = df_labeled["topic"].value_counts().to_dict()
        topic_dist = {str(k): int(v) for k, v in topic_counts.items()}
        status.log(f"  {len(topic_dist)} topics found (including -1 outlier)")

    status.stage("label")
    if config.topic_model.label_model and tm is not None and tm.model is not None:
        tm.label_topics(
            model=config.topic_model.label_model,
            api_base=config.topic_model.label_api_base,
            label_context=config.topic_model.label_context,
            reporter=status,
        )
    df_labeled = ds.update_df(df_labeled, reporter=status)
    ds._save_labeled_dataframe(cache_dir)
    row_counts["retweets_labeled"] = len(df_labeled)

    from topiclayers.core.network import NetworkCreator
    nw = NetworkCreator(df_labeled, config.name, config.output_dir)
    dropped_edges = 0

    status.stage("network", total=len(df_labeled))
    network_type = config.network.type
    if network_type == "multilayer_repost":
        nw.create_retweet_network(reporter=status)
        try:
            nw.create_retweet_ml(reporter=status)
        except ValueError as e:
            status.log(f"  Skipping multilayer network: {e}")
    elif network_type == "retweet":
        nw.create_retweet_network(reporter=status)
    elif network_type == "ttn":
        nw.create_ttnetwork(project=config.network.project_ttn)
    elif network_type == "all":
        nw.create_retweet_network(reporter=status)
        try:
            nw.create_retweet_ml(reporter=status)
        except ValueError as e:
            status.log(f"  Skipping multilayer network: {e}")
        nw.create_ttnetwork(project=config.network.project_ttn)
    else:
        status.log(f"Unknown network type: '{network_type}'. "
                    f"Use one of: multilayer_repost, retweet, ttn, all.")
        sys.exit(1)

    status.stage("manifest")

    params_hash = hashlib.sha256(
        str(config).encode()
    ).hexdigest()[:16]

    manifest = RunManifest.create(
        params={
            "name": config.name,
            "input_format": config.input.format,
            "input_posts": str(config.input.posts),
            "embedder": config.topic_model.embedder,
            "min_topic_size": config.topic_model.min_topic_size,
            "nr_topics": config.topic_model.nr_topics,
            "random_state": config.topic_model.random_state,
            "network_type": config.network.type,
            "exact_probabilities": config.topic_model.exact_probabilities,
            "repro_mode": config.topic_model.repro_mode,
            "umap_init": config.topic_model.umap_init,
        },
        row_counts=row_counts,
        topic_dist=topic_dist,
        dropped=dropped_edges,
        status="success",
    )
    manifest.write(config.output_dir / "run_manifest.json")
    status.finish("success")
    print(f"\nDone. Output in {config.output_dir.resolve()}")
    print(f"  Run manifest: {config.output_dir / 'run_manifest.json'}")

def _cmd_init(dest: Path) -> None:
    import shutil
    from importlib.resources import as_file, files

    if dest.exists():
        print(f"Directory '{dest}' already exists — remove it or pass another path.",
              file=sys.stderr)
        sys.exit(1)

    with as_file(files("topiclayers") / "examples") as bundled:
        shutil.copytree(bundled, dest)

    print(f"Examples copied to {dest.resolve()}")
    print("Try it:")
    print(f"  topiclayers run {dest / 'generic.yml'}")


def main() -> None:
    if len(sys.argv) < 2 or (
        sys.argv[1] not in ("run", "status", "init")
        or (sys.argv[1] != "init" and len(sys.argv) < 3)
    ):
        print("Usage: topiclayers run <config.yaml>", file=sys.stderr)
        print("       topiclayers status <output_dir>", file=sys.stderr)
        print("       topiclayers init [dest=./examples]", file=sys.stderr)
        sys.exit(1)

    if sys.argv[1] == "init":
        _cmd_init(Path(sys.argv[2]) if len(sys.argv) > 2 else Path("examples"))
        return

    if sys.argv[1] == "status":
        from topiclayers.status import print_status

        sys.exit(print_status(Path(sys.argv[2])))

    config_path = Path(sys.argv[2])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = RunConfig.from_yaml(config_path)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
