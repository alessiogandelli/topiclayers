# topiclayers

> From posts to topical multilayer networks — hardened and social-scientist-friendly.

## Quickstart

```bash
# Requires Python 3.10–3.13
pip install -e .
topiclayers run examples/generic.yml
```

That's it. You get a GML network file in `./out/generic_toy/networks/`.

## What does it do?

1. **Loads** your posts (Twitter JSONL, CSV, Parquet).
2. **Cleans** text (removes URLs, @mentions, newlines).
3. **Models topics** using BERTopic with SentenceTransformer embeddings.
4. **Labels retweets** by propagating topic labels through retweet chains.
5. **Builds networks**: single-layer retweet, multilayer per-topic, and bipartite temporal-text networks.

## Minimal config

Save this as `my_config.yml`:

```yaml
input:
  format: csv_posts
  posts: my_data.csv

name: my_dataset

topic_model:
  embedder: all-MiniLM-L6-v2
  min_topic_size: 50

network:
  type: multilayer_repost

output_dir: ./out/my_dataset
```

Run it: `topiclayers run my_config.yml`

## Checking progress

Long runs write a live status file to `<output_dir>/run_status.json`. Check the current stage, elapsed time, and ETA from another terminal without touching the running job:

```bash
topiclayers status out/my_dataset
```

Example output:

```
Stage 3/7: embed
Status: running
Run: my_dataset
Progress: 412000/1200000 (34%)
Stage elapsed: 18m 12s
Stage ETA: ~35m 20s
Total elapsed: 21m 05s
```

Stages are `load → data → embed → topic_model → label → network → manifest`. Stages with a per-item counter (data, embed, network) report a live ETA. Opaque stages (BERTopic's UMAP+HDBSCAN) fall back to the median duration of past runs, stored in `<output_dir>/cache/stage_timings.json`.

## Input formats

| Format | Extension | Description |
|--------|-----------|-------------|
| `twitter_jsonl` | `.json`, `.jsonl` | Twitter/X API v2 JSONL |
| `csv_posts` | `.csv` | Generic CSV with `post_id`, `user_id`, `text` columns |
| `parquet_posts` | `.parquet` | Same schema as CSV, Parquet format |

CSV optional columns: `created_at`, `lang`, `interaction_type` (`original`/`repost`/`quote`/`reply`), `target_post_id`, `mentions` (semicolon-separated), `extra_*` passthrough columns.

## What if something fails?

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| "All topics are -1" / "no topics found" | `min_topic_size` too high or dataset too small | Halve `min_topic_size` in config, or use a larger dataset (>100 posts) |
| "Module not found" | topiclayers not installed | `pip install -e .` |
| "OpenAI API key not set" | Using OpenAI embedder without key | Switch `embedder` to `all-MiniLM-L6-v2` (default, works offline) |
| "Qdrant connection refused" | Qdrant vector DB not running | Ignore — Qdrant is optional. Set `QDRANT_URL` in `.env` to enable |
| "Cannot create multilayer network" | All posts are outliers | Reduce `min_topic_size` or provide more data |
| "UMAP spectral layout failed" | Dataset too small (<10 posts) | Topic modeling needs more data; consider using topic labels from elsewhere |

## Output files

For a dataset named `<name>` (e.g. `my_dataset`), output goes to `<output_dir>/`:

```
<output_dir>/
├── run_manifest.json              # Reproducibility metadata
├── run_status.json                # Live stage/ETA tracking (while running)
├── cache/
│   └── data/
│       ├── tweets_<name>.pkl/.csv          # Full tweet table
│       ├── retweet_labeled_<name>.pkl/.csv # Retweets with topic labels
│       └── manifest_<name>.json            # Cache validity key
│   └── stage_timings.json         # Per-stage durations for ETA prediction
└── networks/
    ├── <name>_retweet.gml                  # Single-layer retweet network
    ├── <name>_retweet_network_ml.gml       # Multilayer (uunet format)
    ├── <name>_ttt.gml                      # Temporal-text bipartite network
    └── projected/
        └── <name>__prj_<topic>.gml         # Per-topic projected networks
```

## Requirements

- Python 3.10–3.13
- Optional: Docker (for Qdrant vector search), OpenAI API key (for OpenAI embeddings / topic labeling)

## Planned: JOSS software paper

Once the API stabilises and the PLOS ONE core paper results are regenerated with this library,
a short JOSS (Journal of Open Source Software) paper will be submitted with a Zenodo DOI.
