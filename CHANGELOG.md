# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-08-29

### Added
- Live console progress: `topiclayers run` now renders colored stage banners, an in-place progress line (spinner, counter, percent, ETA, elapsed time) that keeps ticking even during stages without a per-item counter, and a per-stage duration summary when the run finishes. `RunStatus` gained a `log()` method so intermediate messages no longer clobber the progress line.
- Optional `cache_dir` config key: where reusable artifacts (processed data, embeddings, topic model) live. Defaults to a `cache/` folder next to the input data file, so the cache persists regardless of `output_dir` and can be shared between runs and configs (files are prefixed with the dataset `name`).

### Changed
- `output_dir` now contains only research outputs (`networks/`, `run_manifest.json`, `run_status.json`); `stage_timings.json` moved from `<output_dir>/cache/` to `<output_dir>/` directly. Existing caches under an old `output_dir/cache/` are not migrated; delete them or point `cache_dir` at them explicitly.
- Bundled runnable examples: `topiclayers init` copies example configs and toy datasets (CSV + Twitter JSONL) into `./examples/`, so a fresh pip install works end-to-end out of the box.

### Fixed
- README quickstart referenced `examples/` that was not shipped with the package; troubleshooting entry suggested `pip install -e .` instead of installing from PyPI.

## [0.2.0] — 2026-08-25

### Added
- LLM topic labeling: after clustering, an LLM writes one short label per topic. Default is local Ollama (`ollama/qwen2.5:7b`); any LiteLLM model string works (OpenAI, Anthropic, ...). Configurable via `topic_model.label_model` in YAML (set to `""` to disable). Requires the optional `labeling` extra (`pip install 'topiclayers[labeling]'`). Non-fatal: if the server is unreachable, keyword labels are kept.
- Batched labeling: topics are labeled ~40 per LLM call instead of one call each (~40x fewer requests). The prompt carries dataset-level context (`topic_model.label_context`, auto-derived for cop* names) so labels describe the specific subtopic within the general domain rather than the domain itself. `max_topics` parameter allows a cheap preview on the largest topics.
- Custom OpenAI-compatible endpoints for labeling: `topic_model.label_api_base` (or `LABEL_API_BASE` env) targets self-hosted servers (Unsloth Studio, llama.cpp, vLLM...); API key resolved from `LABEL_API_KEY` -> `OPENAI_LIKE_API_KEY` -> `OPENAI_API_KEY`. On "No model loaded" errors the pipeline auto-resolves the local GGUF file and loads the model server-side, then retries.
- Live run-status tracking: `run_status.json` written during each run with the current stage, per-item progress, elapsed time, and ETA. New `topiclayers status <output_dir>` command to inspect a running or finished run.
- Historical ETA: per-stage durations recorded in `cache/stage_timings.json`, used to predict ETA for stages without a per-item counter (e.g. BERTopic).

### Changed
- Stack upgraded to 2026 releases: bertopic 0.17.4, sentence-transformers 6, transformers 5, torch 2.13, numpy 2.5, pandas 3, openai 3, qdrant-client 1.19, uunet 2.2.1; Python 3.12. See docs/UPGRADE_2026-08.md.
- HDBSCAN now comes from scikit-learn instead of the deprecated standalone package.
- Fixed fast mode crashing with duplicate `random_state` kwarg passed to UMAP (fast mode had never run successfully).
- UMAP init default changed from `spectral` to `pca`: seeded pca is bit-reproducible and ~20x faster with equal or better clustering quality (docs/UMAP_INIT_BENCHMARK.md).
- Package reorganisation: pipeline stages moved to `topiclayers.core` (`data`, `topic`, `network`), small helpers to `topiclayers.support` (`cache`, `repro`). Public classes re-exported from `topiclayers.core`. No behavioural changes.
- `TASKS.md` and `CONTRACT.md` moved to `docs/`.
- SentenceTransformer embeddings now encode in explicit batches to report progress.

## [0.1.0] — 2026-08-04

### Notes
- Hardened greenfield rewrite of `tweets-to-topic-network` (see docs/TASKS.md).
- Renamed to `topiclayers` to reflect input-expansion beyond Twitter.
- Preserves backward-compatible artefact contract with climate-network-analysis harness.
