# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Repository layout tidy-up. The research corpora and write-ups moved from `test_data/` to a structured `experiments/` folder (`datasets/`, `configs/`, `scripts/`, `reports/`, `outputs/`, plus the self-contained `synth_llm/`). Only the heavy corpora, caches and generated outputs stay gitignored; configs, scripts and reports are tracked.
- `src/topiclayers/examples/` is now the single canonical examples source (what `topiclayers init` ships); the duplicate top-level `examples/` folder was removed and the test fixtures point at the packaged copy.
- Untracked previously committed generated artifacts under `data/` (test networks, plots, toy cache) so the `data/` ignore rule actually applies. The files remain on disk.

## [0.3.1] — 2026-10-13

### Fixed
- `.env` files are now actually loaded. `python-dotenv` was a dependency and the docs told users to put `OPENAI_API_KEY` / `QDRANT_URL` / `LABEL_API_BASE` in `.env`, but `load_dotenv()` was never called, so those values were silently ignored. It is now loaded once at package import with `override=False` (real environment variables win).
- `topiclayers.__version__` is now read from the installed package metadata instead of a hardcoded string that had gone stale (`0.2.1`), so `run_manifest.json` reports the real version.
- Topic modeling no longer silently collapses every document to the outlier topic (`-1`) on ordinary corpora. With the default `nr_topics="auto"`, BERTopic re-fits the `CountVectorizer` on one document per topic during topic reduction; the library's `min_df`/`max_df` heuristic could then raise `max_df corresponds to < documents than min_df`. The vectorizer now uses BERTopic's safe defaults (`min_df=1, max_df=1.0`).
- Empty input no longer crashes with `KeyError: 'referenced_type'` / `'topic'`; the suite now handles an empty dataset and produces an empty run cleanly.
- The CSV/Parquet adapters no longer turn empty text cells into the literal string `"nan"` (they become empty strings, matching the JSONL adapter).
- **Large numeric ids are preserved by the CSV/Parquet adapters.** Pandas inferred `target_post_id` as float when the column mixed big integer ids with blanks, turning `1622232152543821824` into `1.6222201547708006e+18`; every reference then failed to resolve and multilayer networks came out empty (nodes but zero edges). Ids are now read as strings. JSONL was unaffected, which is why the COP22 pipeline worked but CSV/Parquet datasets did not.

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
