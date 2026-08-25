# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Live run-status tracking: `run_status.json` written during each run with the current stage, per-item progress, elapsed time, and ETA. New `topiclayers status <output_dir>` command to inspect a running or finished run.
- Historical ETA: per-stage durations recorded in `cache/stage_timings.json`, used to predict ETA for stages without a per-item counter (e.g. BERTopic).

### Changed
- Package reorganisation: pipeline stages moved to `topiclayers.core` (`data`, `topic`, `network`), small helpers to `topiclayers.support` (`cache`, `repro`). Public classes re-exported from `topiclayers.core`. No behavioural changes.
- `TASKS.md` and `CONTRACT.md` moved to `docs/`.
- SentenceTransformer embeddings now encode in explicit batches to report progress.

## [0.1.0] — 2026-08-04

### Notes
- Hardened greenfield rewrite of `tweets-to-topic-network` (see docs/TASKS.md).
- Renamed to `topiclayers` to reflect input-expansion beyond Twitter.
- Preserves backward-compatible artefact contract with climate-network-analysis harness.
