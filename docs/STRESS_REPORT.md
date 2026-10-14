# topiclayers — stress & showcase report

**Date:** 2026-10-13 (overnight run)
**Subject:** `topiclayers` (local build of the fixed `0.3.1`; PyPI ships `0.3.0`)
**Harness:** `tmp/tl_stress/` (gitignored scratch) — `setup.sh`, `gen_data.py`, `run_matrix.py`, `configs/`, `logs/`, `results.json`

---

## TL;DR

- The pipeline runs end-to-end on JSONL, CSV and Parquet and produces the documented
  networks. Same records in all three formats now yield **byte-identical topic
  assignments** (`topics_hash=650a67b3…`), which they did **not** before the fixes.
- **6 bugs** were found and fixed (2 you asked for + 4 uncovered by the stress run).
  All fixes are regression-tested: `45 passed, 2 skipped`.
- **LLM labeling verified end-to-end** with `gpt-6-luna` (OpenAI): topic 0 →
  "Loss and damage funding", topic 1 → "Football match plans"; the outlier topic stays
  unlabeled. Wording varies slightly run-to-run (LLM). Earlier attempts failed as
  `insufficient_quota` on the old key — the new key in `.env` works.
- Fresh venv built `topiclayers==0.3.1`; `run_manifest.json` now reports `0.3.1`
  (was `0.2.1`). Wheel built at `dist/topiclayers-0.3.1-py3-none-any.whl`.

---

## Environment

| | |
|---|---|
| Machine | Apple M1 Pro, 8 cores, 16 GB, MPS available |
| Interpreter | Python 3.12.7 (fresh venv at `tmp/tl_stress/venv`) |
| Install | `pip install '<repo>[labeling]'` → `topiclayers 0.3.1`, litellm 1.83, bertopic 0.17.4, sentence-transformers 6.1, torch 2.14 |
| Extra | `pyarrow` installed manually — **not a declared dependency** but required for the documented `parquet_posts` support |

---

## Bugs found & fixed

### Requested (2)

| # | Bug | Evidence (0.3.0 wheel) | Fix |
|---|-----|------------------------|-----|
| 1 | `.env` never loaded | `load_dotenv` absent from the whole wheel although `python-dotenv` is a dependency and README documents `.env` keys | `load_dotenv(override=False)` at package import (`__init__.py`) |
| 2 | `__version__` stale | hardcoded `"0.2.1"` in `__init__.py` while `METADATA`/pyproject say `0.3.0`; `run_manifest.json` misreported it | read from `importlib.metadata.version("topiclayers")` |

### Discovered by the stress run (4)

| # | Bug | Evidence | Fix |
|---|-----|----------|-----|
| 3 | Topic modeling silently collapses to **all-outliers** on ordinary corpora | `nr_topics="auto"` makes BERTopic re-fit the `CountVectorizer` on one doc/topic; `min_df=4, max_df=0.95` → `ValueError: max_df corresponds to < documents than min_df` → every doc `-1`, status still "success". First seen on the diverse synthetic corpus; also made real `cop22` fail at `min_topic_size=20`. | `CountVectorizer(min_df=1, max_df=1.0)` (BERTopic defaults). After fix: synthetic 2 topics, `cop22 min=20` 3 topics / `min=10` 6 topics. |
| 4 | Empty input crashes | `KeyError: 'referenced_type'` in `Dataset._create_dataframes`; after fixing that, `KeyError: 'topic'` in `update_df` | canonical empty frames + early return in `update_df` |
| 5 | CSV/Parquet adapter turns empty text into the literal `"nan"` | `text=str(row.get("text",""))` on a NaN cell → `"nan"`; explains CSV ≠ JSONL results | NaN text cells → `""`, matching the JSONL adapter |
| 6 | `run_manifest.json` misreports version | (same root cause as #2) | (same fix) |

Regression tests added: `tests/test_versions_and_env.py`, `tests/test_robustness.py`.
Full suite: **45 passed, 2 skipped**.

---

## What the pipeline does (observed)

Stages: `load → data → embed → topic_model → label → network → manifest`, with a live
`run_status.json` and per-stage durations in `stage_timings.json`.

Representative run (`syn_jsonl_ml`, 589 posts / 406 originals):

```
load          0.0s
data          0.9s
embed         7.5s
topic_model  11.5s
label         0.1s
network       0.3s
manifest      0.4s
```

The 6 import-heavy deps (torch/bertopic/umap) dominate short runs: total wall was
~20–26 s for every corpus size from 100 to 2 000 posts, and peak RSS was ~1.0–1.1 GB
regardless. The actual embedding/clustering is seconds — startup is the fixed cost.

**Caching works:** the warm-cache re-run dropped 23 s → 16 s and reused
`embeddings_*`, `model_*`, `topics_*` from `<cache_dir>/tm/<embedder>/`.

---

## Results matrix (19 runs, 0 unexpected failures)

| id | wall | peak RSS | #gml | ml layers | topics | outliers | topics_hash |
|----|-----:|--------:|-----:|----------:|-------:|---------:|-------------|
| syn_jsonl_ml | 23.0s | 1075 MB | 2 | 2 | 2 | 10 | 650a67b3… |
| syn_jsonl_ml_warm | 16.2s | 822 MB | 2 | 2 | 2 | 10 | 650a67b3… |
| syn_csv_ml | 21.7s | 1073 MB | 2 | 2 | 2 | 10 | 650a67b3… |
| syn_parquet_ml | 21.1s | 1089 MB | 2 | 2 | 2 | 10 | 650a67b3… |
| syn_retweet | 20.5s | 1063 MB | 1 | – | 2 | 10 | 650a67b3… |
| syn_all_ttn | 20.9s | 1057 MB | 6 | 2 | 2 | 10 | 650a67b3… |
| syn_spectral | 20.9s | 1070 MB | 2 | 3 | 3 | 3 | 1dd1f319… |
| syn_fast | 21.2s | 1085 MB | 2 | 2 | 2 | 3 | 2a1ad0d8… |
| syn_min_high (min=400) | 21.0s | 1067 MB | 1 | – | 0 | 406 | – |
| cop22_slice_ml10 | 21.1s | 1086 MB | 2 | 6 | 6 | 14 | b7e6969b… |
| cop22_slice_ml20 | 21.3s | 1073 MB | 2 | 3 | 3 | 45 | 7e7e7420… |
| scale_100 | 20.5s | 1061 MB | 2 | 3 | 3 | 0 | a7f1be8f… |
| scale_500 | 21.3s | 1071 MB | 2 | 2 | 2 | 0 | cd5e080c… |
| scale_2000 | 25.8s | 1090 MB | 2 | 2 | 2 | 54 | 8b311edb… |
| neg_empty | 12.6s | 876 MB | 1 | – | 0 | 0 | – |
| neg_bad_ext | 0.4s | 97 MB | 0 | – | – | – | exit 1 (ValueError) |
| neg_missing_cols | 0.4s | 99 MB | 0 | – | – | – | exit 1 (ValueError) |
| neg_tiny (5 rows) | 19.3s | 1065 MB | 1 | – | 0 | 5 | – |
| label_openai (`gpt-6-luna`) | 27.0s | 1077 MB | 2 | 2 | 2 | 10 | 1a67aa3a… (LLM, varies) |

`650a67b3…` across JSONL/CSV/Parquet + cold/warm + retweet/all runs = **deterministic
and format-independent** after the CSV fix. `spectral` and `fast` differ as expected
(different UMAP init / non-seeded UMAP).

---

## Showcases

**Synthetic multilayer (`syn_jsonl_ml`)** — 6 seeded themes (COP negotiations,
renewables, denial, extreme weather, activism, noise), 406 originals, 136 retweets,
20 quotes, 27 replies, ~80 authors. Output: `syn_jsonl_ml_retweet.gml` +
`syn_jsonl_ml_retweet_network_ml.gml` (2 layers).

**All network types (`syn_all_ttn`)** — 6 GML files:
`*_retweet.gml`, `*_retweet_network_ml.gml`, `*_ttt.gml` (bipartite, 178 KB) and 3
per-topic projections under `networks/projected/`. Note the projected set includes a
`__prj_-1.0.gml` (the outlier topic is projected too — arguably should be excluded).

**Real data (`cop22_slice_ml10`)** — real multilingual COP22 tweets
(430 unique originals, en/fr/es/ru/…). English-only MiniLM merges the non-English
content into one "French words" topic (`la en le les`) — a modeling caveat, not a
crash. `min_topic_size=20` → 3 topics/45 outliers; `=10` → 6 topics/14 outliers.

**Robustness** — bad extension and missing CSV columns fail fast with clear
`ValueError`s and exit 1; empty input now completes cleanly (exit 0, empty network);
min_topic_size(400) → 0 topics, 406 outliers, skipped multilayer, status success.

**LLM labeling (`label_openai`)** — ran with `gpt-6-luna` via litellm: 2 topics in 1
batched call (~2 s). The `LLM_label` column appears in `topics_label_openai.csv`:

| topic | n | c-TF-IDF keywords | `gpt-6-luna` label (varies per run) |
|------:|---:|-------------------|--------------------|
| 0 | 318 | climate, cop2023, cop2021, green, glasgow | Loss and damage funding |
| 1 | 74 | grabbed, booked, team, getaway, weekend | Football match plans |
| -1 | 10 | language, phaseout, fuel, fossil | *(outliers — not labeled)* |

This also proves fix #1: the `.env` key was read without exporting it (an initial run
against the old key returned `insufficient_quota`, handled as non-fatal — the labels
above come from the new key).

---

## Remaining observations / recommendations

1. **Labeling verified.** Works with a funded key (`gpt-6-luna`). Model is configurable via the `LABEL_MODEL` env var (default `gpt-6-luna`); re-run with `--only label_openai`.
2. **`pyarrow` is undocumented.** `parquet_posts` is advertised but `pyarrow`/`fastparquet` isn't a declared dependency; `pd.read_parquet` fails without it.
3. **Relative `output_dir` resolves against CWD, not the config file**, unlike `input.posts` (which resolves against the config dir). Surprising; make both consistent.
4. **Empty input returns "success"** with an empty network. Consider a clearer warning/exit code.
5. **Non-English content + English embedder.** Default `all-MiniLM-L6-v2` merges/degrades non-English topics (≈35% of cop22). Consider a multilingual embedder for such data.
6. **BERTopic `nr_topics="auto"`** can still reduce to very few topics; the vectorizer fix removes the crash but not the aggressiveness.
7. **Projected outliers**: `_save_projected` skips `"nan"` but emits a `-1` projection.
8. **`QDRANT_URL` now loaded** (fix #1) — running the test suite with a dead Qdrant URL adds retry latency; harmless but noisy.

---

## Reproduce

```bash
# from repo root
bash tmp/tl_stress/setup.sh                 # fresh venv + install fixed build
tmp/tl_stress/venv/bin/pip install pyarrow
.venv/bin/python tmp/tl_stress/run_matrix.py --include-warm   # ~10 min, writes results.json
# artifacts: tmp/tl_stress/out/<run_id>/, logs/, configs/, results.json
```

Fix-only verification: `.venv/bin/python -m pytest tests/test_versions_and_env.py tests/test_robustness.py -q`.
