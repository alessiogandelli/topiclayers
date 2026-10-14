# PIPELINE_REPORT — test data through topiclayers → netpol → PRISM

**Date:** 2026-10-13
**Data:** `topiclayers/experiments/datasets/swiss/SwissPoliticians-tweets.json.gz`, `experiments/datasets/rumoureval/rumoureval2019.tar.bz2`
**Servers:** Swiss → http://127.0.0.1:8010 · RumourEval → http://127.0.0.1:8001

---

## TL;DR

Both new datasets run end-to-end through **topiclayers → netpol → PRISM** and are
served in the UI. Doing so surfaced one serious, previously invisible bug:

> **The CSV/Parquet adapters corrupted large numeric ids.** Pandas read
> `target_post_id` as `float64` (because the column mixes big ids with blanks),
> so `1622232152543821824` became `1.6222201547708006e+18`. Every reference then
> failed to resolve and the multilayer network came out with **nodes but zero
> edges**. JSONL was unaffected — which is why the COP22 pipeline always worked
> while any CSV/Parquet dataset silently produced empty networks.

Fixed in `src/topiclayers/adapters/csv.py` (+ regression test). This is the most
impactful fix of the whole stress effort.

---

## Data conversion

Both corpora are **Twitter API v1.1**; topiclayers' `twitter_jsonl` adapter is
**API v2-only**, so converters were required.

| | Swiss | RumourEval 2019 |
|---|---|---|
| Source | `SwissPoliticians-tweets.json.gz` (gzip JSONL) | `rumoureval2019.tar.bz2` (nested zips of per-thread JSON) |
| Rows written | 69,179 | 6,634 |
| originals | 18,411 (600 in-file + **17,811 synthesised** from `retweeted_status`) | 384 (thread roots) |
| repost / quote / reply | 50,608 / 59 / 101 | 6,238* / 6 / 0 |
| users | 5,118 | 4,752 |
| references resolvable | **99.5%** (19,102/19,207) | **99.9%** (6,241/6,250) |

Converters: `experiments/scripts/converters/swiss_to_topiclayers.py`,
`experiments/scripts/converters/rumoureval_to_topiclayers.py`. Both emit
`csv_posts` + a PRISM-format `*_users.jsonl`.

Two adaptations worth knowing:
- **Swiss is ~99% retweets**, so the tweets behind retweets/quotes are
  **synthesised** as originals — otherwise there is almost nothing to cluster and
  no edge resolves.
- **RumourEval is reply threads**, but topiclayers only propagates topics and
  builds the multilayer network from `repost` rows; replies/quotes are dropped at
  the network stage. The converter re-tags replies as reposts (documented,
  `--keep-replies` to opt out) so the interaction network exists.

## topiclayers runs

| | topics | unresolved | labeled | network | wall |
|---|---|---|---|---|---|
| swiss | 50 | 127 / 50,608 | 49 (gpt-6-luna) | 36,709 rows; 5,068 nodes / 9,325 edges | 1m27s |
| rumoureval | 8 | 6 / 6,244 | 7 (gpt-6-luna) | 6,622 rows; 4,746 nodes / 5,196 edges | 29s |

Configs: `experiments/configs/swiss.yml`, `experiments/configs/rumoureval.yml` (both `all-MiniLM-L6-v2`
per request, `network: all`, `label_model: gpt-6-luna`; Swiss skips the O(N) TTN
projection). Run with the fixed package from the repo `.venv`.

## netpol (latent-ideology + Hartigan dip test, FDR)

`experiments/scripts/run_netpol.py <name>`; per-topic results in `experiments/outputs/netpol_<name>.csv`.

- **Swiss** — 49 layers, 44 analyzed, **5 polarized**: e.g. `Ehe für alle`,
  `Ersatz fossiler Heizungen`, `Vaterschaftsurlaub`, `Frauenrenten und AHV21`.
- **RumourEval** — 7 layers, all analyzed, **3 polarized**: `Charlie Hebdo
  newsroom shooting`, `Ferguson police chief disclosure`, `Sydney Lindt cafe siege`.

## PRISM (bundles + UI)

Configs `prism/config/swiss.yml`, `prism/config/rumoureval.yml`; bundles in
`prism/.data/<name>/`.

| | posts | topics | layers | agg edges | communities | analyzed / polarized (screened) |
|---|---|---|---|---|---|---|
| swiss | 36,709 | 50 | 49 | 9,325 | 91 | 44 / **2** |
| rumoureval | 6,622 | 8 | 7 | 5,196 | 120 | 7 / **3** |

- Swiss UI: **http://127.0.0.1:8010** · RumourEval UI: **http://127.0.0.1:8001**
  (port 8000 was already in use by your own PRISM).
- PRISM shows topiclayers' **keyword** labels (by design); the LLM labels are in
  `topics_*.csv` and used by the netpol report.

## Bugs / findings from this exercise

1. **CSV/Parquet id corruption → empty networks** (fixed, above). Root cause: no
   string dtype on id columns; pandas float coercion of large integers.
2. **topiclayers' network stage ignores replies/quotes** — only `repost` rows
   propagate topics and form edges. Reply-thread corpora (RumourEval) need the
   converter workaround; consider propagating across all reference types.
3. **Adapters are API v2-only** — v1.1 corpora (both test sets) need conversion;
   `full_text`/`retweeted_status`/`in_reply_to_status_id_str` are not recognised.
4. **English-only default embedder** — Swiss is ~90% German. It still produced
   sensible topic clusters (shared proper nouns/hashtags), but a multilingual
   embedder is the correct choice for production.
5. Cache key for the processed-data stage is derived from **post ids only**, so
   re-running a converter that changes interaction types reuses a stale cache —
   delete `cache/<name>/` when the schema changes.

## Reproduce

```bash
cd topiclayers/experiments
../.venv/bin/python scripts/converters/swiss_to_topiclayers.py
../.venv/bin/python scripts/converters/rumoureval_to_topiclayers.py
QDRANT_URL= LABEL_API_BASE= ../.venv/bin/topiclayers run configs/swiss.yml
QDRANT_URL= LABEL_API_BASE= ../.venv/bin/topiclayers run configs/rumoureval.yml
../.venv/bin/python scripts/run_netpol.py swiss       # and rumoureval

cd ../../prism
uv run prism build -c config/swiss.yml        # and rumoureval.yml
uv run prism serve -c config/swiss.yml  --port 8010
uv run prism serve -c config/rumoureval.yml --port 8001
```

## Caveats

- Swiss `network: all` was reduced to `multilayer_repost` (TTN projection is
  O(N) recursive and 69k posts is too much for a first pass).
- RumourEval's 384 originals make topic granularity coarse by construction.
- PRISM/PORT: leave the two servers running; stop with `pkill -f "prism serve"`.
