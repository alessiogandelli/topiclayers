# CONCLUSIONS — synthetic-corpus study of `topiclayers`

One-page synthesis of everything in `experiments/synth_llm/`. Details and numbers
live in `TUNING_NOTES.md`, `SYNTH_LLM_REPORT.md`, `SYNTH_POL_REPORT.md`,
`SWEEP_RESULTS.md`, `SWEEP_POLARIZATION.md`, `POLARIZATION_REPORT.md`,
`ORACLE_POLARIZATION_REPORT.md` and the `*.json` metrics files.

> Caveat: all of this is on **synthetic** corpora, so the *directions* are
> trustworthy but the *magnitudes* are hypotheses about real data.

## The headline
1. **The pipeline runs cleanly end-to-end** (load → topics → labels → networks)
   on JSONL/CSV/Parquet, retweets/quotes/replies, and scales to ~16k posts in
   ~1 min. No structural failures.
2. **The topic model was the weak link — and the embedder was its weak point.**
   Swapping the default `all-MiniLM-L6-v2` for **`Qwen/Qwen3-Embedding-0.6B`**
   roughly doubles clustering quality (ARI 0.31 → 0.59) and, on a real COP22
   slice, goes from **2 topics** (a junk French-stopword blob) to **14 coherent
   ones**. Details: `EMBEDDER_REPORT.md`, `EMBEDDER_TUNING.md`,
   `COP22_VALIDATION.md`, §7 of `TUNING_NOTES.md`.
3. Even with much better topics, **polarization analysis through the automatic
   pipeline stays unreliable** (netpol recall ≤ 23% vs 83% on known topics).

## What the corpora showed
3. **Realism matters enormously.** Polished synthetic text: purity 0.90, ARI
   0.58. Realistic short/noisy/multilingual tweets: purity 0.44–0.64, ARI
   0.15–0.33. Any tuning done on clean text overstates performance — benchmark
   on realistic data.
4. **`nr_topics: "auto"` is a trap.** On v1 it merged 11 real topics into 3
   (ARI 0.031). Use `nr_topics: null` or an explicit k.
5. **Off-topic posts are not flagged `-1`.** Rejection recall is only ~20–44%;
   the `-1` topic is largely filled with *on-topic* posts. Do **not** use `-1`
   as an off-topic detector without checking its composition.
6. **Multilingual embedder helps** for non-English data (this corpus ~19%
   fr/es/de): ARI 0.17 → 0.29 at matched granularity, and the best outlier
   recall. It needs `nr_topics` control (over-splits otherwise).

## What the parameter sweep showed (18 configs)
7. **`min_topic_size` is the main knob, and there is no single best value.**
   - Best clustering: `min_topic_size=50`, `nr_topics=null` → ARI 0.331.
   - Best balanced (outlier recall + multilingual): `min_topic_size=20`,
     `nr_topics=12` → ARI 0.290, outlier recall 0.41.
   - Small `min_topic_size` (10/20 with `null`) explodes into 100–238 clusters.
8. **UMAP defaults won** (n_neighbors 15, n_components 5); every variation was
   worse. But these are **not exposed in the pipeline config** — `config.py`
   accepts `umap_n_neighbors`/`umap_n_components` and silently ignores them.
   That is a real bug to fix if you want to tune them from YAML.

## The conclusion that matters for the paper
9. **Polarization cannot be recovered through the automatic pipeline.** On a
   corpus deliberately built with two opposed camps and same-side retweeting,
   netpol recall is **11–23% across every config**, versus **83% on
   ground-truth topics** (0% false positives). The topic model splits/merges
   designed topics, so a predicted layer mixes camps sitting on opposite sides
   and the retweet graph stops looking bimodal.
   → To study polarization you must analyze **per known / coarse topic**, or fix
   clustering first (better embedder, semi-supervised seeding). The pipeline's
   raw topic output is too noisy on realistic short multilingual tweets.

## Methodological lessons
10. **Match the pipeline exactly when benchmarking.** Only *reposts* form
    edges — quotes/replies are dropped. An early version of the polarization
    sweep counted them and reported 67% recall; the corrected sweep reproduces
    the full-pipeline netpol run bit-for-bit (20%). Interaction semantics change
    the result, so benchmark against the real construction.
11. **Generation is cheap and reproducible**: ~cents per corpus, raw LLM replies
    cached (`llm_cache*/`) and resumable; structural parts are seeded.

## If you want a next move
- **Adopt `Qwen/Qwen3-Embedding-0.6B`** (config `synth_pol_best.yml`) as the
  topic-model embedder; keep `all-mpnet-base-v2` as the English-only fallback
  and multilingual MiniLM as the lightweight one.
- ~~Fix the UMAP-param passthrough bug~~ **done**; `n_components=10` is now the
  recommended value.
- Try `seed_topic_list` / semi-supervised clustering to map clusters back onto
  the ~10 known topics, and re-measure polarization recall toward the 83% oracle.
- Confirm wall-time on the full `cop22.json` (181k docs) — Qwen3 embedding is
  the slow part; batch/cache or a GPU may be needed.
