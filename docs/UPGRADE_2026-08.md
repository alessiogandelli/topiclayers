# Stack upgrade — August 2026

Full dependency refresh of the `topiclayers` pipeline, from a mid-2024 stack to the
latest releases available as of August 2026. This document records what changed,
why each decision was made, and how the result was validated, so that no choice is
left unexplained.

The UMAP init optimization is documented separately in
[UMAP_INIT_BENCHMARK.md](UMAP_INIT_BENCHMARK.md).

## 1. Motivation

The previous lock file was generated around May 2024 and had drifted far behind:

| Package | Old (locked) | New (Aug 2026) |
|---|---|---|
| python | 3.10–3.11 | **3.12** (constraint `>=3.12,<3.14`) |
| bertopic | 0.16.2 | **0.17.4** |
| sentence-transformers | 2.7.0 | **6.0.0** |
| transformers | 4.39 | **5.15** |
| torch | 2.3 | **2.13** |
| numpy | 1.26.4 | **2.5.2** |
| pandas | 2.2.2 | **3.0.5** |
| openai | 1.30.5 | **3.3.1** |
| qdrant-client | 1.9.1 | **1.19.0** |
| scikit-learn | 1.5.0 | **1.9.0** |
| umap-learn | 0.5.6 | **0.5.12** |
| networkx | 3.3 | **3.6.1** |
| igraph | 0.11.5 | **1.0.0** |
| uunet | 2.1.1 | **2.2.1** |

Running two major versions behind everything (numpy 1.x vs 2.x, transformers 4 vs 5)
meant accumulating migration debt that would only get more expensive to pay later,
plus missing fixes relevant to our usage (e.g. BERTopic's handling of OpenAI
responses returning `None`, which affects the `text-embedding-*` embedder path).

## 2. Dependency constraints and why

```toml
python = ">=3.12,<3.14"
pandas = ">=2.2,<4"
numpy = ">=1.26,<3"
sentence-transformers = "^6.0.0"
bertopic = "^0.17.4"
scikit-learn = ">=1.5"
umap-learn = "^0.5.12"
networkx = "^3.6"
igraph = ">=0.11.4"
qdrant-client = "^1.19.0"
openai = "^3.0.0"
```

- **Python `>=3.12,<3.14`**: numpy 2.5 requires Python >=3.12; networkx 3.6 requires
  >=3.11. Upper bound kept at `<3.14` until numba/llvmlite officially support 3.14.
  The venv was recreated on CPython 3.12.7.
- **numpy `>=1.26,<3` / pandas `>=2.2,<4`**: wide ranges instead of caret pins so
  poetry resolves the newest version compatible with the interpreter. On Python
  3.12 this resolves to numpy 2.5.2 and pandas 3.0.5.
- **`hdbscan` removed**: the standalone package is deprecated in favour of
  scikit-learn's implementation (see §3).
- **`scikit-learn` added as explicit dependency**: we now import from it directly;
  it should not be an implicit transitive dep of bertopic.
- **Dev deps**: pytest `>=8`, pytest-cov `>=6`.

## 3. Code changes

All changes are in `src/topiclayers/core/topic.py` unless noted.

### 3.1 HDBSCAN: standalone package → sklearn

```python
# before
from hdbscan import HDBSCAN
HDBSCAN(..., prediction_data=True, core_dist_n_jobs=-1)

# after
from sklearn.cluster import HDBSCAN
HDBSCAN(..., n_jobs=-1)   # fast mode only
```

Why:
- The standalone `hdbscan` package is deprecated upstream; sklearn ships a
  maintained implementation with the same algorithm and parameters we use.
- Parameter mapping:
  - `core_dist_n_jobs=-1` → `n_jobs=-1`. The parallelism is preserved; sklearn just
    names it differently and applies it to the distance/mutual-reachability phase.
    Clustering itself was always deterministic given embeddings, in both libraries.
  - `prediction_data=True` removed: sklearn does not need it for our code path
    (it existed for the old library's `approximate_predict`, which topiclayers
    never calls).
- Unchanged semantics: `min_cluster_size = max(2, min_topic_size)` and
  `min_samples = max(1, min_topic_size // 5)` are passed explicitly exactly as before.

### 3.2 Bug fix: duplicate `random_state` in fast mode (crash)

The fast branch built UMAP kwargs containing `random_state=None` while also passing
`random_state=None if fast else random_state` as an explicit argument. Python raises
`TypeError: got multiple values for keyword argument 'random_state'`, meaning
**fast mode had never actually been executed successfully** — the bug surfaced on
the first real cop22 run after the upgrade. Parameters are now assembled in a single
dict with no duplicates.

### 3.3 UMAP init `spectral` → `pca` (new default in all modes)

`pca` replaced `spectral` as the default `umap_init`, **including strict mode**:
UMAP seeds its pca solver with the run's `random_state`
(`PCA/TruncatedSVD(random_state=...)`), so seeded pca is bit-reproducible — there
is no reproducibility trade-off, only the removal of the Laplacian eigensolver
(~9-15 min per strict run on cop22-sized data). Benchmark, quality metrics and the
auditable rationale: [UMAP_INIT_BENCHMARK.md](UMAP_INIT_BENCHMARK.md).

Passing `umap_init="spectral"` explicitly restores the old behaviour.

### 3.4 `manifest.py`

Version recording now reports `sklearn` instead of `hdbscan`.

## 4. Validation

- **Unit/integration tests**: 30 passed (including new `test_fast_mode_produces_topics`),
  covering load → data → embed → BERTopic → network end-to-end on the new stack
  (bertopic 0.17.4 + sentence-transformers 6 + transformers 5 + sklearn HDBSCAN +
  pandas 3 + numpy 2.5).
- **Full real-data run** (`examples/cop22.yml`, 768,174 tweets, 180,926 originals):
  succeeded end-to-end including multilayer network export via uunet 2.2.1.
  420 topics found, ~40% outliers, stage timings recorded in
  `out/cop22/cache/stage_timings.json`.
- No pandas-3 breaking patterns (chained assignment, removed APIs) were found in
  the codebase; the only adjustments needed were the ones listed above.

Expected timing after all optimizations (fast mode): embed ~3 min, UMAP+HDBSCAN
~2.5 min (was ~18 min), i.e. roughly **7 min end-to-end** vs ~20 min before.

## 5. Open items / known consequences

- **cop27 raw data missing**: `data/cop27/cop27.pkl` is a fitted BERTopic model from
  the *old* stack (pickled with joblib under Python ≤3.10 + sklearn ≤1.2); it does
  not contain the original texts and cannot be migrated or re-run. To rerun cop27
  the raw tweet dump (like cop22.json / cop21.dat) is required.
- **Old runs are not numerically comparable**: version changes in umap/sklearn/
  bertopic alter results even in strict repro mode. Any figure reproduced from
  pre-upgrade artifacts must be regenerated with the current stack.
- **Old caches**: pre-upgrade pickles (`data/cache/tm/*` from March 2025, written by
  pandas 2.x / transformers 4.x) have **not** been validated against the new stack
  and may fail to load (the cop27 model pkl does). The manifest-based cache
  invalidation means any stale key is simply regenerated rather than reused.
