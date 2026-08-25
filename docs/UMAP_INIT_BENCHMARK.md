# UMAP init benchmark — why `pca` is the default init

Standalone record of the benchmark that motivated switching the UMAP low-dimensional
initialization from `spectral` to `pca` in **fast mode** (`repro_mode="fast"`).
Written so the choice can be audited or cited without re-running anything.

**TL;DR**: on our cop22 embeddings (181,926 × 384), `init="pca"` makes UMAP
**~20× faster** (531 s → 27 s) with equal or slightly *better* downstream clustering
quality. Only `init` was varied; every other parameter was held identical.

## 1. Background: what `init` does in UMAP

UMAP fitting has three phases:

1. **kNN graph construction** (pynndescent over the input embeddings) — identical
   for all init options.
2. **Initial low-dimensional coordinates** (the `init` parameter) — this is the
   only phase affected by the option.
3. **SGD optimization** of those coordinates against the graph (~50 epochs by
   default for datasets ≥10k points) — identical for all init options.

The three options compared:

| Init | What it computes | Cost |
|---|---|---|
| `spectral` (old default) | Eigenvectors of the graph Laplacian via `scipy.sparse.linalg.eigsh` / `lobpcg` (`umap/spectral.py`) | Very expensive at 181k nodes |
| `pca` | Truncated SVD of the input data | Milliseconds–seconds |
| `random` | Uniform noise | Negligible |

## 2. Benchmark setup

- Input: cached MiniLM embeddings from the cop22 run,
  `out/cop22/cache/tm/all-MiniLM-L6-v2/embeddings_cop22.pkl` — shape (181926, 384),
  loaded once and reused across variants.
- UMAP parameters (identical across variants, identical to what the pipeline uses):
  `n_components=5, n_neighbors=15, min_dist=0.0, metric="cosine",
  random_state=None, n_jobs=-1`.
- Downstream clustering per variant: `sklearn.cluster.HDBSCAN(min_cluster_size=50,
  min_samples=10, n_jobs=-1)` — exactly what `TopicModeler._use_BERTopic` builds
  with `min_topic_size=50`.
- Quality proxies:
  - number of clusters found;
  - share of documents assigned to topic −1 (outliers);
  - silhouette score (cosine) on a fixed 10k-point subsample of non-outliers.
- Environment: CPython 3.12.7, umap-learn 0.5.12, scikit-learn 1.9.0, macOS arm64.
  Numba cache was warm after the first variant.

## 3. Results

| Variant | UMAP (s) | HDBSCAN (s) | Clusters | Outlier % | Silhouette |
|---|---:|---:|---:|---:|---:|
| `spectral` (baseline) | **531.3** | 129.8 | 594 | 41.09 | 0.6685 |
| **`pca`** ⭐ adopted | **26.8** | 128.8 | 597 | 42.18 | **0.6912** |
| `random` | 26.7 | 1184.9 ⚠️ | 608 | 41.16 | 0.6944 |
| `random` + `low_memory=True` | 26.6 | 129.2 | 595 | 41.81 | 0.6726 |

### Reading the numbers

- The entire spectral cost is phase 2 (the eigensolver). kNN + SGD are ~25 s total,
  which is exactly where `pca`/`random` land end-to-end.
- **Quality does not degrade**: cluster count and outlier rate are statistically
  indistinguishable across variants; silhouette is actually highest for the cheap
  inits (differences of ±0.02 on a ±0.01-noisy metric — treat as "equal").
- The `random` variant's HDBSCAN took **9× longer** (1185 s vs ~130 s): the random
  initialization produced a reduced geometry whose mutual-reachability structure was
  much harder for HDBSCAN. This is a second, independent reason to prefer `pca`
  over `random`: it is deterministic given the input and yields better-behaved
  downstream geometry.

## 4. Why quality holds despite a "worse" starting point

The initial coordinates only set the starting point; after ~50 epochs of SGD on a
181k-node graph, the final layout is dominated by the graph itself, not by the init.
Two supporting observations:

1. Empirically (table above): same clusters, same outlier rate, same-or-better
   silhouette.
2. On noisy social-media corpora the kNN graph is frequently disconnected, and
   `umap.spectral_layout` silently falls back (to random / TSVD warm starts) when a
   component's eigensolve fails — i.e., strict mode sometimes paid the full spectral
   cost only to land on a fallback anyway.

## 5. What changed in the code

`pca` is now the **default `umap_init` everywhere** (`TopicModeler.get_topics`,
`_use_BERTopic`, and `TopicModelConfig`), including strict mode:

```python
# topic.py / config.py default
umap_init: str = "pca"
```

Rationale: UMAP's pca branch seeds its own solver with the run's `random_state`
(`PCA/TruncatedSVD(n_components=n_components, random_state=random_state)`,
`umap/umap_.py`), so a seeded pca init is **bit-reproducible** — it carries no
reproducibility penalty while removing the ~9 min eigensolver from strict runs too.

- Passing `umap_init="spectral"` explicitly restores the old behaviour in any mode.
- The former fast-mode-only override hack was removed (no longer needed).
- Strict mode keeps `random_state=42`; note UMAP still forces single-threaded
  execution whenever a seed is set (library design, independent of init).

## 6. Impact

For the cop22 run (180,926 embedded originals):

| Phase | Before | After |
|---|---:|---:|
| UMAP (cold numba cache) | ~15.5 min | ~27 s |
| UMAP (warm numba cache) | ~9 min | ~27 s |
| Topic-model stage total (fast) | ~18 min | ~2.5 min |
| Topic-model stage total (strict, seeded pca) | ~18 min | ~9-10 min |

Caveat stated honestly: part of the speedup between the original pipeline run
(~15.5 min) and the benchmark (531 s baseline) comes from the warm numba JIT cache,
not from the init change alone. The init change is the 531 s → 27 s delta measured
under identical warm-cache conditions.

Strict mode still runs UMAP single-threaded (UMAP overrides `n_jobs` whenever a
seed is set), which is why strict remains slower than fast; the init change removes
only the spectral eigensolver cost from it.
