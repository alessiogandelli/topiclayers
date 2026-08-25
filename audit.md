# Codebase Audit & Fix Plan — tweets-to-topic-network

**TL;DR:** The repo has multiple correctness bugs that would produce wrong results in a paper, several broken dependency declarations, no real tests, and 6 architectural problems that cause silent wrong results. Fixes are grouped by severity below, with validation as Phase 4 (tomorrow).

---

## Confirmed Bugs — CRITICAL (wrong results in the paper)

**BUG-1 · data.py line 13 — `self.n_cop = None` (DATA CORRUPTION)**
The `n_cop` parameter is silently overwritten with `None`. Every single tweet gets `cop=None` instead of the dataset label (e.g. `'cop26'`). Downstream filtering and dataset identification are broken.
→ Fix: change `self.n_cop = None` to `self.n_cop = n_cop`.

**BUG-2 · data.py `update_df` — infinite loop risk**
The `while isinstance(value, str)` chain-follower has no `visited` set. A circular reference in the data (`A→B→A`) will infinite-loop forever.
→ Fix: add cycle detection with a `visited` set.

**BUG-3 · network.py line 48 — unguarded `KeyError` in `create_retweet_network`**
`self.df.loc[str(ref_id)]` crashes if the referenced tweet is not in the dataset, which is extremely common with real Twitter/X data.
→ Fix: wrap in `try/except KeyError`.

**BUG-4 · network.py line 48 — NaN slips past the `None` guard**
`if ref_id is not None` does NOT catch `NaN` (they are not equal). `str(NaN)` becomes `'nan'` and causes a `KeyError`.
→ Fix: `if ref_id is not None and not pd.isna(ref_id)`.

**BUG-5 · topic.py line 34 — module-level side effects at import time**
`openai_client = OpenAI()` and `qdrant_client = QdrantClient(os.getenv("QDRANT_URL"))` execute every time the module is imported. If env vars are missing or services are down, the import crashes. A library must never do this.
→ Fix: lazy-initialize both inside the methods that need them.

**BUG-6 · topic.py line 212 — `AttributeError` when OpenAI is unavailable**
`openai_client.embeddings.create(...)` is called with no null-check. If `openai_client = None`, this crashes during embedding generation.
→ Fix: add `if openai_client is None: raise RuntimeError(...)` before use.

**BUG-7 · topic.py line 13 — broken langchain imports**
`from langchain import PromptTemplate` and `from langchain.llms import OpenAI as LLMOpenAI` use the 0.1.x API. In langchain 0.2+ (current), these raise `ImportError`. `label_topics()` itself is entirely unimplemented (`pass`), so all this code is dead weight.
→ Fix: remove dead langchain imports, stub `label_topics` as `raise NotImplementedError`.

---

## Major Bugs (silent failures, wrong output)

**BUG-8 · data.py line 17 — hardcoded `/` separators**
`file_tweets.split('/')` breaks on Windows or cloud runners. Use `pathlib.Path` throughout.

**BUG-9 · network.py line 153 — bare `except:` in `create_retweet_ml`**
All errors silently swallowed; edges are dropped without any counter. At minimum `except KeyError:`.

**BUG-10 · network.py line 169 — `recursive_explore` inconsistent `graph` vs `g`**
Function parameter is `graph` but internally uses the closure `g`. Same object currently, but fragile and confusing.

**BUG-11 · network.py line 196 — `None` crash in projected network merge**
`result.get(key, [])` crashes with `AttributeError` if `recursive_explore` returns `None` (isolated user nodes). Guard: `if result is None: continue`.

**BUG-12 · topic.py line 126 — `reset_index` missing after filtering**
`self.df = self.df[self.df['new_text'] != '']` leaves a sparse index. A `reset_index(drop=False)` would preserve traceability.

---

## Structural / Architectural Issues

### ARCH-1 · No cache invalidation — biggest silent risk
All caches (JSON, embeddings, BERTopic model, labeled df) use existence-only checks. If you change `min_topic_size`, `embedder_name`, or the input data, cached results from the old run are silently reused. This WILL produce wrong results when iterating on parameters.
→ Fix: Include parameter hash in cache filename, or add a manifest JSON next to each cache file.

### ARCH-2 · Shared mutable df references across classes
`Data_processor.df_original` is passed by reference to `Topic_modeler`. Topic modeler mutates it (filters rows, adds columns). Caller-side code still holds the same object. `update_df()` is then called on the mutated df. This coupling is fragile and calling methods a second time corrupts state.
→ Fix: Topic_modeler should `.copy()` the df on init, and return the result. Don't mutate the caller's data.

### ARCH-3 · `update_df()` is not idempotent
Calling it twice overwrites `df_retweets['topic']` with `referenced_id` on first call, then on second call the column already has numeric topics, breaking the chain resolution logic.
→ Fix: Operate on a copy of `df_retweets`, never mutate the original.

### ARCH-4 · Failed topic modeling poisons the cache
If BERTopic fails (exception or only outlier topics), `self.df['topic'] = -1` is set and then saved to cache. Next run loads this poisoned cache and never retries. No flag or log indicating the cache contains a failed run.
→ Fix: Don't save to cache on failure. Save a `status: failed` flag in a manifest.

### ARCH-5 · No reproducibility guarantees
GPU UMAP has no `random_state`. HDBSCAN has no seed. No global `np.random.seed()` or `torch.manual_seed()`. OpenAI embeddings are API-dependent. Different runs produce different topic assignments.
→ Fix: Add `random_state=42` everywhere, set global seeds, document that GPU path is not reproducible.

### ARCH-6 · Empty network created silently when all topics = -1
`create_retweet_ml()` filters out topic `-1`, leaving no topics. Loop never executes. An empty multilayer network is saved without warning.
→ Fix: Raise or warn if filtered topics list is empty.

---

## Dependency Issues

| # | Issue | Fix |
|---|-------|-----|
| DEP-1 | `__init__.py` says `0.1.3`, pyproject says `0.1.10` | Sync to `0.1.10` |
| DEP-2 | `requirements.txt` pins `openai==0.27.8` (old pre-v1 API) and `bertopic==0.14.1` — incompatible | Delete or regenerate from pyproject.toml |
| DEP-3 | `flask`, `tiktoken`, `pipdeptree`, `fastembed` in pyproject but never imported | Remove |
| DEP-4 | `langchain = "^0.1.16"` incompatible with current langchain | Remove (or upgrade) |
| DEP-5 | `python = ">=3.10,<3.13"` excludes stable Python 3.13 | Update to `<3.14` |
| DEP-6 | Makefile uses `python3.9` and points to non-existent `src/main.py` | Fix to `python3` + correct entry point |

---

## Implementation Order

### Phase 1 — Critical Fixes (must do before running on real data)
1. Fix BUG-1: `self.n_cop = n_cop` in data.py
2. Fix BUG-2: Add visited-set cycle detection in `update_df`
3. Fix BUG-3 + BUG-4: NaN guard + try/except in `create_retweet_network` and `create_retweet_ml`
4. Fix BUG-5: Move `openai_client` and `qdrant_client` out of module level into lazy-init
5. Fix BUG-6: Add null guard before `openai_client.embeddings.create()`
6. Fix BUG-7: Remove broken langchain imports; mark `label_topics` as NotImplementedError

### Phase 2 — Robustness & Architecture
7. Fix BUG-8: Replace `/`-based path splitting with `os.path` throughout data.py
8. Fix BUG-9: Replace bare `except:` with `except KeyError:` in `create_retweet_ml`
9. Fix BUG-10: Use `graph` parameter consistently in `recursive_explore`
10. Fix BUG-11: Guard `result is None` in `_project_network`
11. Fix BUG-12: `reset_index(drop=False)` after filtering in `_preprocess`
12. Fix ARCH-2: `.copy()` df in `Topic_modeler.__init__`
13. Fix ARCH-3: Operate on copy in `update_df`
14. Fix ARCH-6: Warn/raise on empty topic list in network creation

### Phase 3 — Cleanup
15. Fix DEP-1: Sync `__init__.py` version to `0.1.10`
16. Fix DEP-2: Delete `requirements.txt` or regenerate
17. Fix DEP-3: Remove `flask`, `tiktoken`, `pipdeptree`, `fastembed` from pyproject.toml
18. Fix DEP-4: Remove langchain from dependencies
19. Fix DEP-5: Update Python constraint to `<3.14`
20. Fix DEP-6: Update Makefile to use `python3` and correct entry point

### Phase 4 — Validation (tomorrow)
21. Create `tests/` directory with pytest
22. Write `tests/test_data.py`: test `Data_processor` with `data/toy.json` fixture
23. Write `tests/test_network.py`: test `Network_creator` with minimal df fixture
24. Write `tests/test_topic.py`: test `Topic_modeler._preprocess` and `_use_BERTopic` with toy data
25. Add `pytest` and `pytest-cov` to dev dependencies
26. Add ARCH-1 cache invalidation (parameter-aware caching)
27. Add ARCH-5 reproducibility (global seeds, `random_state` everywhere)
28. Add pipeline metadata / run manifest

---

## What a reliable, traceable pipeline should look like

1. **Immutable data flow** — each step takes input, returns output, never mutates shared state
2. **Parameter-aware caching** — cache keys include parameter fingerprints
3. **Pipeline metadata** — each run produces a `run_manifest.json` recording inputs, parameters, software versions, row counts at each stage, number of dropped tweets, topic distribution
4. **Assertions between stages** — e.g. after topic modeling: assert topics != all `-1`, assert len(df) > 0; after network: assert G.number_of_edges() > 0
5. **Proper test suite** — pytest with toy fixtures that verify the full pipeline end-to-end

---

## Relevant Files

- `tweets_to_topic_network/data.py` — BUG-1, 2, 8; ARCH-3
- `tweets_to_topic_network/topic.py` — BUG-5, 6, 7, 12; ARCH-2, 4, 5
- `tweets_to_topic_network/network.py` — BUG-3, 4, 9, 10, 11; ARCH-6
- `tweets_to_topic_network/__init__.py` — DEP-1
- `pyproject.toml` — DEP-3, 4, 5
- `requirements.txt` — DEP-2
- `Makefile` — DEP-6
- `test/test.py` — no assertions
- `test/main.py` — hardcoded paths
