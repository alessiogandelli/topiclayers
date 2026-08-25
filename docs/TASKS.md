# topiclayers — Implementation Task List

> **For the executing model:** follow each task in order. Every task names the exact file(s) to create/edit, the function signatures, reference the legacy code to study, and lists an acceptance check. Tasks show `[ ]` — mark `[x]` when done.

---

## Milestone 0 — Repo scaffold & contract lock (T0.1–T0.3)

### [x] T0.1 — Verify scaffold is importable
**Files to check:** `pyproject.toml`, `src/topiclayers/__init__.py`, `src/topiclayers/adapters/__init__.py`, `src/topiclayers/adapters/registry.py`
**What to do:**
```bash
cd topiclayers && pip install -e .
python -c "import topiclayers; print(topiclayers.__version__)"
python -c "from topiclayers.adapters import AdapterRegistry; print('ok')"
```
**Acceptance:** Both print without error; version == "0.1.0".

### [x] T0.2 — Read CONTRACT.md and understand the artefact contract
**File:** `CONTRACT.md`
**What to do:** Read the entire file. It locks every filename, directory layout, dataframe column, GML vertex attribute, and behavioral rule this library MUST produce. Every task below that writes output references this contract.

**Acceptance:** You can answer: what GML vertex attribute distinguishes user nodes from tweet nodes in the temporal text network? (Answer: `bipartite` — `0.0` for user, `1.0` for tweet).

### [x] T0.3 — Verify toy fixtures exist
**Files:** `examples/twitter/toy.json`, `examples/generic/posts.csv`
**What to do:**
```bash
ls -la examples/twitter/toy.json examples/generic/posts.csv
wc -l examples/generic/posts.csv   # should be 8 (header + 7 rows)
head -1 examples/twitter/toy.json  # prints a JSON object
```
**Acceptance:** Both files exist, csv has header + 7 rows.

---

## Milestone 1 — Generic schema & input adapters (T1.1–T1.3)

### [x] T1.1 — Create the Post schema
**File to create:** `src/topiclayers/schema.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/data.py:105-122` (the tweet dict builder)

**What to write:**
```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class InteractionType(str, Enum):
    ORIGINAL = "original"
    REPOST = "repost"
    QUOTE = "quote"
    REPLY = "reply"

@dataclass
class Post:
    post_id: str                              # unique id (string, for safety)
    user_id: str                              # author id
    text: str                                 # body text
    created_at: Optional[str] = None          # ISO-format string
    lang: Optional[str] = None                # language code
    interaction_type: Optional[InteractionType] = None  # None means original
    target_post_id: Optional[str] = None      # referenced post id
    target_user_id: Optional[str] = None      # referenced author id (optional)
    mentions: list[str] = field(default_factory=list)     # mentioned usernames
    extra: dict = field(default_factory=dict) # passthrough for adapter-specific fields
```

**IMPORTANT:** The module-level import must have ZERO side effects. No `os.getenv`, no network calls, no print statements. This is a plain dataclass file.

**Acceptance:**
```python
from topiclayers.schema import Post, InteractionType
p = Post(post_id="1", user_id="a", text="hello", interaction_type=InteractionType.REPOST, target_post_id="2")
print(p.post_id)  # "1"
```

### [x] T1.2 — Implement Twitter JSONL adapter
**File to create:** `src/topiclayers/adapters/twitter.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/data.py:100-128` (`_load_tweets_json`)

**What to write:**
1. Function `load(path: Path) -> List[Post]` that reads a JSONL file line by line.
2. For each line, parse the JSON and extract:
   - `post_id` ← `obj.get("id", ...)` (convert to str)
   - `user_id` ← `obj.get("author_id", ...)` (convert to str)
   - `text` ← `obj.get("text", "")`
   - `created_at` ← `obj.get("created_at", ...)`
   - `lang` ← `obj.get("lang", ...)`
   - `interaction_type` ← map from `obj["referenced_tweets"][0]["type"]`:
     - `"retweeted"` → `InteractionType.REPOST`
     - `"quoted"` → `InteractionType.QUOTE`
     - `"replied_to"` → `InteractionType.REPLY`
     - missing/other → `None` (original)
   - `target_post_id` ← from `referenced_tweets[0]["id"]` if exists
   - `mentions` ← list of `username` from `obj["entities"]["mentions"]` if exists
   - `extra` ← catch-all with `conversation_id`, `context_annotations`, etc.
3. Use `pathlib.Path` throughout — NO `str.split("/")`.
4. At the BOTTOM of the file, register the adapter:
   ```python
   from topiclayers.adapters.registry import register
   register("twitter_jsonl", load)
   ```

**Acceptance:**
```python
from topiclayers.adapters.twitter import load
posts = load(Path("examples/twitter/toy.json"))
assert len(posts) > 0
assert posts[0].post_id is not None
assert posts[0].user_id is not None

# A retweet in the data should have REPOST interaction_type
reposts = [p for p in posts if p.interaction_type == InteractionType.REPOST]
assert len(reposts) > 0, "Expected at least one retweet/repost"
```

### [x] T1.3 — Implement generic CSV adapter + verify registry
**File to edit/create:** `src/topiclayers/adapters/csv.py`

**What to write:**
1. Function `load_csv(path: Path) -> List[Post]`:
   - Read with `pd.read_csv(path)`
   - Required columns: `post_id`, `user_id`, `text`
   - Optional columns: `created_at`, `lang`, `interaction_type`, `target_post_id`, `target_user_id`, `mentions` (semicolon-separated string → list), `extra_*` columns → `extra` dict
   - Build `Post` dataclass per row
2. Register: `register("csv_posts", load_csv)`
3. `load_parquet` stub (same logic, different reader): `register("parquet_posts", load_parquet)`

**Acceptance:**
```python
from topiclayers.adapters.registry import resolve_adapter
fmt, fn = resolve_adapter(Path("examples/generic/posts.csv"))
posts = fn(Path("examples/generic/posts.csv"))
assert len(posts) == 7
assert posts[0].text == "tweet1"

# Also test twitter auto-detect
fmt, fn = resolve_adapter(Path("examples/twitter/toy.json"))
assert fmt == "twitter_jsonl"
```

---

## Milestone 2 — Dataset module (T2.1–T2.3)

### [x] T2.1 — Create Dataset class (`src/topiclayers/data.py`)
**File to create:** `src/topiclayers/data.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/data.py` (the whole file)

**What to write — class `Dataset`:**
```python
class Dataset:
    def __init__(self, posts: list[Post], n_cop: str = "dataset",
                 file_user: Optional[Path] = None, load_users: bool = True):
        # FIX BUG-1: self.n_cop = n_cop  # NOT None!
        # Store paths using pathlib
        # Initialize dataframes as None

    def process(self, cache_dir: Optional[Path] = None):
        # 1. Build tweets dict from posts -> df_tweets (indexed by post_id)
        #    Every row gets: author, author_name, text, date, lang,
        #    conversation_id, referenced_type, referenced_id,
        #    mentions_name (list), mentions_id (list), cop = self.n_cop
        # 2. Build users dict from file_user (optional)
        # 3. Cache as pkl + csv (same filenames as CONTRACT.md)
        # 4. Call _create_dataframes() to split into original/retweet/quote/reply
```

**Detailed steps for `process()`:**
1. Convert each Post to a dict matching the CONTRACT.md **tweets_<name>.pkl** columns. Map:
   - `referenced_type`: `InteractionType.REPOST` → `"retweeted"`, `.QUOTE` → `"quoted"`, `.REPLY` → `"replied_to"`, `None` → `None`
   - `author_name`: if user dict available, get username; else None
   - `cop`: **always `self.n_cop`** (BUG-1 fix)
2. Build DataFrame, set index = post_id string.
3. Split into `df_original` (referenced_type is None/NaN), `df_retweets` (== "retweeted"), `df_quotes`, `df_reply`.
4. Cache logic uses the filenames from CONTRACT.md. Use `pathlib`.

**Acceptance:**
```python
from topiclayers.schema import Post
from topiclayers.data import Dataset
posts = [Post("1", "u1", "hello"), Post("2", "u2", "world", interaction_type="repost", target_post_id="1")]
ds = Dataset(posts, n_cop="cop_test")
ds.process()
# BUG-1 check:
sample = ds.df_tweets.loc["1"]
assert sample["cop"] == "cop_test", f"BUG-1: cop should be 'cop_test', got {sample['cop']}"
assert sample["cop"] is not None, "BUG-1: cop is None!"
```

### [x] T2.2 — Implement `update_df` with cycle detection
**File to edit:** `src/topiclayers/data.py` (add method to Dataset)
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/data.py:139-165` (`update_df`)

**What to write:**
```python
def update_df(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
    # 1. Take a COPY of self.df_retweets (ARCH-3 fix)
    # 2. Set topic = referenced_id initially
    # 3. Concat with df_labeled (labeled originals)
    # 4. RESOLVE CHAINS with visited set (BUG-2 fix):
    #    For each row, if topic is a string (unresolved ref),
    #    follow the chain until a) integer topic found, or b) cycle detected
    # 5. Count & report unresolved; drop them (don't crash)
    # 6. Save as retweet_labeled_<name>.pkl/.csv
    # 7. Return the labeled df
```

**Pseudo-code for cycle-safe resolution:**
```python
resolved = {}
unresolved_count = 0
for idx, value in topic_dict.items():
    visited = set()
    current = idx
    while isinstance(value, str):
        if value in visited:
            # cycle detected
            value = None
            unresolved_count += 1
            break
        visited.add(value)
        if value not in topic_dict:
            unresolved_count += 1
            value = None
            break
        value = topic_dict[value]
    resolved[idx] = value
print(f"Unresolved references: {unresolved_count}")  # Not an error, just info
```

**Acceptance:**
```python
# Test with cycle fixture from conftest.py
from tests.conftest import toy_df_with_cycle
df_original = toy_df_with_cycle.iloc[:3]  # simulate labeled originals
df_original["topic"] = [1, 1, 1]
result = ds.update_df(df_original)  # should not infinite-loop
assert isinstance(result, pd.DataFrame)
# Second call gives same result (idempotency)
result2 = ds.update_df(df_original)
assert result.equals(result2), "update_df is not idempotent"
```

### [x] T2.3 — Parameter-aware cache for data stage
**File to edit:** `src/topiclayers/data.py`
**What to do:**
- Compute a cache key: `hash(input_file_contents) + hash(str(adapter_name)) + hash(n_cop)`
- Store a `cache/<name>.manifest.json` sidecar next to each cached file: `{key, timestamp, n_cop, n_rows, n_users}`
- In `process()`, check if cache exists AND the manifest key matches before loading.
- If cache present but key wrong → log warning, regenerate.

**Acceptance:**
```bash
# First run creates cache
python -c "..." # run with n_cop='test1'
# Second run with SAME n_cop reuses cache
# Third run with n_cop='test2' invalidates cache
```

---

## Milestone 3 — Topic modeling (T3.1–T3.4)

### [x] T3.1 — Create TopicModeler class (no import-time side effects)
**File to create:** `src/topiclayers/topic.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/topic.py` (study for logic, rewrite from scratch)

**What to write:**

```python
class TopicModeler:
    """Topic modeling on a dataframe. Depends on BERTopic."""
    def __init__(self, df: pd.DataFrame, *, embedder_name: str = "all-MiniLM-L6-v2",
                 path_cache: Path = Path("./cache"), name: str = "dataset"):
        self.df = df.copy()  # FIX ARCH-2: operate on a copy, never mutate caller data
        self.embedder_name = embedder_name
        self.name = name
        self.path_cache = Path(path_cache)
        # ... compute derived paths using Path (pathlib, not os.path)
        self.model = None
        self.embeddings = None
    
    def get_topics(self, min_topic_size=50, nr_topics="auto", random_state=42,
                   umap_model=None, hdbscan_model=None):
        # 1. Check cache: if df_labeled.pkl exists AND manifest key matches, load it
        # 2. Else: _preprocess() -> _get_embeddings() -> _use_BERTopic() -> _save_files()
        # 3. Return self.df (the copy)
```

**CRITICAL rules for this file:**
- NO module-level code except imports and class definiton.
- NO `openai_client = OpenAI()` at module level (BUG-5).
- NO `qdrant_client = QdrantClient(os.getenv(...))` at module level (BUG-5).
- NO `os.environ["TOKENIZERS_PARALLELISM"] = "false"` at module level — do it inside the method that calls SentenceTransformer.
- NO `from langchain import ...` (BUG-7).
- NO `from cumulative.ulc import HDBSCAN` or similar GPU imports at module level.
- `label_topics()` method: raise `NotImplementedError("TODO: topic labeling via LLM")`.

**Acceptance:**
```bash
python -c "from topiclayers.topic import TopicModeler; print('import OK')"
# Must work with no .env, no Qdrant, no OpenAI key, no CUDA.
```

### [x] T3.2 — Implement `_preprocess` + `_get_embeddings`
**File to edit:** `src/topiclayers/topic.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/topic.py:116-168`

**`_preprocess(docs)` — what to write:**
1. `self.df["new_text"] = self.df["text"].str.replace(r"http\S+", "", regex=True)`
2. `.str.replace(r"@\S+", "", regex=True)`
3. `.str.replace(r"\n", "", regex=True)`
4. `.str.strip()`
5. `self.df = self.df[self.df["new_text"] != ""]`
6. **FIX BUG-12:** `self.df.reset_index(drop=False, inplace=True)` after filtering
7. Return `self.df["new_text"].tolist()`

**`_get_embeddings(docs)` — what to write:**
1. Set `os.environ["TOKENIZERS_PARALLELISM"] = "false"` HERE (inside method, not at module level)
2. Check embeddings cache: if `.pkl` exists AND manifest key matches → load
3. If `embedder_name.startswith("text-embedding")`:
   - **FIX BUG-6:** First check if `OPENAI_API_KEY` is set. If not → `raise RuntimeError("OpenAI embedder selected but OPENAI_API_KEY is not set. Set it in a .env file or use a SentenceTransformer model like 'all-MiniLM-L6-v2' instead.")`
   - Lazily create `OpenAI()` client inside this method
   - Batch through `client.embeddings.create()`
4. Else (SentenceTransformer):
   - `model = SentenceTransformer(self.embedder_name)`
   - `embeddings = model.encode(docs, show_progress_bar=True)`
   - Set `self.embedder = model`
5. Save `.pkl` with embeddings

**`_save_files()` — what to write:**
1. `self.df.to_pickle(df_labeled_path)` — where df_labeled_path matches CONTRACT.md filename
2. `self.model.save(model_path, serialization="safetensors", save_ctfidf=True, save_embedding_model=self.embedder_name)`

**Acceptance:**
- Toy fixture: `tm.get_topics(min_topic_size=2)` produces >= 1 non-outlier topic.
- Second run uses cache, prints "using cached topics".

### [x] T3.3 — Implement `_use_BERTopic` with seeds + failure-safe cache
**File to edit:** `src/topiclayers/topic.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/topic.py:170-268`

**What to write — `_use_BERTopic(docs, nr_topics, min_topic_size, random_state, ...)`:**
1. Import `umap.UMAP`, `hdbscan.HDBSCAN`, `CountVectorizer`, `ClassTfidfTransformer`, `BERTopic` **inside** the method (or at module top — but NOT cuml or GPU versions at module level).
2. Set global seeds (FIX ARCH-5):
   ```python
   import numpy as np
   import random
   np.random.seed(random_state)
   random.seed(random_state)
   try:
       import torch
       torch.manual_seed(random_state)
   except ImportError:
       pass
   ```
3. Build UMAP with `random_state=random_state`.
4. Build HDBSCAN with `min_cluster_size=max(2, min_topic_size)`.
5. Guard: `if not docs or len(docs) == 0: raise ValueError("No documents for topic modeling.")`
6. `model.fit_transform(docs, embeddings=self.embeddings)`
7. **FIX ARCH-4:** If ANY exception occurs OR `len(set(topics)) <= 1` (all outliers):
   - Do NOT write `df_labeled.pkl` cache
   - Write a `status: "failed"` manifest entry
   - `raise RuntimeError(f"Topic modeling failed. {len(set(topics))-1} topics found. Try reducing min_topic_size below current value ({min_topic_size}).")`
   - Specifically: if all topics == -1, the error message should be: "All documents assigned to topic -1 (outlier). This means BERTopic found no clusters. Common causes: <1> min_topic_size too high (current: {min_topic_size}) — try half of it. <2> dataset too small — need at least a few hundred diverse posts. <3> all documents are too similar. See docs at https://maartengr.github.io/BERTopic/faq.html"
8. If successful: `self.df["topic"] = topics; self.df["topic_prob"] = probs; self.model = model`

**Acceptance:**
- Toy fixture with `min_topic_size=50` (too high for 7 rows): raises RuntimeError with clear message.
- Toy fixture with `min_topic_size=2`: works, topics assigned, not all -1.

### [x] T3.4 — Qdrant upload optional and isolated
**File to edit:** `src/topiclayers/topic.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/topic.py:270-303`

**What to write — method `_save_embeddings_qdrant(docs)`:**
1. Check `os.getenv("QDRANT_URL")` — if not set, print "Qdrant not configured — skipping. Set QDRANT_URL in .env to enable vector search." and return.
2. Lazily create `QdrantClient(url)` inside this method (not at module level).
3. Wrap ALL qdrant calls in:
   ```python
   try:
       client.create_collection(...)
       client.upload_points(...)
   except Exception as e:
       print(f"Qdrant upload failed (non-fatal): {type(e).__name__}: {e}")
       print("The pipeline continues — networks and topic labels are unaffected.")
   ```
4. Never let a Qdrant error crash the pipeline.

**Acceptance:** Pipeline completes without `QDRANT_URL` set; no crash, helpful message printed.

---

## Milestone 4 — Network creation (T4.1–T4.4)

### [x] T4.1 — Create `create_retweet_network` with NaN guard
**File to create:** `src/topiclayers/network.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/network.py:7-47`

**What to write — class `NetworkCreator`:**
```python
class NetworkCreator:
    def __init__(self, df: pd.DataFrame, name: str, output_dir: Path):
        self.df = df
        self.name = name
        self.output_dir = Path(output_dir)
        self.graph_dir = self.output_dir / "networks"
    
    def create_retweet_network(self) -> nx.DiGraph:
        """Single-layer retweet/repost network. Nodes = users, edges = interactions."""
        G = nx.DiGraph()
        G.add_nodes_from(self.df["author"].unique())
        missing_refs = 0
        author_lookup = self.df["author"].to_dict()  # post_id -> author
        for idx, row in self.df.iterrows():
            ref_id = row.get("referenced_id")
            # FIX BUG-4: NaN != None
            if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
                continue
            # FIX BUG-3: use dict lookup, not .loc[str(ref_id)]
            target_author = author_lookup.get(ref_id)
            if target_author is None:
                missing_refs += 1
                continue
            source = row["author"]
            if G.has_edge(source, target_author):
                G[source][target_author]["weight"] += 1
            else:
                G.add_edge(source, target_author, weight=1)
        if missing_refs:
            print(f"  {missing_refs} retweets reference tweets not in dataset — skipped.")
        self._save_graph(G, "retweet")
        return G
```

**Acceptance:**
```python
from topiclayers.network import NetworkCreator
nw = NetworkCreator(toy_df, "test", Path("out_test"))
G = nw.create_retweet_network()
assert G.number_of_nodes() > 0
assert G.number_of_edges() > 0
# Check file exists at CONTRACT location:
assert (Path("out_test/networks/test_retweet.gml")).exists()
```

### [x] T4.2 — Create `create_retweet_ml` with empty-topic guard
**File to edit:** `src/topiclayers/network.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/network.py:122-167`

**What to write:**
```python
def create_retweet_ml(self) -> "ml.Network":
    topics = self.df["topic"].unique()
    topics = topics[topics != -1]
    # FIX ARCH-6: empty topic list
    if len(topics) == 0:
        raise ValueError(
            "Cannot create multilayer network: all topics are -1 (outliers). "
            "This means BERTopic found no topic clusters. "
            "Try: <1> reduce min_topic_size, <2> use a larger dataset, "
            "<3> check the 'text' column is not empty."
        )
    ml_network = ml.empty()
    dropped_edges_total = 0
    for topic in topics:
        G = nx.DiGraph()
        df_tmp = self.df[self.df["topic"] == topic]
        G.add_nodes_from(df_tmp["author"].unique())
        for i, row in df_tmp.iterrows():
            ref_id = row["referenced_id"]
            # BUG-4 (NaN guard) applies here too
            if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
                continue
            try:
                target_author = df_tmp.loc[str(ref_id)]["author"]
                # ... edge creation
            except KeyError:  # FIX BUG-9: typed exception, not bare except
                dropped_edges_total += 1
        ml.add_nx_layer(ml_network, G, str(topic))  # use str(topic) for uunet layer name
    if dropped_edges_total:
        print(f"  {dropped_edges_total} edges dropped (referenced tweet missing in topic slice).")
    self._save_multilayer(ml_network)
    return ml_network
```

**Acceptance:**
- `toy_df` → 2 layers (topics 1 and 2), loads with `uunet.multinet.read`.
- `toy_df_all_outliers` → raises `ValueError` with remediation hint.

### [x] T4.3 — Create `create_ttnetwork` + `_project_network`
**File to edit:** `src/topiclayers/network.py`
**Legacy ref:** `tweets-to-topic-network/tweets_to_topic_network/network.py:49-119,169-253`

**What to write — `create_ttnetwork(project: bool = True)`:**
1. Build bipartite graph: user nodes (bipartite=0) + tweet nodes (bipartite=1).
2. Edges: author→tweet (weight=10), tweet→referenced_tweet (weight=1), tweet→mentioned_user.
3. Set vertex attrs matching CONTRACT.md: `label`, `bipartite`, `author`, `text`, `topics`, `is_retweet`.
4. Date lookup: set `date` edge attr.
5. Save as `<name>_ttt.gml`.
6. If `project=True`, call `_project_network()`.

**What to write — `_project_network(title)`:**
- Reimplement `recursive_explore` correctly:
  - **FIX BUG-10:** use the function parameter `graph` (not closure variable `g`) consistently.
  - **FIX BUG-11:** guard `if result is None: continue` when merging.
  - Pop `None` key from edges dict.
- Save per-topic networks as `networks/projected/<name>__prj_<topic>.gml`.

**Acceptance:**
- `create_ttnetwork` on toy_df produces `<name>_ttt.gml` with vertex attr `bipartite` = `0.0` or `1.0`.
- `_project_network` produces `projected/*.gml` files.

### [x] T4.4 — Add unit tests for projection edge cases
**File to create:** `tests/test_network.py`
**What to write:**
1. `test_create_retweet_network_edges` — use `toy_df`, verify weights > 0.
2. `test_create_retweet_ml_all_outlier_raises` — use `toy_df_all_outliers`, verify `ValueError`.
3. `test_missing_referenced_tweet_skipped` — add a row with `referenced_id` not in index, verify no crash, missing count > 0.
4. `test_nan_referenced_id` — inject `float("nan")` as `referenced_id`, verify no crash.

**Acceptance:**
```bash
pytest tests/test_network.py -v
```

---

## Milestone 5 — Reproducibility (T5.1–T5.3)

### [x] T5.1 — Create `run_manifest.json` writer
**File to create:** `src/topiclayers/manifest.py`

**What to write:**
```python
@dataclass
class RunManifest:
    topiclayers_version: str
    git_hash: Optional[str]       # git rev-parse HEAD, best-effort
    python_version: str            # sys.version
    library_versions: dict         # {pkg: version} for key deps
    params: dict                   # all user params
    input_hash: str                # sha256 of input file
    row_counts: dict               # {stage: n_rows}
    dropped_edges: int
    topic_distribution: dict       # {topic_id: count}
    status: str                    # "success" | "failed"
    timestamp: str                 # ISO-8601

    @classmethod
    def create(cls, params, row_counts, topic_dist, dropped, status):
        # gather versions, git hash, format manifest
    
    def write(self, path: Path):
        with open(path, "w") as f:
            json.dumps(f, asdict(self), indent=2, default=str)
    
    @classmethod
    def load(cls, path: Path): ...
```

**Acceptance:**
- Full pipeline run produces `output_dir/run_manifest.json` with all fields populated.
- `manifest["status"] == "success"` after a clean run.

### [x] T5.2 — Centralize seed setting
**File to create:** `src/topiclayers/repro.py`

**What to write:**
```python
def set_global_seed(seed: int) -> None:
    """Set numpy, random, and torch seeds for reproducibility.
    
    NOTE: GPU UMAP (cuML) does not respect random_state.
    BERTopic results on GPU are NOT bit-reproducible.
    For reproducible results, use the CPU path (default).
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
```

Call this at the start of `TopicModeler.get_topics()` passing the `random_state` param.

**Acceptance:** `set_global_seed(42)` sets seeds; rerunning topic modeling on the same docs twice with same seed gives identical topic assignments (on CPU path).

### [x] T5.3 — Parameter-aware cache keys on all caches
**Files to edit:** `src/topiclayers/data.py`, `src/topiclayers/topic.py`, `src/topiclayers/network.py`

**What to do:**
- Write a helper `cache_key(*args) -> str` in a new file `src/topiclayers/cache.py`:
  ```python
  import hashlib
  import json
  def cache_key(*parts: str) -> str:
      """Stable hex digest of concatenated arguments."""
      h = hashlib.sha256()
      for p in parts:
          h.update(json.dumps(p, sort_keys=True).encode())
      return h.hexdigest()[:16]
  ```
- Apply to: data cache (input_hash + adapter + n_cop), embeddings cache (docs_hash + embedder_name), topic model cache (embeddings_hash + min_topic_size + nr_topics + random_state).
- Each cache file gets a sibling `.manifest.json` with `{key, params, timestamp, status}`.

**Acceptance:** Changing `min_topic_size` in config triggers re-run of topic modeling (not just loading stale cache). Changing nothing reuses all caches.

---

## Milestone 6 — Idiot-proof UX (T6.1–T6.3)

### [x] T6.1 — YAML config schema
**File to create:** `src/topiclayers/config.py`

**What to write:**
```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml

@dataclass
class InputConfig:
    format: str = "twitter_jsonl"
    posts: Path = Path(".")
    users: Optional[Path] = None

@dataclass
class TopicModelConfig:
    embedder: str = "all-MiniLM-L6-v2"
    min_topic_size: int = 50
    nr_topics: str = "auto"
    random_state: int = 42
    umap_n_neighbors: int = 15
    umap_n_components: int = 5

@dataclass
class NetworkConfig:
    type: str = "multilayer_repost"  # multilayer_repost | retweet | ttn
    project_ttn: bool = True

@dataclass  
class RunConfig:
    input: InputConfig = field(default_factory=InputConfig)
    name: str = "default"
    topic_model: TopicModelConfig = field(default_factory=TopicModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output_dir: Path = Path("./out")

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(
            input=InputConfig(**raw.get("input", {})),
            name=raw.get("name", "default"),
            topic_model=TopicModelConfig(**raw.get("topic_model", {})),
            network=NetworkConfig(**raw.get("network", {})),
            output_dir=Path(raw.get("output_dir", "./out")),
        )
```

**Acceptance:**
```python
from topiclayers.config import RunConfig
cfg = RunConfig.from_yaml(Path("examples/twitter.yml"))
assert cfg.topic_model.min_topic_size == 3
```

### [x] T6.2 — CLI entry point
**File to create:** `src/topiclayers/cli.py`

**What to write:**
```python
import sys
from pathlib import Path
from topiclayers.config import RunConfig

def run_pipeline(config: RunConfig):
    # 1. Validate input file exists
    # 2. Resolve adapter from config.input.format
    # 3. Load posts via adapter
    # 4. Dataset.process()
    # 5. TopicModeler.get_topics(params from config.topic_model)
    # 6. Dataset.update_df()
    # 7. NetworkCreator.create_<type>()
    # 8. Write run_manifest.json
    pass

def main():
    if len(sys.argv) < 3 or sys.argv[1] != "run":
        print("Usage: topiclayers run <config.yaml>", file=sys.stderr)
        sys.exit(1)
    config_path = Path(sys.argv[2])
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    cfg = RunConfig.from_yaml(config_path)
    run_pipeline(cfg)
```

**Acceptance:**
```bash
topiclayers run examples/twitter.yml  # or: python -m topiclayers run examples/twitter.yml
```

Runs end-to-end and writes `./out/toy/` with all CONTRACT.m matching artefacts.

### [x] T6.3 — One-command demo + path safety audit
**File to create:** `examples/demo.sh`

**What to write:**
```bash
#!/bin/bash
set -e
echo "=== topiclayers demo ==="
echo "1. Installing..."
pip install -e .
echo "2. Running Twitter JSONL demo..."
topiclayers run examples/twitter.yml
echo "3. Running generic CSV demo..."
topiclayers run examples/generic.yml
echo "=== Done! Output in ./out/ ==="
```

**Path safety audit:** Grep entire `src/topiclayers/` for `/` in path construction strings (not in imports/comments). Every path operation MUST use `pathlib.Path`, NEVER `str.split("/")`, `"prefix/" + var`, or `os.path.join`.

**Acceptance:**
```bash
bash examples/demo.sh    # completes without error
rg 'split\("/"\)' src/   # returns zero matches
rg 'os\.path\.join' src/ # returns zero matches (pathlib only)
```

---

## Milestone 7 — Tests (T7.1–T7.3)

### [x] T7.1 — Verify pytest scaffold works
**Files:** `tests/conftest.py` (already written), `pyproject.toml` (already has pytest in dev deps)

**Acceptance:**
```bash
pip install -e ".[dev]"
pytest --co  # lists collected tests; at minimum the conftest fixtures load
```

### [x] T7.2 — Write tests/test_schema_adapters.py + tests/test_data.py
**File to create:** `tests/test_schema_adapters.py`

Write at minimum:
1. `test_post_creation` — instantiate Post with required fields.
2. `test_interaction_type_enum` — verify `InteractionType("repost")` works, `InteractionType("invalid")` raises.
3. `test_twitter_adapter_loads_toy` — load `examples/twitter/toy.json`, assert non-empty, check `post_id` and `user_id`.
4. `test_generic_csv_adapter_loads_7_rows` — load `examples/generic/posts.csv`, assert `len(posts) == 7`.
5. `test_auto_detect_format` — `resolve_adapter(Path("something.csv"))` gives `"csv_posts"`.
6. `test_unknown_extension_raises` — `resolve_adapter(Path("file.xyz"))` raises ValueError.

**File to create:** `tests/test_data.py`

Write at minimum:
1. `test_dataset_cop_not_none` — create Dataset with `n_cop="cop26"`, process, check `df_tweets.loc[something]["cop"] == "cop26"` (BUG-1 regression).
2. `test_update_df_no_infinite_loop` — use `toy_df_with_cycle` fixture that has A→B→A cycle; `update_df` returns without hanging (BUG-2 regression).
3. `test_update_df_idempotent` — calling twice returns identical results (ARCH-3 regression).
4. `test_get_original_retweet_split` — `Dataset.process()` populates `df_original` and `df_retweets`.

**Acceptance:**
```bash
pytest tests/test_schema_adapters.py tests/test_data.py -v
```
All pass.

### [x] T7.3 — Write tests/test_topic.py + tests/test_network.py + test_e2e
**File to create:** `tests/test_topic.py`

Write at minimum:
1. `test_import_no_side_effects` — re-`import topiclayers.topic` with no OPENAI_API_KEY, no QDRANT_URL, no CUDA — passes (BUG-5).
2. `test_all_outlier_raises` — TopicModeler on data where `get_topics` would produce only -1 → raises with message (ARCH-4).
3. `test_normal_run_produces_topics` — on `toy_df`: topics not all -1, df gains `topic` column.

**File to create:** `tests/test_network.py`

Write at minimum:
1. `test_retweet_network_has_edges` — toy_df → retweet network → at least 1 edge.
2. `test_nan_referenced_id_is_skipped` — inject `pd.NA` or `float("nan")` (BUG-4).
3. `test_missing_ref_does_not_crash` — `referenced_id` points to missing tweet → counted, not crash (BUG-3).
4. `test_create_retweet_ml_all_outlier_raises` — all `topic=-1` → ValueError.
5. `test_bare_except_removed` — grep `network.py` for `except:` (no Exception class specified) → zero matches (BUG-9).

**File to create:** `tests/test_e2e.py`

Write:
1. `test_twitter_toy_e2e` — load toy.json → Dataset → TopicModeler → update_df → NetworkCreator → assert `retweet.gml` exists, edges > 0, topics not all -1.
2. `test_csv_toy_e2e` — same for generic.csv.

**Acceptance:**
```bash
pytest tests/ -v --cov=src/topiclayers
```
All pass. Coverage > 70%.

---

## Milestone 8 — Docs, packaging, JOSS-readiness (T8.1–T8.3)

### [x] T8.1 — Write the social-scientist quickstart in README
**File to edit:** `README.md`

Replace the stub with:
1. Install: `pip install -e .` (with Python version note: 3.10–3.13)
2. Demo command: `topiclayers run examples/twitter.yml`
3. Annotated minimal config (5 fields: input format+path, name, embedder, min_topic_size, output_dir).
4. "What if something fails?" section mapping common errors to fixes:
   - "All topics are -1" → min_topic_size too high / dataset too small
   - "Module not found" → `pip install -e .`
   - "OpenAI API key not set" → optional; use SentenceTransformer embedder instead
   - "Qdrant connection refused" → optional; set QDRANT_URL or ignore
5. Input format table: Twitter JSONL, CSV, Parquet.
6. Output files section summarizing what is produced and where.

**Acceptance:** A person who has never used Python can follow the README and get a GML file in <= 15 minutes.

### [x] T8.2 — Error messages audit + CHANGELOG entry + version sync
**File to edit:** all `src/topiclayers/*.py` where exceptions are raised.

**What to do:**
- Every `raise` statement must include a remediation hint in the message. Examples:
  - ❌ `raise ValueError("empty df")`
  - ✅ `raise ValueError("DataFrame is empty after preprocessing. This may happen if all posts have URLs/mentions as their only text. Check the 'text' column of your input.")`
- **Version sync:** ensure `src/topiclayers/__init__.py :: __version__` matches `pyproject.toml :: version`.
- **CHANGELOG:** add date to `[0.1.0]` line.

**Acceptance:** Grep for `raise ` in src/ — every one includes a human-readable remediation hint.

### [x] T8.3 — CI configuration + JOSS outline note
**File to create:** `.github/workflows/ci.yml`

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
```

**JOSS outline note:** at the bottom of README.md add:
```markdown
## Planned: JOSS software paper

Once the API stabilises and the PLOS ONE core paper results are regenerated with this library,
a short JOSS (Journal of Open Source Software) paper will be submitted with a Zenodo DOI.
```

**Acceptance:** CI file exists, pytest green on all three Python versions (test locally on at least one).

---

## Quick reference: which legacy bug goes where

| Bug ID | Fixed in task | File |
|--------|--------------|------|
| BUG-1 (cop=None) | T2.1 | `data.py` |
| BUG-2 (cycle loop) | T2.2 | `data.py` |
| BUG-3 (KeyError ref) | T4.1 | `network.py` |
| BUG-4 (NaN guard) | T4.1, T4.2 | `network.py` |
| BUG-5 (import side effects) | T3.1 | `topic.py` |
| BUG-6 (no null guard on OpenAI) | T3.2 | `topic.py` |
| BUG-7 (langchain dead imports) | T3.1 | — removed entirely |
| BUG-8 (hardcoded /) | T6.3 audit | all files |
| BUG-9 (bare except) | T4.2 | `network.py` |
| BUG-10 (g vs graph) | T4.3 | `network.py` |
| BUG-11 (None crash projection) | T4.3 | `network.py` |
| BUG-12 (reset_index missing) | T3.2 | `topic.py` |
| ARCH-1 (cache invalidation) | T5.3 | `cache.py` |
| ARCH-2 (shared mutable df) | T3.1 | `topic.py` |
| ARCH-3 (update_df not idempotent) | T2.2 | `data.py` |
| ARCH-4 (failed cache poison) | T3.3 | `topic.py` |
| ARCH-5 (no seeds) | T5.2 | `repro.py` |
| ARCH-6 (empty ml silent) | T4.2 | `network.py` |
| DEP-1 (version mismatch) | T8.2 | `__init__.py` + `pyproject.toml` |
| DEP-2 (bad requirements.txt) | — N/A (no requirements.txt) | |
| DEP-3/4 (dead deps) | T0.1 | `pyproject.toml` |
| DEP-5 (Python <3.13) | T0.1 | `pyproject.toml` |
| DEP-6 (Makefile) | — N/A (no Makefile in new repo) | |
