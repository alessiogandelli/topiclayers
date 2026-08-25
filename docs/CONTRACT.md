# Artefact Compatibility Contract

This document locks the **exact filesystem artefacts** that `topiclayers` must emit for backward compatibility with the `climate-network-analysis` experiment harness. Every adapter and output formatter MUST conform to this contract.

The contract was reverse-engineered from the harness in `climate-network-analysis/src/RQ/scripts/common.py`, its notebooks, and verified against the legacy `tweets-to-topic-network` 0.1.10 output.

---

## Directory layout

For a dataset named `<name>` (e.g. `cop26`) processed from input files in `<data_folder>`:

```
<data_folder>/
├── cache/
│   ├── data/
│   │   ├── tweets_<name>.csv
│   │   ├── tweets_<name>.pkl
│   │   ├── users_<name>.csv        (only if user file provided)
│   │   ├── users_<name>.pkl
│   │   ├── retweet_labeled_<name>.csv
│   │   └── retweet_labeled_<name>.pkl
│   ├── <name>.manifest.json        (new in topiclayers — run metadata)
│   ├── labels_<name>.json          (topic id → human label mapping)
│   └── tm/
│       └── <embedder_name>/
│           ├── embeddings_<name>.pkl
│           ├── topics_<name>.csv
│           └── model_<name>/       (BERTopic safetensors dir)
│               ├── config.json
│               ├── ctfidf.safetensors
│               ├── ctfidf_config.json
│               ├── topic_embeddings.safetensors
│               └── topics.json
└── networks/
    ├── <name>_retweet.gml
    ├── <name>_retweet_network_ml.gml   (uunet multilayer format)
    ├── <name>_ttt.gml
    └── projected/
        ├── <name>__prj_-1.0.gml
        ├── <name>__prj_0.0.gml
        └── <name>__prj_<topic>.gml
```

---

## Dataframe columns (pickle + CSV)

### `tweets_<name>.pkl` / `tweets_<name>.csv`

Index: tweet id (string or int).

| Column | Type | Notes |
|--------|------|-------|
| `author` | str/int | User id of the tweet author |
| `author_name` | str or NaN | Username if user file was loaded; NaN otherwise |
| `text` | str | Tweet body |
| `date` | str | ISO-format created_at |
| `lang` | str | Language code |
| `conversation_id` | str/int or None | |
| `referenced_type` | str or None | One of: `'retweeted'`, `'quoted'`, `'replied_to'`, or None (original) |
| `referenced_id` | str/int or None | Tweet id of referenced tweet; None for originals |
| `mentions_name` | list of str | May be empty list `[]` |
| `mentions_id` | list of str/int | May be empty list |
| `attachments` | dict or None | Media attachments |
| `cop` | str | Dataset label, e.g. `'cop26'`. **Must not be None** (bug in legacy). |

### `users_<name>.pkl`

| Column | Type |
|--------|------|
| `username` | str |
| `tweet_count` | int |
| `followers` | int |
| `following` | int |

### `retweet_labeled_<name>.pkl` / `.csv`

Subset of `tweets_<name>` columns after topic propagation, **plus**:

| Column | Type | Notes |
|--------|------|-------|
| `topic` | int | Topic id. **-1 = outlier/no topic**. Must be int, never float or str. |
| `topic_prob` | float or None | Probability of assignment to that topic. Float, may be NaN. |

### `topics_<name>.csv`

Standard BERTopic `get_topic_info()` output. Columns: `Topic`, `Count`, `Name`, `Representation`, `Representative_Docs`.

### `labels_<name>.json`

```json
{"0": "Climate science consensus", "1": "Fossil fuel divestment", ...}
```

Keys are topic ids as **strings** that json.dump produces for int keys. Load-side casts them to int.

---

## Network vertex attributes (GML)

### Single-layer retweet network: `<name>_retweet.gml`

Directed. Nodes = user ids (strings). Edges = retweet/repost interactions.

| Attribute | Type | On |
|-----------|------|----|
| `label` | str | Nodes: user id |
| (none — no bipartite, author, topics on this file) | | |

Other attributes possible but these are the only ones consumed.

### Temporal text network: `<name>_ttt.gml`

Directed bipartite graph. Nodes of two types:

**User nodes (bipartite=0):**

| Attribute | Type |
|-----------|------|
| `label` | str — user id |
| `bipartite` | int — `0` |
| `author` | str — user id (redundant with label, but consumed as `g.vs['author']`) |

**Tweet nodes (bipartite=1):**

| Attribute | Type |
|-----------|------|
| `label` | str — tweet id |
| `bipartite` | int — `1` |
| `text` | str — tweet body |
| `topics` | int or float — topic id (consumed as `g.vs.select(topics=23)`) |
| `author` | str — author user id |
| `is_retweet` | str — `'original'` or `'retweet'` (consumed as `g.vs.select(is_retweet='original')`) |

### Multilayer network: `<name>_retweet_network_ml.gml`

Format: **uunet.multinet** GML. Read with `ml.read(path)`. Layer names are topic id **strings** (e.g. `"0"`, `"1"`). The downstream code casts with `int(float(k))`.

### Per-topic projected networks: `<name>__prj_<topic>.gml`

Single-layer NetworkX GML. Directed. Nodes = user ids. No special attributes consumed beyond node labels.

---

## Interaction type mapping (legacy ↔ generic)

| `referenced_type` (legacy df column) | `interaction_type` (generic Post) |
|--------------------------------------|----------------------------------|
| `None` (NaN in CSV) | `None` / `"original"` |
| `"retweeted"` | `"repost"` |
| `"quoted"` | `"quote"` |
| `"replied_to"` | `"reply"` |
| `"original"` (never in real data, appears in test fixture) | `"original"` |

---

## Critical behavioral contract

1. **-1 = outlier**: topic `-1` always means "no topic assigned". Filtered out before network creation.
2. **NaN != None**: `pd.isna()` guards MUST be used before checking `is not None` on `referenced_id` (BUG-4 in legacy).
3. **cop column never None**: The dataset label must be set to the value passed by the user, never overwritten with None (BUG-1 legacy).
4. **ttt.gml bipartite attr**: `bipartite` is stored as `float` in GML (`0.0` / `1.0`), accessed as `node['bipartite'] == 0.0` in downstream igraph code.
5. **Topic ids in ML layers**: stored as `str(topic)` when creating `uunet` layer. Downstream reads with `ml.to_nx_dict()` then casts keys `int(float(k))`.
