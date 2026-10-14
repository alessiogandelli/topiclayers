# TUNING_NOTES — what the synthetic corpora taught us about `topiclayers`

Living document. Everything here is backed by a reproducible run; metrics are in
`tuning_metrics_v1.json`, `tuning_metrics_v2.json`, `sweep_results.json`,
`sweep_polarization.json`, `netpol_synth_pol.csv`, and the reports
(`SYNTH_LLM_REPORT.md`, `SYNTH_POL_REPORT.md`, `SWEEP_RESULTS.md`,
`SWEEP_POLARIZATION.md`, `POLARIZATION_REPORT.md`, `ORACLE_POLARIZATION_REPORT.md`).

---

## 1. The two corpora

| | **v1 `synth_llm`** | **v2 `synth_pol`** (use this) |
|---|---|---|
| Generator | `gen_llm_tweets.py` | `gen_llm_tweets_v2.py` |
| Style | polished, essay-like | **realistic**: short, messy, hashtags, @mentions, `RT @user:`, t.co links, typos, ALL-CAPS, fragments |
| Originals | 2,440 | **7,011** (+ 9,114 retweets/quotes/replies) |
| Languages | ~en only | en 81%, fr 10%, es 6%, de 3% |
| Median length | ~130 chars | **66 chars** (62% < 70) |
| Structure | 10 topics + bridges + 240 off-topic | 10 topics + bridges + 300 off-topic; **2,000 authors each with a pro/anti side; same-side retweeting** |
| Purpose | first benchmark; the `nr_topics: auto` collapse | **realism + polarization** |

Both share the same topic ids, so `eval_synth_llm.py` works on either. Ground
truth: `truth_<name>.csv` (per post) and `authors_<name>.csv` (per author:
`side`, `home_topics`). `topics_<name>.csv` marks which topics are designed
`polarized`.

### Reproduce
```bash
# v2 corpus (~280 cached gpt-6-luna batches; resumable via llm_cache_v2/)
.venv/bin/python experiments/synth_llm/gen_llm_tweets_v2.py --out-dir experiments/synth_llm
.venv/bin/topiclayers run experiments/synth_llm/synth_pol.yml          # granular (67 topics)
.venv/bin/topiclayers run experiments/synth_llm/synth_pol_coarse.yml   # coarse (12 topics)
.venv/bin/python experiments/synth_llm/eval_synth_llm.py --cache experiments/cache \
    --truth experiments/synth_llm/truth_synth_pol.csv --names synth_pol synth_pol_coarse \
    --out experiments/reports/SYNTH_POL_REPORT.md --json experiments/synth_llm/tuning_metrics_v2.json
.venv/bin/python experiments/synth_llm/eval_polarization.py --name synth_pol_coarse \
    --ml out/synth_pol_coarse/networks/synth_pol_coarse_retweet_network_ml.gml \
    --labeled experiments/cache/synth_pol_coarse/tm/all-MiniLM-L6-v2/tweets_synth_pol_coarse_labeled.pkl
.venv/bin/python experiments/synth_llm/oracle_polarization.py
```

---

## 2. Findings (ranked by impact)

### F1. `nr_topics: "auto"` (the default) destroys fine-grained topics  ⚠️ biggest
On v1, auto-reduction merged **11 real topics into 3 clusters** (purity 0.195,
ARI 0.031) — all climate subtopics became one blob. HDBSCAN's natural structure
was fine (84 clusters); the *reduction* did the damage.
**Fix:** set `nr_topics: null` and control granularity with `min_topic_size`
(or set an explicit k). Never trust `auto` for this kind of corpus.

### F2. Realistic tweets are *much* harder to cluster than polished ones
| corpus / config | k | purity | ARI |
|---|---|---|---|
| v1, `min_topic_size=30, nr_topics=null` | 28 | **0.895** | **0.575** |
| v1, `nr_topics=12` | 11 | 0.512 | 0.405 |
| v2, `nr_topics=null` | 67 | 0.644 | 0.146 |
| v2, `nr_topics=12` | 11 | 0.444 | 0.212 |

Short, noisy, hashtag/multilingual tweets share vocabulary across topics, so the
English `all-MiniLM-L6-v2` embeddings overlap heavily. **Any tuning conclusion
drawn on clean synthetic text overstates real performance** — always validate on
the realistic corpus.

### F3. The English-only embedder is a real limitation here
v2 is ~19% French/Spanish/German. `all-MiniLM-L6-v2` is English-only; the
non-English share degrades clusters (same caveat the real `cop22` run showed).
**Try:** a multilingual embedder (`paraphrase-multilingual-MiniLM-L12-v2`,
`intfloat/multilingual-e5-*`) and compare ARI/purity on the same corpus.

### F4. Over-segmentation vs completeness trade-off
At natural granularity (`nr_topics=null`) you get many pure but tiny clusters
(v2: k=67, homogeneity 0.576 but completeness 0.394). Forcing k=12 raises
completeness but mixes topics (purity 0.444). There is no single setting; pick by
downstream need:
- **topic labeling / interpretation** → `nr_topics=null` (pure clusters).
- **polarization per layer (netpol)** → **coarser** (big layers), see F7.

### F5. Off-topic posts are *not* reliably flagged as `-1`
Rejection recall of the designed off-topic posts is only **21–24%** (v1 and v2
alike); the `-1` topic is mostly filled with *on-topic* posts. The off-topic
posts share an "everyday life" register, so MiniLM clusters them into a topic
instead of isolating them. Don't rely on `-1` as an off-topic detector without
checking its composition (the eval reports it: `-1 composition by true family`).

### F6. Tricky near-duplicate topics: separable only at the right granularity
On v1 at natural granularity, **0/4** tricky pairs merged (carbon-pricing vs COP,
grid vs renewables, doom vs denial, health vs weather). On the harder v2 text,
several pairs are absorbed into `-1` or a neighbour. The *tricky-pair table* in
the eval report is the thing to watch when tuning.

### F7. Polarization detection is bottlenecked by topic quality, not by netpol
The v2 corpus is built so 6 topics have two opposing camps with same-side
retweeting (96% within-side edges) and 4 are consensual. Running netpol
(latent ideology + Hartigan dip, FDR):

| layers used | recall (designed-polarized) | false-positive (consensual) |
|---|---|---|
| **ground-truth topics (oracle)** | **83%** (5/6) | **0%** |
| pipeline `nr_topics=12` (coarse) | 43% (3/6) | 25% |

So when topics are clean, netpol recovers polarization well. Through the pipeline
it loses signal because topics merge/split: the coarse run put `activism` +
`ev_transport` + others into one 1,219-node mega-layer, which no longer looks
bimodal. **To see polarization through the pipeline you need topic layers that
match the real topics** — i.e. better clustering (multilingual embedder, tuned
granularity), or analyze per *known* topic.

`carbon_pricing` was the one designed-polarized topic missed even by the oracle
(its camps overlap with `cop_negotiations`, its tricky neighbour) — a good
stress case to keep.

---

## 3. Recommended settings (starting points)

Based on the 18-config sweep (`SWEEP_RESULTS.md`) + polarization sweep
(`SWEEP_POLARIZATION.md`), **on the realistic `synth_pol` corpus**:

| goal | config | result |
|---|---|---|
| **best clustering (recommended)** | `Qwen/Qwen3-Embedding-0.6B`, `min_topic_size: 50`, `nr_topics: null`, `umap_n_components: 10` | **ARI 0.587**, purity 0.839 |
| **English-only fallback** | `all-mpnet-base-v2` | ARI 0.325, en-ARI 0.433 (see §7) |
| lightweight fallback | `paraphrase-multilingual-MiniLM-L12-v2`, `min_topic_size: 50` | ARI 0.425 (fast, 384-d) |
| fewer, bigger topics | `nr_topics: 12` (or `min_topic_size` ≥ 50) | — |
| **polarization via netpol** | **no config fixes it** — recall ~0–23% through the pipeline; use ground-truth/known topics (oracle 83%) | see §5, §7 |
| always | `QDRANT_URL= LABEL_API_BASE=` when not using those services; `repro_mode: strict` for determinism | — |

> Caveat: absolute ARI stays ≤ 0.33 — realistic short/noisy/multilingual tweets
> are genuinely hard. Treat these as relative rankings on this corpus, not
> expected real-world scores.

## 4. How to use this for polarization later
- `authors_synth_pol.csv` → each author's `side` (pro/anti) and `home_topics`.
- `truth_synth_pol.csv` → per post `true_topic`, `true_side`, `topic_polarized`.
- `topics_synth_pol.csv` → which topics are designed polarized (the answer key).
- `oracle_polarization.py` → netpol's ceiling on clean topics.
- `eval_polarization.py` → netpol vs the answer key on any pipeline run.

## 5. Parameter sweep results (v2, realistic corpus)

`python experiments/synth_llm/sweep_params.py` — 18 configs, embeddings computed
once per embedder (fast). Top rows by ARI:

| id | embedder | min_topic_size | nr_topics | k | ARI | purity | tricky merged | outlier recall |
|---|---|---|---|---|---|---|---|---|
| min50_kall | MiniLM | 50 | null | 24 | **0.331** | 0.602 | 1/4 | 0.17 |
| multi_min20_k12 | multilingual | 20 | 12 | 11 | 0.290 | 0.557 | 0/4 | **0.41** |
| min50_k12 | MiniLM | 50 | 12 | 11 | 0.267 | 0.439 | 2/4 | 0.17 |
| min30_k12 | MiniLM | 30 | 12 | 11 | 0.212 | 0.444 | 0/4 | 0.24 |
| min20_k12 | MiniLM | 20 | 12 | 11 | 0.173 | 0.436 | 0/4 | 0.44 |
| min20_auto | MiniLM | 20 | auto | 51 | 0.115 | 0.542 | 0/4 | 0.44 |
| min20_kall | MiniLM | 20 | null | 121 | 0.051 | 0.689 | 3/4 | 0.44 |

Findings:
- **`min_topic_size: 50` beats 10/20/30** on this noisy corpus: fewer spurious
  micro-clusters → best ARI. Small `min_topic_size` explodes into 100–238
  clusters (high purity, low ARI — over-segmentation).
- **Multilingual embedder helps** even at matched granularity: `min20_k12`
  MiniLM ARI 0.173 → `multi_min20_k12` ARI 0.290, and outlier recall 0.44 vs
  0.41. But multilingual with `nr_topics: null` over-splits (112 clusters, ARI
  0.069) — it **needs** `nr_topics` control.
- **`nr_topics: 12` vs `null`**: at `min_topic_size=50` null wins (0.331 vs
  0.267); at 10/20/30, k=12 wins. There is no universal winner.
- **UMAP `n_neighbors`/`n_components`**: defaults (15/5) beat every variation
  tried (5/10/30/50 neighbours, 2/10 components) — and note these are **not
  exposed in the pipeline config** (the pipeline hardcodes `n_neighbors=15`,
  `n_components=5`); the sweep passed a custom `umap_model` to `TopicModeler`.
  `umap_init: spectral` was worse than `pca` and slower.
- `nr_topics: auto` is still poor (ARI 0.115, 51 clusters) but no longer the
  v1-style total collapse — realistic text has more structure.

### Polarization recovery (`sweep_polarization.py`)
Layers built from the pipeline's predicted topics **exactly as the pipeline
builds them** (only *reposts* make edges; quotes/replies dropped; node set =
originals+reposts). netpol recall/FPR vs the designed answer key:

| id | k | ARI | netpol TP | FP | FN | TN | recall | FPR |
|---|---|---|---|---|---|---|---|---|
| min50_kall | 24 | 0.331 | 3 | 1 | 11 | 4 | 21% | 20% |
| multi_min20_k12 | 11 | 0.290 | 1 | 1 | 4 | 3 | 20% | 25% |
| multi_min20 | 112 | 0.069 | 5 | 3 | 17 | 9 | 23% | 25% |
| min20_auto | 51 | 0.115 | 3 | 1 | 13 | 6 | 19% | 14% |
| min30_kall | 67 | 0.146 | 3 | 1 | 21 | 8 | 13% | 11% |
| min50_k12 | 11 | 0.267 | 1 | 0 | 8 | 1 | 11% | 0% |

- **No config recovers polarization well through the pipeline** — recall is
  11–23% across all settings, versus **83% on ground-truth topics** (oracle).
  The bottleneck is *topic-model error*, not the knobs: predicted layers mix
  designed topics whose camps sit on opposite sides, so the retweet graph no
  longer looks bimodal.
- **Methodological trap (corrected):** an earlier version of this sweep built
  layers from *all* derived posts (including quotes/replies) and got recall 67%.
  The pipeline only uses **reposts** for edges, so that was inflated. The table
  above mirrors the pipeline and reproduces the full-pipeline netpol run exactly.
- **Implication for the paper:** to measure polarization you must analyze
  per *known/coarse* topic (or fix clustering first); the automatic pipeline
  output is too noisy on realistic short multilingual tweets.

## 7. Embedder decision — the biggest single lever

Swapping the embedder roughly **doubles** clustering quality, far more than any
clustering knob. Bake-off on `synth_pol` (`EMBEDDER_REPORT.md`), min50/null:

| model | dim | ARI | purity | en-ARI | local |
|---|---|---|---|---|---|
| **Qwen/Qwen3-Embedding-0.6B** | 1024 | **0.568** | 0.829 | 0.553 | yes |
| text-embedding-3-small (OpenAI) | 1536 | 0.551 | 0.805 | 0.646 | no (API) |
| paraphrase-multilingual-MiniLM-L12-v2 | 384 | 0.425 | 0.722 | 0.407 | yes |
| paraphrase-multilingual-mpnet-base-v2 | 768 | 0.380 | 0.669 | 0.361 | yes |
| all-mpnet-base-v2 | 768 | 0.325 | 0.543 | 0.433 | yes |
| all-MiniLM-L6-v2 (old default) | 384 | 0.313 | 0.588 | 0.441 | yes |
| intfloat/multilingual-e5-small | 384 | 0.301 | 0.681 | 0.414 | yes |
| intfloat/multilingual-e5-base | 768 | 0.298 | 0.728 | 0.396 | yes |

- **Qwen3-Embedding-0.6B is the winner** — best on full *and* English-only
  subsets, so it is not just better at fr/es/de. It ties the OpenAI API model,
  so **no API needed**. Phase 3 (`EMBEDDER_TUNING.md`): `umap_n_components=10`
  lifts it to **ARI 0.587**; larger `n_neighbors` hurts; the instruction prompt
  and normalization barely matter for Qwen3.
- **E5 underperforms here** (0.30) despite its reputation — the symmetric
  `"query: "` prefix and these older checkpoints did not beat multilingual
  MiniLM. Worth re-testing only with proper asymmetric handling.
- **Normalization**: helps multilingual MiniLM (0.425 vs 0.406 without);
  near-neutral for Qwen3. Default is on (correct for cosine/UMAP).
- **Cost**: Qwen3-0.6B is 0.6B params and slower to embed (~3.5 min for 7k docs
  on MPS, one-time and cached); text-embedding-3-small is 19 s but not local.

### Real-data gate (the actual decision) — `COP22_VALIDATION.md`
On a real 6,000-line COP22 slice (en 900 / fr 230 / es 70 / … originals):
- **MiniLM → 2 topics**: one giant English blob + a junk `la en les le`
  French-stopword topic (the known failure).
- **multilingual MiniLM → 9 topics**, still a `googlealerts libros novelas` junk
  cluster.
- **Qwen3 → 14 coherent topics** with good LLM labels: *India's renewable energy
  transition*, *COP22 Marrakech participation*, *Climate impacts on
  agriculture*, *Trump and climate science*, *Australian oil politics*,
  *Arctic ice melt*, *Forêts et biodiversité* (mixing en/fr/de sensibly).
Winner confirmed on real data, not just synthetic ARI.

### But it does not fix polarization
With Qwen3 the clustering is far better (ARI 0.587) yet netpol recall on the
designed polarization stays low: **23% at 28 clusters, 0% at 12 clusters**
(vs 83% oracle). Better topics did not unblock polarization — the layer must
still match one designed topic *and* keep the two camps inside it. See §5/§6.

## 8. Open questions / next steps
1. ~~Swap in a multilingual embedder~~ **done** (§7): Qwen3-Embedding-0.6B wins;
   it also beats the OpenAI API reference, so no API needed.
2. ~~Expose UMAP params in the pipeline config~~ **done**: `umap_n_neighbors` /
   `umap_n_components` now flow through; `n_components=10` helped Qwen3.
3. ~~English-only fallback~~ identified: `all-mpnet-base-v2` (§7).
4. **Re-test E5 properly.** It underperformed with the symmetric `"query: "`
   prefix; try the documented asymmetric treatment / newer e5 checkpoints.
5. **Semi-supervised topic seeding** (`seed_topic_list`) to map clusters back
   onto the ~10 known topics — the most promising way to unblock polarization
   (oracle 83% vs pipeline ≤23%).
6. **Outlier test**: add truly idiosyncratic singleton posts to see whether
   `-1` can work at all.
7. Validate the whole config on the full `cop22.json` (181k docs) — Qwen3 is
   slow to embed; confirm wall-time is acceptable or batch/cache it.
