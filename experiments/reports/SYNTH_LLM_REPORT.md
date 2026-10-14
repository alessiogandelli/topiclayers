# SYNTH_LLM_REPORT (v1) — LLM-synthetic tweets through topiclayers

First-generation corpus (v1): polished, essay-like tweets, 2440 originals, 10 topics (6 clear + 4 tricky) + bridges + 240 off-topic posts. Superseded by v2 (SYNTH_POL_REPORT) for realism/polarization; kept for the auto-collapse finding.

2440 originals scored (derived reposts/quotes/replies excluded).

Reproduce:

```bash
gen: .venv/bin/python experiments/synth_llm/gen_llm_tweets.py --out-dir experiments/synth_llm; run: .venv/bin/topiclayers run experiments/synth_llm/synth_llm.yml  (also _resolved, _t12, _overgranular); eval: .venv/bin/python experiments/synth_llm/eval_synth_llm.py --names synth_llm synth_llm_t12 synth_llm_resolved synth_llm_overgranular
```

## Key findings

- **`nr_topics: auto` over-merges**: run `synth_llm` found only **3** clusters (purity 0.195, ARI 0.031); the fine-grained subtopics collapse into one blob and the tricky pairs become indistinguishable.
- Best run by ARI is **`synth_llm_resolved`** (k=28, purity 0.895, ARI 0.575, homogeneity 0.86); **0/4 tricky pairs merged**.
- **Off-topic posts are not reliably flagged `-1`**: rejection recall is 0%–21%, and the `-1` topic is largely filled with *on-topic* posts. The off-topic posts share an 'everyday life' register, so MiniLM groups them into a topic instead of isolating them.

## Summary

| name | n | k | ari | homogeneity | completeness | v_measure | purity | best_mapping_accuracy | outlier_rate | true_outlier_reject_recall | false_outlier_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synth_llm | 2440 | 3 | 0.031 | 0.108 | 0.433 | 0.172 | 0.195 | 0.179 | 0.102 | 0.212 | 0.09 |
| synth_llm_t12 | 2440 | 11 | 0.405 | 0.553 | 0.697 | 0.617 | 0.512 | 0.471 | 0.101 | 0.192 | 0.091 |
| synth_llm_resolved | 2440 | 28 | 0.575 | 0.86 | 0.673 | 0.755 | 0.895 | 0.563 | 0.075 | 0.0 | 0.084 |
| synth_llm_overgranular | 2440 | 52 | 0.323 | 0.845 | 0.56 | 0.674 | 0.877 | 0.348 | 0.102 | 0.212 | 0.09 |


## Run `synth_llm`

- originals scored: **2440**, predicted clusters (excl. -1): **3**
- ARI **0.031**, homogeneity **0.108**, completeness **0.433**, purity **0.195**, best-mapping accuracy **0.179**
- true off-topic posts rejected to -1: **21.2%**; on-topic posts wrongly marked outlier: **9.0%**
- `-1` composition by true family: `{'clear': 123, 'tricky': 56, 'outlier': 51, 'bridge': 18}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 250 | 0 | 0.992 | 0.008 |
| bridge_cop_pricing | 15 | 0 | 1.0 | 0.0 |
| bridge_denial_doom | 15 | -1 | 0.533 | 0.533 |
| bridge_renew_grid | 15 | 0 | 0.8 | 0.2 |
| bridge_weather_health | 15 | 0 | 0.533 | 0.467 |
| carbon_pricing | 160 | 0 | 0.994 | 0.006 |
| climate_denial | 250 | 0 | 0.9 | 0.1 |
| climate_health | 160 | 0 | 0.869 | 0.131 |
| cop_negotiations | 250 | 0 | 0.924 | 0.072 |
| doom_anxiety | 160 | 0 | 0.888 | 0.113 |
| ev_transport | 250 | 0 | 0.868 | 0.132 |
| extreme_weather | 250 | 0 | 0.856 | 0.144 |
| grid_storage | 160 | 0 | 0.9 | 0.1 |
| outlier | 240 | 1 | 0.638 | 0.212 |
| renewables | 250 | 0 | 0.964 | 0.036 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 0 | 0.994 | 0 | 0.924 | True | 0.994 |
| grid_storage / renewables | 0 | 0.9 | 0 | 0.964 | True | 0.9 |
| doom_anxiety / climate_denial | 0 | 0.888 | 0 | 0.9 | True | 0.887 |
| climate_health / extreme_weather | 0 | 0.869 | 0 | 0.856 | True | 0.869 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 15 | 0 | activism | 0.123 |
| bridge_renew_grid | renewables + grid_storage | 15 | 0 | activism | 0.123 |
| bridge_denial_doom | climate_denial + doom_anxiety | 15 | -1 | outlier | 0.206 |
| bridge_weather_health | extreme_weather + climate_health | 15 | 0 | activism | 0.123 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

```
pred_cluster           -1    0    1   2
true_topic                             
activism                2  248    0   0
bridge_cop_pricing      0   15    0   0
bridge_denial_doom      8    7    0   0
bridge_renew_grid       3   12    0   0
bridge_weather_health   7    8    0   0
carbon_pricing          1  159    0   0
climate_denial         25  225    0   0
climate_health         21  139    0   0
cop_negotiations       18  231    1   0
doom_anxiety           18  142    0   0
ev_transport           33  217    0   0
extreme_weather        36  214    0   0
grid_storage           16  144    0   0
outlier                51   11  153  25
renewables              9  241    0   0
```

## Run `synth_llm_t12`

- originals scored: **2440**, predicted clusters (excl. -1): **11**
- ARI **0.405**, homogeneity **0.553**, completeness **0.697**, purity **0.512**, best-mapping accuracy **0.471**
- true off-topic posts rejected to -1: **19.2%**; on-topic posts wrongly marked outlier: **9.1%**
- `-1` composition by true family: `{'clear': 154, 'outlier': 46, 'tricky': 28, 'bridge': 18}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 250 | 1 | 0.828 | 0.156 |
| bridge_cop_pricing | 15 | 2 | 1.0 | 0.0 |
| bridge_denial_doom | 15 | 1 | 0.467 | 0.4 |
| bridge_renew_grid | 15 | 0 | 0.6 | 0.333 |
| bridge_weather_health | 15 | -1 | 0.467 | 0.467 |
| carbon_pricing | 160 | 2 | 0.994 | 0.006 |
| climate_denial | 250 | 2 | 0.936 | 0.052 |
| climate_health | 160 | 4 | 0.706 | 0.075 |
| cop_negotiations | 250 | 3 | 0.78 | 0.068 |
| doom_anxiety | 160 | 1 | 0.925 | 0.062 |
| ev_transport | 250 | 0 | 0.964 | 0.032 |
| extreme_weather | 250 | 1 | 0.728 | 0.188 |
| grid_storage | 160 | 0 | 0.969 | 0.031 |
| outlier | 240 | 5 | 0.471 | 0.192 |
| renewables | 250 | 0 | 0.7 | 0.12 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 2 | 0.994 | 3 | 0.78 | False | 0.0 |
| grid_storage / renewables | 0 | 0.969 | 0 | 0.7 | True | 0.969 |
| doom_anxiety / climate_denial | 1 | 0.925 | 2 | 0.936 | False | 0.0 |
| climate_health / extreme_weather | 4 | 0.706 | 1 | 0.728 | False | 0.075 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 15 | 2 | climate_denial | 0.515 |
| bridge_renew_grid | renewables + grid_storage | 15 | 0 | ev_transport | 0.413 |
| bridge_denial_doom | climate_denial + doom_anxiety | 15 | 1 | activism | 0.367 |
| bridge_weather_health | extreme_weather + climate_health | 15 | -1 | extreme_weather | 0.191 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

```
pred_cluster           -1    0    1    2    3    4    5    6    7    8    9    10
true_topic                                                                       
activism                39    1  207    3    0    0    0    0    0    0    0    0
bridge_cop_pricing       0    0    0   15    0    0    0    0    0    0    0    0
bridge_denial_doom       6    0    7    2    0    0    0    0    0    0    0    0
bridge_renew_grid        5    9    0    0    0    1    0    0    0    0    0    0
bridge_weather_health    7    0    6    0    0    2    0    0    0    0    0    0
carbon_pricing           1    0    0  159    0    0    0    0    0    0    0    0
climate_denial          13    0    2  234    1    0    0    0    0    0    0    0
climate_health          12    0   12    0    0  113    0    0    0   23    0    0
cop_negotiations        17    0    0   37  195    0    1    0    0    0    0    0
doom_anxiety            10    0  148    0    1    1    0    0    0    0    0    0
ev_transport             8  241    0    1    0    0    0    0    0    0    0    0
extreme_weather         47    1  182    0    0   20    0    0    0    0    0    0
grid_storage             5  155    0    0    0    0    0    0    0    0    0    0
outlier                 46    1    0    2    0    1  113   30   25    0   12   10
renewables              30  175    0    1    0   44    0    0    0    0    0    0
```

## Run `synth_llm_resolved`

- originals scored: **2440**, predicted clusters (excl. -1): **28**
- ARI **0.575**, homogeneity **0.86**, completeness **0.673**, purity **0.895**, best-mapping accuracy **0.563**
- true off-topic posts rejected to -1: **0.0%**; on-topic posts wrongly marked outlier: **8.4%**
- `-1` composition by true family: `{'clear': 111, 'tricky': 61, 'bridge': 12}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 250 | 2 | 0.78 | 0.012 |
| bridge_cop_pricing | 15 | 5 | 0.4 | 0.133 |
| bridge_denial_doom | 15 | 6 | 0.8 | 0.133 |
| bridge_renew_grid | 15 | -1 | 0.467 | 0.467 |
| bridge_weather_health | 15 | 3 | 0.4 | 0.067 |
| carbon_pricing | 160 | 5 | 0.581 | 0.0 |
| climate_denial | 250 | 7 | 0.38 | 0.068 |
| climate_health | 160 | 4 | 0.538 | 0.212 |
| cop_negotiations | 250 | 9 | 0.272 | 0.036 |
| doom_anxiety | 160 | 6 | 0.588 | 0.05 |
| ev_transport | 250 | 0 | 0.992 | 0.0 |
| extreme_weather | 250 | 3 | 0.476 | 0.224 |
| grid_storage | 160 | 15 | 0.338 | 0.119 |
| outlier | 240 | 1 | 0.925 | 0.0 |
| renewables | 250 | 8 | 0.344 | 0.104 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 5 | 0.581 | 9 | 0.272 | False | 0.0 |
| grid_storage / renewables | 15 | 0.338 | 8 | 0.344 | False | 0.0 |
| doom_anxiety / climate_denial | 6 | 0.588 | 7 | 0.38 | False | 0.0 |
| climate_health / extreme_weather | 4 | 0.538 | 3 | 0.476 | False | 0.0 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 15 | 5 | carbon_pricing | 0.823 |
| bridge_renew_grid | renewables + grid_storage | 15 | -1 | extreme_weather | 0.304 |
| bridge_denial_doom | climate_denial + doom_anxiety | 15 | 6 | doom_anxiety | 0.887 |
| bridge_weather_health | extreme_weather + climate_health | 15 | 3 | extreme_weather | 0.937 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

```
pred_cluster           -1    0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27
true_topic                                                                                                                                                            
activism                 3    1    0  195    1    0    1    0    0    0    0    0    0    1    0    0    0    0   48    0    0    0    0    0    0    0    0    0    0
bridge_cop_pricing       2    0    0    0    0    0    6    0    0    0    0    0    0    0    0    0    0    0    0    0    0    4    0    0    0    0    0    0    3
bridge_denial_doom       2    0    0    0    0    0    0   12    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0
bridge_renew_grid        7    0    0    0    0    0    0    0    0    2    0    0    0    0    1    0    0    3    0    0    0    0    0    0    2    0    0    0    0
bridge_weather_health    1    0    0    0    6    4    0    0    0    0    0    0    0    0    0    0    0    0    0    0    2    0    2    0    0    0    0    0    0
carbon_pricing           0    1    0    0    0    0   93    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   37    0   29
climate_denial          17    0    0    3    0    1   11    0   95    0    0    0   65   56    0    0    0    0    0    0    0    1    0    0    0    0    0    1    0
climate_health          34    0    0    0    0   86    0    0    0    0    0    0    0    0    0    3    0    0    0    0    0    0   37    0    0    0    0    0    0
cop_negotiations         9    0    1    0    0    0    0    0    0    0   68   66    0    0    0    0    0    0    0    0    0   36    0    0    0   38    0   32    0
doom_anxiety             8    0    0    0    0    1    0   94    0    0    0    0    0    0    0   55    0    0    1    0    0    0    0    0    0    0    0    1    0
ev_transport             0  248    0    0    0    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0    1    0    0    0    0
extreme_weather         56    0    0    0  119   29    0    0    0    0    0    0    0    0    0    0    0    0    0    0   45    0    1    0    0    0    0    0    0
grid_storage            19    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   54    0    0   48    0    0    0   39    0    0    0    0    0
outlier                  0    6  222    7    1    0    2    0    0    1    0    0    0    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0    0
renewables              26    0    0    0    0    0    0    0    0   86    0    0    0    0   57    0    0   46    0    0    0    0    0    0   35    0    0    0    0
```

## Run `synth_llm_overgranular`

- originals scored: **2440**, predicted clusters (excl. -1): **52**
- ARI **0.323**, homogeneity **0.845**, completeness **0.56**, purity **0.877**, best-mapping accuracy **0.348**
- true off-topic posts rejected to -1: **21.2%**; on-topic posts wrongly marked outlier: **9.0%**
- `-1` composition by true family: `{'clear': 123, 'tricky': 56, 'outlier': 51, 'bridge': 18}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 250 | 0 | 0.78 | 0.008 |
| bridge_cop_pricing | 15 | 1 | 0.4 | 0.0 |
| bridge_denial_doom | 15 | -1 | 0.533 | 0.533 |
| bridge_renew_grid | 15 | 44 | 0.267 | 0.2 |
| bridge_weather_health | 15 | -1 | 0.467 | 0.467 |
| carbon_pricing | 160 | 1 | 0.581 | 0.006 |
| climate_denial | 250 | 2 | 0.404 | 0.1 |
| climate_health | 160 | 19 | 0.231 | 0.131 |
| cop_negotiations | 250 | 3 | 0.272 | 0.072 |
| doom_anxiety | 160 | 5 | 0.362 | 0.113 |
| ev_transport | 250 | 8 | 0.224 | 0.132 |
| extreme_weather | 250 | 6 | 0.22 | 0.144 |
| grid_storage | 160 | 9 | 0.344 | 0.1 |
| outlier | 240 | 11 | 0.212 | 0.212 |
| renewables | 250 | 7 | 0.228 | 0.036 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 1 | 0.581 | 3 | 0.272 | False | 0.0 |
| grid_storage / renewables | 9 | 0.344 | 7 | 0.228 | False | 0.0 |
| doom_anxiety / climate_denial | 5 | 0.362 | 2 | 0.404 | False | 0.0 |
| climate_health / extreme_weather | 19 | 0.231 | 6 | 0.22 | False | 0.0 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 15 | 1 | carbon_pricing | 0.823 |
| bridge_renew_grid | renewables + grid_storage | 15 | 44 | renewables | 0.783 |
| bridge_denial_doom | climate_denial + doom_anxiety | 15 | -1 | outlier | 0.206 |
| bridge_weather_health | extreme_weather + climate_health | 15 | -1 | outlier | 0.206 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

_(52 clusters — too many to print; see `netpol`/cache for the full table.)_
