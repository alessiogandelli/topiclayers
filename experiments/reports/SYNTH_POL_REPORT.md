# SYNTH_LLM_REPORT — LLM-synthetic tweets through topiclayers

7011 originals scored (derived reposts/quotes/replies excluded).

## Key findings

- Best run by ARI is **`synth_pol_best`** (k=28, purity 0.839, ARI 0.587, homogeneity 0.755); **0/4 tricky pairs merged**.
- **Off-topic posts are not reliably flagged `-1`**: rejection recall is 12%–24%, and the `-1` topic is largely filled with *on-topic* posts. The off-topic posts share an 'everyday life' register, so MiniLM groups them into a topic instead of isolating them.

## Summary

| name | n | k | ari | homogeneity | completeness | v_measure | purity | best_mapping_accuracy | outlier_rate | true_outlier_reject_recall | false_outlier_rate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| synth_pol | 7011 | 67 | 0.146 | 0.576 | 0.394 | 0.468 | 0.644 | 0.264 | 0.164 | 0.24 | 0.161 |
| synth_pol_coarse | 7011 | 11 | 0.212 | 0.389 | 0.421 | 0.405 | 0.444 | 0.427 | 0.164 | 0.24 | 0.161 |
| synth_pol_best | 7011 | 28 | 0.587 | 0.755 | 0.619 | 0.68 | 0.839 | 0.606 | 0.08 | 0.12 | 0.079 |


## Run `synth_pol`

- originals scored: **7011**, predicted clusters (excl. -1): **67**
- ARI **0.146**, homogeneity **0.576**, completeness **0.394**, purity **0.644**, best-mapping accuracy **0.264**
- true off-topic posts rejected to -1: **24.0%**; on-topic posts wrongly marked outlier: **16.1%**
- `-1` composition by true family: `{'clear': 768, 'tricky': 265, 'outlier': 72, 'bridge': 47}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 717 | -1 | 0.215 | 0.215 |
| bridge_cop_pricing | 40 | 2 | 0.4 | 0.15 |
| bridge_denial_doom | 40 | -1 | 0.575 | 0.575 |
| bridge_renew_grid | 40 | -1 | 0.425 | 0.425 |
| bridge_weather_health | 40 | 1 | 0.55 | 0.025 |
| carbon_pricing | 480 | 2 | 0.652 | 0.071 |
| climate_denial | 658 | -1 | 0.284 | 0.284 |
| climate_health | 590 | 1 | 0.69 | 0.056 |
| cop_negotiations | 863 | 9 | 0.123 | 0.098 |
| doom_anxiety | 587 | 8 | 0.187 | 0.17 |
| ev_transport | 866 | -1 | 0.206 | 0.206 |
| extreme_weather | 598 | 3 | 0.313 | 0.052 |
| grid_storage | 594 | -1 | 0.165 | 0.165 |
| outlier | 300 | 6 | 0.623 | 0.24 |
| renewables | 598 | -1 | 0.222 | 0.222 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 2 | 0.652 | 9 | 0.123 | False | 0.0 |
| grid_storage / renewables | -1 | 0.165 | -1 | 0.222 | True | 0.165 |
| doom_anxiety / climate_denial | 8 | 0.187 | -1 | 0.284 | False | 0.17 |
| climate_health / extreme_weather | 1 | 0.69 | 3 | 0.313 | False | 0.003 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 40 | 2 | carbon_pricing | 0.902 |
| bridge_renew_grid | renewables + grid_storage | 40 | -1 | climate_denial | 0.162 |
| bridge_denial_doom | climate_denial + doom_anxiety | 40 | -1 | climate_denial | 0.162 |
| bridge_weather_health | extreme_weather + climate_health | 40 | 1 | climate_health | 0.656 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

_(67 clusters — too many to print; see `netpol`/cache for the full table.)_

## Run `synth_pol_coarse`

- originals scored: **7011**, predicted clusters (excl. -1): **11**
- ARI **0.212**, homogeneity **0.389**, completeness **0.421**, purity **0.444**, best-mapping accuracy **0.427**
- true off-topic posts rejected to -1: **24.0%**; on-topic posts wrongly marked outlier: **16.1%**
- `-1` composition by true family: `{'clear': 768, 'tricky': 265, 'outlier': 72, 'bridge': 47}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 717 | 2 | 0.434 | 0.215 |
| bridge_cop_pricing | 40 | 3 | 0.55 | 0.15 |
| bridge_denial_doom | 40 | -1 | 0.575 | 0.575 |
| bridge_renew_grid | 40 | -1 | 0.425 | 0.425 |
| bridge_weather_health | 40 | 1 | 0.55 | 0.025 |
| carbon_pricing | 480 | 3 | 0.721 | 0.071 |
| climate_denial | 658 | 6 | 0.412 | 0.284 |
| climate_health | 590 | 1 | 0.737 | 0.056 |
| cop_negotiations | 863 | 2 | 0.419 | 0.098 |
| doom_anxiety | 587 | 9 | 0.232 | 0.17 |
| ev_transport | 866 | 4 | 0.56 | 0.206 |
| extreme_weather | 598 | 5 | 0.38 | 0.052 |
| grid_storage | 594 | 7 | 0.431 | 0.165 |
| outlier | 300 | 8 | 0.623 | 0.24 |
| renewables | 598 | 5 | 0.4 | 0.222 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 3 | 0.721 | 2 | 0.419 | False | 0.0 |
| grid_storage / renewables | 7 | 0.431 | 5 | 0.4 | False | 0.076 |
| doom_anxiety / climate_denial | 9 | 0.232 | 6 | 0.412 | False | 0.201 |
| climate_health / extreme_weather | 1 | 0.737 | 5 | 0.38 | False | 0.003 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 40 | 3 | carbon_pricing | 0.501 |
| bridge_renew_grid | renewables + grid_storage | 40 | -1 | climate_denial | 0.162 |
| bridge_denial_doom | climate_denial + doom_anxiety | 40 | -1 | climate_denial | 0.162 |
| bridge_weather_health | extreme_weather + climate_health | 40 | 1 | climate_health | 0.594 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

```
pred_cluster           -1    0    1    2    3    4    5    6    7    8    9    10
true_topic                                                                       
activism               154  137    5  311    1    3    5   92    1    0    8    0
bridge_cop_pricing       6    9    0    2   22    0    0    0    0    0    0    1
bridge_denial_doom      23    8    0    0    0    0    0    5    0    0    4    0
bridge_renew_grid       17    8    4    0    0    1    7    0    3    0    0    0
bridge_weather_health    1    9   22    0    0    0    3    5    0    0    0    0
carbon_pricing          34   99    0    0  346    1    0    0    0    0    0    0
climate_denial         187  121    3    0   73    0    1  271    0    0    2    0
climate_health          33  118  435    0    0    2    2    0    0    0    0    0
cop_negotiations        85  160    0  362  215    1    0    0    1    0    1   38
doom_anxiety           100  119   42   11    6    2   47  118    3    3  136    0
ev_transport           178  170    2    2   22  485    2    0    4    1    0    0
extreme_weather         31  121  144    1    0    0  227   73    0    1    0    0
grid_storage            98  120    4    1    0   70   45    0  256    0    0    0
outlier                 72    0    0    7    5    7   12    4    1  187    5    0
renewables             133  117   71    0    0   33  239    0    2    3    0    0
```

## Run `synth_pol_best`

- originals scored: **7011**, predicted clusters (excl. -1): **28**
- ARI **0.587**, homogeneity **0.755**, completeness **0.619**, purity **0.839**, best-mapping accuracy **0.606**
- true off-topic posts rejected to -1: **12.0%**; on-topic posts wrongly marked outlier: **7.9%**
- `-1` composition by true family: `{'tricky': 308, 'clear': 188, 'outlier': 36, 'bridge': 31}`

**Per true topic**

| true_topic | n | dominant_cluster | share | outlier_frac |
|---|---|---|---|---|
| activism | 717 | 2 | 0.937 | 0.026 |
| bridge_cop_pricing | 40 | 3 | 0.85 | 0.0 |
| bridge_denial_doom | 40 | 4 | 0.4 | 0.0 |
| bridge_renew_grid | 40 | -1 | 0.35 | 0.35 |
| bridge_weather_health | 40 | -1 | 0.425 | 0.425 |
| carbon_pricing | 480 | 3 | 0.933 | 0.033 |
| climate_denial | 658 | 11 | 0.236 | 0.07 |
| climate_health | 590 | 5 | 0.525 | 0.158 |
| cop_negotiations | 863 | 1 | 0.72 | 0.02 |
| doom_anxiety | 587 | 4 | 0.722 | 0.121 |
| ev_transport | 866 | 0 | 0.968 | 0.009 |
| extreme_weather | 598 | 6 | 0.378 | 0.082 |
| grid_storage | 594 | 8 | 0.357 | 0.215 |
| outlier | 300 | 19 | 0.297 | 0.12 |
| renewables | 598 | 7 | 0.358 | 0.082 |

**Tricky pairs (merged = same dominant cluster)**

| pair | a_dominant | a_share | b_dominant | b_share | merged | a_in_b_cluster |
|---|---|---|---|---|---|---|
| carbon_pricing / cop_negotiations | 3 | 0.933 | 1 | 0.72 | False | 0.013 |
| grid_storage / renewables | 8 | 0.357 | 7 | 0.358 | False | 0.002 |
| doom_anxiety / climate_denial | 4 | 0.722 | 11 | 0.236 | False | 0.002 |
| climate_health / extreme_weather | 5 | 0.525 | 6 | 0.378 | False | 0.0 |

**Bridge posts**

| bridge | links | n | top_cluster | owner_true_topic | owner_share |
|---|---|---|---|---|---|
| bridge_cop_pricing | cop_negotiations + carbon_pricing | 40 | 3 | carbon_pricing | 0.9 |
| bridge_renew_grid | renewables + grid_storage | 40 | -1 | grid_storage | 0.227 |
| bridge_denial_doom | climate_denial + doom_anxiety | 40 | 4 | doom_anxiety | 0.857 |
| bridge_weather_health | extreme_weather + climate_health | 40 | -1 | grid_storage | 0.227 |

**Confusion matrix** (rows = true topic, cols = predicted cluster)

```
pred_cluster           -1    0    1    2    3    4    5    6    7    8    9    10   11   12   13   14   15   16   17   18   19   20   21   22   23   24   25   26   27
true_topic                                                                                                                                                            
activism                19    5    4  672    0   14    0    0    1    0    0    0    0    0    0    0    0    0    0    0    1    0    0    0    0    0    0    1    0
bridge_cop_pricing       0    0    0    0   34    0    0    0    0    0    0    0    0    0    0    0    0    0    0    5    0    0    0    0    0    0    0    1    0
bridge_denial_doom       0    0    0    0    0   16    0    0    0    0    0    0    5   12    1    0    2    4    0    0    0    0    0    0    0    0    0    0    0
bridge_renew_grid       14    0    0    0    0    0    0    0    2    2    0    4    0    0    0    3    0    0    7    0    0    5    0    0    0    0    3    0    0
bridge_weather_health   17    0    0    0    0    0    8    0    0    0    6    0    0    0    0    0    0    0    0    0    0    0    0    9    0    0    0    0    0
carbon_pricing          16    2    6    0  448    0    1    0    0    0    0    0    0    1    0    1    0    0    0    1    0    0    0    0    1    0    0    3    0
climate_denial          46    2   15    1    9   19    0    0    2    0    0    1  155  112   79    0  114  100    0    0    0    0    0    2    0    1    0    0    0
climate_health          93    2    0    3    0    2  310    0    0    0  173    0    0    0    1    0    0    0    0    0    0    0    4    2    0    0    0    0    0
cop_negotiations        17    0  621    1    1    3    0    0    0    0    0    0    1    1    0    0    0    0    0   91    0    0    0    0    5   69    0   53    0
doom_anxiety            71    2   42   11    0  424    9    6    1    0    0    0    1    0    2    0    1    1    0    0    0    0    1    7    2    1    0    1    4
ev_transport             8  838    2    3    2    1    0    0    0    1    0    1    0    3    0    1    0    0    2    0    0    0    0    0    0    0    4    0    0
extreme_weather         49    0    0    7    0    6   69  226    0    0   12    0    0    0   38    0    0    1    1    0    1    0   85   56    0    0    0    0   47
grid_storage           128    7   15    8    0    3    3    1    1  212    0    0    0    0    2  113    0    1    0    0    0   78    0    1    2    0   19    0    0
outlier                 36   39   27   17    4    7    0    6    0    1    0    3    0    0    0    1    0    0    0    0   89    0    0    0   67    1    0    0    2
renewables              49    5    3    1    0    0    1    0  214    3    0  170    0    0    0    3    0    0   94    0    4    9    0    0    0    0   42    0    0
```
