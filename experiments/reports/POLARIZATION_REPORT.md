# POLARIZATION_REPORT — netpol vs designed polarization (synth_pol_best)

- multilayer layers: **28**; analyzed: **24**; polarized (FDR): **4**
- layers mapped to a designed topic: **24**
- confusion (detected vs designed): TP=3 FP=1 FN=10 TN=10
- recall on designed-polarized: **23%**
- false-positive rate on consensual: **9%**

## Per-layer detail

| layer | n_nodes | n_edges | was_analyzed | dip | adj_p | detected_polarized | true_topic | designed_polarized | layer_purity | skip_reason |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 446 | 908 | True | 0.092 | 0.0 | True | ev_transport | True | 0.929 | nan |
| 1 | 479 | 752 | True | 0.063 | 0.0523 | False | cop_negotiations | True | 0.845 | nan |
| 2 | 383 | 741 | True | 0.042 | 0.4066 | False | activism | True | 0.928 | nan |
| 3 | 365 | 474 | True | 0.047 | 0.7264 | False | carbon_pricing | True | 0.9 | nan |
| 4 | 407 | 537 | True | 0.084 | 0.0207 | True | doom_anxiety | True | 0.857 | nan |
| 5 | 368 | 385 | True | 0.057 | 0.605 | False | climate_health | False | 0.773 | nan |
| 6 | 248 | 263 | True | 0.085 | 0.0523 | False | extreme_weather | False | 0.946 | nan |
| 7 | 264 | 276 | True | 0.062 | 0.4066 | False | renewables | False | 0.968 | nan |
| 8 | 219 | 203 | True | 0.071 | 0.879 | False | grid_storage | False | 0.968 | nan |
| 9 | 219 | 216 | True | 0.146 | 0.0004 | True | climate_health | False | 0.906 | nan |
| 10 | 203 | 184 | True | 0.113 | 0.0523 | False | renewables | False | 0.95 | nan |
| 11 | 189 | 159 | True | 0.131 | 0.1945 | False | climate_denial | True | 0.957 | nan |
| 12 | 158 | 157 | True | 0.077 | 0.3486 | False | climate_denial | True | 0.868 | nan |
| 13 | 138 | 93 | True | 0.125 | 0.5053 | False | climate_denial | True | 0.642 | nan |
| 14 | 148 | 119 | True | 0.125 | 0.1653 | False | grid_storage | False | 0.926 | nan |
| 15 | 146 | 149 | True | 0.08 | 0.32 | False | climate_denial | True | 0.974 | nan |
| 16 | 146 | 125 | True | 0.127 | 0.0523 | False | climate_denial | True | 0.935 | nan |
| 17 | 154 | 125 | True | 0.136 | 0.1105 | False | renewables | False | 0.904 | nan |
| 18 | 136 | 124 | True | 0.067 | 0.8886 | False | cop_negotiations | True | 0.938 | nan |
| 19 | 123 | 85 | False | nan | nan | None | outlier | False | 0.937 | scoring_failed: ValueError: need at least 2 users after filtering, got 1 |
| 20 | 107 | 78 | True | 0.167 | 0.1105 | False | grid_storage | False | 0.848 | nan |
| 21 | 116 | 87 | True | 0.13 | 0.2532 | False | extreme_weather | False | 0.944 | nan |
| 22 | 112 | 82 | True | 0.143 | 0.2182 | False | extreme_weather | False | 0.727 | nan |
| 23 | 96 | 66 | False | nan | nan | None | outlier | False | 0.87 | dip_test_failed: ValueError: dip test requires at least 4 scores, got 2 |
| 24 | 100 | 73 | True | 0.083 | 0.9 | False | cop_negotiations | True | 0.958 | nan |
| 25 | 81 | 52 | False | nan | nan | None | renewables | False | 0.618 | scoring_failed: ValueError: need at least 2 users after filtering, got 1 |
| 26 | 72 | 62 | True | 0.214 | 0.0 | True | cop_negotiations | True | 0.898 | nan |
| 27 | 66 | 47 | False | nan | nan | None | extreme_weather | False | 0.887 | dip_test_failed: ValueError: dip test requires at least 4 scores, got 3 |
