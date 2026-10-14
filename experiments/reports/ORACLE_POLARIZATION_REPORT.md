# ORACLE_POLARIZATION_REPORT — netpol on ground-truth topics (synth_pol)

- layers: **15**; analyzed: **11**; polarized (FDR): **5**
- confusion (detected vs designed): TP=5 FP=0 FN=1 TN=4
- recall on designed-polarized: **83%**
- false-positive rate on consensual: **0%**

## Per-layer detail

| true_topic | n_nodes | n_edges | was_analyzed | dip | adj_p | detected_polarized | designed_polarized | skip_reason |
|---|---|---|---|---|---|---|---|---|
| activism | 352 | 930 | True | 0.125 | 0.0 | True | True | nan |
| bridge_cop_pricing | 88 | 51 | False | nan | nan | None | None | scoring_failed: ValueError: need at least 2 users after filtering, got 1 |
| bridge_denial_doom | 91 | 53 | False | nan | nan | None | None | scoring_failed: ValueError: need at least 2 users after filtering, got 1 |
| bridge_renew_grid | 85 | 47 | False | nan | nan | None | None | scoring_failed: ValueError: need at least 2 users after filtering, got 0 |
| bridge_weather_health | 94 | 55 | False | nan | nan | None | None | scoring_failed: ValueError: need at least 2 users after filtering, got 1 |
| carbon_pricing | 355 | 572 | True | 0.029 | 0.9988 | False | True | nan |
| climate_denial | 362 | 858 | True | 0.122 | 0.0 | True | True | nan |
| climate_health | 359 | 760 | True | 0.017 | 0.9988 | False | False | nan |
| cop_negotiations | 339 | 1128 | True | 0.083 | 0.0 | True | True | nan |
| doom_anxiety | 339 | 762 | True | 0.133 | 0.0 | True | True | nan |
| ev_transport | 343 | 1065 | True | 0.13 | 0.0 | True | True | nan |
| extreme_weather | 305 | 741 | True | 0.023 | 0.9988 | False | False | nan |
| grid_storage | 360 | 758 | True | 0.03 | 0.9988 | False | False | nan |
| outlier | 569 | 375 | True | 0.112 | 0.9988 | False | None | nan |
| renewables | 381 | 865 | True | 0.024 | 0.9988 | False | False | nan |
