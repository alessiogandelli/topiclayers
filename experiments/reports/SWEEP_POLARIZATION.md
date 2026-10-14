# SWEEP_POLARIZATION — netpol recovery by config (`synth_pol`)

Layers built from the pipeline's predicted topics (retweets inherit the target's topic). `recall` = designed-polarized layers netpol flags; `fpr` = consensual layers it wrongly flags.

| id | k | ari | purity | tricky_merged | layers | analyzed | polarized_detected | TP | FP | FN | TN | recall | fpr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| min50_kall | 24 | 0.331 | 0.602 | 1 | 24 | 20 | 5 | 3 | 1 | 11 | 4 | 0.214 | 0.2 |
| min50_k12 | 11 | 0.267 | 0.439 | 2 | 11 | 11 | 2 | 1 | 0 | 8 | 1 | 0.111 | 0.0 |
| min30_kall | 67 | 0.146 | 0.644 | 1 | 67 | 34 | 5 | 3 | 1 | 21 | 8 | 0.125 | 0.111 |
| multi_min20_k12 | 11 | 0.29 | 0.557 | 0 | 11 | 9 | 2 | 1 | 1 | 4 | 3 | 0.2 | 0.25 |
| multi_min20 | 112 | 0.069 | 0.708 | 2 | 112 | 34 | 8 | 5 | 3 | 17 | 9 | 0.227 | 0.25 |
| min20_auto | 51 | 0.115 | 0.542 | 0 | 51 | 23 | 4 | 3 | 1 | 13 | 6 | 0.188 | 0.143 |
