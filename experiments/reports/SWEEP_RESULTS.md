# SWEEP_RESULTS — parameter sweep on `synth_pol` (realistic corpus)

18 configurations, ranked by ARI. Originals: 7011; ground truth: 10 topics + bridges + off-topic.

**Best by ARI:** `min50_kall` — ARI 0.331, purity 0.602, k=24, tricky pairs merged 1/4.

| id | embedder | min_topic_size | nr_topics | umap_nn | umap_nc | umap_init | k | ari | homogeneity | completeness | v_measure | purity | best_mapping_accuracy | outlier_recall | false_outlier_rate | tricky_merged | runtime_s | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| min50_kall | all-MiniLM-L6-v2 | 50 | None | nan | nan | pca | 24 | 0.331 | 0.546 | 0.469 | 0.504 | 0.602 | 0.458 | 0.17 | 0.08 | 1 | 8.2 | ok |
| multi_min20_k12 | paraphrase-multilingual-MiniLM-L12-v2 | 20 | 12 | nan | nan | pca | 11 | 0.29 | 0.47 | 0.528 | 0.497 | 0.557 | 0.546 | 0.413 | 0.242 | 0 | 8.7 | ok |
| min50_k12 | all-MiniLM-L6-v2 | 50 | 12 | nan | nan | pca | 11 | 0.267 | 0.404 | 0.441 | 0.422 | 0.439 | 0.411 | 0.17 | 0.08 | 2 | 8.2 | ok |
| min30_k12 | all-MiniLM-L6-v2 | 30 | 12 | nan | nan | pca | 11 | 0.212 | 0.389 | 0.421 | 0.405 | 0.444 | 0.427 | 0.24 | 0.161 | 0 | 8.2 | ok |
| min20_k12 | all-MiniLM-L6-v2 | 20 | 12 | nan | nan | pca | 11 | 0.173 | 0.356 | 0.397 | 0.375 | 0.436 | 0.418 | 0.443 | 0.243 | 0 | 8.2 | ok |
| min20_nn5 | all-MiniLM-L6-v2 | 20 | None | 5.0 | nan | pca | 119 | 0.167 | 0.697 | 0.398 | 0.507 | 0.755 | 0.222 | 0.273 | 0.127 | 1 | 5.7 | ok |
| min10_k12 | all-MiniLM-L6-v2 | 10 | 12 | nan | nan | pca | 11 | 0.164 | 0.297 | 0.365 | 0.327 | 0.35 | 0.341 | 0.323 | 0.182 | 1 | 8.4 | ok |
| min20_nc2 | all-MiniLM-L6-v2 | 20 | None | nan | 2.0 | pca | 113 | 0.147 | 0.681 | 0.402 | 0.505 | 0.735 | 0.215 | 0.343 | 0.166 | 1 | 7.7 | ok |
| min30_kall | all-MiniLM-L6-v2 | 30 | None | nan | nan | pca | 67 | 0.146 | 0.576 | 0.394 | 0.468 | 0.644 | 0.264 | 0.24 | 0.161 | 1 | 8.3 | ok |
| min20_auto | all-MiniLM-L6-v2 | 20 | auto | nan | nan | pca | 51 | 0.115 | 0.467 | 0.366 | 0.411 | 0.542 | 0.301 | 0.443 | 0.243 | 0 | 8.4 | ok |
| min20_nc10 | all-MiniLM-L6-v2 | 20 | None | nan | 10.0 | pca | 111 | 0.073 | 0.646 | 0.388 | 0.485 | 0.702 | 0.174 | 0.247 | 0.214 | 3 | 8.9 | ok |
| multi_min20 | paraphrase-multilingual-MiniLM-L12-v2 | 20 | None | nan | nan | pca | 112 | 0.069 | 0.643 | 0.396 | 0.49 | 0.708 | 0.181 | 0.413 | 0.242 | 2 | 8.6 | ok |
| min20_nn10 | all-MiniLM-L6-v2 | 20 | None | 10.0 | nan | pca | 116 | 0.065 | 0.639 | 0.38 | 0.477 | 0.702 | 0.149 | 0.453 | 0.206 | 3 | 7.4 | ok |
| min20_spectral | all-MiniLM-L6-v2 | 20 | None | nan | nan | spectral | 113 | 0.065 | 0.646 | 0.391 | 0.487 | 0.708 | 0.156 | 0.38 | 0.23 | 4 | 8.5 | ok |
| min20_nn50 | all-MiniLM-L6-v2 | 20 | None | 50.0 | nan | pca | 81 | 0.054 | 0.504 | 0.35 | 0.413 | 0.579 | 0.179 | 0.317 | 0.248 | 2 | 13.9 | ok |
| min20_nn30 | all-MiniLM-L6-v2 | 20 | None | 30.0 | nan | pca | 105 | 0.052 | 0.596 | 0.376 | 0.461 | 0.654 | 0.158 | 0.453 | 0.258 | 4 | 11.0 | ok |
| min20_kall | all-MiniLM-L6-v2 | 20 | None | nan | nan | pca | 121 | 0.051 | 0.627 | 0.378 | 0.472 | 0.689 | 0.144 | 0.443 | 0.243 | 3 | 8.2 | ok |
| min10_kall | all-MiniLM-L6-v2 | 10 | None | nan | nan | pca | 238 | 0.044 | 0.709 | 0.363 | 0.48 | 0.755 | 0.117 | 0.323 | 0.182 | 3 | 27.2 | ok |
