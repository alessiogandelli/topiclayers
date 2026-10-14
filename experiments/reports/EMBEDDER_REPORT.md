# EMBEDDER_REPORT — bake-off on `synth_pol` (realistic corpus)

16 successful (model, config) runs. Ranked by full-corpus ARI. `en_*` columns are the English-only subset (to separate model quality from multilingual handling).

| model | config | dim | k | ari | homogeneity | completeness | purity | best_mapping_accuracy | outlier_recall | tricky_merged | en_k | en_ari | en_purity | embed_s | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | min50_null | 1024 | 28 | 0.568 | 0.746 | 0.614 | 0.829 | 0.592 | 0.163 | 0 | 28 | 0.553 | 0.816 | 219.0 | 8.9 |
| text-embedding-3-small | min50_null | 1536 | 32 | 0.551 | 0.779 | 0.621 | 0.805 | 0.583 | 0.017 | 1 | 23 | 0.646 | 0.815 | 19.0 | 9.0 |
| paraphrase-multilingual-MiniLM-L12-v2 | min50_null | 384 | 27 | 0.425 | 0.653 | 0.559 | 0.722 | 0.511 | 0.113 | 1 | 27 | 0.407 | 0.702 | 12.0 | 8.4 |
| paraphrase-multilingual-mpnet-base-v2 | min50_null | 768 | 24 | 0.38 | 0.616 | 0.584 | 0.669 | 0.464 | 0.22 | 1 | 24 | 0.361 | 0.654 | 55.2 | 10.0 |
| text-embedding-3-small | min20_k12 | 1536 | 11 | 0.355 | 0.514 | 0.572 | 0.539 | 0.503 | 0.32 | 1 | 9 | 0.36 | 0.552 | 0.1 | 8.9 |
| all-mpnet-base-v2 | min50_null | 768 | 20 | 0.325 | 0.519 | 0.489 | 0.543 | 0.434 | 0.1 | 2 | 18 | 0.433 | 0.641 | 40.2 | 8.5 |
| all-MiniLM-L6-v2 | min50_null | 384 | 27 | 0.313 | 0.536 | 0.456 | 0.588 | 0.422 | 0.18 | 2 | 26 | 0.441 | 0.696 | 0.0 | 30.3 |
| Qwen/Qwen3-Embedding-0.6B | min20_k12 | 1024 | 11 | 0.308 | 0.474 | 0.557 | 0.509 | 0.489 | 0.263 | 1 | 11 | 0.287 | 0.494 | 0.0 | 8.8 |
| intfloat/multilingual-e5-small | min50_null | 384 | 37 | 0.301 | 0.64 | 0.506 | 0.681 | 0.42 | 0.213 | 1 | 29 | 0.414 | 0.726 | 0.0 | 8.1 |
| intfloat/multilingual-e5-base | min50_null | 768 | 43 | 0.298 | 0.683 | 0.515 | 0.728 | 0.408 | 0.09 | 2 | 32 | 0.396 | 0.76 | 50.6 | 8.6 |
| paraphrase-multilingual-MiniLM-L12-v2 | min20_k12 | 384 | 11 | 0.275 | 0.456 | 0.508 | 0.515 | 0.484 | 0.34 | 0 | 11 | 0.271 | 0.515 | 0.0 | 8.4 |
| paraphrase-multilingual-mpnet-base-v2 | min20_k12 | 768 | 11 | 0.246 | 0.431 | 0.515 | 0.47 | 0.422 | 0.373 | 1 | 11 | 0.226 | 0.46 | 0.0 | 9.6 |
| all-mpnet-base-v2 | min20_k12 | 768 | 11 | 0.207 | 0.351 | 0.397 | 0.387 | 0.366 | 0.307 | 1 | 10 | 0.28 | 0.445 | 0.0 | 8.5 |
| intfloat/multilingual-e5-base | min20_k12 | 768 | 11 | 0.205 | 0.324 | 0.394 | 0.346 | 0.334 | 0.07 | 3 | 10 | 0.247 | 0.381 | 0.0 | 8.7 |
| all-MiniLM-L6-v2 | min20_k12 | 384 | 11 | 0.193 | 0.369 | 0.418 | 0.432 | 0.406 | 0.367 | 1 | 11 | 0.282 | 0.498 | 0.0 | 9.2 |
| intfloat/multilingual-e5-small | min20_k12 | 384 | 11 | 0.052 | 0.149 | 0.231 | 0.253 | 0.241 | 0.217 | 3 | 9 | 0.061 | 0.274 | 0.0 | 8.2 |
