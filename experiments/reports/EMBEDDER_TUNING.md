# EMBEDDER_TUNING — clustering knobs on the top embedders (`synth_pol`)

min_topic_size=50, nr_topics=null. **Best:** `Qwen/Qwen3-Embedding-0.6B` [default] nn=15 nc=10 → ARI 0.587, purity 0.839.

| model | variant | umap_nn | umap_nc | k | ari | homogeneity | completeness | purity | outlier_recall | en_ari | en_purity | runtime_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-Embedding-0.6B | default | 15 | 10 | 28 | 0.587 | 0.755 | 0.619 | 0.839 | 0.12 | 0.57 | 0.824 | 9.0 |
| Qwen/Qwen3-Embedding-0.6B | default | 15 | 5 | 28 | 0.568 | 0.746 | 0.614 | 0.829 | 0.163 | 0.553 | 0.816 | 29.7 |
| Qwen/Qwen3-Embedding-0.6B | prompt='',norm=None | 15 | 5 | 28 | 0.568 | 0.746 | 0.614 | 0.829 | 0.163 | 0.553 | 0.816 | 8.9 |
| Qwen/Qwen3-Embedding-0.6B | prompt=None,norm=False | 15 | 5 | 27 | 0.568 | 0.737 | 0.613 | 0.818 | 0.237 | 0.549 | 0.802 | 9.8 |
| Qwen/Qwen3-Embedding-0.6B | default | 30 | 5 | 26 | 0.566 | 0.746 | 0.626 | 0.825 | 0.237 | 0.547 | 0.81 | 11.7 |
| text-embedding-3-small | default | 30 | 5 | 29 | 0.558 | 0.757 | 0.616 | 0.798 | 0.14 | 0.64 | 0.809 | 12.3 |
| text-embedding-3-small | default | 15 | 5 | 32 | 0.551 | 0.779 | 0.621 | 0.805 | 0.017 | 0.646 | 0.815 | 8.7 |
| text-embedding-3-small | default | 50 | 5 | 31 | 0.525 | 0.771 | 0.619 | 0.792 | 0.09 | 0.614 | 0.797 | 19.6 |
| text-embedding-3-small | default | 15 | 10 | 33 | 0.523 | 0.772 | 0.608 | 0.793 | 0.047 | 0.603 | 0.801 | 9.0 |
| text-embedding-3-small | default | 50 | 10 | 30 | 0.522 | 0.765 | 0.62 | 0.78 | 0.077 | 0.602 | 0.792 | 18.2 |
| Qwen/Qwen3-Embedding-0.6B | default | 30 | 10 | 30 | 0.494 | 0.737 | 0.596 | 0.807 | 0.313 | 0.474 | 0.788 | 12.4 |
| text-embedding-3-small | default | 30 | 10 | 33 | 0.488 | 0.751 | 0.595 | 0.775 | 0.15 | 0.566 | 0.784 | 22.0 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 15 | 5 | 27 | 0.425 | 0.653 | 0.559 | 0.722 | 0.113 | 0.407 | 0.702 | 8.7 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 15 | 10 | 29 | 0.424 | 0.663 | 0.56 | 0.728 | 0.11 | 0.406 | 0.71 | 9.1 |
| Qwen/Qwen3-Embedding-0.6B | default | 50 | 5 | 31 | 0.415 | 0.701 | 0.574 | 0.773 | 0.347 | 0.391 | 0.753 | 15.3 |
| paraphrase-multilingual-MiniLM-L12-v2 | prompt=None,norm=False | 15 | 5 | 21 | 0.406 | 0.592 | 0.605 | 0.637 | 0.07 | 0.39 | 0.623 | 8.9 |
| Qwen/Qwen3-Embedding-0.6B | default | 50 | 10 | 26 | 0.397 | 0.648 | 0.603 | 0.686 | 0.22 | 0.387 | 0.677 | 16.0 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 30 | 10 | 26 | 0.396 | 0.661 | 0.543 | 0.74 | 0.153 | 0.381 | 0.725 | 12.5 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 30 | 5 | 31 | 0.355 | 0.66 | 0.528 | 0.746 | 0.19 | 0.338 | 0.728 | 11.3 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 50 | 10 | 28 | 0.34 | 0.635 | 0.528 | 0.711 | 0.257 | 0.317 | 0.695 | 15.3 |
| paraphrase-multilingual-MiniLM-L12-v2 | default | 50 | 5 | 27 | 0.336 | 0.625 | 0.526 | 0.71 | 0.23 | 0.318 | 0.695 | 14.5 |
