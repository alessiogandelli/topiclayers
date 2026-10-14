#!/usr/bin/env python3
"""Parameter sweep for topiclayers' topic model on the realistic synth_pol corpus.

Runs many BERTopic configurations through the same code path the pipeline uses
(``TopicModeler``), scores each against the ground-truth topics with
``eval_synth_llm.score_assignments``, and writes a ranked report.

Embeddings are computed once per embedder and reused across configs, so only the
clustering (UMAP + HDBSCAN + c-TF-IDF) is re-run per config.

Usage:
    .venv/bin/python experiments/synth_llm/sweep_params.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from eval_synth_llm import score_assignments

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CACHE = REPO / "experiments/cache/synth_pol"
BASE_PKL = CACHE / "tm/all-MiniLM-L6-v2/tweets_synth_pol_labeled.pkl"
TRUTH = HERE / "truth_synth_pol.csv"
MULTI = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------- config grid
CONFIGS: list[dict] = []
for mts in (10, 20, 30, 50):
    for nt in (None, 12):
        CONFIGS.append(dict(id=f"min{mts}_k{nt or 'all'}", mts=mts, nt=nt))
CONFIGS.append(dict(id="min20_auto", mts=20, nt="auto"))
for nn in (5, 10, 30, 50):
    CONFIGS.append(dict(id=f"min20_nn{nn}", mts=20, nt=None, nn=nn))
for nc in (2, 10):
    CONFIGS.append(dict(id=f"min20_nc{nc}", mts=20, nt=None, nc=nc))
CONFIGS.append(dict(id="min20_spectral", mts=20, nt=None, init="spectral"))
CONFIGS.append(dict(id="multi_min20", mts=20, nt=None, embedder=MULTI))
CONFIGS.append(dict(id="multi_min20_k12", mts=20, nt=12, embedder=MULTI))

EMBEDDER = "all-MiniLM-L6-v2"


def embed_docs(docs: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    out = []
    for i in range(0, len(docs), 256):
        out.append(model.encode(docs[i:i + 256], show_progress_bar=False))
    return np.vstack(out)


def build_umap(cfg: dict):
    if cfg.get("nn") is None and cfg.get("nc") is None:
        return None  # let TopicModeler use its defaults
    from umap import UMAP
    return UMAP(
        n_components=cfg.get("nc", 5),
        n_neighbors=cfg.get("nn", 15),
        min_dist=0.0,
        metric="cosine",
        init=cfg.get("init", "pca"),
        random_state=42,
    )


def run_config(cfg: dict, base: pd.DataFrame, embeddings: dict, truth: pd.DataFrame) -> dict:
    from topiclayers.core.topic import TopicModeler

    embedder = cfg.get("embedder", EMBEDDER)
    tm = TopicModeler(base, embedder_name=embedder, path_cache=CACHE,
                      name=f"sweep_{cfg['id']}")
    docs = tm._preprocess()
    tm.embeddings = embeddings[embedder]

    t0 = time.monotonic()
    status = "ok"
    try:
        tm._use_BERTopic(
            docs, cfg["nt"], cfg["mts"], 42,
            build_umap(cfg), None, None,
            exact_probabilities=False, repro_mode="strict",
            umap_init=cfg.get("init", "pca"),
        )
        assign = tm.df[["id", "topic"]].rename(columns={"topic": "pred"})
        res = score_assignments(assign, truth)
    except Exception as e:  # all-outlier / fit failure
        status = f"failed: {type(e).__name__}"
        res = {"k": 0, "metrics": {k: 0 for k in
               ("ari", "homogeneity", "completeness", "v_measure", "purity",
                "best_mapping_accuracy")},
               "outlier": {"true_outlier_reject_recall": 0, "false_outlier_rate": 0},
               "tricky": []}

    merged = sum(1 for t in res.get("tricky", []) if t["merged"])
    return {
        "id": cfg["id"], "embedder": embedder, "min_topic_size": cfg["mts"],
        "nr_topics": cfg["nt"], "umap_nn": cfg.get("nn"), "umap_nc": cfg.get("nc"),
        "umap_init": cfg.get("init", "pca"), "status": status,
        "runtime_s": round(time.monotonic() - t0, 1),
        "k": res["k"], **res["metrics"],
        "outlier_recall": res["outlier"]["true_outlier_reject_recall"],
        "false_outlier_rate": res["outlier"]["false_outlier_rate"],
        "tricky_merged": merged,
    }


def main() -> None:
    truth = pd.read_csv(TRUTH)
    base = pd.read_pickle(BASE_PKL).set_index("id")[["text"]]
    print(f"base originals: {len(base)}")

    # preprocess once to get the docs; embeddings computed once per embedder
    from topiclayers.core.topic import TopicModeler
    probe = TopicModeler(base, embedder_name=EMBEDDER, path_cache=CACHE, name="probe")
    docs = probe._preprocess()

    embeddings: dict[str, np.ndarray] = {}
    for embedder in {c.get("embedder", EMBEDDER) for c in CONFIGS}:
        print(f"embedding with {embedder} …", flush=True)
        embeddings[embedder] = embed_docs(docs, embedder)

    rows = []
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"[{i}/{len(CONFIGS)}] {cfg['id']} …", flush=True)
        r = run_config(cfg, base, embeddings, truth)
        print(f"    k={r['k']} ari={r['ari']} purity={r['purity']} "
              f"tricky_merged={r['tricky_merged']} ({r['runtime_s']}s) {r['status']}", flush=True)
        rows.append(r)
        (HERE / "sweep_results.json").write_text(json.dumps(rows, indent=2, default=str))

    df = pd.DataFrame(rows).sort_values("ari", ascending=False)
    cols = ["id", "embedder", "min_topic_size", "nr_topics", "umap_nn", "umap_nc",
            "umap_init", "k", "ari", "homogeneity", "completeness", "v_measure",
            "purity", "best_mapping_accuracy", "outlier_recall", "false_outlier_rate",
            "tricky_merged", "runtime_s", "status"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    best = df.iloc[0]
    md = ["# SWEEP_RESULTS — parameter sweep on `synth_pol` (realistic corpus)", "",
          f"{len(df)} configurations, ranked by ARI. Originals: {len(base)}; "
          "ground truth: 10 topics + bridges + off-topic.", "",
          f"**Best by ARI:** `{best['id']}` — ARI {best['ari']}, purity {best['purity']}, "
          f"k={best['k']}, tricky pairs merged {best['tricky_merged']}/4.", "",
          "\n".join(lines), ""]
    (HERE.parent / "reports" / "SWEEP_RESULTS.md").write_text("\n".join(md))
    print(f"\n-> {HERE.parent / 'reports' / 'SWEEP_RESULTS.md'}")
    print(df[["id", "k", "ari", "purity", "tricky_merged", "outlier_recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
