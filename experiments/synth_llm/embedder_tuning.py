#!/usr/bin/env python3
"""Clustering-knob grid on the top embedders (Phase 3).

For the bake-off winners, sweep UMAP ``n_neighbors`` / ``n_components`` (now
wired through the pipeline) and, for the winners, the embedder treatment
(instruction prompt on/off, normalisation on/off). Scores on the full and
English-only subsets. Writes ``EMBEDDER_TUNING.md`` / ``embedder_tuning.json``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from embedder_bakeoff import BASE_PKL, CACHE, EMB_CACHE, TRUTH, _slug, compute_embeddings
from eval_synth_llm import score_assignments

HERE = Path(__file__).resolve().parent

MODELS = [
    "Qwen/Qwen3-Embedding-0.6B",
    "text-embedding-3-small",
    "paraphrase-multilingual-MiniLM-L12-v2",
]
NEIGHBORS = [15, 30, 50]
COMPONENTS = [5, 10]

# extra treatment variants: (model, prompt, normalize) with None = family default
VARIANTS = [
    ("Qwen/Qwen3-Embedding-0.6B", "", None),        # no instruction
    ("Qwen/Qwen3-Embedding-0.6B", None, False),     # no normalisation
    ("paraphrase-multilingual-MiniLM-L12-v2", None, False),
]


def _variant_path(model: str, prompt, normalize):
    default = (prompt is None and normalize is None)
    if default:
        return EMB_CACHE / f"{_slug(model)}.npy"
    tag = hashlib.md5(f"{prompt}|{normalize}".encode()).hexdigest()[:8]
    return EMB_CACHE / f"{_slug(model)}__{tag}.npy"


def embeddings_for(model: str, docs, prompt, normalize) -> np.ndarray:
    path = _variant_path(model, prompt, normalize)
    if path.exists():
        return np.load(path)
    if model.startswith("text-embedding"):
        return compute_embeddings(model, docs)  # API: treatment N/A
    from topiclayers.support.embedders import encode_documents, resolve_profile
    prof = resolve_profile(model, prompt or "", normalize)
    emb = encode_documents(model, docs, prof)
    np.save(path, emb)
    return emb


def score(model: str, emb, base, truth, nn, nc, tag) -> dict:
    from topiclayers.core.topic import TopicModeler
    tm = TopicModeler(base, embedder_name=model, path_cache=CACHE,
                      name=f"tune_{_slug(model)}_{tag}")
    docs = tm._preprocess()
    tm.embeddings = emb
    t0 = time.monotonic()
    tm._use_BERTopic(docs, None, 50, 42, None, None, None,
                     exact_probabilities=False, repro_mode="strict",
                     umap_init="pca", umap_n_neighbors=nn, umap_n_components=nc)
    assign = tm.df[["id", "topic"]].rename(columns={"topic": "pred"})
    full = score_assignments(assign, truth)
    en = score_assignments(assign, truth[truth["lang"] == "en"])
    return {
        "model": model, "variant": tag, "umap_nn": nn, "umap_nc": nc,
        "k": full["k"], "ari": full["metrics"]["ari"], "purity": full["metrics"]["purity"],
        "homogeneity": full["metrics"]["homogeneity"], "completeness": full["metrics"]["completeness"],
        "outlier_recall": full["outlier"]["true_outlier_reject_recall"],
        "en_ari": en["metrics"]["ari"], "en_purity": en["metrics"]["purity"],
        "runtime_s": round(time.monotonic() - t0, 1),
    }


def main() -> None:
    truth = pd.read_csv(TRUTH)
    base = pd.read_pickle(BASE_PKL).set_index("id")[["text"]]
    from topiclayers.core.topic import TopicModeler
    docs = TopicModeler(base, embedder_name="all-MiniLM-L6-v2", path_cache=CACHE,
                        name="probe")._preprocess()

    rows = []
    for model in MODELS:
        emb = embeddings_for(model, docs, None, None)
        for nn in NEIGHBORS:
            for nc in COMPONENTS:
                r = score(model, emb, base, truth, nn, nc, "default")
                print(f"{model} nn={nn} nc={nc}: ari={r['ari']} purity={r['purity']}", flush=True)
                rows.append(r)
                (HERE / "embedder_tuning.json").write_text(json.dumps(rows, indent=2, default=str))

    for model, prompt, norm in VARIANTS:
        tag = f"prompt={prompt!r},norm={norm}"
        emb = embeddings_for(model, docs, prompt, norm)
        r = score(model, emb, base, truth, 15, 5, tag)
        print(f"{model} [{tag}]: ari={r['ari']} purity={r['purity']}", flush=True)
        rows.append(r)
        (HERE / "embedder_tuning.json").write_text(json.dumps(rows, indent=2, default=str))

    rows.sort(key=lambda r: r["ari"], reverse=True)
    cols = ["model", "variant", "umap_nn", "umap_nc", "k", "ari", "homogeneity",
            "completeness", "purity", "outlier_recall", "en_ari", "en_purity", "runtime_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    best = rows[0]
    md = ["# EMBEDDER_TUNING — clustering knobs on the top embedders (`synth_pol`)", "",
          f"min_topic_size=50, nr_topics=null. **Best:** `{best['model']}` "
          f"[{best['variant']}] nn={best['umap_nn']} nc={best['umap_nc']} → "
          f"ARI {best['ari']}, purity {best['purity']}.", "",
          "\n".join(lines), ""]
    (HERE.parent / "reports" / "EMBEDDER_TUNING.md").write_text("\n".join(md))
    print(f"\n-> {HERE.parent / 'reports' / 'EMBEDDER_TUNING.md'}")


if __name__ == "__main__":
    main()
