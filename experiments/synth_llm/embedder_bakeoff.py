#!/usr/bin/env python3
"""Embedder bake-off on the realistic `synth_pol` corpus.

For each candidate model: compute embeddings once (with the model-specific
prefix/normalization from ``topiclayers.support.embedders``), cluster with two
fixed BERTopic configs via the pipeline's own ``TopicModeler``, and score
against the ground-truth topics — on the **full corpus and the English-only
subset** — so we can tell "better model" apart from "handles fr/es/de".

Writes ``EMBEDDER_REPORT.md`` and ``embedder_results.json``; embeddings are
cached under ``<cache>/bakeoff_embeddings/<slug>.npy`` for reuse.

Usage:
    .venv/bin/python experiments/synth_llm/embedder_bakeoff.py
    .venv/bin/python experiments/synth_llm/embedder_bakeoff.py --models all-MiniLM-L6-v2 intfloat/multilingual-e5-base
"""
from __future__ import annotations

import argparse
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
EMB_CACHE = CACHE / "bakeoff_embeddings"

# sweep order agreed with the user
MODELS = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "paraphrase-multilingual-mpnet-base-v2",
    "Qwen/Qwen3-Embedding-0.6B",
    "all-mpnet-base-v2",
    "text-embedding-3-small",
]

CONFIGS = [
    {"id": "min50_null", "mts": 50, "nt": None},
    {"id": "min20_k12", "mts": 20, "nt": 12},
]


def _slug(model: str) -> str:
    return model.replace("/", "__")


def compute_embeddings(model: str, docs: list[str], device: str = "") -> np.ndarray:
    """Embeddings with the correct input treatment; cached on disk."""
    EMB_CACHE.mkdir(parents=True, exist_ok=True)
    path = EMB_CACHE / f"{_slug(model)}.npy"
    if path.exists():
        return np.load(path)

    if model.startswith("text-embedding"):
        import os
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(REPO / ".env")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        out = []
        for i in range(0, len(docs), 1000):
            resp = client.embeddings.create(input=docs[i:i + 1000], model=model)
            out.extend(np.array(d.embedding) for d in resp.data)
        emb = np.vstack(out)
    else:
        from topiclayers.support.embedders import encode_documents, resolve_profile
        profile = resolve_profile(model)
        emb = encode_documents(model, docs, profile, device=device)

    np.save(path, emb)
    return emb


def fit_and_score(model: str, emb: np.ndarray, base: pd.DataFrame, truth: pd.DataFrame,
                  cfg: dict) -> dict:
    from topiclayers.core.topic import TopicModeler

    tm = TopicModeler(base, embedder_name=model, path_cache=CACHE,
                      name=f"bakeoff_{_slug(model)}_{cfg['id']}")
    docs = tm._preprocess()
    tm.embeddings = emb
    t0 = time.monotonic()
    tm._use_BERTopic(docs, cfg["nt"], cfg["mts"], 42, None, None, None,
                     exact_probabilities=False, repro_mode="strict",
                     umap_init="pca", umap_n_neighbors=15, umap_n_components=5)
    assign = tm.df[["id", "topic"]].rename(columns={"topic": "pred"})
    runtime = round(time.monotonic() - t0, 1)

    full = score_assignments(assign, truth, f"{model}::{cfg['id']}")
    en_truth = truth[truth["lang"] == "en"]
    en = score_assignments(assign, en_truth, f"{model}::{cfg['id']}::en")
    return {
        "k": full["k"],
        "ari": full["metrics"]["ari"],
        "homogeneity": full["metrics"]["homogeneity"],
        "completeness": full["metrics"]["completeness"],
        "purity": full["metrics"]["purity"],
        "best_mapping_accuracy": full["metrics"]["best_mapping_accuracy"],
        "outlier_recall": full["outlier"]["true_outlier_reject_recall"],
        "tricky_merged": sum(1 for t in full["tricky"] if t["merged"]),
        "en_k": en["k"],
        "en_ari": en["metrics"]["ari"],
        "en_purity": en["metrics"]["purity"],
        "en_outlier_recall": en["outlier"]["true_outlier_reject_recall"],
        "runtime_s": runtime,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    truth = pd.read_csv(TRUTH)
    base = pd.read_pickle(BASE_PKL).set_index("id")[["text"]]

    from topiclayers.core.topic import TopicModeler
    docs = TopicModeler(base, embedder_name="all-MiniLM-L6-v2", path_cache=CACHE,
                        name="probe")._preprocess()

    rows = []
    for model in args.models:
        for cfg in CONFIGS:
            rid = f"{model}::{cfg['id']}"
            print(f"[{rid}] …", flush=True)
            try:
                t0 = time.monotonic()
                emb = compute_embeddings(model, docs, args.device)
                embed_s = round(time.monotonic() - t0, 1)
                r = fit_and_score(model, emb, base, truth, cfg)
                r.update({"model": model, "config": cfg["id"], "dim": int(emb.shape[1]),
                          "embed_s": embed_s, "status": "ok"})
                print(f"    dim={r['dim']} k={r['k']} ari={r['ari']} purity={r['purity']} "
                      f"en_ari={r['en_ari']} ({r['runtime_s']}s fit)", flush=True)
            except Exception as e:
                r = {"model": model, "config": cfg["id"], "status": f"failed: {type(e).__name__}: {str(e)[:120]}"}
                print(f"    FAILED: {r['status']}", flush=True)
            rows.append(r)
            (HERE / "embedder_results.json").write_text(json.dumps(rows, indent=2, default=str))

    ok = [r for r in rows if r.get("status") == "ok"]
    ok.sort(key=lambda r: r["ari"], reverse=True)
    cols = ["model", "config", "dim", "k", "ari", "homogeneity", "completeness",
            "purity", "best_mapping_accuracy", "outlier_recall", "tricky_merged",
            "en_k", "en_ari", "en_purity", "embed_s", "runtime_s"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in ok:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")

    failed = [r for r in rows if r.get("status") != "ok"]
    md = ["# EMBEDDER_REPORT — bake-off on `synth_pol` (realistic corpus)", "",
          f"{len(ok)} successful (model, config) runs. Ranked by full-corpus ARI. "
          "`en_*` columns are the English-only subset (to separate model quality "
          "from multilingual handling).", "",
          "\n".join(lines), ""]
    if failed:
        md += ["## Failed", ""] + [f"- `{r['model']}` / {r['config']}: {r['status']}" for r in failed] + [""]
    (HERE.parent / "reports" / "EMBEDDER_REPORT.md").write_text("\n".join(md))
    print(f"\n-> {HERE.parent / 'reports' / 'EMBEDDER_REPORT.md'}")


if __name__ == "__main__":
    main()
