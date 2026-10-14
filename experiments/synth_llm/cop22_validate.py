#!/usr/bin/env python3
"""Real-data validation gate: run candidate embedders on a real COP22 slice.

There is no ground truth here, so we judge on what matters for the paper:
topic count, top c-TF-IDF terms, LLM labels (interpretability), and the
**language composition per topic** (the English MiniLM default is known to merge
non-English COP22 content into a junk "French words" topic).

Writes ``COP22_VALIDATION.md``.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SLICE = REPO / "tmp/cop22_val/cop22_slice.jsonl"
CACHE = REPO / "tmp/cop22_val/cache"

MODELS = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "Qwen/Qwen3-Embedding-0.6B",
]
MIN_TOPIC_SIZE = 30


def load_originals():
    from topiclayers.adapters.registry import resolve_adapter
    from topiclayers.core.data import Dataset
    import topiclayers.adapters.csv  # noqa: F401
    import topiclayers.adapters.twitter  # noqa: F401

    fmt, loader = resolve_adapter(SLICE)
    posts = loader(SLICE)
    ds = Dataset(posts, n_cop="cop22val")
    ds.process(cache_dir=CACHE, adapter_name=fmt)
    return ds.df_original


def run(model: str, df: pd.DataFrame, label: bool) -> dict:
    from topiclayers.core.topic import TopicModeler
    tm = TopicModeler(df, embedder_name=model, path_cache=CACHE, name=f"cop22_{model.split('/')[-1]}")
    df_lab = tm.get_topics(min_topic_size=MIN_TOPIC_SIZE, nr_topics=None, random_state=42,
                           repro_mode="strict", umap_init="pca",
                           umap_n_neighbors=15, umap_n_components=10)
    if label:
        tm.label_topics(model=os.getenv("LABEL_MODEL", "gpt-6-luna"),
                        label_context="posts about the UN climate conference (COP) and climate politics")
    info = tm.model.get_topic_info()
    label_col = "LLM_label" if "LLM_label" in info.columns else "Name"

    # language composition per topic
    lang_by_topic = {}
    for topic, sub in df_lab.groupby("topic"):
        if topic == -1:
            continue
        langs = Counter(sub["lang"].fillna("?"))
        lang_by_topic[int(topic)] = ", ".join(f"{k}:{v}" for k, v in langs.most_common(3))

    topics = []
    for _, row in info.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        terms = ", ".join(list(row["Representation"])[:8])
        topics.append({
            "topic": tid, "n": int(row["Count"]),
            "label": str(row.get(label_col, ""))[:60], "terms": terms,
            "langs": lang_by_topic.get(tid, ""),
        })
    topics.sort(key=lambda t: t["n"], reverse=True)
    return {"model": model, "n_topics": len(topics),
            "outliers": int((df_lab["topic"] == -1).sum()), "topics": topics}


def main() -> None:
    df = load_originals()
    print(f"originals: {len(df)}; langs: {Counter(df['lang'].fillna('?')).most_common(6)}")
    results = []
    for model in MODELS:
        print(f"== {model} ==", flush=True)
        try:
            r = run(model, df, label=True)
            print(f"  {r['n_topics']} topics, {r['outliers']} outliers", flush=True)
            results.append(r)
        except Exception as e:
            print(f"  FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)

    md = ["# COP22_VALIDATION — real-data gate for embedder choice", "",
          f"Slice: first 6,000 lines of `data/cop22.json`; originals clustered; "
          f"min_topic_size={MIN_TOPIC_SIZE}, nr_topics=null, umap 15/10. "
          "Languages: " + ", ".join(f"{k} {v}" for k, v in Counter(df["lang"].fillna("?")).most_common(6)) + ".", ""]
    for r in results:
        md += [f"## {r['model']}", "",
               f"**{r['n_topics']} topics**, {r['outliers']} outliers.", "",
               "| topic | n | lang mix | label | top c-TF-IDF terms |",
               "|---|---|---|---|---|"]
        for t in r["topics"]:
            md.append(f"| {t['topic']} | {t['n']} | {t['langs']} | {t['label']} | {t['terms']} |")
        md.append("")
    (HERE.parent / "reports" / "COP22_VALIDATION.md").write_text("\n".join(md))
    print(f"-> {HERE.parent / 'reports' / 'COP22_VALIDATION.md'}")


if __name__ == "__main__":
    main()
