#!/usr/bin/env python3
"""Score promising configs on the *end goal*: does netpol recover the designed
polarization through the pipeline's own topics?

For each config it fits the topic model (reusing ``sweep_params`` helpers),
builds per-predicted-topic retweet layers from the ground-truth interaction
graph, runs netpol, and compares the verdict with the designed ``polarized``
flag. Also reports the clustering metrics for context.

Writes ``SWEEP_POLARIZATION.md`` and ``sweep_polarization.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep_params as sp  # noqa: E402
from eval_synth_llm import score_assignments  # noqa: E402

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO.parent / "netpol"))
import netpol  # noqa: E402

# the promising configs from SWEEP_RESULTS.md (+ a baseline)
CONFIGS = [
    dict(id="min50_kall", mts=50, nt=None),
    dict(id="min50_k12", mts=50, nt=12),
    dict(id="min30_kall", mts=30, nt=None),
    dict(id="multi_min20_k12", mts=20, nt=12, embedder=sp.MULTI),
    dict(id="multi_min20", mts=20, nt=None, embedder=sp.MULTI),
    dict(id="min20_auto", mts=20, nt="auto"),
]


def _bool(x) -> bool:
    return str(x).strip().lower() == "true"


def load_kinds() -> dict:
    """post_id -> referenced type ('retweeted'/'quoted'/'replied_to') from the jsonl."""
    import json
    kinds = {}
    for line in open(sp.HERE / "synth_pol.jsonl"):
        o = json.loads(line)
        ref = o.get("referenced_tweets")
        if ref:
            kinds[o["id"]] = ref[0]["type"]
    return kinds


def build_predicted_layers(assign: pd.DataFrame, truth: pd.DataFrame, kinds: dict) -> dict:
    """Mirror the pipeline: only *reposts* make edges, and the node set is the
    pipeline's df_labeled (originals + reposts) — quotes/replies are dropped."""
    pred = dict(zip(assign["id"], assign["pred"]))
    t = truth.copy()
    t["pred"] = t["post_id"].map(pred)
    target_pred = t.set_index("post_id")["pred"]
    mask = (t["is_derived"] == True) & (t["pred"].isna())  # noqa: E712
    t.loc[mask, "pred"] = t.loc[mask, "target_id"].map(target_pred)
    t = t[t["pred"].notna()]
    t["kind"] = t["post_id"].map(kinds)
    t = t[(t["is_derived"] == False) | (t["kind"] == "retweeted")]  # noqa: E712

    author = dict(zip(t["post_id"], t["author_id"]))
    target_author = t["target_id"].map(author)
    layers = {}
    for topic, sub in t.groupby("pred"):
        if int(topic) == -1:
            continue
        G = nx.DiGraph()
        G.add_nodes_from(sub["author_id"].unique())
        rts = sub[sub["kind"] == "retweeted"]
        edges = (pd.DataFrame({"a": rts["author_id"], "b": target_author.loc[rts.index]})
                 .dropna().groupby(["a", "b"]).size().reset_index(name="w"))
        G.add_weighted_edges_from((a, b, int(w)) for a, b, w in edges.itertuples(index=False, name=None))
        layers[str(int(topic))] = G
    return layers


def designed_by_layer(assign: pd.DataFrame, truth: pd.DataFrame, topics: pd.DataFrame) -> dict:
    pol = {r["topic"]: _bool(r["polarized"]) for _, r in topics.iterrows()}
    m = assign.merge(truth, left_on="id", right_on="post_id")
    m = m[m["is_derived"] == False]  # noqa: E712
    out = {}
    for layer, sub in m.groupby("pred"):
        vt = sub["true_topic"].value_counts()
        out[int(layer)] = {"true_topic": vt.index[0], "designed": pol.get(vt.index[0]),
                           "purity": round(vt.iloc[0] / len(sub), 3)}
    return out


def main() -> None:
    truth = pd.read_csv(sp.TRUTH)
    topics = pd.read_csv(HERE / "topics_synth_pol.csv")
    base = pd.read_pickle(sp.BASE_PKL).set_index("id")[["text"]]

    from topiclayers.core.topic import TopicModeler
    docs = TopicModeler(base, embedder_name=sp.EMBEDDER, path_cache=sp.CACHE, name="probe")._preprocess()
    embeddings = {}
    for e in {c.get("embedder", sp.EMBEDDER) for c in CONFIGS}:
        print(f"embedding {e} …", flush=True)
        embeddings[e] = sp.embed_docs(docs, e)

    config = netpol.PolarizationConfig(
        influencer_strategy="in_degree", n_influencers=50, ideology_dimensions=1,
        significance_level=0.05, fdr_correction=True, min_edges=20, exclude_layers=(-1,))
    kinds = load_kinds()

    rows = []
    for cfg in CONFIGS:
        cached = HERE / "sweep_assignments" / f"{cfg['id']}.csv"
        if cached.exists():
            assign = pd.read_csv(cached)
        else:
            e = cfg.get("embedder", sp.EMBEDDER)
            tm = TopicModeler(base, embedder_name=e, path_cache=sp.CACHE, name=f"polsweep_{cfg['id']}")
            d = tm._preprocess()
            tm.embeddings = embeddings[e]
            tm._use_BERTopic(d, cfg["nt"], cfg["mts"], 42, sp.build_umap(cfg), None, None,
                             exact_probabilities=False, repro_mode="strict",
                             umap_init=cfg.get("init", "pca"))
            assign = tm.df[["id", "topic"]].rename(columns={"topic": "pred"})
            assign.to_csv(cached, index=False)

        clus = score_assignments(assign, truth, cfg["id"])
        layers = build_predicted_layers(assign, truth, kinds)
        design = designed_by_layer(assign, truth, topics)
        results = netpol.analyze_layers(layers, config)

        tp = fp = fn = tn = 0
        for lid, r in results.items():
            info = design.get(int(float(lid)))
            if info is None or not r.was_analyzed or info["designed"] is None:
                continue
            det = bool(r.is_polarized)
            if det and info["designed"]:
                tp += 1
            elif det and not info["designed"]:
                fp += 1
            elif not det and info["designed"]:
                fn += 1
            else:
                tn += 1
        rows.append({
            "id": cfg["id"], "k": clus["k"], "ari": clus["metrics"]["ari"],
            "purity": clus["metrics"]["purity"], "tricky_merged": sum(1 for t in clus["tricky"] if t["merged"]),
            "layers": len(results), "analyzed": int(sum(1 for r in results.values() if r.was_analyzed)),
            "polarized_detected": int(sum(1 for r in results.values() if r.is_polarized)),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": round(tp / (tp + fn), 3) if (tp + fn) else None,
            "fpr": round(fp / (fp + tn), 3) if (fp + tn) else None,
        })
        print(f"{cfg['id']}: k={clus['k']} ari={clus['metrics']['ari']} "
              f"netpol TP={tp} FP={fp} FN={fn} TN={tn}", flush=True)
        (HERE / "sweep_polarization.json").write_text(json.dumps(rows, indent=2, default=str))

    cols = list(rows[0].keys())
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md = ["# SWEEP_POLARIZATION — netpol recovery by config (`synth_pol`)", "",
          "Layers built from the pipeline's predicted topics (retweets inherit the "
          "target's topic). `recall` = designed-polarized layers netpol flags; "
          "`fpr` = consensual layers it wrongly flags.", "",
          "\n".join(lines), ""]
    (HERE.parent / "reports" / "SWEEP_POLARIZATION.md").write_text("\n".join(md))
    print(f"-> {HERE.parent / 'reports' / 'SWEEP_POLARIZATION.md'}")


if __name__ == "__main__":
    (HERE / "sweep_assignments").mkdir(exist_ok=True)
    main()
