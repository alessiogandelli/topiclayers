#!/usr/bin/env python3
"""Polarization on *ground-truth* topics: netpol's ceiling for synth_pol.

The pipeline's own topics are noisy (over/under-segmented), which caps netpol.
This script builds the per-topic multilayer retweet network directly from the
ground-truth ``true_topic`` labels — so each layer is exactly one designed topic
— runs netpol, and compares the verdict with the designed ``polarized`` flag.
The gap between this and the pipeline run is the cost of topic-model error.

Writes ``experiments/reports/ORACLE_POLARIZATION_REPORT.md``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import uunet.multinet as ml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "netpol"))
import netpol  # noqa: E402


def build_layers(truth: pd.DataFrame) -> dict:
    authors = dict(zip(truth["post_id"], truth["author_id"]))
    target_author = truth["target_id"].map(authors)
    layers = {}
    for topic, sub in truth.groupby("true_topic"):
        G = nx.DiGraph()
        G.add_nodes_from(sub["author_id"].unique())
        edges = (
            pd.DataFrame({"a": sub["author_id"], "b": target_author.loc[sub.index]})
            .dropna()
            .groupby(["a", "b"]).size().reset_index(name="w")
        )
        G.add_weighted_edges_from((a, b, int(w)) for a, b, w in edges.itertuples(index=False, name=None))
        layers[str(topic)] = G
    return layers


def main() -> None:
    truth = pd.read_csv(HERE / "truth_synth_pol.csv")
    topics = pd.read_csv(HERE / "topics_synth_pol.csv")
    designed = dict(zip(topics["topic"], topics["polarized"]))

    layers = build_layers(truth)
    config = netpol.PolarizationConfig(
        influencer_strategy="in_degree", n_influencers=50, ideology_dimensions=1,
        significance_level=0.05, fdr_correction=True, min_edges=20,
    )
    results = netpol.analyze_layers(layers, config)

    rows = []
    for lid, r in results.items():
        d = designed.get(lid)
        rows.append({
            "true_topic": lid, "n_nodes": r.n_nodes, "n_edges": r.n_edges,
            "was_analyzed": r.was_analyzed,
            "dip": None if r.dip_statistic is None else round(r.dip_statistic, 3),
            "adj_p": None if r.adjusted_p_value is None else round(r.adjusted_p_value, 4),
            "detected_polarized": r.is_polarized,
            "designed_polarized": bool(d) if d is not None and not pd.isna(d) else None,
            "skip_reason": r.skip_reason,
        })
    df = pd.DataFrame(rows).sort_values("true_topic")
    lab = df[df["designed_polarized"].notna() & (df["was_analyzed"] == True)]  # noqa: E712
    tp = int(((lab["detected_polarized"] == True) & (lab["designed_polarized"] == True)).sum())  # noqa: E712
    fp = int(((lab["detected_polarized"] == True) & (lab["designed_polarized"] == False)).sum())  # noqa: E712
    fn = int(((lab["detected_polarized"] != True) & (lab["designed_polarized"] == True)).sum())  # noqa: E712
    tn = int(((lab["detected_polarized"] != True) & (lab["designed_polarized"] == False)).sum())  # noqa: E712

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    md = ["# ORACLE_POLARIZATION_REPORT — netpol on ground-truth topics (synth_pol)", "",
          f"- layers: **{len(results)}**; analyzed: **{int((df['was_analyzed'] == True).sum())}**; "  # noqa: E712
          f"polarized (FDR): **{int((df['detected_polarized'] == True).sum())}**",  # noqa: E712
          f"- confusion (detected vs designed): TP={tp} FP={fp} FN={fn} TN={tn}",
          f"- recall on designed-polarized: **{tp/(tp+fn):.0%}**" if (tp + fn) else "- recall: n/a",
          f"- false-positive rate on consensual: **{fp/(fp+tn):.0%}**" if (fp + tn) else "- FPR: n/a",
          "",
          "## Per-layer detail", "", "\n".join(lines), ""]
    (HERE.parent / "reports" / "ORACLE_POLARIZATION_REPORT.md").write_text("\n".join(md))
    print(f"oracle: layers={len(results)} TP={tp} FP={fp} FN={fn} TN={tn}")
    print(df[["true_topic", "n_nodes", "n_edges", "dip", "adj_p",
              "detected_polarized", "designed_polarized"]].to_string(index=False))


if __name__ == "__main__":
    main()
