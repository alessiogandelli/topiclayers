#!/usr/bin/env python3
"""Check whether netpol recovers the *designed* polarization in synth_pol.

The corpus is built with a ground truth: 6 topics are written with two opposing
camps and same-side retweeting (should look polarized), 4 are consensual (should
not). This script:

1. loads a run's per-topic multilayer retweet network (``*_retweet_network_ml.gml``);
2. runs ``netpol`` (latent ideology + Hartigan dip test, FDR);
3. maps each predicted layer back to its majority ground-truth topic (from the
   cached ``tweets_<name>_labeled.pkl``);
4. compares netpol's ``is_polarized`` verdict with the designed flag.

Usage:
    .venv/bin/python experiments/synth_llm/eval_polarization.py \
        --name synth_pol_coarse \
        --ml out/synth_pol_coarse/networks/synth_pol_coarse_retweet_network_ml.gml \
        --labeled experiments/cache/synth_pol_coarse/tm/all-MiniLM-L6-v2/tweets_synth_pol_coarse_labeled.pkl \
        --truth experiments/synth_llm/truth_synth_pol.csv \
        --out experiments/reports/POLARIZATION_REPORT.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent  # .../topiclayers/experiments/synth_llm
ROOT = HERE.parents[2]  # .../climate_polarization_paper
sys.path.insert(0, str(ROOT / "netpol"))
import netpol  # noqa: E402


def _majority_topic_by_layer(labeled_pkl: Path, truth: pd.DataFrame) -> dict:
    df = pd.read_pickle(labeled_pkl)[["id", "topic"]].rename(columns={"topic": "layer"})
    m = df.merge(truth, left_on="id", right_on="post_id")
    m = m[m["is_derived"] == False]  # noqa: E712
    out = {}
    for layer, sub in m.groupby("layer"):
        vt = sub["true_topic"].value_counts()
        vp = sub["topic_polarized"].value_counts()
        out[int(layer)] = {
            "true_topic": vt.index[0],
            "purity": round(vt.iloc[0] / len(sub), 3),
            "n": len(sub),
            "designed_polarized": bool(vp.index[0]) if len(vp) else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ml", type=Path, required=True)
    ap.add_argument("--labeled", type=Path, required=True)
    ap.add_argument("--truth", type=Path, default=HERE / "truth_synth_pol.csv")
    ap.add_argument("--out", type=Path, default=HERE.parent / "reports" / "POLARIZATION_REPORT.md")
    ap.add_argument("--n-influencers", type=int, default=50)
    ap.add_argument("--min-edges", type=int, default=20)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    layers = netpol.load_layers(str(args.ml))
    mapping = _majority_topic_by_layer(args.labeled, truth)

    config = netpol.PolarizationConfig(
        influencer_strategy="in_degree",
        n_influencers=args.n_influencers,
        ideology_dimensions=1,
        significance_level=0.05,
        fdr_correction=True,
        min_edges=args.min_edges,
        exclude_layers=(-1,),
    )
    results = netpol.analyze_layers(layers, config)

    rows = []
    for lid, r in results.items():
        info = mapping.get(int(float(lid)), {})
        rows.append({
            "layer": int(float(lid)),
            "n_nodes": r.n_nodes,
            "n_edges": r.n_edges,
            "was_analyzed": r.was_analyzed,
            "dip": None if r.dip_statistic is None else round(r.dip_statistic, 3),
            "adj_p": None if r.adjusted_p_value is None else round(r.adjusted_p_value, 4),
            "detected_polarized": r.is_polarized,
            "true_topic": info.get("true_topic", ""),
            "designed_polarized": info.get("designed_polarized"),
            "layer_purity": info.get("purity", ""),
            "skip_reason": r.skip_reason,
        })
    df = pd.DataFrame(rows).sort_values("layer")

    analyzed = df[df["was_analyzed"] == True]  # noqa: E712
    lab = analyzed[analyzed["designed_polarized"].notna()]
    tp = int(((lab["detected_polarized"] == True) & (lab["designed_polarized"] == True)).sum())  # noqa: E712
    fp = int(((lab["detected_polarized"] == True) & (lab["designed_polarized"] == False)).sum())  # noqa: E712
    fn = int(((lab["detected_polarized"] != True) & (lab["designed_polarized"] == True)).sum())  # noqa: E712
    tn = int(((lab["detected_polarized"] != True) & (lab["designed_polarized"] == False)).sum())  # noqa: E712

    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")

    md = [f"# POLARIZATION_REPORT — netpol vs designed polarization ({args.name})", "",
          f"- multilayer layers: **{len(results)}**; analyzed: **{len(analyzed)}**; "
          f"polarized (FDR): **{int((analyzed['detected_polarized'] == True).sum())}**",  # noqa: E712
          f"- layers mapped to a designed topic: **{len(lab)}**",
          f"- confusion (detected vs designed): TP={tp} FP={fp} FN={fn} TN={tn}",
          f"- recall on designed-polarized: **{tp/(tp+fn):.0%}**" if (tp + fn) else "- recall: n/a",
          f"- false-positive rate on consensual: **{fp/(fp+tn):.0%}**" if (fp + tn) else "- FPR: n/a",
          "",
          "## Per-layer detail", "",
          "\n".join(lines), ""]
    args.out.write_text("\n".join(md))
    print(f"{args.name}: layers={len(results)} analyzed={len(analyzed)} "
          f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
