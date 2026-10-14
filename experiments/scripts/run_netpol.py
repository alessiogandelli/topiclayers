#!/usr/bin/env python3
"""Run netpol polarisation analysis on a topiclayers multilayer network.

Usage:
    .venv/bin/python run_netpol.py swiss
    .venv/bin/python run_netpol.py rumoureval
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent  # topiclayers/experiments/
ROOT = HERE.parents[2]  # climate_polarization_paper/
sys.path.insert(0, str(ROOT / "netpol"))

import netpol  # noqa: E402
import pandas as pd  # noqa: E402


def load_labels(out: Path, name: str) -> dict[int, str]:
    patterns = [
        str(EXPERIMENTS / "cache" / name / "tm" / "*" / f"topics_{name}.csv"),
        str(out / "cache" / "tm" / "*" / f"topics_{name}.csv"),
    ]
    for pattern in patterns:
        for csv in glob.glob(pattern):
            df = pd.read_csv(csv)
            col = "LLM_label" if "LLM_label" in df.columns else "Name"
            return {int(r["Topic"]): str(r[col]) for _, r in df.iterrows() if pd.notna(r["Topic"])}
    return {}


def main() -> None:
    name = sys.argv[1]
    out = EXPERIMENTS.parent / "out" / name
    gml = out / "networks" / f"{name}_retweet_network_ml.gml"
    if not gml.exists():
        sys.exit(f"not found: {gml}")

    layers = netpol.load_layers(gml)
    labels = load_labels(out, name)
    print(f"{name}: {len(layers)} layers")

    config = netpol.PolarizationConfig(
        influencer_strategy="in_degree",
        n_influencers=100,
        ideology_dimensions=1,
        significance_level=0.05,
        fdr_correction=True,
        min_edges=10,
        exclude_layers=(-1,),
    )
    results = netpol.analyze_layers(layers, config)

    analyzed = [(lid, r) for lid, r in results.items() if r.was_analyzed]
    polarized = [(lid, r) for lid, r in analyzed if r.is_polarized]
    polarized.sort(key=lambda kv: kv[1].dip_statistic, reverse=True)

    print(f"  analyzed {len(analyzed)}/{len(results)} layers; {len(polarized)} polarized (FDR-adjusted)")
    print(f"  {'topic':>6}  {'n_edges':>7}  {'dip':>6}  {'adj_p':>9}  label")
    for lid, r in polarized[:15]:
        lbl = labels.get(int(float(lid)), "?")[:44]
        print(f"  {int(float(lid)):>6}  {r.n_edges:>7}  {r.dip_statistic:>6.3f}  "
              f"{(r.adjusted_p_value or float('nan')):>9.3g}  {lbl}")

    skipped = {lid: r.skip_reason for lid, r in results.items() if not r.was_analyzed}
    if skipped:
        print(f"  skipped {len(skipped)} layers, e.g. " +
              "; ".join(f"{lid}: {reason}" for lid, reason in list(skipped.items())[:3]))

    rows = []
    for lid, r in results.items():
        rows.append({
            "topic": int(float(lid)),
            "label": labels.get(int(float(lid)), ""),
            "n_nodes": r.n_nodes,
            "n_edges": r.n_edges,
            "dip_statistic": r.dip_statistic,
            "p_value": r.p_value,
            "adjusted_p_value": r.adjusted_p_value,
            "is_polarized": r.is_polarized,
            "skip_reason": r.skip_reason,
        })
    dest = EXPERIMENTS / "outputs" / f"netpol_{name}.csv"
    pd.DataFrame(rows).sort_values("topic").to_csv(dest, index=False)
    print(f"  -> {dest}")


if __name__ == "__main__":
    main()
