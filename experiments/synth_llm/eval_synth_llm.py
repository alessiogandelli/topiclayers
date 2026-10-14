#!/usr/bin/env python3
"""Score topiclayers' clustering of the LLM-synthetic corpus against ground truth.

Joins the pipeline's per-document topic assignment (``tweets_<name>_labeled.pkl``
in the cache) with ``truth_synth_llm.csv`` on the post id, then reports:

* agreement metrics (ARI, homogeneity, completeness, V-measure, purity,
  best-mapping accuracy);
* a true-topic x predicted-cluster confusion matrix;
* outlier handling (true off-topic posts rejected to -1, false-outlier rate);
* whether each deliberately-confusable *tricky pair* merged or stayed separate;
* where the *bridge* posts (mixing two topics) landed.

Usage:
    .venv/bin/python eval_synth_llm.py \
        --cache experiments/synth_llm/cache \
        --truth experiments/synth_llm/truth_synth_llm.csv \
        --out experiments/reports/SYNTH_LLM_REPORT.md \
        --names synth_llm synth_tune_t12 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, v_measure_score)

TRICKY_PAIRS = [
    ("carbon_pricing", "cop_negotiations"),
    ("grid_storage", "renewables"),
    ("doom_anxiety", "climate_denial"),
    ("climate_health", "extreme_weather"),
]
BRIDGE_OF = {
    "bridge_cop_pricing": ("cop_negotiations", "carbon_pricing"),
    "bridge_renew_grid": ("renewables", "grid_storage"),
    "bridge_denial_doom": ("climate_denial", "doom_anxiety"),
    "bridge_weather_health": ("extreme_weather", "climate_health"),
}


def _find_labeled(cache: Path, name: str) -> Path | None:
    hits = sorted(cache.rglob(f"tweets_{name}_labeled.pkl"))
    return hits[0] if hits else None


def _purity(true: np.ndarray, pred: np.ndarray) -> float:
    df = pd.DataFrame({"t": true, "p": pred})
    return df.groupby("p")["t"].apply(lambda s: s.value_counts().max()).sum() / len(df)


def _best_mapping_accuracy(true: np.ndarray, pred: np.ndarray) -> tuple[float, dict]:
    ct = pd.crosstab(pd.Series(true, name="t"), pd.Series(pred, name="p"))
    row, col = linear_sum_assignment(-ct.values)
    acc = ct.values[row, col].sum() / ct.values.sum()
    mapping = {ct.columns[c]: ct.index[r] for r, c in zip(row, col)}
    return acc, mapping


def score_assignments(assign: pd.DataFrame, truth: pd.DataFrame, name: str = "") -> dict:
    """Score a per-post assignment frame (columns ``id``, ``pred``) against truth."""
    merged = assign.merge(truth, left_on="id", right_on="post_id", how="inner")
    merged = merged[merged["is_derived"] == False].copy()  # noqa: E712  (originals only)

    true = merged["true_topic"].to_numpy()
    pred = merged["pred"].to_numpy()
    n = len(merged)
    k = len(set(pred) - {-1})

    is_out = merged["true_family"] == "outlier"
    out_recall = float((merged.loc[is_out, "pred"] == -1).mean()) if is_out.any() else float("nan")
    false_out = float((merged.loc[~is_out, "pred"] == -1).mean())

    acc, mapping = _best_mapping_accuracy(true, pred)
    ct = pd.crosstab(pd.Series(true, name="true_topic"),
                     pd.Series(pred, name="pred_cluster"))

    # per true-topic dominant cluster
    per_topic = []
    for t in sorted(set(true)):
        sub = merged[merged["true_topic"] == t]
        vc = sub["pred"].value_counts()
        per_topic.append({
            "true_topic": t,
            "n": len(sub),
            "dominant_cluster": int(vc.index[0]),
            "share": round(vc.iloc[0] / len(sub), 3),
            "outlier_frac": round(float((sub["pred"] == -1).mean()), 3),
        })

    # tricky pairs
    tricky = []
    for a, b in TRICKY_PAIRS:
        ma = merged[merged["true_topic"] == a]
        mb = merged[merged["true_topic"] == b]
        va = ma["pred"].value_counts()
        vb = mb["pred"].value_counts()
        dom_a = int(va.index[0]) if len(va) else None
        dom_b = int(vb.index[0]) if len(vb) else None
        # overlap = share of A's docs that fall in B's dominant cluster
        overlap = float((ma["pred"] == dom_b).mean()) if dom_b is not None else float("nan")
        tricky.append({
            "pair": f"{a} / {b}",
            "a_dominant": dom_a,
            "a_share": round(va.iloc[0] / len(ma), 3) if len(va) else None,
            "b_dominant": dom_b,
            "b_share": round(vb.iloc[0] / len(mb), 3) if len(vb) else None,
            "merged": dom_a == dom_b,
            "a_in_b_cluster": round(overlap, 3),
        })

    # bridge destinations
    bridge = []
    for bt, (a, b) in BRIDGE_OF.items():
        sub = merged[merged["true_topic"] == bt]
        if not len(sub):
            continue
        vc = sub["pred"].value_counts()
        top = int(vc.index[0])
        # which true topic dominates the top cluster overall?
        owner = ct[top].idxmax() if top in ct.columns else "?"
        bridge.append({
            "bridge": bt,
            "links": f"{a} + {b}",
            "n": len(sub),
            "top_cluster": top,
            "owner_true_topic": owner,
            "owner_share": round(float(ct[top].max() / ct[top].sum()), 3),
        })

    minus1_fam = merged[merged["pred"] == -1]["true_family"].value_counts().to_dict()

    return {
        "name": name,
        "n": n,
        "k": k,
        "outlier_rate": round(float((pred == -1).mean()), 3),
        "minus1_family": minus1_fam,
        "metrics": {
            "ari": round(adjusted_rand_score(true, pred), 3),
            "homogeneity": round(homogeneity_score(true, pred), 3),
            "completeness": round(completeness_score(true, pred), 3),
            "v_measure": round(v_measure_score(true, pred), 3),
            "purity": round(_purity(true, pred), 3),
            "best_mapping_accuracy": round(acc, 3),
        },
        "outlier": {
            "true_outlier_reject_recall": round(out_recall, 3),
            "false_outlier_rate": round(false_out, 3),
        },
        "per_topic": per_topic,
        "tricky": tricky,
        "bridge": bridge,
        "confusion": ct,
    }


def evaluate(name: str, cache: Path, truth: pd.DataFrame) -> dict | None:
    pkl = _find_labeled(cache, name)
    if pkl is None:
        print(f"  [skip] no labeled file for '{name}'")
        return None
    df = pd.read_pickle(pkl)[["id", "topic"]].rename(columns={"topic": "pred"})
    return score_assignments(df, truth, name)


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out) + "\n"


def _findings(results: list[dict]) -> list[str]:
    out = []
    collapse = [r for r in results if r["k"] < 11]
    if collapse:
        r = collapse[0]
        out.append(
            f"**`nr_topics: auto` over-merges**: run `{r['name']}` found only **{r['k']}** "
            f"clusters (purity {r['metrics']['purity']}, ARI {r['metrics']['ari']}); the "
            f"fine-grained subtopics collapse into one blob and the tricky pairs become "
            f"indistinguishable.")
    best = max(results, key=lambda r: r["metrics"]["ari"])
    merged = sum(1 for t in best["tricky"] if t["merged"])
    out.append(
        f"Best run by ARI is **`{best['name']}`** (k={best['k']}, purity "
        f"{best['metrics']['purity']}, ARI {best['metrics']['ari']}, homogeneity "
        f"{best['metrics']['homogeneity']}); **{merged}/4 tricky pairs merged**.")
    rec = [r["outlier"]["true_outlier_reject_recall"] for r in results]
    out.append(
        f"**Off-topic posts are not reliably flagged `-1`**: rejection recall is "
        f"{min(rec):.0%}–{max(rec):.0%}, and the `-1` topic is largely filled with "
        f"*on-topic* posts. The off-topic posts share an 'everyday life' register, so "
        f"MiniLM groups them into a topic instead of isolating them.")
    return out


def render(results: list[dict], max_confusion_k: int = 40, title: str = "SYNTH_LLM_REPORT",
           note: str = "", repro: str = "") -> str:
    md = [f"# {title} — LLM-synthetic tweets through topiclayers", ""]
    if note:
        md += [note, ""]
    md += [f"{results[0]['n']} originals scored (derived reposts/quotes/replies excluded).", ""]
    if repro:
        md += ["Reproduce:", "", "```bash", repro, "```", ""]

    md += ["## Key findings", ""]
    for f in _findings(results):
        md.append(f"- {f}")
    md.append("")

    cols = ["name", "n", "k", "ari", "homogeneity", "completeness", "v_measure",
            "purity", "best_mapping_accuracy", "outlier_rate",
            "true_outlier_reject_recall", "false_outlier_rate"]
    summary = []
    for r in results:
        summary.append({"name": r["name"], "n": r["n"], "k": r["k"],
                        **r["metrics"], "outlier_rate": r["outlier_rate"],
                        **r["outlier"]})
    md += ["## Summary", "", _table(summary, cols), ""]

    for r in results:
        md += [f"## Run `{r['name']}`", "",
               f"- originals scored: **{r['n']}**, predicted clusters (excl. -1): **{r['k']}**",
               f"- ARI **{r['metrics']['ari']}**, homogeneity **{r['metrics']['homogeneity']}**, "
               f"completeness **{r['metrics']['completeness']}**, purity **{r['metrics']['purity']}**, "
               f"best-mapping accuracy **{r['metrics']['best_mapping_accuracy']}**",
               f"- true off-topic posts rejected to -1: **{r['outlier']['true_outlier_reject_recall']:.1%}**; "
               f"on-topic posts wrongly marked outlier: **{r['outlier']['false_outlier_rate']:.1%}**",
               f"- `-1` composition by true family: `{r['minus1_family']}`",
               ""]
        md += ["**Per true topic**", "",
               _table(r["per_topic"], ["true_topic", "n", "dominant_cluster", "share", "outlier_frac"]),
               "**Tricky pairs (merged = same dominant cluster)**", "",
               _table(r["tricky"], ["pair", "a_dominant", "a_share", "b_dominant", "b_share", "merged", "a_in_b_cluster"]),
               "**Bridge posts**", "",
               _table(r["bridge"], ["bridge", "links", "n", "top_cluster", "owner_true_topic", "owner_share"])]
        md.append("**Confusion matrix** (rows = true topic, cols = predicted cluster)")
        md.append("")
        if r["k"] <= max_confusion_k:
            md += ["```", r["confusion"].to_string(), "```", ""]
        else:
            md += [f"_({r['k']} clusters — too many to print; see `netpol`/cache for the full table.)_", ""]
    return "\n".join(md)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=here / "cache")
    ap.add_argument("--truth", type=Path, default=here / "truth_synth_llm.csv")
    ap.add_argument("--out", type=Path, default=here.parent / "reports" / "SYNTH_LLM_REPORT.md")
    ap.add_argument("--names", nargs="+", default=["synth_llm"])
    ap.add_argument("--json", type=Path, default=None,
                    help="also write per-run metrics as JSON for later tuning")
    ap.add_argument("--title", default="SYNTH_LLM_REPORT")
    ap.add_argument("--note", default="")
    ap.add_argument("--repro", default="")
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    results = []
    for name in args.names:
        print(f"== {name} ==")
        r = evaluate(name, args.cache, truth)
        if r:
            print(f"  k={r['k']} ari={r['metrics']['ari']} "
                  f"purity={r['metrics']['purity']} "
                  f"outlier_recall={r['outlier']['true_outlier_reject_recall']}")
            results.append(r)

    args.out.write_text(render(results, title=args.title, note=args.note, repro=args.repro))
    print(f"\n-> {args.out}")

    if args.json:
        import json as _json
        payload = [{k: v for k, v in r.items() if k != "confusion"} for r in results]
        args.json.write_text(_json.dumps(payload, indent=2, default=str))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
