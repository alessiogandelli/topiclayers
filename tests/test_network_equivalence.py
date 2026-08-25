"""Equivalence tests: vectorized network creation vs the original row-loop code.

Uses the real COP22 run cache when available (slow, ~1 min); otherwise a
synthetic frame with the same schema. The old implementation is inlined here
on purpose: it is the reference whose output the optimized code must match.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest

from topiclayers.core.network import NetworkCreator

CACHE = (
    __file__.split("tests")[0] + "out/cop22/cache/data/retweet_labeled_cop22.pkl"
)


def _load_df() -> pd.DataFrame:
    import os

    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    pytest.skip("cop22 cache not available")


def old_retweet_network(df: pd.DataFrame) -> tuple[nx.DiGraph, int]:
    G = nx.DiGraph()
    G.add_nodes_from(df["author"].unique())

    author_lookup = df["author"].to_dict()
    missing_refs = 0

    for idx, row in df.iterrows():
        ref_id = row.get("referenced_id")
        if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
            continue

        target_author = author_lookup.get(str(ref_id))
        if target_author is None:
            missing_refs += 1
            continue

        source = row["author"]
        if G.has_edge(source, target_author):
            G[source][target_author]["weight"] += 1
        else:
            G.add_edge(source, target_author, weight=1)

    return G, missing_refs


def old_layer_edges(df_slice: pd.DataFrame) -> tuple[dict, int]:
    """Old per-topic edge aggregation: {(src, tgt): weight}, dropped count."""
    author_lookup = df_slice["author"].to_dict()
    edges: dict = {}
    dropped = 0

    for idx, row in df_slice.iterrows():
        ref_id = row.get("referenced_id")
        if ref_id is None or (isinstance(ref_id, float) and pd.isna(ref_id)):
            continue

        target_author = author_lookup.get(str(ref_id))
        if target_author is None:
            dropped += 1
            continue

        key = (row["author"], target_author)
        edges[key] = edges.get(key, 0) + 1

    return edges, dropped


def _assert_same_graph(a: nx.DiGraph, b: nx.DiGraph) -> None:
    assert set(a.nodes) == set(b.nodes)
    assert set(a.edges) == set(b.edges)
    for u, v in a.edges:
        assert a[u][v]["weight"] == b[u][v]["weight"]


def test_single_layer_equivalence_real_cache(tmp_path):
    df = _load_df()
    nw = NetworkCreator(df, "equiv_test", tmp_path)
    new_G = nw.create_retweet_network()
    old_G, old_missing = old_retweet_network(df)

    _assert_same_graph(old_G, new_G)
    # the reference run's 52,032 unresolved retweets were already excluded
    # during label propagation, so the cached frame has 0 missing refs
    assert old_missing == 0


def test_multilayer_slices_equivalence_real_cache(tmp_path):
    df = _load_df()
    rt = df.loc[df["referenced_id"].notna(), ["author", "referenced_id", "topic"]]

    topics = [t for t in sorted(df["topic"].unique()) if t != -1]
    # sample slices across the size spectrum to keep runtime reasonable
    sizes = df["topic"].value_counts()
    sampled = list(sizes.head(3).index) + list(sizes.tail(3).index) + topics[::50]

    for topic in set(sampled):
        if topic == -1:
            continue
        df_slice = df[df["topic"] == topic]
        rt_slice = rt[rt["topic"] == topic]

        edges, dropped = NetworkCreator._aggregate_slice(df_slice, rt_slice)
        old_edges, old_dropped = old_layer_edges(df_slice)

        assert dropped == old_dropped, f"topic {topic}: dropped mismatch"
        new_map = {
            (r.author, r.target): int(r.weight)
            for r in edges.itertuples(index=False)
        }
        assert new_map == old_edges, f"topic {topic}: edges mismatch"
