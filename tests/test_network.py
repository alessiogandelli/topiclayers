from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from topiclayers.core.network import NetworkCreator


def _toy_df():
    return pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "author": ["a", "b", "c", "c", "a", "d", "d"],
            "text": ["tweet1", "tweet2", "tweet3", "tweet4", "tweet5", "tweet6", "tweet7"],
            "date": pd.date_range(start="2022-01-01", periods=7).strftime("%Y-%m-%d").tolist(),
            "lang": ["en"] * 7,
            "referenced_type": ["retweeted", "retweeted", "retweeted", None, "retweeted", "retweeted", None],
            "referenced_id": ["t1", "t2", "t1", None, "t7", "t7", None],
            "topic": [1, 1, 1, 1, 2, 2, 2],
            "mentions_name": [["b"], [], [], [], [], [], []],
        }
    ).set_index("id")


def test_retweet_network_has_edges():
    with tempfile.TemporaryDirectory() as d:
        nw = NetworkCreator(_toy_df(), "test", Path(d))
        G = nw.create_retweet_network()
        assert G.number_of_edges() > 0


def test_nan_referenced_id_is_skipped():
    df = _toy_df()
    df.loc["t2", "referenced_id"] = float("nan")
    with tempfile.TemporaryDirectory() as d:
        nw = NetworkCreator(df, "test", Path(d))
        G = nw.create_retweet_network()
        assert G.number_of_edges() >= 0


def test_missing_ref_does_not_crash():
    df = _toy_df()
    df.loc["t2", "referenced_id"] = "missing_xyz"
    with tempfile.TemporaryDirectory() as d:
        nw = NetworkCreator(df, "test", Path(d))
        G = nw.create_retweet_network()
        assert G.number_of_nodes() > 0


def test_create_retweet_ml_all_outlier_raises():
    df = _toy_df()
    df["topic"] = -1
    with tempfile.TemporaryDirectory() as d:
        nw = NetworkCreator(df, "test", Path(d))
        with pytest.raises(ValueError):
            nw.create_retweet_ml()
