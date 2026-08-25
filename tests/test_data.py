from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from topiclayers.core.data import Dataset
from topiclayers.schema import InteractionType, Post


def test_dataset_cop_not_none():
    posts = [
        Post("1", "u1", "hello"),
        Post("2", "u2", "world", interaction_type=InteractionType.REPOST, target_post_id="1"),
    ]
    ds = Dataset(posts, n_cop="cop26")
    with tempfile.TemporaryDirectory() as d:
        ds.process(cache_dir=Path(d) / "cache")
        assert ds.df_tweets.loc["1"]["cop"] == "cop26"


def test_update_df_no_infinite_loop():
    posts = [
        Post("t1", "a", "t1", interaction_type=InteractionType.REPOST, target_post_id="t2"),
        Post("t2", "b", "t2", interaction_type=InteractionType.REPOST, target_post_id="t1"),
        Post("t3", "c", "t3"),
    ]
    ds = Dataset(posts, n_cop="cycle_test")
    with tempfile.TemporaryDirectory() as d:
        ds.process(cache_dir=Path(d) / "cache")
        df_orig = ds.df_original.copy()
        df_orig["topic"] = [1]
        result = ds.update_df(df_orig)
        assert isinstance(result, pd.DataFrame)


def test_update_df_idempotent():
    posts = [
        Post("1", "a", "hello"),
        Post("2", "b", "world", interaction_type=InteractionType.REPOST, target_post_id="1"),
    ]
    ds = Dataset(posts, n_cop="test")
    with tempfile.TemporaryDirectory() as d:
        ds.process(cache_dir=Path(d) / "cache")
        df_orig = ds.df_original.copy()
        df_orig["topic"] = [1]
        r1 = ds.update_df(df_orig)
        r2 = ds.update_df(df_orig)
        assert r1.equals(r2)


def test_get_original_retweet_split():
    posts = [
        Post("1", "a", "hello"),
        Post("2", "b", "world", interaction_type=InteractionType.REPOST, target_post_id="1"),
        Post("3", "c", "hi", interaction_type=InteractionType.QUOTE, target_post_id="1"),
    ]
    ds = Dataset(posts, n_cop="test")
    with tempfile.TemporaryDirectory() as d:
        ds.process(cache_dir=Path(d) / "cache")
        assert len(ds.df_original) == 1
        assert len(ds.df_retweets) == 1
        assert len(ds.df_quotes) == 1
