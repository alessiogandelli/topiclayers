"""Robustness regressions found by the stress run."""

from __future__ import annotations

from pathlib import Path

from topiclayers.adapters.csv import load_csv
from topiclayers.core.data import Dataset


def test_csv_large_numeric_ids_preserved(tmp_path):
    path = Path(tmp_path) / "ids.csv"
    path.write_text(
        "post_id,user_id,text,interaction_type,target_post_id\n"
        "1622232152543821824,25254764,hello world,repost,1622232152543821824\n"
        "2,3,another post,original,\n"
    )
    posts = load_csv(path)
    assert posts[0].post_id == "1622232152543821824"
    assert posts[0].target_post_id == "1622232152543821824"
    assert posts[1].target_post_id is None


def test_csv_empty_text_is_empty_not_nan(tmp_path):
    path = Path(tmp_path) / "posts.csv"
    path.write_text("post_id,user_id,text\n1,a,\n2,b,hello world\n")
    posts = load_csv(path)
    assert posts[0].text == ""
    assert posts[1].text == "hello world"


def test_empty_input_does_not_crash(tmp_path):
    ds = Dataset([], n_cop="empty_input")
    ds.process(cache_dir=Path(tmp_path) / "cache", adapter_name="twitter_jsonl")

    assert len(ds.df_tweets) == 0
    assert "referenced_type" in ds.df_tweets.columns
    for frame in (ds.df_original, ds.df_retweets, ds.df_quotes, ds.df_reply):
        assert len(frame) == 0
        assert "author" in frame.columns
