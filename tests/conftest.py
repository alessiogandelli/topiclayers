"""Shared pytest fixtures for topiclayers tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
EXAMPLES_DIR = REPO_ROOT / "examples"


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """7-row fixture matching legacy test/test.py and examples/generic/posts.csv."""
    df = pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "author": ["a", "b", "c", "c", "a", "d", "d"],
            "text": ["tweet1", "tweet2", "tweet3", "tweet4", "tweet5", "tweet6", "tweet7"],
            "date": pd.date_range(start="2022-01-01", periods=7).strftime("%Y-%m-%d").tolist(),
            "lang": ["en"] * 7,
            "referenced_type": [
                "retweeted",
                "retweeted",
                "retweeted",
                None,
                "retweeted",
                "retweeted",
                None,
            ],
            "referenced_id": [None, "t1", "t2", None, "t7", "t7", None],
            "topic": [1, 1, 1, 1, 2, 2, 2],
            "mentions_name": [["b"], [], [], [], [], [], []],
        }
    ).set_index("id")
    return df


@pytest.fixture
def toy_df_with_cycle() -> pd.DataFrame:
    """7-row fixture with a reference cycle A->B->A."""
    return pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "author": ["a", "b", "c", "c", "a", "d", "d"],
            "text": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "referenced_type": [
                "retweeted",
                "retweeted",
                "retweeted",
                None,
                "retweeted",
                "retweeted",
                None,
            ],
            "referenced_id": ["t2", "t1", "t1", None, "t7", "t4", None],
            "topic": [1, 1, 1, 1, 2, 2, 2],
        }
    ).set_index("id")


@pytest.fixture
def toy_df_all_outliers() -> pd.DataFrame:
    """7-row fixture where every tweet has topic=-1."""
    df = pd.DataFrame(
        {
            "id": ["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
            "author": ["a", "b", "c", "c", "a", "d", "d"],
            "text": ["x"] * 7,
            "topic": [-1] * 7,
            "referenced_type": ["retweeted"] * 7,
            "referenced_id": [None] * 7,
        }
    ).set_index("id")
    return df


@pytest.fixture
def twitter_jsonl_path() -> Path:
    """Path to the toy Twitter JSONL fixture."""
    p = EXAMPLES_DIR / "twitter" / "toy.json"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture
def generic_csv_path() -> Path:
    """Path to the generic CSV fixture."""
    p = EXAMPLES_DIR / "generic" / "posts.csv"
    assert p.exists(), f"Missing fixture: {p}"
    return p


@pytest.fixture
def tmp_output_dir() -> Path:
    """Temporary directory for test outputs. Automatically cleaned up."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)
