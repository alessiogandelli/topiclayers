from __future__ import annotations

from pathlib import Path

import pytest

import topiclayers.adapters.csv as _csv_adapter
import topiclayers.adapters.twitter as _twitter_adapter
from topiclayers.adapters.registry import resolve_adapter
from topiclayers.schema import InteractionType, Post


def test_post_creation():
    p = Post(post_id="1", user_id="a", text="hello")
    assert p.post_id == "1"
    assert p.user_id == "a"
    assert p.text == "hello"
    assert p.interaction_type is None


def test_interaction_type_enum():
    assert InteractionType("repost") == InteractionType.REPOST
    with pytest.raises(ValueError):
        InteractionType("invalid")


def test_twitter_adapter_loads_toy(twitter_jsonl_path):
    from topiclayers.adapters.twitter import load

    posts = load(twitter_jsonl_path)
    assert len(posts) > 0
    assert posts[0].post_id is not None
    assert posts[0].user_id is not None


def test_generic_csv_adapter_loads_7_rows(generic_csv_path):
    from topiclayers.adapters.csv import load_csv

    posts = load_csv(generic_csv_path)
    assert len(posts) == 7


def test_auto_detect_format():
    fmt, _ = resolve_adapter(Path("something.csv"))
    assert fmt == "csv_posts"


def test_unknown_extension_raises():
    with pytest.raises(ValueError):
        resolve_adapter(Path("file.xyz"))
