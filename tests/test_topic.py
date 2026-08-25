from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_import_no_side_effects():
    import importlib
    import sys

    if "topiclayers.core.topic" in sys.modules:
        del sys.modules["topiclayers.core.topic"]
    import topiclayers.core.topic


def test_all_outlier_raises():
    from topiclayers.core.topic import TopicModeler

    df = pd.DataFrame({"text": ["a", "b", "c", "d", "e", "f", "g"]})
    with tempfile.TemporaryDirectory() as d:
        tm = TopicModeler(df, embedder_name="all-MiniLM-L6-v2", path_cache=Path(d), name="test")
        with pytest.raises(RuntimeError):
            tm.get_topics(min_topic_size=50, random_state=42)


def test_normal_run_produces_topics():
    from topiclayers.core.topic import TopicModeler

    df = pd.DataFrame({"text": ["hello world", "hello there", "world today", "python code", "python test", "java code", "java test"]})
    with tempfile.TemporaryDirectory() as d:
        tm = TopicModeler(df, embedder_name="all-MiniLM-L6-v2", path_cache=Path(d), name="test")
        result = tm.get_topics(min_topic_size=2, random_state=42)
        topics = result["topic"].unique()
        assert len(topics) > 0
        assert "topic" in result.columns


def test_fast_mode_produces_topics():
    from topiclayers.core.topic import TopicModeler

    df = pd.DataFrame({"text": ["hello world", "hello there", "world today", "python code", "python test", "java code", "java test"]})
    with tempfile.TemporaryDirectory() as d:
        tm = TopicModeler(df, embedder_name="all-MiniLM-L6-v2", path_cache=Path(d), name="test")
        result = tm.get_topics(min_topic_size=2, random_state=42, repro_mode="fast")
        topics = result["topic"].unique()
        assert len(topics) > 0
        assert "topic" in result.columns
