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


TOY_DF = pd.DataFrame({"text": ["hello world", "hello there", "world today", "python code", "python test", "java code", "java test"]})


def _fit_toy(tmpdir):
    from topiclayers.core.topic import TopicModeler

    tm = TopicModeler(TOY_DF.copy(), embedder_name="all-MiniLM-L6-v2", path_cache=Path(tmpdir), name="test")
    tm.get_topics(min_topic_size=2, random_state=42)
    return tm


def test_label_topics_skips_without_model():
    from topiclayers.core.topic import TopicModeler

    with tempfile.TemporaryDirectory() as d:
        tm = TopicModeler(TOY_DF.copy(), embedder_name="all-MiniLM-L6-v2", path_cache=Path(d), name="test")
        assert tm.label_topics() == {}


def test_label_topics_non_fatal_when_ollama_down(monkeypatch):
    import os

    monkeypatch.setenv("OLLAMA_HOST", "localhost")
    monkeypatch.setenv("OLLAMA_PORT", "1")  # nothing listens here
    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(model="ollama/qwen2.5:7b")
        assert labels == {}
        # the model still has its keyword labels
        assert tm.model.get_topic_info()["Name"].notna().all()


def _fake_completion_response(content):
    import types

    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def test_label_topics_with_mock_llm(monkeypatch):
    calls = {"n": 0}

    def fake_completion(**kwargs):
        calls["n"] += 1
        user_msg = kwargs["messages"][1]["content"]
        # reply with a JSON object covering the topic ids requested in the prompt
        ids = [
            part.splitlines()[0].strip()
            for part in user_msg.split("### Topic ")[1:]
        ]
        import json

        return _fake_completion_response(
            json.dumps({tid: f"llm label {tid}" for tid in ids})
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(model="mock/test", overwrite=True)
        assert len(labels) > 0
        generated = [v for v in labels.values() if v.startswith("llm label")]
        assert len(generated) > 0
        info = tm.model.get_topic_info()
        assert "LLM_label" in info.columns
        assert info["LLM_label"].notna().any()
        assert calls["n"] >= 1  # batched: far fewer calls than topics


def test_label_topics_batches_requests(monkeypatch):
    """batch_size=2 on a model with N topics -> ceil(N/2) LLM calls."""
    n_topics_seen = {"n": 0}
    import json

    def fake_completion(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        ids = [part.splitlines()[0].strip() for part in user_msg.split("### Topic ")[1:]]
        n_topics_seen["n"] += len(ids)
        return _fake_completion_response(json.dumps({tid: f"label {tid}" for tid in ids}))

    monkeypatch.setattr("litellm.completion", fake_completion)
    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        total = int((tm.model.get_topic_info()["Topic"] != -1).sum())
        tm.label_topics(model="mock/test", overwrite=True, batch_size=2)
        assert n_topics_seen["n"] == total


def test_label_topics_passes_api_base_and_key(monkeypatch):
    captured = {}

    def fake_completion(**kwargs):
        captured["model"] = kwargs["model"]
        captured["api_base"] = kwargs.get("api_base")
        captured["api_key"] = kwargs.get("api_key")
        import json

        user_msg = kwargs["messages"][1]["content"]
        ids = [part.splitlines()[0].strip() for part in user_msg.split("### Topic ")[1:]]
        return _fake_completion_response(json.dumps({tid: f"llm label {tid}" for tid in ids}))

    monkeypatch.setattr("litellm.completion", fake_completion)
    monkeypatch.delenv("LABEL_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_LIKE_API_KEY", "secret-token")

    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(
            model="openai/unsloth/Qwen3.8-27B-GGUF",
            api_base="http://localhost:8888/v1",
            overwrite=True,
        )
        assert len(labels) > 0
        assert captured["model"] == "openai/unsloth/Qwen3.8-27B-GGUF"
        assert captured["api_base"] == "http://localhost:8888/v1"
        assert captured["api_key"] == "secret-token"


def test_label_topics_skips_when_api_base_without_key(monkeypatch):
    called = {"n": 0}

    def fake_completion(**kwargs):
        called["n"] += 1
        return _fake_completion_response("{}")

    monkeypatch.setattr("litellm.completion", fake_completion)
    for var in ("LABEL_API_KEY", "OPENAI_LIKE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(model="openai/some-model", api_base="http://x/v1")
        assert labels == {}
        assert called["n"] == 0


def test_find_gguf_file_skips_mmproj_and_picks_largest(tmp_path):
    from topiclayers.core.topic import _find_gguf_file

    d = tmp_path / "snapshots" / "abc"
    d.mkdir(parents=True)
    (d / "mmproj-F16.gguf").write_bytes(b"x" * 100)          # vision projector
    (d / "model-Q4_K_M.gguf").write_bytes(b"x" * 5000)
    small = tmp_path / "other" / "tiny.gguf"                 # nested elsewhere
    small.parent.mkdir(parents=True)
    small.write_bytes(b"x" * 10)

    found = _find_gguf_file(str(tmp_path))
    assert found == str(d / "model-Q4_K_M.gguf")


def test_find_gguf_file_returns_none_without_gguf(tmp_path):
    from topiclayers.core.topic import _find_gguf_file

    assert _find_gguf_file(str(tmp_path)) is None


def test_label_topics_autoload_on_no_model_loaded(monkeypatch):
    calls = {"load": 0, "calls": 0}

    def fake_completion(**kwargs):
        calls["calls"] += 1
        if calls["calls"] == 1:
            raise RuntimeError("No model loaded. Call POST /inference/load first.")
        import json

        user_msg = kwargs["messages"][1]["content"]
        ids = [part.splitlines()[0].strip() for part in user_msg.split("### Topic ")[1:]]
        return _fake_completion_response(json.dumps({tid: f"llm label {tid}" for tid in ids}))

    monkeypatch.setattr("litellm.completion", fake_completion)

    import topiclayers.core.topic as topic_mod

    def fake_loader(model, api_base, api_key):
        calls["load"] += 1
        return True

    monkeypatch.setattr(topic_mod, "_load_openai_like_model", fake_loader)
    monkeypatch.setenv("OPENAI_LIKE_API_KEY", "secret-token")

    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(
            model="openai/unsloth/Qwen3.8-27B-GGUF",
            api_base="http://localhost:8888/v1",
            overwrite=True,
        )
        assert calls["load"] == 1
        assert len(labels) > 0
        assert any(v.startswith("llm label") for v in labels.values())


def test_label_topics_failed_batch_keeps_keyword_labels(monkeypatch):
    def fake_completion(**kwargs):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr("litellm.completion", fake_completion)

    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        keyword_names = list(tm.model.get_topic_info()["Name"])
        labels = tm.label_topics(model="mock/test", overwrite=True)
        assert labels == {}
        # keyword labels untouched
        assert list(tm.model.get_topic_info()["Name"]) == keyword_names


def test_derive_label_context():
    from topiclayers.core.topic import _derive_label_context

    assert "COP22" in _derive_label_context("cop22")
    assert "climate change" in _derive_label_context("cop22")
    assert _derive_label_context("my_dataset") == ""


def test_label_topics_max_topics_preview(monkeypatch):
    import json

    def fake_completion(**kwargs):
        user_msg = kwargs["messages"][1]["content"]
        ids = [p.splitlines()[0].strip() for p in user_msg.split("### Topic ")[1:]]
        return _fake_completion_response(json.dumps({tid: f"label {tid}" for tid in ids}))

    monkeypatch.setattr("litellm.completion", fake_completion)
    with tempfile.TemporaryDirectory() as d:
        tm = _fit_toy(d)
        labels = tm.label_topics(model="mock/test", overwrite=True, max_topics=1)
        labeled_llm = [v for v in labels.values() if v.startswith("label ")]
        assert len(labeled_llm) == 1
