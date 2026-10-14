from __future__ import annotations

import pytest

from topiclayers.support.embedders import EmbedderProfile, resolve_profile


def test_minilm_profile_has_no_prompt():
    p = resolve_profile("all-MiniLM-L6-v2")
    assert p.prompt == ""
    assert p.normalize is True
    assert p.trust_remote_code is False


def test_e5_gets_query_prefix():
    for name in ("intfloat/multilingual-e5-base", "intfloat/multilingual-e5-small"):
        p = resolve_profile(name)
        assert p.prompt == "query: "
        assert p.normalize is True


def test_qwen3_gets_instruction_and_trust_remote_code():
    p = resolve_profile("Qwen/Qwen3-Embedding-0.6B")
    assert p.prompt.startswith("Instruct:")
    assert p.normalize is True
    assert p.trust_remote_code is True


def test_overrides_take_precedence():
    p = resolve_profile("intfloat/multilingual-e5-base", prompt_override="passage: ",
                        normalize_override=False)
    assert p.prompt == "passage: "
    assert p.normalize is False


def test_profile_key_distinguishes_treatments():
    a = resolve_profile("intfloat/multilingual-e5-base")
    b = resolve_profile("intfloat/multilingual-e5-base", normalize_override=False)
    assert a.key() != b.key()
    assert EmbedderProfile().key() == resolve_profile("all-MiniLM-L6-v2").key()


def test_unknown_model_is_plain_text():
    p = resolve_profile("some/random-model")
    assert p.prompt == ""
    assert p.normalize is True
