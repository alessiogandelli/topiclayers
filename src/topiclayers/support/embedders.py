"""Model-specific input treatment for sentence embedders.

Different sentence-embedding families expect different inputs: E5 wants a
``"query: "`` prefix, Qwen3-Embedding expects its instruction format, and the
MiniLM/MPNet families want plain text. Getting this wrong silently degrades
quality, so the treatment lives in one place, is derived from the model name,
can be overridden in the config, and is folded into the cache keys.

Usage::

    profile = resolve_profile("intfloat/multilingual-e5-base")
    emb = encode_documents("intfloat/multilingual-e5-base", docs, profile)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderProfile:
    """How to feed documents to a sentence embedder."""

    prompt: str = ""              # string prepended to every document
    normalize: bool = True        # L2-normalise (recommended for cosine/UMAP)
    trust_remote_code: bool = False

    def key(self) -> str:
        return f"prompt={self.prompt!r};norm={self.normalize}"


# Family defaults. E5 uses the "query: " prefix on all documents for symmetric
# (document-to-document) tasks like clustering; Qwen3 uses its instruction
# format. MiniLM/MPNet/paraphrase-* take plain text. Normalisation is on for
# every family because the pipeline clusters with cosine/UMAP.
def resolve_profile(
    model_name: str,
    prompt_override: str = "",
    normalize_override: bool | None = None,
) -> EmbedderProfile:
    """Return the input treatment for ``model_name``.

    ``prompt_override`` (non-empty) and ``normalize_override`` (not ``None``)
    take precedence over the family default.
    """
    name = model_name.lower()

    if "e5" in name:  # intfloat/multilingual-e5-*, e5-*
        base = EmbedderProfile(prompt="query: ", normalize=True)
    elif "qwen3-embedding" in name or "qwen3_embedding" in name:
        base = EmbedderProfile(
            prompt="Instruct: Retrieve semantically similar text\nQuery: ",
            normalize=True,
            trust_remote_code=True,
        )
    elif "bge" in name:  # BAAI/bge-*, bge-m3
        base = EmbedderProfile(normalize=True)
    else:  # all-MiniLM-*, all-mpnet-*, paraphrase-multilingual-*, ...
        base = EmbedderProfile(normalize=True)

    prompt = prompt_override if prompt_override else base.prompt
    normalize = base.normalize if normalize_override is None else normalize_override
    return EmbedderProfile(
        prompt=prompt,
        normalize=normalize,
        trust_remote_code=base.trust_remote_code,
    )


def pick_device(explicit: str = "") -> str | None:
    """Return the torch device string, or ``None`` to let ST choose."""
    if explicit:
        return explicit
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return None


def encode_documents(model_name: str, docs: list[str], profile: EmbedderProfile,
                     device: str = "", batch_size: int = 256):
    """Encode ``docs`` with the profile applied. Returns a numpy array."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=pick_device(device),
        trust_remote_code=profile.trust_remote_code or None,
    )
    chunks = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        chunks.append(
            model.encode(
                batch,
                show_progress_bar=False,
                normalize_embeddings=profile.normalize,
                prompt=profile.prompt or None,
            )
        )
    return np.vstack(chunks) if chunks else np.empty((0,))
