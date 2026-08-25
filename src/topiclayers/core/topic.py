from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from topiclayers.support.cache import cache_key


class TopicModeler:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        embedder_name: str = "all-MiniLM-L6-v2",
        path_cache: Path = Path("./cache"),
        name: str = "dataset",
    ):
        self.df = df.copy()
        self.embedder_name = embedder_name
        self.name = name
        self.path_cache = Path(path_cache)

        self.tm_path = self.path_cache / "tm" / self.embedder_name
        self.df_labeled_path = self.tm_path / f"tweets_{self.name}_labeled.pkl"
        self.model_path = self.tm_path / f"model_{self.name}"
        self.embeddings_path = self.tm_path / f"embeddings_{self.name}.pkl"
        self.topic_path = self.tm_path / f"topics_{self.name}.csv"

        self.model = None
        self.embeddings = None
        self.docs = None

    def get_topics(
        self,
        min_topic_size: int = 50,
        nr_topics: str = "auto",
        random_state: int = 42,
        umap_model: Optional[object] = None,
        hdbscan_model: Optional[object] = None,
        reporter=None,
        exact_probabilities: bool = False,
        repro_mode: str = "strict",
        # "pca" (seeded) is deterministic AND ~20x faster than "spectral";
        # pass umap_init="spectral" explicitly to restore the old behaviour
        umap_init: str = "pca",
    ) -> pd.DataFrame:
        from topiclayers.support.repro import set_global_seed

        set_global_seed(random_state)

        self.tm_path.mkdir(parents=True, exist_ok=True)

        manifest_path = self.tm_path / f"manifest_{self.name}.json"
        docs_hash = _docs_hash(self.df["text"].tolist())
        expected_key = cache_key(
            docs_hash, self.embedder_name, str(min_topic_size),
            str(nr_topics), str(random_state),
            f"exact_prob={exact_probabilities}", f"repro={repro_mode}",
            f"umap_init={umap_init}",
        )

        if self.df_labeled_path.exists() and self.model_path.exists():
            cache_valid = False
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        manifest = json.load(f)
                    if manifest.get("key") == expected_key and manifest.get("status") == "success":
                        cache_valid = True
                except Exception:
                    pass
            if cache_valid:
                print("using cached topics")
                self.df = pd.read_pickle(self.df_labeled_path)
                from bertopic import BERTopic
                self.model = BERTopic.load(self.model_path)
                if "new_text" in self.df.columns:
                    self.docs = self.df["new_text"].tolist()
                return self.df

        print("running topic modeling")
        docs = self._preprocess()

        if self.embeddings_path.exists():
            embed_manifest = self.tm_path / f"embeddings_manifest_{self.name}.json"
            embed_key = cache_key(_docs_hash(docs), self.embedder_name)
            use_cached = False
            if embed_manifest.exists():
                try:
                    with open(embed_manifest) as f:
                        em = json.load(f)
                    if em.get("key") == embed_key:
                        use_cached = True
                except Exception:
                    pass
            if use_cached:
                print("  loading embeddings from cache")
                with open(self.embeddings_path, "rb") as f:
                    self.embeddings = pickle.load(f)
            else:
                self._get_embeddings(docs, reporter)
        else:
            self._get_embeddings(docs, reporter)

        self._use_BERTopic(
            docs, nr_topics, min_topic_size, random_state,
            umap_model, hdbscan_model, reporter,
            exact_probabilities=exact_probabilities, repro_mode=repro_mode,
            umap_init=umap_init,
        )

        self._save_qdrant(docs, reporter)
        self._save_files()

        manifest = {
            "key": expected_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "min_topic_size": min_topic_size,
            "nr_topics": nr_topics,
            "random_state": random_state,
            "embedder": self.embedder_name,
            "n_rows": len(self.df),
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        return self.df

    def _preprocess(self) -> list[str]:
        self.df["new_text"] = self.df["text"]
        self.df["new_text"] = self.df["new_text"].str.replace(r"http\S+", "", regex=True)
        self.df["new_text"] = self.df["new_text"].str.replace(r"@\S+", "", regex=True)
        self.df["new_text"] = self.df["new_text"].str.replace(r"\n", "", regex=True)
        self.df["new_text"] = self.df["new_text"].str.strip()
        self.df = self.df[self.df["new_text"] != ""]
        self.df.reset_index(drop=False, inplace=True)
        docs = self.df["new_text"].tolist()
        self.docs = docs
        return docs

    def _get_embeddings(self, docs: list[str], reporter=None) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if reporter is not None:
            reporter.stage("embed", total=len(docs))

        if self.embedder_name.startswith("text-embedding"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OpenAI embedder selected but OPENAI_API_KEY is not set. "
                    "Set it in a .env file or use a SentenceTransformer model like 'all-MiniLM-L6-v2' instead."
                )

            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            batch_size = 1000
            all_embeddings = []

            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                response = client.embeddings.create(input=batch, model=self.embedder_name)
                all_embeddings.extend([np.array(d.embedding) for d in response.data])
                if reporter is not None:
                    reporter.update(min(i + batch_size, len(docs)))

            self.embeddings = np.array(all_embeddings)
            self.embedder = None
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.embedder_name)
            batch_size = 256
            all_embeddings = []

            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                batch_emb = model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_emb)
                if reporter is not None:
                    reporter.update(min(i + batch_size, len(docs)))

            self.embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0,))
            self.embedder = model

        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.embeddings, f)

        embed_manifest = self.tm_path / f"embeddings_manifest_{self.name}.json"
        embed_key = cache_key(_docs_hash(docs), self.embedder_name)
        with open(embed_manifest, "w") as f:
            json.dump({"key": embed_key, "embedder": self.embedder_name}, f)

    def _use_BERTopic(
        self,
        docs: list[str],
        nr_topics: str = "auto",
        min_topic_size: int = 50,
        random_state: int = 42,
        umap_model: Optional[object] = None,
        hdbscan_model: Optional[object] = None,
        reporter=None,
        exact_probabilities: bool = False,
        repro_mode: str = "strict",
        umap_init: str = "pca",
    ) -> None:
        import random

        from umap import UMAP
        from sklearn.cluster import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer

        if reporter is not None:
            reporter.stage("topic_model")

        np.random.seed(random_state)
        random.seed(random_state)
        try:
            import torch
            torch.manual_seed(random_state)
        except ImportError:
            pass

        if not docs or len(docs) == 0:
            raise ValueError(
                "No documents for topic modeling. "
                "The DataFrame is empty after preprocessing. "
                "This may happen if all posts have URLs/mentions as their only text. "
                "Check the 'text' column of your input."
            )

        if len(docs) < 10:
            vectorizer_model = CountVectorizer(stop_words="english", min_df=1, max_df=1.0)
        else:
            min_df_value = max(1, min(5, len(docs) // 100))
            vectorizer_model = CountVectorizer(
                stop_words="english", min_df=min_df_value, max_df=0.95
            )
        ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

        fast = repro_mode == "fast"
        if umap_model is None:
            n_neighbors = min(15, max(2, len(docs) - 1))
            n_components = min(5, len(docs) - 1)
            if n_components + 1 >= len(docs):
                n_components = max(1, len(docs) - 3)
            umap_params = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "min_dist": 0.0,
                "metric": "cosine",
                "init": umap_init,
            }
            if fast:
                # a seeded UMAP forces single-threaded execution; fast mode
                # trades reproducibility for parallelism
                umap_params.update({"random_state": None, "n_jobs": -1})
            else:
                umap_params["random_state"] = random_state
            umap_model = UMAP(**umap_params)
        else:
            fast = False

        if hdbscan_model is None:
            min_cluster_size = max(2, min_topic_size)
            hdbscan_kwargs = {"n_jobs": -1} if fast else {}
            hdbscan_model = HDBSCAN(
                min_samples=max(1, min_topic_size // 5),
                min_cluster_size=min_cluster_size,
                **hdbscan_kwargs,
            )

        model = BERTopic(
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            # exact probabilities require the full membership matrix (slow on
            # large corpora); the default computes the probability of the
            # assigned topic only — topics themselves are identical
            calculate_probabilities=exact_probabilities,
            verbose=True,
        )

        try:
            topics, probs = model.fit_transform(docs, embeddings=self.embeddings)
        except Exception as e:
            self._write_failed_manifest(min_topic_size, nr_topics, random_state, str(e))
            raise RuntimeError(
                f"Topic modeling failed with error: {e}. "
                f"Try reducing min_topic_size below current val ue ({min_topic_size}). "
                f"See docs at https://maartengr.github.io/BERTopic/faq.html"
            ) from e

        unique_topics = len(set(topics))
        if unique_topics <= 1:
            self._write_failed_manifest(min_topic_size, nr_topics, random_state, "All topics are -1 (outlier)")
            raise RuntimeError(
                f"All documents assigned to topic -1 (outlier). "
                f"This means BERTopic found no clusters. "
                f"Common causes: "
                f"<1> min_topic_size too high (current: {min_topic_size}) — try half of it. "
                f"<2> dataset too small — need at least a few hundred diverse posts. "
                f"<3> all documents are too similar. "
                f"See docs at https://maartengr.github.io/BERTopic/faq.html"
            )

        self.df["topic"] = topics
        if hasattr(probs, "ndim") and probs.ndim == 2:
            self.df["topic_prob"] = probs.max(axis=1)
        else:
            self.df["topic_prob"] = probs

        topic_info = model.get_topic_info()
        self.topic_path.parent.mkdir(parents=True, exist_ok=True)
        topic_info.to_csv(self.topic_path)
        self.model = model

    def _write_failed_manifest(self, min_topic_size: int, nr_topics: str, random_state: int, error: str) -> None:
        self.tm_path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.tm_path / f"manifest_{self.name}.json"
        manifest = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "error": error,
            "params": {
                "min_topic_size": min_topic_size,
                "nr_topics": nr_topics,
                "random_state": random_state,
                "embedder": self.embedder_name,
            },
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

    def _save_qdrant(self, docs: list[str], reporter=None) -> None:
        qdrant_url = os.getenv("QDRANT_URL")
        if not qdrant_url:
            print("Qdrant not configured — skipping. Set QDRANT_URL in .env to enable vector search.")
            return

        try:
            from qdrant_client import QdrantClient, models as qmodels

            client = QdrantClient(url=qdrant_url)
            ids = self.df.index.tolist()
            vectors = self.embeddings.tolist()
            topics = self.df["topic"].tolist()
            probs = self.df["topic_prob"].tolist()
            docs = self.docs

            try:
                client.create_collection(
                    collection_name=self.name,
                    vectors_config=qmodels.VectorParams(
                        size=len(self.embeddings[0]),
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            except Exception:
                pass

            points = [
                qmodels.PointStruct(
                    id=int(idx),
                    vector=vector,
                    payload={"text": text, "topic": topic, "prob": prob},
                )
                for idx, vector, text, topic, prob in zip(ids, vectors, docs, topics, probs)
            ]
            client.upload_points(self.name, points)
            if reporter is not None:
                reporter.update(len(points), total=len(points))
        except Exception as e:
            print(f"Qdrant upload failed (non-fatal): {type(e).__name__}: {e}")
            print("The pipeline continues — networks and topic labels are unaffected.")

    def _save_files(self) -> None:
        self.df_labeled_path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_pickle(self.df_labeled_path)
        if self.model is not None:
            self.model.save(
                self.model_path,
                serialization="safetensors",
                save_ctfidf=True,
                save_embedding_model=self.embedder_name,
            )

    def label_topics(
        self,
        model: str = "ollama/qwen2.5:7b",
        n_representative_docs: int = 3,
        overwrite: bool = False,
        api_base: str = "",
        batch_size: int = 40,
        label_context: str = "",
        max_topics: int = 0,
    ) -> dict[int, str]:
        """Label topics with an LLM, batching many topics per request.

        Non-fatal by design: any failure (missing litellm, unreachable Ollama,
        provider error) prints a warning and returns {} so the pipeline can
        continue with the c-TF-IDF labels. Chunks whose request or JSON parse
        fails keep their keyword labels; the rest are labeled.

        Args:
            model: LiteLLM model string, e.g. "ollama/qwen2.5:7b",
                "gpt-4o-mini", "anthropic/claude-3-haiku-20240307". For a custom
                OpenAI-compatible endpoint use "openai/<model-name>" together
                with api_base.
            n_representative_docs: docs per topic passed to the prompt.
            overwrite: re-label even if labels were already generated.
            api_base: base URL of an OpenAI-compatible endpoint (e.g.
                "http://localhost:8888/v1"). API key resolution order:
                LABEL_API_KEY -> OPENAI_LIKE_API_KEY -> OPENAI_API_KEY.
            batch_size: topics per LLM call (~40 keeps prompts well within
                small-context models while cutting request count ~40x).
            label_context: dataset-level context sentence describing what all
                documents have in common, so the LLM labels the *subtopic*
                rather than the general domain. Derived from the dataset name
                when it matches "cop<number>"; empty otherwise.
            max_topics: label only the N largest topics (0 = all). Useful for
                a cheap preview before running the full labeling.
        """
        import http.client
        import json as _json
        import socket

        api_base = api_base or os.getenv("LABEL_API_BASE", "")

        if self.model is None:
            print("  No fitted topic model — skipping LLM labeling.")
            return {}

        topic_info = self.model.get_topic_info()
        if "LLM_label" in topic_info.columns and not overwrite:
            print("  LLM labels already present — skipping (use overwrite=True).")
            return {}

        try:
            import litellm  # noqa: F401
        except ImportError:
            print(
                "  LLM labeling skipped: 'litellm' is not installed.\n"
                "    Install it with: pip install 'topiclayers[labeling]'\n"
                "    Topics keep their c-TF-IDF keyword labels."
            )
            return {}

        if model.startswith("ollama/"):
            host = os.getenv("OLLAMA_HOST", "localhost")
            port = int(os.getenv("OLLAMA_PORT", "11434"))
            try:
                conn = http.client.HTTPConnection(host, port, timeout=2)
                conn.request("GET", "/api/tags")
                conn.getresponse().read()
                conn.close()
            except (OSError, socket.timeout):
                print(
                    f"  LLM labeling skipped: Ollama not reachable at {host}:{port}.\n"
                    f"    Start it with: ollama serve   (then pull the model:\n"
                    f"    ollama pull {model.split('/', 1)[1]})\n"
                    f"    Or point TOPIC_LLM to another model (e.g. gpt-4o-mini)."
                )
                return {}

        generator_kwargs = {}
        if api_base:
            api_key = (
                os.getenv("LABEL_API_KEY")
                or os.getenv("OPENAI_LIKE_API_KEY")
                or os.getenv("OPENAI_API_KEY")
            )
            if not api_key:
                print(
                    "  LLM labeling skipped: custom api_base set but no API key found.\n"
                    "    Set LABEL_API_KEY (or OPENAI_LIKE_API_KEY / OPENAI_API_KEY) in .env."
                )
                return {}
            generator_kwargs = {"api_base": api_base, "api_key": api_key}

        context = label_context or _derive_label_context(self.name)

        # collect per-topic prompt material from the fitted model
        entries = []  # (topic_id, keywords_str, docs_str)
        for _, row in topic_info.iterrows():
            tid = int(row["Topic"])
            if tid == -1:
                continue
            keywords = ", ".join(list(row["Representation"])[:8])
            docs_list = row.get("Representative_Docs")
            if isinstance(docs_list, str):
                try:
                    docs_list = eval(docs_list)
                except Exception:
                    docs_list = [docs_list]
            if not isinstance(docs_list, list):
                docs_list = []
            docs_list = [str(d)[:280] for d in docs_list[:n_representative_docs]]
            docs_str = "\n".join(f"  {i+1}. {d}" for i, d in enumerate(docs_list)) or "  (none)"
            entries.append((tid, keywords, docs_str))

        if not entries:
            print("  No non-outlier topics to label.")
            return {}

        if max_topics and max_topics < len(entries):
            entries = entries[:max_topics]  # topic_info is sorted by size
            print(f"  Preview mode: labeling only the {max_topics} largest topics.")

        system_msg = (
            "You annotate discussion topics from social media.\n"
            + (f"All documents are {context}.\n" if context else "")
            + (
                "For each numbered topic below, generate ONE concise label "
                "(max 5 words) that describes the SPECIFIC subtopic being "
                "discussed WITHIN this general domain — never the domain itself.\n"
                "Labels like \"climate change\", \"environment\", \"social media "
                "discussion\" are WRONG because they describe the domain, not the "
                "specific subtopic.\n"
                "Write each label in the dominant language of that topic's "
                "documents.\n"
                "Reply ONLY with a single JSON object mapping every topic id to "
                'its label, e.g. {"0": "...", "7": "..."} — no other text.'
            )
        )

        def call_chunk(chunk):
            blocks = []
            for tid, keywords, docs_str in chunk:
                blocks.append(f"### Topic {tid}\nKeywords: {keywords}\nDocuments:\n{docs_str}")
            user_msg = "\n\n".join(blocks)
            resp = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                **generator_kwargs,
            )
            text = resp.choices[0].message.content.strip()
            # tolerate markdown fences around the JSON object
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = _json.loads(text)
            return {int(k): str(v).strip() for k, v in parsed.items()}

        llm_labels: dict[int, str] = {}
        n_chunks = (len(entries) + batch_size - 1) // batch_size
        failed = 0
        for i in range(0, len(entries), batch_size):
            chunk = entries[i : i + batch_size]
            for attempt in (1, 2):
                try:
                    llm_labels.update(call_chunk(chunk))
                    break
                except Exception as e:
                    # Unsloth Studio-style servers require an explicit model
                    # load; resolve the local GGUF path and load once, then retry
                    if api_base and "no model loaded" in str(e).lower():
                        _load_openai_like_model(model, api_base, generator_kwargs.get("api_key", ""))
                        continue
                    if attempt == 1:
                        print(f"  Batch {i // batch_size + 1}/{n_chunks}: {type(e).__name__}, retrying once.")
                        continue
                    failed += 1
                    print(
                        f"  Batch {i // batch_size + 1}/{n_chunks} failed (non-fatal): "
                        f"{type(e).__name__}: {str(e)[:120]} — those topics keep keyword labels."
                    )

        if not llm_labels:
            print("  No labels generated — topics keep their c-TF-IDF keyword labels.")
            return {}

        # expose labels as an aspect column named LLM_label in get_topic_info()
        self.model.topic_aspects_["LLM_label"] = {
            tid: [label] for tid, label in llm_labels.items()
        }

        info = self.model.get_topic_info()
        labels = {}
        for _, row in info.iterrows():
            tid = int(row["Topic"])
            name = row.get("LLM_label")
            if name is None or (isinstance(name, float) and pd.isna(name)):
                name = row["Name"]
            if isinstance(name, (list, tuple)):
                name = " ".join(str(x) for x in name)
            labels[tid] = str(name)

        self.topic_path.parent.mkdir(parents=True, exist_ok=True)
        info.to_csv(self.topic_path)
        ok = len(llm_labels)
        suffix = f", {failed} batch(es) failed" if failed else ""
        print(f"  Labeled {ok} topics with '{model}' in {n_chunks} batch(es){suffix}.")
        return labels


def _derive_label_context(name: str) -> str:
    """Dataset-level context sentence for LLM labeling.

    Recognizes COP dataset names (cop21, cop22, ...) and describes the shared
    domain, so the model labels the specific subtopic instead of it.
    """
    import re

    m = re.fullmatch(r"cop(\d+)", str(name).strip().lower())
    if m:
        return (
            f"social media posts about climate change and the "
            f"COP{m.group(1)} UN Climate Change Conference"
        )
    return ""


def _find_gguf_file(model_dir: str) -> str | None:
    """Find the main GGUF file under a model directory (largest one, skipping
    mmproj/vision projections)."""
    from pathlib import Path as _Path

    candidates = [
        p for p in _Path(model_dir).rglob("*.gguf")
        if "mmproj" not in p.name.lower()
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: p.stat().st_size))


def _load_openai_like_model(model: str, api_base: str, api_key: str) -> bool:
    """Try to load a model on an Unsloth Studio-style OpenAI-compatible server.

    Returns True if the model appears loaded afterwards. Best-effort only:
    any failure returns False (labeling stays non-fatal).
    """
    import json
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    bare = model.split("/", 1)[1] if model.startswith("openai/") else model
    root = api_base.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]

    def request(path: str, payload=None, timeout: int = 1800):
        req = urllib.request.Request(
            f"{root}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
        return json.loads(body) if body.strip().startswith(("{", "[")) else body

    try:
        # locate the local model entry and its GGUF file
        models = request("/api/models/local")
        entry = next(
            (
                m for m in models.get("models", [])
                if m.get("id") == bare or m.get("model_id") == bare
            ),
            None,
        )
        model_path = entry.get("path") if entry else None
        gguf = _find_gguf_file(model_path) if model_path else None

        load_payload = {"model": bare}
        if gguf:
            load_payload["model_path"] = gguf
        elif not model_path:
            print(f"  Model '{bare}' not found on the server — cannot auto-load.")
            return False

        print(f"  Loading model '{bare}' on the server...")
        request("/v1/load", load_payload)
        print(f"  Model '{bare}' loaded.")
        return True
    except Exception as e:
        print(f"  Auto-load failed: {type(e).__name__}: {e}")
        return False


def _docs_hash(docs: list[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for doc in sorted(docs):
        h.update(doc.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]
