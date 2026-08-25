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

    def label_topics(self):
        raise NotImplementedError("TODO: topic labeling via LLM")


def _docs_hash(docs: list[str]) -> str:
    import hashlib

    h = hashlib.sha256()
    for doc in sorted(docs):
        h.update(doc.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]
