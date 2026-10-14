from __future__ import annotations

import json
import hashlib
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from topiclayers.support.cache import cache_key
from topiclayers.status import log_via
from topiclayers.schema import InteractionType, Post

_INTERACTION_REVERSE_MAP = {
    InteractionType.REPOST: "retweeted",
    InteractionType.QUOTE: "quoted",
    InteractionType.REPLY: "replied_to",
}


def _ref_type(post: Post) -> str | None:
    if post.interaction_type is None:
        return None
    return _INTERACTION_REVERSE_MAP.get(post.interaction_type, None)


class Dataset:
    def __init__(
        self,
        posts: list[Post],
        n_cop: str = "dataset",
        file_user: Optional[Path] = None,
        load_users: bool = True,
    ):
        self.posts = posts
        self.n_cop = n_cop
        self.file_user = Path(file_user) if file_user is not None else None
        self.load_users = load_users
        self.name = n_cop

        self.df_tweets: Optional[pd.DataFrame] = None
        self.df_users: Optional[pd.DataFrame] = None
        self.df_original: Optional[pd.DataFrame] = None
        self.df_retweets: Optional[pd.DataFrame] = None
        self.df_quotes: Optional[pd.DataFrame] = None
        self.df_reply: Optional[pd.DataFrame] = None
        self.df_retweets_labeled: Optional[pd.DataFrame] = None

    def process(
        self,
        cache_dir: Optional[Path] = None,
        adapter_name: str = "unknown",
        reporter=None,
    ) -> None:
        cache_dir = Path(cache_dir) if cache_dir is not None else Path("./cache")
        data_dir = cache_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        tweets_pkl = data_dir / f"tweets_{self.name}.pkl"
        users_pkl = data_dir / f"users_{self.name}.pkl"
        manifest_path = data_dir / f"manifest_{self.name}.json"

        posts_hash = hashlib.sha256(
            json.dumps([p.post_id for p in self.posts], sort_keys=True).encode()
        ).hexdigest()[:16]
        expected_key = cache_key(posts_hash, adapter_name, self.n_cop)

        cache_valid = False
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                if manifest.get("key") == expected_key:
                    cache_valid = True
            except Exception:
                pass

        if cache_valid and tweets_pkl.exists():
            self.df_tweets = pd.read_pickle(tweets_pkl)
            if users_pkl.exists():
                self.df_users = pd.read_pickle(users_pkl)
        else:
            if tweets_pkl.exists():
                log_via(reporter, "Cache key mismatch — regenerating data cache.")
            users_dict = self._load_user_dict(reporter=reporter) if self.file_user is not None else {}
            tweets_dict = {}
            for i, post in enumerate(self.posts):
                if reporter is not None:
                    reporter.update(i + 1)
                author_name = None
                if users_dict and post.user_id in users_dict:
                    author_name = users_dict[post.user_id].get("username")

                mentions_id_raw = post.extra.get("mentions_id", []) if isinstance(post.extra, dict) else []
                mentions_id = [str(m) for m in mentions_id_raw] if mentions_id_raw else []
                mentions_name = [str(m) for m in post.mentions] if post.mentions else []

                tweets_dict[post.post_id] = {
                    "author": post.user_id,
                    "author_name": author_name,
                    "text": post.text,
                    "date": post.created_at,
                    "lang": post.lang,
                    "conversation_id": post.extra.get("conversation_id") if isinstance(post.extra, dict) else None,
                    "referenced_type": _ref_type(post),
                    "referenced_id": post.target_post_id,
                    "mentions_name": mentions_name,
                    "mentions_id": mentions_id,
                    "attachments": post.extra.get("attachments") if isinstance(post.extra, dict) else None,
                    "cop": self.n_cop,
                }

            self.df_tweets = pd.DataFrame(tweets_dict).T
            self.df_tweets.index.name = "id"

            if users_dict:
                self.df_users = pd.DataFrame(users_dict).T
                self.df_users.to_csv(data_dir / f"users_{self.name}.csv")
                self.df_users.to_pickle(users_pkl)

            self.df_tweets.to_csv(data_dir / f"tweets_{self.name}.csv")
            self.df_tweets.to_pickle(tweets_pkl)

            manifest = {
                "key": expected_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "n_cop": self.n_cop,
                "n_rows": len(self.df_tweets),
                "n_users": len(self.df_users) if self.df_users is not None else 0,
                "adapter": adapter_name,
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2, default=str)

        self._create_dataframes()

    def _load_user_dict(self, reporter=None) -> dict:
        import json

        if self.file_user is None:
            return {}
        if not self.file_user.exists():
            log_via(reporter, f"User file not found: {self.file_user} — skipping user metadata.")
            return {}

        users = {}
        with open(self.file_user) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                uid = str(obj.get("id", ""))
                users[uid] = {
                    "username": obj.get("username", ""),
                    "tweet_count": obj.get("public_metrics", {}).get("tweet_count", 0),
                    "followers": obj.get("public_metrics", {}).get("followers_count", 0),
                    "following": obj.get("public_metrics", {}).get("following_count", 0),
                }
        return users

    _TWEET_COLUMNS = [
        "author", "author_name", "text", "date", "lang", "conversation_id",
        "referenced_type", "referenced_id", "mentions_name", "mentions_id",
        "attachments", "cop",
    ]

    def _create_dataframes(self) -> None:
        df = self.df_tweets
        if df is None or len(df) == 0 or "referenced_type" not in df.columns:
            # No posts (empty input) leaves df_tweets without any columns;
            # create empty, correctly-typed frames so downstream stages
            # (network building, manifest) don't crash with a KeyError.
            empty = pd.DataFrame({c: pd.Series(dtype="object") for c in self._TWEET_COLUMNS})
            empty.index.name = "id"
            if df is None or len(df) == 0:
                self.df_tweets = empty
            self.df_original = empty.copy()
            self.df_retweets = empty.copy()
            self.df_quotes = empty.copy()
            self.df_reply = empty.copy()
            return

        self.df_original = df[df["referenced_type"].isna()].copy()
        self.df_retweets = df[df["referenced_type"] == "retweeted"].copy()
        self.df_quotes = df[df["referenced_type"] == "quoted"].copy()
        self.df_reply = df[df["referenced_type"] == "replied_to"].copy()

    def update_df(self, df_labeled: pd.DataFrame, reporter=None) -> pd.DataFrame:
        if "id" in df_labeled.columns and df_labeled.index.name != "id":
            df_labeled = df_labeled.set_index("id")

        if len(df_labeled) == 0 and len(self.df_retweets) == 0:
            # Empty dataset (no posts at all): return an empty labeled frame
            # with the expected column instead of crashing on .notna().
            out = df_labeled.copy()
            if "topic" not in out.columns:
                out["topic"] = pd.Series(dtype="object")
            self.df_retweets_labeled = out
            return out

        df_retweets_copy = self.df_retweets.copy()
        df_retweets_copy["topic"] = df_retweets_copy["referenced_id"]
        self.df_retweets_labeled = pd.concat([df_labeled, df_retweets_copy])

        topic_dict = self.df_retweets_labeled["topic"].to_dict()

        resolved: dict = {}
        unresolved_count = 0
        for idx, value in topic_dict.items():
            visited: set = set()
            current = idx
            val = value
            while isinstance(val, str):
                if val in visited:
                    val = None
                    unresolved_count += 1
                    break
                visited.add(val)
                if val not in topic_dict:
                    unresolved_count += 1
                    val = None
                    break
                val = topic_dict[val]
            resolved[idx] = val

        self.df_retweets_labeled["topic"] = self.df_retweets_labeled.index.map(resolved)
        log_via(reporter, f"Unresolved references: {unresolved_count}")
        if unresolved_count > 0:
            log_via(reporter, f"  {unresolved_count} retweets could not be resolved to a topic. These will be excluded from the labeled output.")

        self.df_retweets_labeled = self.df_retweets_labeled[
            self.df_retweets_labeled["topic"].apply(lambda x: not isinstance(x, str))
        ]
        self.df_retweets_labeled = self.df_retweets_labeled[self.df_retweets_labeled["topic"].notna()]

        return self.df_retweets_labeled

    def _save_labeled_dataframe(self, cache_dir: Optional[Path] = None) -> None:
        cache_dir = Path(cache_dir) if cache_dir is not None else Path("./cache")
        data_dir = cache_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        labeled_file = data_dir / f"retweet_labeled_{self.name}"
        self.df_retweets_labeled.to_csv(f"{labeled_file}.csv")
        self.df_retweets_labeled.to_pickle(f"{labeled_file}.pkl")
