#!/usr/bin/env python3
"""Convert the Swiss politicians tweets (Twitter API v1.1 JSONL, gzipped) into
the topiclayers `csv_posts` schema plus a PRISM-compatible users JSONL.

Why a converter: topiclayers' `twitter_jsonl` adapter expects API v2 fields
(`referenced_tweets`, `author_id`, ...), while this corpus is v1.1
(`retweeted_status`, `quoted_status`, `in_reply_to_status_id_str`, `user`).

It also *synthesises* the original tweets embedded in `retweeted_status` /
`quoted_status`. The corpus is ~99% retweets, so without synthesis topiclayers
would have almost no originals to cluster and almost no retweet edges that
resolve to a target author inside the dataset.

Usage:
    python swiss_to_topiclayers.py
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[1] / "datasets" / "swiss"
SRC = DATASET / "SwissPoliticians-tweets.json.gz"
OUT_CSV = DATASET / "swiss.csv"
OUT_USERS = DATASET / "swiss_users.jsonl"

CSV_COLS = ["post_id", "user_id", "text", "created_at", "lang",
            "interaction_type", "target_post_id", "mentions"]


def _full_text(obj: dict) -> str:
    if obj.get("truncated") and isinstance(obj.get("extended_tweet"), dict):
        return obj["extended_tweet"].get("full_text") or obj.get("full_text") or obj.get("text", "")
    return obj.get("full_text") or obj.get("text", "")


def _user_id(obj: dict) -> str:
    user = obj.get("user") or {}
    return str(user.get("id_str") or user.get("id") or "")


def _mentions(obj: dict) -> str:
    ents = obj.get("entities") or {}
    return ";".join(m.get("screen_name", "") for m in ents.get("user_mentions", []) if m.get("screen_name"))


def _classify(obj: dict) -> tuple[str, str | None]:
    if obj.get("retweeted_status"):
        rt = obj["retweeted_status"]
        return "repost", str(rt.get("id_str") or rt.get("id") or "")
    if obj.get("quoted_status") or obj.get("is_quote_status"):
        q = obj.get("quoted_status") or {}
        tid = q.get("id_str") or obj.get("quoted_status_id_str")
        return ("quote", str(tid)) if tid else (None, None)
    if obj.get("in_reply_to_status_id_str") or obj.get("in_reply_to_status_id"):
        tid = obj.get("in_reply_to_status_id_str") or obj.get("in_reply_to_status_id")
        return "reply", str(tid)
    return None, None


def _row(obj: dict, post_id: str, interaction: str | None, target: str | None) -> dict:
    return {
        "post_id": post_id,
        "user_id": _user_id(obj),
        "text": _full_text(obj),
        "created_at": obj.get("created_at", ""),
        "lang": obj.get("lang", ""),
        "interaction_type": interaction or "",
        "target_post_id": target or "",
        "mentions": _mentions(obj),
    }


def _user_record(user: dict) -> dict:
    return {
        "id": str(user.get("id_str") or user.get("id") or ""),
        "username": user.get("screen_name"),
        "name": user.get("name"),
        "location": user.get("location"),
        "description": user.get("description"),
        "verified": bool(user.get("verified")),
        "protected": bool(user.get("protected")),
        "public_metrics": {
            "followers_count": user.get("followers_count"),
            "following_count": user.get("friends_count"),
            "tweet_count": user.get("statuses_count"),
            "listed_count": user.get("listed_count"),
        },
    }


def main() -> None:
    seen_posts: set[str] = set()
    parents: dict[str, dict] = {}
    users: dict[str, dict] = {}
    n_rows = n_repost = n_quote = n_reply = n_original = 0

    with gzip.open(SRC, "rt", encoding="utf-8") as fin, open(OUT_CSV, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_COLS)
        writer.writeheader()

        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            post_id = str(obj.get("id_str") or obj.get("id") or "")
            if not post_id or post_id in seen_posts:
                continue
            seen_posts.add(post_id)

            user = obj.get("user")
            if isinstance(user, dict) and _user_id(obj):
                users.setdefault(_user_id(obj), _user_record(user))

            interaction, target = _classify(obj)
            writer.writerow(_row(obj, post_id, interaction, target))

            if interaction == "repost":
                n_repost += 1
                rt = obj["retweeted_status"]
                pid = str(rt.get("id_str") or rt.get("id") or "")
                if pid and pid not in parents:
                    parents[pid] = rt
            elif interaction == "quote":
                n_quote += 1
                q = obj.get("quoted_status")
                if isinstance(q, dict):
                    pid = str(q.get("id_str") or q.get("id") or "")
                    if pid and pid not in parents:
                        parents[pid] = q
            elif interaction == "reply":
                n_reply += 1
            else:
                n_original += 1

        n_synth = 0
        for pid, obj in parents.items():
            if pid in seen_posts:
                continue
            seen_posts.add(pid)
            writer.writerow(_row(obj, pid, None, None))
            user = obj.get("user")
            if isinstance(user, dict) and _user_id(obj):
                users.setdefault(_user_id(obj), _user_record(user))
            n_synth += 1

        n_rows = len(seen_posts)

    with open(OUT_USERS, "w") as f:
        for rec in users.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # how many retweets point at a post that now exists in the table
    resolvable = 0
    total_refs = 0
    if True:
        import pandas as pd
        df = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
        ids = set(df["post_id"])
        refs = df.loc[df["target_post_id"] != "", "target_post_id"]
        total_refs = len(refs)
        resolvable = int(refs.isin(ids).sum())

    print(f"rows written        : {n_rows}")
    print(f"  originals (in-file): {n_original}")
    print(f"  originals (synth)  : {n_synth}")
    print(f"  reposts/quotes/replies: {n_repost}/{n_quote}/{n_reply}")
    print(f"users              : {len(users)}")
    print(f"refs resolvable    : {resolvable}/{total_refs} "
          f"({100*resolvable/total_refs:.1f}%)" if total_refs else "no references")
    print(f"-> {OUT_CSV}\n-> {OUT_USERS}")


if __name__ == "__main__":
    main()
