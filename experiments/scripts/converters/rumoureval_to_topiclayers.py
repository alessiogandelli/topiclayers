#!/usr/bin/env python3
"""Convert the RumourEval-2019 Twitter data into the topiclayers `csv_posts`
schema plus a PRISM-compatible users JSONL.

The archive ships per-thread folders:

    twitter-en-<split>-data/<event>/<thread_id>/source-tweet/<id>.json
                                              /replies/<id>.json
                                              /structure.json

Each tweet is Twitter API v1.1. Source tweets are the thread roots (originals);
replies keep their `in_reply_to_status_id_str` as `target_post_id`, so reply
edges resolve inside the dataset. Reddit data is ignored.

Usage:
    python rumoureval_to_topiclayers.py
"""
from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE.parents[1] / "datasets" / "rumoureval"
SRC = DATASET / "rumoureval2019.tar.bz2"
OUT_CSV = DATASET / "rumoureval.csv"
OUT_USERS = DATASET / "rumoureval_users.jsonl"

CSV_COLS = ["post_id", "user_id", "text", "created_at", "lang",
            "interaction_type", "target_post_id", "mentions"]


def _full_text(obj: dict) -> str:
    if obj.get("truncated") and isinstance(obj.get("extended_tweet"), dict):
        return obj["extended_tweet"].get("full_text") or obj.get("text", "")
    return obj.get("full_text") or obj.get("text", "")


def _user_id(obj: dict) -> str:
    user = obj.get("user") or {}
    return str(user.get("id_str") or user.get("id") or "")


def _mentions(obj: dict) -> str:
    ents = obj.get("entities") or {}
    return ";".join(m.get("screen_name", "") for m in ents.get("user_mentions", []) if m.get("screen_name"))


def _classify(obj: dict, is_source: bool) -> tuple[str, str | None]:
    if is_source:
        return None, None
    if obj.get("retweeted_status"):
        rt = obj["retweeted_status"]
        return "repost", str(rt.get("id_str") or rt.get("id") or "")
    if obj.get("quoted_status") or obj.get("is_quote_status"):
        q = obj.get("quoted_status") or {}
        tid = q.get("id_str") or obj.get("quoted_status_id_str")
        return ("quote", str(tid)) if tid else (None, None)
    if obj.get("in_reply_to_status_id_str") or obj.get("in_reply_to_status_id"):
        return "reply", str(obj.get("in_reply_to_status_id_str") or obj.get("in_reply_to_status_id"))
    return None, None


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


def _iter_tweets():
    with tarfile.open(SRC, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".zip"):
                continue
            zf = zipfile.ZipFile(io.BytesIO(tar.extractfile(member).read()))
            for name in zf.namelist():
                if "twitter" not in name or not name.endswith(".json") or name.endswith("structure.json"):
                    continue
                try:
                    obj = json.loads(zf.read(name).decode("utf-8", "replace"))
                except (json.JSONDecodeError, KeyError):
                    continue
                if not isinstance(obj, dict) or "id" not in obj:
                    continue
                yield obj, "/source-tweet/" in name


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    # topiclayers only propagates topic labels (and builds the multilayer
    # retweet network) from `repost` rows -- replies/quotes are dropped at the
    # network stage.  This corpus is a reply thread dataset, so we optionally
    # re-tag replies as reposts so their edges form an interaction network.
    ap.add_argument("--keep-replies", action="store_true",
                    help="keep replies as `reply` (network will be empty)")
    args = ap.parse_args()
    replies_as_reposts = not args.keep_replies

    seen: set[str] = set()
    users: dict[str, dict] = {}
    counts = {"original": 0, "repost": 0, "quote": 0, "reply": 0}

    with open(OUT_CSV, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=CSV_COLS)
        writer.writeheader()
        for obj, is_source in _iter_tweets():
            post_id = str(obj.get("id_str") or obj.get("id") or "")
            if not post_id or post_id in seen:
                continue
            seen.add(post_id)
            user = obj.get("user")
            if isinstance(user, dict) and _user_id(obj):
                users.setdefault(_user_id(obj), _user_record(user))

            interaction, target = _classify(obj, is_source)
            counts[interaction or "original"] += 1
            if interaction == "reply" and replies_as_reposts:
                # consumed by topiclayers' repost network (see arg help)
                interaction = "repost"
            writer.writerow({
                "post_id": post_id,
                "user_id": _user_id(obj),
                "text": _full_text(obj),
                "created_at": obj.get("created_at", ""),
                "lang": obj.get("lang", ""),
                "interaction_type": interaction or "",
                "target_post_id": target or "",
                "mentions": _mentions(obj),
            })

    with open(OUT_USERS, "w") as f:
        for rec in users.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    import pandas as pd
    df = pd.read_csv(OUT_CSV, dtype=str, keep_default_na=False)
    ids = set(df["post_id"])
    refs = df.loc[df["target_post_id"] != "", "target_post_id"]
    resolvable = int(refs.isin(ids).sum())

    print(f"rows written       : {len(df)}")
    print(f"  original/repost/quote/reply: {counts['original']}/{counts['repost']}/{counts['quote']}/{counts['reply']}")
    print(f"users              : {len(users)}")
    print(f"refs resolvable    : {resolvable}/{len(refs)} "
          f"({100*resolvable/len(refs):.1f}%)" if len(refs) else "no references")
    print(f"-> {OUT_CSV}\n-> {OUT_USERS}")


if __name__ == "__main__":
    main()
