from __future__ import annotations

import json
from pathlib import Path
from typing import List

from topiclayers.adapters.registry import register
from topiclayers.schema import InteractionType, Post


def _map_interaction_type(ref_type: str | None) -> InteractionType | None:
    if ref_type == "retweeted":
        return InteractionType.REPOST
    elif ref_type == "quoted":
        return InteractionType.QUOTE
    elif ref_type == "replied_to":
        return InteractionType.REPLY
    return None


def load(path: Path) -> List[Post]:
    posts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            post_id = str(obj.get("id", ""))
            user_id = str(obj.get("author_id", ""))
            text = obj.get("text", "")
            created_at = obj.get("created_at")
            lang = obj.get("lang")

            ref_tweets = obj.get("referenced_tweets")
            if ref_tweets and isinstance(ref_tweets, list) and len(ref_tweets) > 0:
                ref = ref_tweets[0]
                interaction_type = _map_interaction_type(ref.get("type"))
                target_post_id = str(ref.get("id")) if ref.get("id") is not None else None
            else:
                interaction_type = None
                target_post_id = None

            entities = obj.get("entities", {})
            mentions_raw = entities.get("mentions", []) if isinstance(entities, dict) else []
            mentions = [m.get("username", "") for m in mentions_raw if isinstance(m, dict)]

            extra = {
                "conversation_id": str(obj.get("conversation_id")) if obj.get("conversation_id") is not None else None,
                "context_annotations": obj.get("context_annotations", []),
                "public_metrics": obj.get("public_metrics", {}),
                "possibly_sensitive": obj.get("possibly_sensitive"),
                "edit_history_tweet_ids": obj.get("edit_history_tweet_ids", []),
            }

            post = Post(
                post_id=post_id,
                user_id=user_id,
                text=text,
                created_at=created_at,
                lang=lang,
                interaction_type=interaction_type,
                target_post_id=target_post_id,
                mentions=mentions,
                extra=extra,
            )
            posts.append(post)

    return posts


register("twitter_jsonl", load)
