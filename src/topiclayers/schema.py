"""Generic post schema: Post dataclass and InteractionType enum."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class InteractionType(str, Enum):
    ORIGINAL = "original"
    REPOST = "repost"
    QUOTE = "quote"
    REPLY = "reply"


@dataclass
class Post:
    post_id: str
    user_id: str
    text: str
    created_at: Optional[str] = None
    lang: Optional[str] = None
    interaction_type: Optional[InteractionType] = None
    target_post_id: Optional[str] = None
    target_user_id: Optional[str] = None
    mentions: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
