from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from topiclayers.adapters.registry import register
from topiclayers.schema import InteractionType, Post

_INTERACTION_MAP = {
    "original": None,
    "repost": InteractionType.REPOST,
    "quote": InteractionType.QUOTE,
    "reply": InteractionType.REPLY,
}


def _parse_interaction(value) -> InteractionType | None:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s or s in ("nan", "none", ""):
        return None
    return _INTERACTION_MAP.get(s, None)


def _parse_mentions(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return []
    return [m.strip() for m in s.split(";") if m.strip()]


def _row_to_post(row, extra_cols: list[str]) -> Post:
    extra = {}
    for col in extra_cols:
        val = row.get(col)
        if isinstance(val, float) and pd.isna(val):
            val = None
        key = col.removeprefix("extra_")
        extra[key] = val

    raw_text = row.get("text", "")
    if isinstance(raw_text, float) and pd.isna(raw_text):
        raw_text = ""

    return Post(
        post_id=str(row["post_id"]),
        user_id=str(row["user_id"]),
        text=str(raw_text),
        created_at=row.get("created_at") if not (isinstance(row.get("created_at"), float) and pd.isna(row.get("created_at"))) else None,
        lang=row.get("lang") if not (isinstance(row.get("lang"), float) and pd.isna(row.get("lang"))) else None,
        interaction_type=_parse_interaction(row.get("interaction_type")),
        target_post_id=str(row["target_post_id"]) if row.get("target_post_id") and not (isinstance(row.get("target_post_id"), float) and pd.isna(row.get("target_post_id"))) else None,
        mentions=_parse_mentions(row.get("mentions")),
        extra=extra,
    )


def _strings_only(df: pd.DataFrame) -> pd.DataFrame:
    """Force every column to string dtype, mapping missing values to ''.

    Ids must never be numeric: pandas otherwise infers ``target_post_id`` as
    float when the column mixes large integer ids with blanks, silently
    turning ``1622232152543821824`` into ``1.6222201547708006e+18`` and
    breaking every reference (empty networks).
    """
    for col in df.columns:
        df[col] = df[col].astype("string").fillna("")
    return df


def load_csv(path: Path) -> List[Post]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"post_id", "user_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV file at {path} is missing required columns: {missing}. "
            f"Required columns are: post_id, user_id, text. "
            f"Optional: created_at, lang, interaction_type, target_post_id, target_user_id, mentions, and extra_* columns."
        )
    extra_cols = [c for c in df.columns if c.startswith("extra_")]
    return [_row_to_post(row, extra_cols) for _, row in df.iterrows()]


def load_parquet(path: Path) -> List[Post]:
    df = _strings_only(pd.read_parquet(path))
    required = {"post_id", "user_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Parquet file at {path} is missing required columns: {missing}. "
            f"Required columns are: post_id, user_id, text."
        )
    extra_cols = [c for c in df.columns if c.startswith("extra_")]
    return [_row_to_post(row, extra_cols) for _, row in df.iterrows()]


register("csv_posts", load_csv)
register("parquet_posts", load_parquet)
