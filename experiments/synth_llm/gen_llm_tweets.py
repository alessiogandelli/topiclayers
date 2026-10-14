#!/usr/bin/env python3
"""LLM-generated synthetic climate/politics tweet corpus for topiclayers.

Uses ``gpt-6-luna`` (OpenAI, via litellm) to write realistic, varied tweets
across subjects designed to stress a topic model:

* **clear**    - well-separated topics that should cluster cleanly.
* **tricky**   - deliberately confusable neighbour topics (shared vocabulary,
                 different focus) that probe topic boundaries.
* **bridge**   - posts that mix two neighbour topics, to see where they land.
* **outlier**  - many small, mutually-unrelated off-topic subjects; these should
                 mostly be rejected into the ``-1`` outlier topic.

Reposts / quotes / replies are generated programmatically from the originals so
the interaction networks and topic propagation work, and they inherit the
target's ground-truth topic.

Outputs (in ``--out-dir``):
  <name>.jsonl        - twitter_jsonl (API v2) feed
  <name>.csv          - csv_posts feed (same records)
  truth_<name>.csv    - ground truth keyed by post id (true_topic, true_family,
                        true_stance, is_derived, target_id, author_id)

Raw LLM responses are cached under ``<cache-dir>/`` so re-runs are cheap; pass
``--fresh`` to ignore the cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROMPT_VERSION = "v1"
MODEL = os.getenv("GEN_MODEL", "gpt-6-luna")

# --------------------------------------------------------------------- topics
# desc: what the topic is about (given to the LLM).
# stance: the range of viewpoints to cover, all on the "climate action" axis.

TOPICS = [
    {
        "id": "cop_negotiations",
        "family": "clear",
        "count": 250,
        "desc": ("UN COP climate summit negotiations: plenary deadlocks, the "
                 "presidency, draft cover decisions, Article 6 carbon-market "
                 "rules, the loss-and-damage fund, adaptation finance and "
                 "national delegations horse-trading."),
        "stance": "negotiators and observers who are hopeful, frustrated, or critical of the outcome",
    },
    {
        "id": "renewables",
        "family": "clear",
        "count": 250,
        "desc": ("Deployment of solar panels, wind farms, home batteries and heat "
                 "pumps: record installations, falling costs, manufacturing "
                 "booms and household savings."),
        "stance": "enthusiastic adopters, sceptical ratepayers, and industry boosters",
    },
    {
        "id": "climate_denial",
        "family": "clear",
        "count": 250,
        "desc": ("Claims that human-caused climate change is a hoax or a natural "
                 "cycle, attacks on climate scientists and their models, and "
                 "framing climate policy as a tax grab or a power grab."),
        "stance": "mostly dismissive and contrarian, with a few people pushing back",
    },
    {
        "id": "extreme_weather",
        "family": "clear",
        "count": 250,
        "desc": ("The impact of floods, wildfires, heatwaves, droughts and storms "
                 "on towns, farms and emergency services, and whether climate "
                 "change is to blame."),
        "stance": "shocked residents, volunteers, and commentators debating attribution",
    },
    {
        "id": "activism",
        "family": "clear",
        "count": 250,
        "desc": ("Climate protests, school strikes, road blockades, marches and "
                 "campaigners demanding government and corporate action."),
        "stance": "organisers, sympathetic supporters, and annoyed commuters",
    },
    {
        "id": "ev_transport",
        "family": "clear",
        "count": 250,
        "desc": ("Electric vehicles, charging networks, range anxiety, EV prices "
                 "and clean transport policy for cars, buses and bikes."),
        "stance": "EV owners, would-be buyers and sceptics",
    },
    # ------------------------------------------------------------- tricky
    {
        "id": "carbon_pricing",
        "family": "tricky",
        "count": 160,
        "tricky_with": "cop_negotiations",
        "desc": ("Domestic carbon taxes and cap-and-trade markets: carbon border "
                 "adjustments, emissions trading schemes, the price of carbon "
                 "and where the revenue goes."),
        "stance": "economists for it, households against it, and industry in between",
    },
    {
        "id": "grid_storage",
        "family": "tricky",
        "count": 160,
        "tricky_with": "renewables",
        "desc": ("Electricity grid infrastructure: transmission lines, "
                 "interconnection queues, utility-scale battery storage and "
                 "keeping the lights on when the wind stops."),
        "stance": "grid engineers, planners and impatient developers",
    },
    {
        "id": "doom_anxiety",
        "family": "tricky",
        "count": 160,
        "tricky_with": "climate_denial",
        "desc": ("Eco-anxiety, doomism and climate grief: 'it is too late', "
                 "fatalism, civilisational collapse narratives and young "
                 "people's despair about the future."),
        "stance": "people who feel hopeless, people resisting despair, and therapists",
    },
    {
        "id": "climate_health",
        "family": "tricky",
        "count": 160,
        "tricky_with": "extreme_weather",
        "desc": ("Climate impacts on human health: heat-related deaths, air "
                 "quality and asthma, infectious disease spread and mental "
                 "health, plus hospital preparedness."),
        "stance": "doctors, public-health officials and worried parents",
    },
]

BRIDGES = [
    {"id": "bridge_cop_pricing", "a": "cop_negotiations", "b": "carbon_pricing", "count": 15},
    {"id": "bridge_renew_grid", "a": "renewables", "b": "grid_storage", "count": 15},
    {"id": "bridge_denial_doom", "a": "climate_denial", "b": "doom_anxiety", "count": 15},
    {"id": "bridge_weather_health", "a": "extreme_weather", "b": "climate_health", "count": 15},
]

# Many small, mutually-unrelated subjects: each is too tiny and too incoherent
# with the others to form a real cluster, so most should land in topic -1.
OUTLIER_SUBJECTS = [
    "a football derby result and the league table",
    "a new pasta recipe someone cooked last night",
    "a movie or streaming series someone just watched",
    "a video game release and a boss fight",
    "someone's cat or dog doing something funny",
    "a weekend city-break travel tip",
    "a fashion haul and a pair of sneakers",
    "a concert or a new album release",
    "a celebrity breakup rumour",
    "someone's gym workout and personal best",
    "changing a car's engine oil at the weekend",
    "a DIY shelf-building project gone wrong",
    "houseplants and repotting a monstera",
    "cryptocurrency prices and a token pump",
    "a new phone launch and camera specs",
    "online dating and a bad first date",
    "an internet meme that is going around",
    "rush-hour traffic and a delayed train",
    "exam revision and a stressful deadline",
    "a board-game night with friends",
    "a book recommendation and a plot twist",
    "a knitting or crochet project",
    "a local restaurant's burger review",
    "a basketball playoff game",
    "a photography tip for golden hour",
    "a home-brewed beer experiment",
    "a fantasy football league update",
    "a makeup tutorial and a foundation match",
    "a garden shed clear-out",
    "a vintage record fair find",
    "a marathon training long run",
    "a coding side-project bug at 2am",
    "a fish tank and a new pleco",
    "a skateboarding trick landed at last",
    "a cooking fail that turned out fine",
    "a karaoke night song choice",
    "a comic-con cosplay build",
    "a road-trip playlist",
    "a chess puzzle that took all evening",
    "a sourdough starter finally rising",
]

# --------------------------------------------------------------------- helpers

# short, self-authored comments for quotes/replies (not clustered by the model)
QUOTE_COMMENTS = [
    "Interesting take.", "This is exactly right.", "I could not disagree more.",
    "Worth reading carefully.", "This misses the point.", "Finally someone said it.",
    "Not sure I buy this.", "Spot on.", "Where is the evidence?", "Hmm, complicated.",
]
REPLY_COMMENTS = [
    "Totally agree with this.", "This is exactly right.", "Not sure I buy this take.",
    "Source?", "Great point.", "This aged badly.", "Exactly my worry.",
    "Tell that to my neighbours.", "Well said.", "That is not what the data says.",
]


def _extract_json(text: str):
    """Pull the first JSON array/object out of an LLM reply."""
    if text is None:
        raise ValueError("empty LLM content")
    text = text.strip()
    if text.startswith("```"):
        # strip a fenced block
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    for opener, closer in (("[", "]"), ("{", "}")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    return json.loads(text)


def _cache_path(cache_dir: Path, key: str) -> Path:
    h = hashlib.sha1(f"{PROMPT_VERSION}:{MODEL}:{key}".encode()).hexdigest()[:20]
    return cache_dir / f"{h}.json"


def _complete(prompt: str, max_tokens: int) -> str:
    import litellm
    resp = litellm.completion(
        model=MODEL,
        messages=[
            {"role": "system",
             "content": "You write realistic, diverse social-media posts. "
                        "You output only the requested JSON, with no commentary."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def gen_topic_batch(topic: dict, n: int, batch_idx: int, fresh: bool,
                    cache_dir: Path) -> list[dict]:
    """Generate n tweets for one topic. Returns [{text, stance}, ...]."""
    key = f"topic::{topic['id']}::{n}::b{batch_idx}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        prompt = (
            f"Write {n} DIFFERENT tweets about: {topic['desc']}\n\n"
            "Requirements:\n"
            "- English, each tweet at most 280 characters.\n"
            "- Sound like real people on social media; vary length, tone and format.\n"
            "- Vary hashtags, @mentions, questions, hot takes and emoji (sparingly).\n"
            "- Never repeat a sentence or reuse a template.\n"
            f"- Cover these perspectives: {topic['stance']}.\n"
            "- Stay specifically on this subject; no generic filler.\n\n"
            f"Return ONLY a JSON array of exactly {n} objects:\n"
            '[{"text": "...", "stance": "support|oppose|neutral"}, ...]'
        )
        data = _extract_json(_complete(prompt, max_tokens=12000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))

    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            stance = str(item.get("stance", "neutral")).strip().lower()
            if stance not in ("support", "oppose", "neutral"):
                stance = "neutral"
            out.append({"text": str(item["text"]).strip(), "stance": stance})
    return out


def gen_bridge_batch(bridge: dict, a_desc: str, b_desc: str,
                     n: int, batch_idx: int, fresh: bool, cache_dir: Path) -> list[dict]:
    key = f"bridge::{bridge['id']}::{n}::b{batch_idx}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        prompt = (
            f"Write {n} DIFFERENT tweets that each discuss BOTH of these subjects "
            f"together, linking them naturally:\n"
            f"(1) {a_desc}\n(2) {b_desc}\n\n"
            "Requirements:\n"
            "- English, each tweet at most 280 characters.\n"
            "- Sound like real people; vary tone, format and hashtags.\n"
            "- Each tweet must genuinely touch both subjects.\n\n"
            f"Return ONLY a JSON array of exactly {n} objects:\n"
            '[{"text": "...", "stance": "support|oppose|neutral"}, ...]'
        )
        data = _extract_json(_complete(prompt, max_tokens=8000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))

    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            out.append({"text": str(item["text"]).strip(), "stance": "neutral"})
    return out


def gen_outlier_batch(subject: str, n: int, fresh: bool, cache_dir: Path) -> list[dict]:
    key = f"outlier::{subject}::{n}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        prompt = (
            f"Write {n} DIFFERENT tweets about {subject}.\n\n"
            "Requirements:\n"
            "- English, each tweet at most 280 characters.\n"
            "- Ordinary everyday opinions and chatter; vary tone and format.\n"
            "- Do not mention climate, energy or the environment.\n\n"
            f"Return ONLY a JSON array of exactly {n} objects:\n"
            '[{"text": "...", "stance": "neutral"}, ...]'
        )
        data = _extract_json(_complete(prompt, max_tokens=6000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))

    out = []
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and str(item.get("text", "")).strip():
            out.append({"text": str(item["text"]).strip(), "stance": "neutral"})
    return out


# ------------------------------------------------------------------ assembly

def _dedupe(items: list[dict], seen: set[str], limit: int) -> list[dict]:
    out = []
    for it in items:
        t = it["text"]
        norm = re.sub(r"\s+", " ", t).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def build_originals(scaled: float, workers: int, fresh: bool, cache_dir: Path,
                    rng: random.Random, outlier_subjects: int = 40) -> list[dict]:
    seen: set[str] = set()
    originals: list[dict] = []

    jobs = []
    for topic in TOPICS:
        n = max(10, int(round(topic["count"] * scaled)))
        batch = 20
        for b, off in enumerate(range(0, n, batch)):
            jobs.append(("topic", topic, min(batch, n - off), b))
    desc = {t["id"]: t["desc"] for t in TOPICS}
    for br in BRIDGES:
        n = max(5, int(round(br["count"] * scaled)))
        batch = 15
        for b, off in enumerate(range(0, n, batch)):
            jobs.append(("bridge", br, min(batch, n - off), b))
    n_out = max(2, int(round(6 * scaled)))
    for b, subj in enumerate(OUTLIER_SUBJECTS[:outlier_subjects]):
        jobs.append(("outlier", subj, n_out, b))

    print(f"Generating {len(jobs)} LLM batches with {workers} workers (model={MODEL})…",
          file=sys.stderr)
    results: dict[int, list[dict]] = {}
    done = 0
    lock = threading.Lock()

    def work(idx_job):
        idx, job = idx_job
        kind, payload, n, batch_idx = job
        if kind == "topic":
            return idx, gen_topic_batch(payload, n, batch_idx, fresh, cache_dir)
        if kind == "bridge":
            return idx, gen_bridge_batch(payload, desc[payload["a"]],
                                         desc[payload["b"]], n, batch_idx,
                                         fresh, cache_dir)
        return idx, gen_outlier_batch(payload, n, fresh, cache_dir)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, (i, j)): i for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            try:
                idx, items = fut.result()
            except Exception as e:
                print(f"  batch failed: {type(e).__name__}: {str(e)[:120]}", file=sys.stderr)
                continue
            results[idx] = items
            with lock:
                done += 1
                if done % 10 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} batches done", file=sys.stderr)

    # author pools
    topic_ids = [t["id"] for t in TOPICS]
    authors = {tid: [f"{tid[:8]}_{i:02d}" for i in range(14)] for tid in topic_ids}
    # bridges borrow authors from both linked topics (shared discussants)
    for br in BRIDGES:
        authors[br["id"]] = authors[br["a"]][:7] + authors[br["b"]][:7]
    random_authors = [f"rand_{i:03d}" for i in range(90)]

    bridges_by_id = {b["id"]: b for b in BRIDGES}
    date_rng = random.Random(rng.randrange(1 << 30))

    counter = 0
    for i, job in enumerate(jobs):
        kind, payload, n, _batch_idx = job
        items = _dedupe(results.get(i, []), seen, n)
        for it in items:
            counter += 1
            if kind == "topic":
                tid, fam, stance_hint = payload["id"], payload["family"], payload["stance"]
            elif kind == "bridge":
                tid, fam, stance_hint = payload["id"], "bridge", "mixed"
            else:
                tid, fam, stance_hint = "outlier", "outlier", "neutral"

            if kind == "outlier":
                author = rng.choice(random_authors)
            else:
                author = rng.choice(authors[tid])
            day = date_rng.randrange(0, 30)
            ts = f"2024-11-{1 + day:02d}T{date_rng.randrange(0, 24):02d}:{date_rng.randrange(0, 60):02d}:00Z"
            originals.append({
                "id": f"g{counter:05d}",
                "author_id": author,
                "text": it["text"],
                "created_at": ts,
                "lang": "en",
                "true_topic": tid,
                "true_family": fam,
                "true_stance": it["stance"],
                "mentions": [rng.choice(random_authors)] if rng.random() < 0.15 else [],
                "_desc": stance_hint,
            })

    print(f"  {len(originals)} unique originals assembled", file=sys.stderr)
    return originals


def build_derived(originals: list[dict], ratio: float, rng: random.Random) -> list[dict]:
    n_derived = int(round(len(originals) * ratio))
    all_authors = [o["author_id"] for o in originals]
    derived: list[dict] = []
    for j in range(n_derived):
        target = rng.choice(originals)
        kind = rng.choices(["repost", "quote", "reply"], weights=[0.75, 0.10, 0.15])[0]
        author = rng.choice(all_authors)
        created = f"2024-11-{1 + rng.randrange(0, 30):02d}T{rng.randrange(0, 24):02d}:00:00Z"
        if kind == "repost":
            text = target["text"]
        elif kind == "quote":
            text = f"{rng.choice(QUOTE_COMMENTS)} {target['text']}"
        else:
            text = rng.choice(REPLY_COMMENTS)
        text = text[:280]
        derived.append({
            "id": f"d{j:05d}",
            "author_id": author,
            "text": text,
            "created_at": created,
            "lang": "en",
            "true_topic": target["true_topic"],
            "true_family": target["true_family"],
            "true_stance": target["true_stance"],
            "mentions": [],
            "_kind": kind,
            "_target": target["id"],
        })
    return derived


# --------------------------------------------------------------------- emit

_CSV_KIND = {"repost": "repost", "quote": "quote", "reply": "reply"}
_JSONL_REF = {"repost": "retweeted", "quote": "quoted", "reply": "replied_to"}


def emit_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            obj = {
                "id": r["id"],
                "author_id": r["author_id"],
                "text": r["text"],
                "created_at": r["created_at"],
                "lang": r["lang"],
                "conversation_id": r["id"],
                "public_metrics": {"retweet_count": 0, "like_count": 0,
                                   "reply_count": 0, "quote_count": 0},
            }
            if r.get("_kind"):
                obj["referenced_tweets"] = [{"type": _JSONL_REF[r["_kind"]],
                                             "id": r["_target"]}]
            if r.get("mentions"):
                obj["entities"] = {"mentions": [{"username": m} for m in r["mentions"]]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def emit_csv(records: list[dict], path: Path) -> None:
    cols = ["post_id", "user_id", "text", "created_at", "lang",
            "interaction_type", "target_post_id", "mentions"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "post_id": r["id"],
                "user_id": r["author_id"],
                "text": r["text"],
                "created_at": r["created_at"],
                "lang": r["lang"],
                "interaction_type": _CSV_KIND.get(r.get("_kind"), "original"),
                "target_post_id": r.get("_target", ""),
                "mentions": ";".join(r.get("mentions", [])),
            })


def emit_truth(records: list[dict], path: Path) -> None:
    cols = ["post_id", "true_topic", "true_family", "true_stance",
            "is_derived", "target_id", "author_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "post_id": r["id"],
                "true_topic": r["true_topic"],
                "true_family": r["true_family"],
                "true_stance": r["true_stance"],
                "is_derived": "true" if r.get("_kind") else "false",
                "target_id": r.get("_target", ""),
                "author_id": r["author_id"],
            })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="raw LLM cache (default: <out-dir>/llm_cache)")
    ap.add_argument("--name", default="synth_llm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every topic/outlier count")
    ap.add_argument("--interaction-ratio", type=float, default=0.5)
    ap.add_argument("--outlier-subjects", type=int, default=len(OUTLIER_SUBJECTS),
                    help="cap the number of distinct off-topic subjects")
    ap.add_argument("--fresh", action="store_true", help="ignore the LLM cache")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "llm_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    originals = build_originals(args.scale, args.workers, args.fresh, cache_dir,
                                rng, args.outlier_subjects)
    derived = build_derived(originals, args.interaction_ratio, rng)
    records = originals + derived

    emit_jsonl(records, args.out_dir / f"{args.name}.jsonl")
    emit_csv(records, args.out_dir / f"{args.name}.csv")
    emit_truth(records, args.out_dir / f"truth_{args.name}.csv")

    n_orig = len(originals)
    n_rep = sum(1 for d in derived if d["_kind"] == "repost")
    n_quo = sum(1 for d in derived if d["_kind"] == "quote")
    n_rep_ = sum(1 for d in derived if d["_kind"] == "reply")
    from collections import Counter
    fam = Counter(o["true_family"] for o in originals)
    print(f"{args.name}: {len(records)} records "
          f"({n_orig} originals, {n_rep} reposts, {n_quo} quotes, {n_rep_} replies)")
    print(f"  families: {dict(fam)}")
    print(f"  -> {args.out_dir / (args.name + '.jsonl')}")
    print(f"  -> {args.out_dir / (args.name + '.csv')}")
    print(f"  -> {args.out_dir / ('truth_' + args.name + '.csv')}")


if __name__ == "__main__":
    main()
