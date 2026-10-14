#!/usr/bin/env python3
"""LLM-generated *realistic* climate/politics tweet corpus (v2), built for
topic-model tuning **and** downstream polarization analysis.

Differences from v1 (``gen_llm_tweets.py``):

* **Realistic style** modelled on the real COP22 scrape in ``data/cop22.json``:
  short posts, hashtag stacks, @mentions, fake t.co links, ``RT @user:``
  prefixes, typos, all-caps, fragments and multilingual noise — not polished
  essays.
* **Bigger**: ~8.4k originals + ~11k retweets/quotes/replies.
* **Polarization structure**: every author has a *side* (``pro``/``anti``
  climate action) and home topics; posts are written in-camp, and retweets are
  overwhelmingly same-side, so polarized topics yield two-community retweet
  layers while consensual topics stay one-community. Ground-truth
  ``topic_polarized`` flags let you check whether ``netpol`` detects the right
  layers.

Outputs (in ``--out-dir``):
  <name>.jsonl         twitter_jsonl feed
  <name>.csv           csv_posts feed (same records)
  truth_<name>.csv     per-post ground truth (topic, stance, side, lang, ...)
  authors_<name>.csv   per-author side + home topics
  topics_<name>.csv    per-topic design (family, polarized, camps)

Raw LLM replies are cached under ``<cache-dir>``; ``--fresh`` ignores the cache.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

PROMPT_VERSION = "v2"
MODEL = os.getenv("GEN_MODEL", "gpt-6-luna")
MENTION_RE = re.compile(r"@([A-Za-z0-9_]{2,15})")

# --------------------------------------------------------------------- topics
# family: clear | tricky      polarized: two camps -> netpol should flag it
# camps[].side: pro | anti (relative to climate action); consensual = neutral

TOPICS = [
    {
        "id": "cop_negotiations", "family": "clear", "polarized": True, "count": 900,
        "desc": "the UN COP climate summit: the plenary, the draft cover decision, "
                "loss and damage, Article 6 carbon markets, adaptation finance, delegates",
        "tags": "#COP29 #COP #climatefinance #LossAndDamage #Article6",
        "camps": [
            {"side": "pro", "share": 0.55,
             "voice": "demand far more ambition, real climate finance and hold polluters "
                      "accountable; they defend the process and celebrate wins"},
            {"side": "anti", "share": 0.45,
             "voice": "call the summit a pointless talking shop, a jet-setting elite jamboree, "
                      "or a scam that will raise bills; they mock the pledges and the delegates"},
        ],
    },
    {
        "id": "renewables", "family": "clear", "polarized": False, "count": 600,
        "desc": "solar panels, wind farms, home batteries, heat pumps, clean electricity",
        "tags": "#solar #wind #renewables #cleanenergy #heatpumps",
        "camps": [
            {"side": "pro", "share": 1.0,
             "voice": "ordinary people, installers and engineers sharing clean-energy news, "
                      "record output, falling costs and home upgrades"},
        ],
    },
    {
        "id": "climate_denial", "family": "clear", "polarized": True, "count": 900,
        "desc": "claims that human-caused climate change is a hoax or natural cycle, attacks "
                "on climate scientists and models, climate policy as a tax grab",
        "tags": "#ClimateScam #globalwarming #climategate #NetZero #hoax",
        "camps": [
            {"side": "anti", "share": 0.6,
             "voice": "insist it is a hoax / natural cycle, attack 'climate alarmists', "
                      "scientists' funding and green taxes"},
            {"side": "pro", "share": 0.4,
             "voice": "push back with evidence, mock the conspiracy claims and defend scientists"},
        ],
    },
    {
        "id": "extreme_weather", "family": "clear", "polarized": False, "count": 600,
        "desc": "floods, wildfires, heatwaves, droughts, storms and their impacts; whether "
                "climate change is to blame",
        "tags": "#heatwave #floods #wildfire #extremeweather #climatecrisis",
        "camps": [
            {"side": "pro", "share": 1.0,
             "voice": "shocked residents, volunteers and reporters describing the weather, "
                      "the damage and the link to a warming climate"},
        ],
    },
    {
        "id": "activism", "family": "clear", "polarized": True, "count": 900,
        "desc": "climate protests, school strikes, road blockades, marches, campaigners",
        "tags": "#ClimateStrike #FridaysForFuture #protest #JustStopOil #climateaction",
        "camps": [
            {"side": "pro", "share": 0.55,
             "voice": "support the protests, join the strikes and praise the campaigners"},
            {"side": "anti", "share": 0.45,
             "voice": "are furious about the blockades, call the protesters entitled or "
                      "professional agitators, and complain about disruption"},
        ],
    },
    {
        "id": "ev_transport", "family": "clear", "polarized": True, "count": 900,
        "desc": "electric vehicles, charging networks, range anxiety, EV prices, clean transport",
        "tags": "#EV #electricvehicles #EVcharging #cleanenergy #netzero",
        "camps": [
            {"side": "pro", "share": 0.5,
             "voice": "love their EV, praise cheap charging and clean air, and mock petrol cars"},
            {"side": "anti", "share": 0.5,
             "voice": "slam EV range, charging queues, battery fires and the cost, and defend "
                      "petrol/diesel"},
        ],
    },
    # ------------------------------------------------------------- tricky
    {
        "id": "carbon_pricing", "family": "tricky", "tricky_with": "cop_negotiations",
        "polarized": True, "count": 600,
        "desc": "domestic carbon taxes and cap-and-trade: the price of carbon, emissions "
                "trading, carbon border adjustments, where the revenue goes",
        "tags": "#carbonprice #carbontax #ETS #capandtrade #carbonmarkets",
        "camps": [
            {"side": "pro", "share": 0.5,
             "voice": "argue a carbon price is the cheapest way to cut emissions and want "
                      "revenue recycled as rebates"},
            {"side": "anti", "share": 0.5,
             "voice": "say the carbon tax punishes ordinary families, farmers and drivers and "
                      "want it scrapped"},
        ],
    },
    {
        "id": "grid_storage", "family": "tricky", "tricky_with": "renewables",
        "polarized": False, "count": 600,
        "desc": "electricity grid infrastructure: transmission lines, interconnection queues, "
                "utility-scale batteries, keeping the lights on when the wind stops",
        "tags": "#grid #energystorage #transmission #batteries #blackout",
        "camps": [
            {"side": "pro", "share": 1.0,
             "voice": "grid engineers, planners and developers discussing transmission, "
                      "storage and reliability"},
        ],
    },
    {
        "id": "doom_anxiety", "family": "tricky", "tricky_with": "climate_denial",
        "polarized": True, "count": 600,
        "desc": "eco-anxiety, doomism, 'it is too late', climate grief, collapse narratives",
        "tags": "#ecoanxiety #climategrief #doom #collapse #ClimateCrisis",
        "camps": [
            {"side": "anti", "share": 0.5,
             "voice": "believe it is already too late, civilisation is collapsing and there is "
                      "no point trying"},
            {"side": "pro", "share": 0.5,
             "voice": "resist despair, argue action still matters and encourage others to keep going"},
        ],
    },
    {
        "id": "climate_health", "family": "tricky", "tricky_with": "extreme_weather",
        "polarized": False, "count": 600,
        "desc": "climate impacts on health: heat deaths, air quality and asthma, disease, "
                "mental health, hospital preparedness",
        "tags": "#heathealth #airquality #asthma #publichealth #climatehealth",
        "camps": [
            {"side": "pro", "share": 1.0,
             "voice": "doctors, health officials and parents talking about heat, smoke, "
                      "asthma and hospital pressure"},
        ],
    },
]

BRIDGES = [
    {"id": "bridge_cop_pricing", "a": "cop_negotiations", "b": "carbon_pricing", "count": 40},
    {"id": "bridge_renew_grid", "a": "renewables", "b": "grid_storage", "count": 40},
    {"id": "bridge_denial_doom", "a": "climate_denial", "b": "doom_anxiety", "count": 40},
    {"id": "bridge_weather_health", "a": "extreme_weather", "b": "climate_health", "count": 40},
]

OUTLIER_SUBJECTS = [
    "a football derby result and the league table", "a new pasta recipe",
    "a movie or streaming series someone watched", "a video game release and a boss fight",
    "someone's cat or dog doing something funny", "a weekend city-break travel tip",
    "a fashion haul and sneakers", "a concert or new album", "a celebrity breakup rumour",
    "someone's gym workout and personal best", "changing a car's engine oil",
    "a DIY shelf-building project gone wrong", "houseplants and repotting a monstera",
    "cryptocurrency prices and a token pump", "a new phone launch and camera specs",
    "online dating and a bad first date", "an internet meme going around",
    "rush-hour traffic and a delayed train", "exam revision and a deadline",
    "a board-game night with friends", "a book recommendation and a plot twist",
    "a knitting or crochet project", "a local restaurant's burger review",
    "a basketball playoff game", "a photography tip for golden hour",
    "a home-brewed beer experiment", "a fantasy football league update",
    "a makeup tutorial and a foundation match", "a garden shed clear-out",
    "a vintage record fair find", "a marathon training long run",
    "a coding side-project bug at 2am", "a fish tank and a new pleco",
    "a skateboarding trick landed at last", "a cooking fail that turned out fine",
    "a karaoke night song choice", "a comic-con cosplay build", "a road-trip playlist",
    "a chess puzzle that took all evening", "a sourdough starter finally rising",
    "a parking ticket appeal", "a wedding seating-plan argument",
    "a neighbour's noisy renovation", "a phone battery dying too fast",
    "a supermarket checkout queue", "a gym membership nobody uses",
    "a dog refusing to walk in the rain", "a lost parcel delivery",
    "a pub quiz controversy", "a supermarket meal-deal ranking",
    "a laundry-day sock mystery", "a bus that never showed up",
    "a badly-timed haircut", "a broken dishwasher", "a flat tyre on the motorway",
    "a surprise tax refund", "a new café opening nearby", "a childhood toy found in the loft",
    "a rainy bank-holiday barbecue", "a mystery smell in the fridge",
    "a spreadsheet that will not sum", "a printer that jams every time",
    "a neighbour's new fence", "a karaoke song stuck in the head",
    "a supermarket self-checkout meltdown", "a garden snail invasion",
    "a wardrobe clear-out", "a phone-screen crack", "a subscription nobody remembers",
    "a train strike plan", "a pub with no Wi-Fi", "a scooter commute",
    "a birthday-cake fail", "a leaky roof", "a lost house key",
]

# ---------------------------------------------------------------- author pools
_HANDLE_PRE = ["Green", "Eco", "Climate", "Carbon", "Solar", "Wind", "Earth", "Planet",
               "Fossil", "Oil", "Coal", "NetZero", "Gaia", "Terra", "Blue", "Storm",
               "Heat", "Flood", "Arctic", "Reef", "Canopy", "Tide", "Ember", "Frost",
               "JustTransition", "ZeroCarbon", "Wildfire", "Ice", "Desert", "Prairie"]
_HANDLE_SUF = ["Watcher", "Warrior", "Voice", "Action", "Now", "Future", "Patriot",
               "Realist", "Skeptic", "Truth", "News", "Daily", "Report", "Activist",
               "Engineer", "Scientist", "Farmer", "Teacher", "Parent", "Citizen",
               "Dad", "Mum", "Girl", "Guy", "Doc", "Prof", "Lad", "Nan", "Nerd", "Troll"]

QUOTE_COMMENTS = [
    "this", "spot on", "nonsense", "lol", "read the room", "exactly", "wrong",
    "finally someone said it", "source?", "🤡", "here we go again", "imagine believing this",
    "so true", "copium", "tell that to my bills", "100%", "no thanks",
]
REPLY_COMMENTS = [
    "@{u} source?", "@{u} lol no", "@{u} this is wrong", "@{u} totally agree",
    "@{u} tell that to my neighbours", "@{u} 🤡", "@{u} exactly", "@{u} copium",
    "@{u} well said", "@{u} that aged badly", "🧵", "this", "smh", "nah",
]


# --------------------------------------------------------------------- helpers
def _extract_json(text: str):
    if text is None:
        raise ValueError("empty LLM content")
    text = text.strip()
    if text.startswith("```"):
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
             "content": "You imitate a real Twitter/X dataset. You write short, messy, "
                        "authentic posts (not polished essays) and output only the "
                        "requested JSON, with no commentary."},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=max_tokens,
    )
    return resp.choices[0].message.content


_STYLE = (
    "Style rules (make them look like a real scrape, NOT like essays):\n"
    "- MOST posts are SHORT: aim 15-120 characters; at least a third under 70 chars; "
    "a few may reach 280.\n"
    "- Some posts are barely understandable: terse fragments, hashtag stacks, ALL-CAPS "
    "rants, half-sentences, typos, abbreviations (imo, tbh, smh, ngl, FFS).\n"
    "- Sprinkle hashtags, @mentions, and fake links like https://t.co/Ab3xY9; use emoji "
    "sparingly. Some posts start with 'RT @someone:'.\n"
    "- Vary the voice: activists, journalists, ordinary people, officials, trolls, bots.\n"
    "- Language mix: ~80% en, ~10% fr, ~5% es, ~5% de. Put the language in \"lang\".\n"
    "- Never repeat a sentence; stay on the subject but allow noise.\n"
)


def gen_camp_batch(topic: dict, camp: dict, n: int, batch_idx: int,
                   fresh: bool, cache_dir: Path) -> list[dict]:
    key = f"camp::{topic['id']}::{camp['side']}::{n}::b{batch_idx}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        prompt = (
            f"Write {n} DIFFERENT Twitter/X posts from people who {camp['voice']}.\n"
            f"The subject is: {topic['desc']}.\n"
            f"Suggested hashtags: {topic['tags']}\n\n"
            + _STYLE +
            f"\nSet \"stance\" to \"{camp['side']}\" (or \"neutral\") per post.\n\n"
            f"Return ONLY a JSON array of exactly {n} objects:\n"
            '[{"text":"...","stance":"pro|anti|neutral","lang":"en"}, ...]'
        )
        data = _extract_json(_complete(prompt, max_tokens=14000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    return _clean_items(data)


def gen_bridge_batch(bridge: dict, a_desc: str, b_desc: str, n: int, batch_idx: int,
                     fresh: bool, cache_dir: Path) -> list[dict]:
    key = f"bridge::{bridge['id']}::{n}::b{batch_idx}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        prompt = (
            f"Write {n} DIFFERENT short Twitter/X posts that each touch BOTH subjects at once:\n"
            f"(1) {a_desc}\n(2) {b_desc}\n\n" + _STYLE +
            "\nReturn ONLY a JSON array of exactly "
            f"{n} objects:\n"
            '[{"text":"...","stance":"neutral","lang":"en"}, ...]'
        )
        data = _extract_json(_complete(prompt, max_tokens=10000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    return _clean_items(data)


def gen_outlier_batch(subjects: list[str], n_each: int, batch_idx: int,
                      fresh: bool, cache_dir: Path) -> dict[str, list[dict]]:
    key = f"outlier::{batch_idx}::{'|'.join(subjects)}::{n_each}"
    path = _cache_path(cache_dir, key)
    if path.exists() and not fresh:
        data = json.loads(path.read_text())
    else:
        bullets = "\n".join(f"- {s}" for s in subjects)
        prompt = (
            "Write short, ordinary, unrelated Twitter/X posts about each of these "
            "filler subjects (this is noise from a general scrape):\n"
            f"{bullets}\n\n"
            f"Write exactly {n_each} posts per subject. Keep them short and everyday. "
            "Do NOT mention climate, energy or the environment.\n\n"
            "Return ONLY a JSON object mapping each exact subject string to an array:\n"
            '{"<subject>": [{"text":"...","lang":"en"}, ...], ...}'
        )
        data = _extract_json(_complete(prompt, max_tokens=10000))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=0))
    out = {}
    if isinstance(data, dict):
        for s, items in data.items():
            out[s] = _clean_items(items)
    return out


def _clean_items(data) -> list[dict]:
    out = []
    for it in data if isinstance(data, list) else []:
        if isinstance(it, dict) and str(it.get("text", "")).strip():
            stance = str(it.get("stance", "neutral")).strip().lower()
            if stance not in ("pro", "anti", "neutral"):
                stance = "neutral"
            lang = str(it.get("lang", "en")).strip().lower()[:2] or "en"
            out.append({"text": str(it["text"]).strip()[:280], "stance": stance, "lang": lang})
    return out


# --------------------------------------------------------------- author model
def build_authors(n_authors: int, topics: list[dict], rng: random.Random) -> list[dict]:
    used: set[str] = set()
    authors = []
    for i in range(n_authors):
        while True:
            h = f"{rng.choice(_HANDLE_PRE)}{rng.choice(_HANDLE_SUF)}{rng.randrange(1, 9999)}"
            if h not in used:
                used.add(h)
                break
        side = "pro" if rng.random() < 0.5 else "anti"
        k = rng.choices([1, 2, 3], weights=[0.4, 0.4, 0.2])[0]
        home = rng.sample([t["id"] for t in topics], k)
        authors.append({"id": h, "side": side, "home": home})
    return authors


def _pool(authors: list[dict], topic_id: str, side: str | None) -> list[str]:
    if side is None:
        return [a["id"] for a in authors if topic_id in a["home"]]
    return [a["id"] for a in authors if topic_id in a["home"] and a["side"] == side]


# ------------------------------------------------------------------ assembly
def _dedupe(items: list[dict], seen: set[str], limit: int) -> list[dict]:
    out = []
    for it in items:
        norm = re.sub(r"\s+", " ", it["text"]).lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(it)
        if len(out) >= limit:
            break
    return out


def build_originals(scaled, workers, fresh, cache_dir, rng, outlier_subjects):
    seen: set[str] = set()
    jobs = []  # (kind, payload, n, batch_idx, extra)
    batch = 30
    for topic in TOPICS:
        n = max(20, int(round(topic["count"] * scaled)))
        for camp in topic["camps"]:
            cn = max(5, int(round(n * camp["share"])))
            for b, off in enumerate(range(0, cn, batch)):
                jobs.append(("camp", topic, min(batch, cn - off), b, camp))
    desc = {t["id"]: t["desc"] for t in TOPICS}
    for br in BRIDGES:
        n = max(10, int(round(br["count"] * scaled)))
        for b, off in enumerate(range(0, n, batch)):
            jobs.append(("bridge", br, min(batch, n - off), b, None))
    subjects = OUTLIER_SUBJECTS[:outlier_subjects]
    n_each = max(2, int(round(4 * scaled)))
    per_call = 6
    for b, off in enumerate(range(0, len(subjects), per_call)):
        jobs.append(("outlier", subjects[off:off + per_call], n_each, b, None))

    print(f"Generating {len(jobs)} LLM batches ({MODEL}, {workers} workers)…", file=sys.stderr)
    results: dict[int, object] = {}
    done = 0
    lock = threading.Lock()

    def work(idx_job):
        idx, job = idx_job
        kind, payload, n, b, extra = job
        if kind == "camp":
            return idx, gen_camp_batch(payload, extra, n, b, fresh, cache_dir)
        if kind == "bridge":
            return idx, gen_bridge_batch(payload, desc[payload["a"]], desc[payload["b"]],
                                         n, b, fresh, cache_dir)
        return idx, gen_outlier_batch(payload, n, b, fresh, cache_dir)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(work, (i, j)): i for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            try:
                idx, items = fut.result()
            except Exception as e:
                print(f"  batch failed: {type(e).__name__}: {str(e)[:140]}", file=sys.stderr)
                continue
            results[idx] = items
            with lock:
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    print(f"  {done}/{len(jobs)} batches done", file=sys.stderr)

    # authors
    n_authors = int(round(2000 * min(1.0, scaled)))
    authors = build_authors(n_authors, TOPICS, rng)
    author_side = {a["id"]: a["side"] for a in authors}
    consensual = {t["id"] for t in TOPICS if not t["polarized"]}
    topic_meta = {t["id"]: t for t in TOPICS}

    date_rng = random.Random(rng.randrange(1 << 30))
    counter = 0
    originals = []
    for i, job in enumerate(jobs):
        kind, payload, n, b, extra = job
        raw = results.get(i)
        if kind == "outlier":
            items = []
            if isinstance(raw, dict):
                for s in payload:
                    items += _dedupe(raw.get(s, []), seen, n)
        else:
            items = _dedupe(raw or [], seen, n)

        for it in items:
            counter += 1
            if kind == "camp":
                tid = payload["id"]
                fam = payload["family"]
                polarized = payload["polarized"]
                if tid in consensual:
                    side = "consensus"
                    author = rng.choice(_pool(authors, tid, None))
                else:
                    side = extra["side"]
                    author = rng.choice(_pool(authors, tid, side))
            elif kind == "bridge":
                tid, side, fam, polarized = payload["id"], "neutral", "bridge", None
                author = rng.choice(_pool(authors, tid, None)) if _pool(authors, tid, None) \
                    else rng.choice([a["id"] for a in authors])
            else:
                tid, side, fam, polarized = "outlier", "neutral", "outlier", False
                author = rng.choice([a["id"] for a in authors])

            day = date_rng.randrange(0, 45)
            ts = (f"2024-11-{1 + day:02d}T{date_rng.randrange(0, 24):02d}:"
                  f"{date_rng.randrange(0, 60):02d}:00Z")
            originals.append({
                "id": f"g{counter:05d}",
                "author_id": author,
                "text": it["text"],
                "created_at": ts,
                "lang": it["lang"],
                "true_topic": tid,
                "true_family": fam,
                "true_stance": it["stance"],
                "true_side": side,
                "topic_polarized": polarized,
            })

    print(f"  {len(originals)} unique originals assembled", file=sys.stderr)
    return originals, authors, author_side, topic_meta


def build_derived(originals, authors, ratio, rng, cross_side=0.08):
    n_derived = int(round(len(originals) * ratio))
    # authors who post about a topic (for choosing retweeters)
    topic_authors: dict[str, list[str]] = {}
    for a in authors:
        for t in a["home"]:
            topic_authors.setdefault(t, []).append(a["id"])
    side_map = {a["id"]: a["side"] for a in authors}
    all_ids = [a["id"] for a in authors]

    weights = [rng.lognormvariate(0, 0.9) for _ in originals]
    derived = []
    for j in range(n_derived):
        target = rng.choices(originals, weights=weights, k=1)[0]
        kind = rng.choices(["repost", "quote", "reply"], weights=[0.8, 0.08, 0.12])[0]
        t = target["true_topic"]
        pool = topic_authors.get(t) or all_ids
        if target["topic_polarized"]:
            same = [u for u in pool if side_map.get(u) == target["true_side"]] or pool
        else:
            same = pool
        if rng.random() < cross_side and len(pool) > 1:
            r = rng.choice(pool)
        else:
            r = rng.choice(same)
        created = (f"2024-11-{1 + rng.randrange(0, 45):02d}T"
                   f"{rng.randrange(0, 24):02d}:00:00Z")
        if kind == "repost":
            text = f"RT @{target['author_id']}: {target['text']}"[:280]
        elif kind == "quote":
            text = f"{rng.choice(QUOTE_COMMENTS)} https://t.co/{_tok(rng)}"
        else:
            text = rng.choice(REPLY_COMMENTS).format(u=target["author_id"])[:280]
        derived.append({
            "id": f"d{j:06d}",
            "author_id": r,
            "text": text,
            "created_at": created,
            "lang": "en",
            "true_topic": t,
            "true_family": target["true_family"],
            "true_stance": target["true_stance"],
            "true_side": target["true_side"],
            "topic_polarized": target["topic_polarized"],
            "_kind": kind,
            "_target": target["id"],
        })
    return derived


def _tok(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz0123456789"
    return "".join(rng.choice(alphabet) for _ in range(rng.randrange(8, 11)))


# --------------------------------------------------------------------- emit
_JSONL_REF = {"repost": "retweeted", "quote": "quoted", "reply": "replied_to"}
_CSV_KIND = {"repost": "repost", "quote": "quote", "reply": "reply"}


def emit_jsonl(records, path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            obj = {
                "id": r["id"], "author_id": r["author_id"], "text": r["text"],
                "created_at": r["created_at"], "lang": r["lang"],
                "conversation_id": r["id"],
                "public_metrics": {"retweet_count": 0, "like_count": 0,
                                   "reply_count": 0, "quote_count": 0},
            }
            if r.get("_kind"):
                obj["referenced_tweets"] = [{"type": _JSONL_REF[r["_kind"]], "id": r["_target"]}]
            men = MENTION_RE.findall(r["text"])
            if men:
                obj["entities"] = {"mentions": [{"username": m} for m in men[:5]]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def emit_csv(records, path: Path) -> None:
    cols = ["post_id", "user_id", "text", "created_at", "lang",
            "interaction_type", "target_post_id", "mentions"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "post_id": r["id"], "user_id": r["author_id"], "text": r["text"],
                "created_at": r["created_at"], "lang": r["lang"],
                "interaction_type": _CSV_KIND.get(r.get("_kind"), "original"),
                "target_post_id": r.get("_target", ""),
                "mentions": ";".join(MENTION_RE.findall(r["text"])),
            })


def emit_truth(records, path: Path) -> None:
    cols = ["post_id", "true_topic", "true_family", "true_stance", "true_side",
            "topic_polarized", "lang", "is_derived", "target_id", "author_id"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow({
                "post_id": r["id"], "true_topic": r["true_topic"],
                "true_family": r["true_family"], "true_stance": r["true_stance"],
                "true_side": r["true_side"], "topic_polarized": r["topic_polarized"],
                "lang": r["lang"], "is_derived": "true" if r.get("_kind") else "false",
                "target_id": r.get("_target", ""), "author_id": r["author_id"],
            })


def emit_authors(authors, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["author_id", "side", "home_topics"])
        w.writeheader()
        for a in authors:
            w.writerow({"author_id": a["id"], "side": a["side"],
                        "home_topics": ";".join(a["home"])})


def emit_topics(path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["topic", "family", "polarized", "tricky_with"])
        w.writeheader()
        for t in TOPICS:
            w.writerow({"topic": t["id"], "family": t["family"],
                        "polarized": t["polarized"], "tricky_with": t.get("tricky_with", "")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--name", default="synth_pol")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--interaction-ratio", type=float, default=1.3)
    ap.add_argument("--outlier-subjects", type=int, default=len(OUTLIER_SUBJECTS))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (args.out_dir / "llm_cache_v2")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    originals, authors, _, _ = build_originals(
        args.scale, args.workers, args.fresh, cache_dir, rng, args.outlier_subjects)
    derived = build_derived(originals, authors, args.interaction_ratio, rng)
    records = originals + derived

    emit_jsonl(records, args.out_dir / f"{args.name}.jsonl")
    emit_csv(records, args.out_dir / f"{args.name}.csv")
    emit_truth(records, args.out_dir / f"truth_{args.name}.csv")
    emit_authors(authors, args.out_dir / f"authors_{args.name}.csv")
    emit_topics(args.out_dir / f"topics_{args.name}.csv")

    from collections import Counter
    fam = Counter(o["true_family"] for o in originals)
    n_rep = sum(1 for d in derived if d["_kind"] == "repost")
    n_quo = sum(1 for d in derived if d["_kind"] == "quote")
    n_rpl = sum(1 for d in derived if d["_kind"] == "reply")
    print(f"{args.name}: {len(records)} records ({len(originals)} originals, "
          f"{n_rep} reposts, {n_quo} quotes, {n_rpl} replies; {len(authors)} authors)")
    print(f"  families: {dict(fam)}")
    for suffix in (".jsonl", ".csv"):
        print(f"  -> {args.out_dir / (args.name + suffix)}")
    print(f"  -> {args.out_dir / ('truth_' + args.name + '.csv')}")


if __name__ == "__main__":
    main()
