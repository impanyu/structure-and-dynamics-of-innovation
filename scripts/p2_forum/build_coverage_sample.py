"""Reference paper sample for topic coverage (spec §3.2, §4): the seven venues
x 2023/2024 from Semantic Scholar bulk search, stratified to N papers, labeled
by the experiment's tagger against the frozen topics-v3 list, summarized as
configs/p2_forum/topics-v3.coverage.json.

Every S2 page and every tagger call is disk-cached, and labels.jsonl is
appended chunk by chunk, so a rerun resumes where the last one stopped.
Run: uv run python scripts/p2_forum/build_coverage_sample.py"""
import hashlib
import json
import time
from pathlib import Path

import requests

from innovation.core.config import load_env
from innovation.core.data.s2 import S2_BULK, _cached_call, s2_headers
from innovation.core.fsutil import atomic_write_text
from innovation.core.llm import CachedLLM, RoutedLLM
from innovation.p2_forum.coverage import allocate, coverage_table, sample_stratum
from innovation.p2_forum.literature import Paper
from innovation.p2_forum.tagger import TopicTagger
from innovation.p2_forum.topics import load_topics

VENUES = [  # exact S2 venue names (configs/p1_dial/stage1.yaml recognized_venues)
    "AAAI Conference on Artificial Intelligence",
    "Neural Information Processing Systems",
    "Annual Meeting of the Association for Computational Linguistics",
    "Computer Vision and Pattern Recognition",
    "IEEE International Conference on Computer Vision",
    "International Conference on Machine Learning",
    "International Conference on Learning Representations",
]
YEARS = [2023, 2024]
N, SEED = 4000, 0
FIELDS = "paperId,title,abstract,venue,year,publicationDate"
OUT = Path("data/p2_forum/coverage_sample")
S2_CACHE = Path("data/online_cache/s2bulk_coverage")
LLM_CACHE = Path("data/online_cache/llm")
TOPICS = Path("configs/p2_forum/topics-v3.yaml")
TABLE = Path("configs/p2_forum/topics-v3.coverage.json")
MODEL = "claude-sonnet-5:medium"
CHUNK = 200


def fetch_stratum(venue: str, year: int) -> list[dict]:
    """Every paper S2 files under exactly this venue and year (all pages)."""
    papers, token, page = [], None, 0
    while True:
        params = {"query": "", "venue": venue, "year": str(year), "fields": FIELDS}
        if token:
            params["token"] = token
        key = hashlib.sha256(json.dumps([venue, year, FIELDS, page]).encode()).hexdigest()
        payload = _cached_call(S2_CACHE / f"{key}.json",
                               lambda: requests.get(S2_BULK, params=params,
                                                    headers=s2_headers(), timeout=60),
                               1.1)
        papers.extend(p for p in (payload.get("data") or []) if isinstance(p, dict))
        token, page = payload.get("token"), page + 1
        if not token:
            return papers


def build_sample() -> dict:
    strata, pools = {}, {}
    for venue in VENUES:
        for year in YEARS:
            name = f"{venue}|{year}"
            raw = fetch_stratum(venue, year)
            seen, exact = set(), []
            for p in raw:   # the venue filter is exact; double-check and dedupe
                if p.get("paperId") and p.get("venue") == venue and p.get("year") == year \
                        and p["paperId"] not in seen:
                    seen.add(p["paperId"])
                    exact.append(p)
            with_abs = [p for p in exact if (p.get("abstract") or "").strip()]
            pools[name] = with_abs
            strata[name] = {"venue": venue, "year": year, "returned": len(raw),
                            "total": len(exact), "no_abstract": len(exact) - len(with_abs),
                            "with_abstract": len(with_abs)}
            print(f"{name}: returned {len(raw)}, total {len(exact)}, "
                  f"no abstract {strata[name]['no_abstract']}", flush=True)
    alloc = allocate({k: len(v) for k, v in pools.items()}, N)
    rows = []
    for name, pool in pools.items():
        strata[name]["allocated"] = alloc[name]
        for p in sample_stratum(pool, alloc[name], seed=SEED, stratum=name):
            rows.append({"paperId": p["paperId"], "title": p.get("title") or "",
                         "abstract": p["abstract"], "venue": p["venue"], "year": p["year"],
                         "stratum": name})
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_write_text(OUT / "papers.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    atomic_write_text(OUT / "strata.json", json.dumps(strata, indent=1))
    return strata


def read_labels() -> dict[str, list[int] | None]:
    path, out = OUT / "labels.jsonl", {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["paperId"], r["labels"])
    return out


def label(rows: list[dict], topics) -> TopicTagger:
    tagger = TopicTagger(llm=CachedLLM(RoutedLLM(), LLM_CACHE), model=MODEL, topics=topics,
                         max_labels=8, refusal_log=OUT / "tagger_refusals.jsonl")
    done = read_labels()
    todo = [r for r in rows if r["paperId"] not in done]
    print(f"labels: {len(done)} done, {len(todo)} to go", flush=True)
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        texts = [Paper(paper_id=r["paperId"], title=r["title"], abstract=r["abstract"],
                       year=r["year"], venue=r["venue"], pub_date="", citations=0,
                       branch="venue").text() for r in chunk]
        got = tagger.label_many(texts, workers=8)
        with (OUT / "labels.jsonl").open("a") as f:
            f.writelines(json.dumps({"paperId": r["paperId"], "labels": g}) + "\n"
                         for r, g in zip(chunk, got))
        print(f"labeled {len(done) + i + len(chunk)}/{len(rows)} "
              f"(refusals so far {tagger.refusals})", flush=True)
    return tagger


def main() -> None:
    load_env()
    t0 = time.time()
    if not (OUT / "papers.jsonl").exists():
        build_sample()
    strata = json.loads((OUT / "strata.json").read_text())
    rows = [json.loads(x) for x in (OUT / "papers.jsonl").read_text().splitlines() if x.strip()]
    topics = load_topics(TOPICS, expected=None)
    label(rows, topics)
    done = read_labels()
    ids = [r["paperId"] for r in rows]
    table = coverage_table(ids, [done[i] for i in ids], topics)
    table = {"topics_file": str(TOPICS), "tagger": MODEL, "seed": SEED,
             "strata": strata, **table}
    atomic_write_text(TABLE, json.dumps(table, indent=1) + "\n")
    print(f"wrote {TABLE}: {table['labeled']} labeled, {table['unlabeled']} unlabeled; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
