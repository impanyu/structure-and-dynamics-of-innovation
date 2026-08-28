"""Augment S2 citation edges with OpenAlex references.

S2's reference coverage for NeurIPS/ICLR is incomplete (~35% of papers carry
references), which would leave half the idea network isolated. OpenAlex has
strong reference coverage but broken venue labels — so we match our S2 papers
into OpenAlex by identifier and take referenced_works from there:

  1. MAG id  -> OpenAlex work id is literally "W<mag>"
  2. DOI     -> filter=doi:...
  3. arXiv   -> DataCite DOI "10.48550/arxiv.<id>"

Edges from both sources are unioned. All calls disk-cached (spec §3.2 rules).
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import requests

from innovation.data.s2 import S2_BATCH, _cached_call, s2_headers

OPENALEX_WORKS = "https://api.openalex.org/works"


def s2_fetch_external_ids(paper_ids: list[str], *, cache_dir, http_post=None,
                          delay: float = 4.0, batch_size: int = 500) -> dict:
    """paper_id -> externalIds dict (MAG/DOI/ArXiv/...), via the batch endpoint."""
    http_post = http_post or requests.post
    out: dict[str, dict] = {}
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        key = hashlib.sha256(json.dumps(batch).encode()).hexdigest()[:24]
        cache_file = Path(cache_dir) / f"s2ext_{key}.json"
        payload = _cached_call(
            cache_file,
            lambda: http_post(S2_BATCH, params={"fields": "externalIds"},
                              json={"ids": batch}, headers=s2_headers()), delay)
        for entry in payload or []:
            if isinstance(entry, dict) and entry.get("paperId"):
                out[entry["paperId"]] = entry.get("externalIds") or {}
    return out


def _lookup_key(ext: dict) -> tuple[str, str] | None:
    """Preferred OpenAlex lookup key for one paper: MAG > DOI > arXiv."""
    if ext.get("MAG"):
        return ("openalex", f"W{ext['MAG']}")
    if ext.get("DOI"):
        return ("doi", ext["DOI"].lower())
    if ext.get("ArXiv"):
        return ("doi", f"10.48550/arxiv.{ext['ArXiv']}".lower())
    return None


def _fetch_openalex_chunk(filter_key: str, values: list[str], *, mailto,
                          cache_dir, http_get, delay: float) -> list[dict]:
    filter_str = f"{filter_key}:{'|'.join(values)}"
    key = hashlib.sha256(filter_str.encode()).hexdigest()[:24]
    cache_file = Path(cache_dir) / f"oaw_{key}.json"
    payload = _cached_call(
        cache_file,
        lambda: http_get(OPENALEX_WORKS, params={
            "filter": filter_str, "select": "id,doi,referenced_works",
            "per-page": 200, "mailto": mailto}), delay)
    return payload.get("results", [])


def augment_edges(papers: pd.DataFrame, s2_edges: pd.DataFrame, *,
                  cache_dir, mailto: str, http_get=None, http_post=None,
                  delay: float = 1.0, chunk: int = 50) -> pd.DataFrame:
    """Union of S2 edges and OpenAlex-derived edges over the same paper set."""
    http_get = http_get or requests.get
    ids = list(papers["paper_id"])
    ext = s2_fetch_external_ids(ids, cache_dir=cache_dir, http_post=http_post,
                                delay=delay)

    by_filter: dict[str, dict[str, str]] = {"openalex": {}, "doi": {}}
    oa2s2: dict[str, str] = {}
    for pid in ids:
        e = ext.get(pid) or {}
        if e.get("MAG"):  # free mapping even when we look the paper up by DOI
            oa2s2[f"W{e['MAG']}"] = pid
        lk = _lookup_key(e)
        if lk:
            by_filter[lk[0]][lk[1]] = pid

    refs_by_s2: dict[str, list[str]] = {}
    for fkey, mapping in by_filter.items():
        values = list(mapping)
        for i in range(0, len(values), chunk):
            for work in _fetch_openalex_chunk(fkey, values[i:i + chunk],
                                              mailto=mailto, cache_dir=cache_dir,
                                              http_get=http_get, delay=delay):
                wid = (work.get("id") or "").rsplit("/", 1)[-1]
                doi = (work.get("doi") or "").removeprefix("https://doi.org/").lower()
                pid = oa2s2.get(wid) or mapping.get(doi) or mapping.get(wid)
                if not pid:
                    continue
                oa2s2[wid] = pid
                existing = refs_by_s2.get(pid, [])
                if len(work.get("referenced_works") or []) > len(existing):
                    refs_by_s2[pid] = work["referenced_works"]

    edge_rows = [{"src": pid, "dst": oa2s2[ref.rsplit("/", 1)[-1]]}
                 for pid, refs in refs_by_s2.items()
                 for ref in refs if ref.rsplit("/", 1)[-1] in oa2s2]
    merged = pd.concat([s2_edges, pd.DataFrame(edge_rows, columns=["src", "dst"])],
                       ignore_index=True)
    merged = merged[merged["src"] != merged["dst"]].drop_duplicates()
    return merged.reset_index(drop=True)
