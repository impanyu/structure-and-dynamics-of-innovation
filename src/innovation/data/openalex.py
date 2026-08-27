"""Thin OpenAlex API client with disk caching. https://docs.openalex.org"""
import json
from pathlib import Path

import requests

OPENALEX_BASE = "https://api.openalex.org"


def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; invert back to text."""
    if not inv:
        return ""
    positions = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(positions))


def _cached_json(cache_file: Path, fetch):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    payload = fetch()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload))
    return payload


def find_source_id(name: str, *, mailto: str, cache_dir: Path, http_get=None) -> str:
    """Resolve a venue name to its OpenAlex source id (short form, e.g. 'S999')."""
    http_get = http_get or requests.get
    cache_file = Path(cache_dir) / f"source_{name.replace(' ', '_')}.json"

    def fetch():
        r = http_get(f"{OPENALEX_BASE}/sources",
                     params={"search": name, "mailto": mailto})
        r.raise_for_status()
        return r.json()

    payload = _cached_json(cache_file, fetch)
    full_id = payload["results"][0]["id"]  # top hit; audited via CLI in Task 14
    return full_id.rsplit("/", 1)[-1]


def _fetch_works(filter_str: str, cache_key: str, *, mailto: str,
                 cache_dir: Path, http_get) -> list[dict]:
    """Cursor-paginate /works for a filter, caching each page on disk."""
    works: list[dict] = []
    cursor, page_i = "*", 0
    while cursor:
        cache_file = Path(cache_dir) / f"works_{cache_key}_p{page_i}.json"

        def fetch(cursor=cursor):
            r = http_get(f"{OPENALEX_BASE}/works", params={
                "filter": filter_str, "per-page": 200,
                "cursor": cursor, "mailto": mailto})
            r.raise_for_status()
            return r.json()

        page = _cached_json(cache_file, fetch)
        works.extend(page["results"])
        cursor = page["meta"].get("next_cursor")
        page_i += 1
    return works


def fetch_source_works(source_id: str, year_from: int, year_to: int, *,
                       mailto: str, cache_dir: Path, http_get=None) -> list[dict]:
    """All works of a source in [year_from, year_to]."""
    return _fetch_works(
        (f"primary_location.source.id:{source_id},"
         f"publication_year:{year_from}-{year_to}"),
        f"{source_id}_{year_from}_{year_to}",
        mailto=mailto, cache_dir=cache_dir, http_get=http_get or requests.get)


def fetch_field_works(query: str, year_from: int, year_to: int, *,
                      mailto: str, cache_dir: Path, http_get=None,
                      min_citations: int = 0) -> list[dict]:
    """All works matching a small-field keyword query in [year_from, year_to].
    min_citations > 0 adds a cited_by_count floor to bound corpus size."""
    filter_str = (f"title_and_abstract.search:{query},"
                  f"publication_year:{year_from}-{year_to}")
    if min_citations > 0:
        filter_str += f",cited_by_count:>{min_citations}"
    cache_key = f"field_{query.replace(' ', '_')}_{year_from}_{year_to}_c{min_citations}"
    return _fetch_works(filter_str, cache_key, mailto=mailto,
                        cache_dir=cache_dir, http_get=http_get or requests.get)
