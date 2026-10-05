"""Topics-v3: synonym-only deduplication of the seven venues' official
submission areas (configs/p2_forum/venue_areas.yaml).

Only synonyms and duplicates are merged; umbrella and fine areas stay side
by side, nothing is split or invented, and only literal "Other ..." bins are
dropped. Pure helpers here; the live LLM passes live in
scripts/p2_forum/build_topics_v3.py.

Pipeline: areas -> drop catch-alls -> exact-duplicate units (normalized
string within one domain) -> LLM synonym groups (per domain batch, then one
cross-batch pass) -> LLM names + definitions -> draft YAML."""
import json
import re
from dataclasses import dataclass

import yaml

VENUE_ORDER = ("NeurIPS", "ICML", "ICLR", "AAAI", "ACL", "CVPR", "ICCV")
# Domain of a whole venue; AAAI areas take their keyword prefix instead.
VENUE_DOMAIN = {"NeurIPS": "ML", "ICML": "ML", "ICLR": "ML",
                "ACL": "NLP", "CVPR": "CV", "ICCV": "CV"}
DOMAIN_NAMES = {
    "APP": "Application Domains", "CMS": "Cognitive Modeling & Cognitive Systems",
    "CV": "Computer Vision", "CSO": "Constraint Satisfaction and Optimization",
    "DMKM": "Data Mining & Knowledge Management",
    "GTEP": "Game Theory and Economic Paradigms", "HAI": "Humans and AI",
    "ROB": "Intelligent Robots", "KRR": "Knowledge Representation and Reasoning",
    "ML": "Machine Learning", "MAS": "Multiagent Systems",
    "PEAI": "Philosophy and Ethics of AI", "PRS": "Planning, Routing, and Scheduling",
    "RU": "Reasoning under Uncertainty", "SO": "Search and Optimization",
    "NLP": "Speech & Natural Language Processing"}
# Synonym-pass batches: domains whose areas plausibly share synonyms. A
# cross-batch pass afterwards catches synonyms that straddle batches.
SYNONYM_BATCHES = (
    ("machine learning", ("ML",)),
    ("computer vision", ("CV",)),
    ("natural language processing", ("NLP",)),
    ("agents, games, humans, cognition and ethics", ("GTEP", "MAS", "HAI", "CMS", "PEAI")),
    ("reasoning, planning, search and constraints", ("KRR", "PRS", "RU", "SO", "CSO")),
    ("application domains, data mining and robotics", ("APP", "DMKM", "ROB")),
)
_PREFIX = re.compile(r"^([A-Z]{2,5}):\s*(.*)$")
_VENUE_WORD = re.compile(r"\b(" + "|".join(VENUE_ORDER) + r"|venues?|tracks?)\b", re.I)
DRAFT_HEADER = ("# DRAFT (synonym-only dedupe; umbrellas kept). Review, then rename to\n"
                "# topics-v3.yaml to freeze; ids = list positions.\n")


@dataclass(frozen=True)
class Area:
    venue: str
    area: str      # verbatim from venue_areas.yaml (provenance)
    domain: str    # AAAI keyword prefix, or the venue's domain
    keyword: str   # area without its "ML:"-style prefix

    @property
    def source(self) -> dict:
        return {"venue": self.venue, "area": self.area}


def split_prefix(area: str) -> tuple[str, str]:
    """'ML: Clustering' -> ('ML', 'Clustering'); no prefix -> ('', area)."""
    m = _PREFIX.match(area)
    return (m.group(1), m.group(2).strip()) if m else ("", area.strip())


def normalize(text: str) -> str:
    """Case-, punctuation- and 'and'/'&'-insensitive key: 'Self-& semi-&
    meta-& unsupervised learning' == 'Self-, semi-, meta-, unsupervised
    learning'; 'Vision + graphics' == 'Vision and graphics'."""
    tokens = re.sub(r"[^a-z0-9]+", " ", text.lower().replace("&", " and ")).split()
    return " ".join(t for t in tokens if t != "and")


def load_areas(venues: list[dict]) -> list[Area]:
    """All areas in VENUE_ORDER, then source order (the stable topic order)."""
    by_name = {v["venue"]: v for v in venues}
    if set(by_name) != set(VENUE_ORDER):
        raise ValueError(f"expected venues {VENUE_ORDER}, got {sorted(by_name)}")
    out = []
    for name in VENUE_ORDER:
        for a in by_name[name]["areas"]:
            prefix, kw = split_prefix(a) if name == "AAAI" else ("", a.strip())
            domain = prefix if name == "AAAI" else VENUE_DOMAIN[name]
            if domain not in DOMAIN_NAMES:
                raise ValueError(f"unknown domain {domain!r} for {name}: {a!r}")
            out.append(Area(venue=name, area=a, domain=domain, keyword=kw))
    return out


def is_catch_all(area: Area) -> bool:
    """Literal 'Other ...'/'Miscellaneous' bins (e.g. 'ML: Other Foundations of
    Machine Learning', 'NLP: Other'). 'Machine learning (other than deep
    learning)' is a real area and is kept."""
    key = normalize(area.keyword)
    return key == "other" or key.startswith("other ") or key == "miscellaneous"


def exact_duplicate_units(areas: list[Area], keep: list[int]) -> list[list[int]]:
    """Group the kept area indices whose normalized keyword is identical within
    one domain (e.g. CVPR and ICCV 'Biometrics'; AAAI 'ML: Optimization' and
    ICLR 'optimization'). Across domains, identical strings are left to the
    LLM pass: 'CV: Applications' and 'NLP: Applications' are different areas."""
    units: dict[tuple[str, str], list[int]] = {}
    for i in keep:
        units.setdefault((areas[i].domain, normalize(areas[i].keyword)), []).append(i)
    return sorted(units.values(), key=min)


def validate_partition(groups, n: int) -> list[list[int]]:
    """Every index 0..n-1 exactly once across non-empty integer groups."""
    if not isinstance(groups, list):
        raise ValueError("groups must be a JSON list of lists")
    seen: set[int] = set()
    for g in groups:
        if not isinstance(g, list) or not g:
            raise ValueError(f"each group must be a non-empty list, got {g!r}")
        for i in g:
            if not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < n:
                raise ValueError(f"index {i!r} out of range 0..{n - 1}")
            if i in seen:
                raise ValueError(f"index {i} appears more than once")
            seen.add(i)
    missing = sorted(set(range(n)) - seen)
    if missing:
        raise ValueError(f"indices missing from the groups: {missing}")
    return [list(g) for g in groups]


def _json_span(reply: str, open_ch: str, close_ch: str):
    start, end = reply.find(open_ch), reply.rfind(close_ch)
    if start == -1 or end <= start:
        raise ValueError("no JSON list in reply")
    try:
        return json.loads(reply[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


def _answer(reply: str) -> str:
    """The text inside the last <answer>...</answer>, else the whole reply."""
    m = re.findall(r"<answer>(.*?)</answer>", reply, flags=re.S)
    return m[-1] if m else reply


def parse_groups(reply: str, n: int) -> list[list[int]]:
    """The reply lists only the synonym sets (size >= 2); every unlisted item
    becomes a singleton. The completed groups must partition 0..n-1, so
    duplicate or out-of-range indices are rejected."""
    sets = _json_span(_answer(reply), "[", "]")
    if not isinstance(sets, list) or not all(isinstance(g, list) for g in sets):
        raise ValueError("groups must be a JSON list of lists")
    for g in sets:
        if len(g) < 2:
            raise ValueError(f"a synonym set needs at least two items, got {g!r}")
    listed = {i for g in sets for i in g if isinstance(i, int)}
    return validate_partition(sets + [[i] for i in range(n) if i not in listed], n)


def verify_prompt(candidate: list[int], units: list[list[int]], areas: list[Area],
                  attempt: int = 1, feedback: str = "") -> str:
    """Second, focused look at one proposed synonym set (items renumbered
    0..m-1): split it into the groups that really are synonyms."""
    m = len(candidate)
    lines = "\n".join(f"{j}. {describe_unit(units[u], areas)}" for j, u in enumerate(candidate))
    prompt = (
        f"These {m} official submission areas of top AI venues were proposed as "
        "synonyms. Each line shows '<venue> [<domain>]: <area>'; a line that already "
        "lists several areas separated by ' | ' is one item.\n\n"
        f"{lines}\n\n"
        "Task: decide which of them really are SYNONYMS — the same research area with "
        "the same scope.\n"
        + SYNONYM_RULES
        + "Give a one-line reason per decision, then the final answer inside "
        "<answer></answer> as a JSON list of the synonym sets only, each a list of at "
        f"least two item numbers from 0..{m - 1}; no number may appear twice; items "
        "not listed stay on their own. Example: <answer>[[0, 2]]</answer>, or "
        "<answer>[]</answer> if none of them are synonyms.")
    if attempt > 1:
        prompt += (f"\n\nAttempt {attempt}: your previous answer was rejected: "
                   f"{feedback}. Fix it.")
    return prompt


def candidate_sets(votes: list[list[list[int]]], n: int) -> list[list[int]]:
    """High-recall proposals: every set some vote merged (connected
    components of all proposed pairs), size >= 2. verify_prompt prunes them."""
    return [g for g in majority_groups(votes, n, min_votes=1) if len(g) > 1]


def verified_groups(n: int, candidates: list[list[int]],
                    local: list[list[list[int]]]) -> list[list[int]]:
    """Map each candidate's verified local partition back to unit indices;
    items outside every candidate stay singletons."""
    for c, lg in zip(candidates, local, strict=True):
        validate_partition(lg, len(c))
    groups = [[c[j] for j in g] for c, lg in zip(candidates, local) for g in lg]
    listed = {i for g in groups for i in g}
    return validate_partition(groups + [[i] for i in range(n) if i not in listed], n)


def majority_groups(votes: list[list[list[int]]], n: int, min_votes: int) -> list[list[int]]:
    """Self-consistency: items i, j end up together when at least min_votes of
    the independent partitions put them in one group (connected components
    of the agreed pairs)."""
    count: dict[tuple[int, int], int] = {}
    for groups in votes:
        validate_partition(groups, n)
        for g in groups:
            for a in g:
                for b in g:
                    if a < b:
                        count[(a, b)] = count.get((a, b), 0) + 1
    parent = list(range(n))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (a, b), c in count.items():
        if c >= min_votes:
            parent[root(a)] = root(b)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(root(i), []).append(i)
    return sorted(comps.values(), key=min)


def apply_groups(units: list[list[int]], groups: list[list[int]]) -> list[list[int]]:
    """Merge units by groups of unit indices; each merged unit keeps every
    area index it absorbed, sorted. Result ordered by first area index."""
    validate_partition(groups, len(units))
    merged = [sorted(i for u in g for i in units[u]) for g in groups]
    return sorted(merged, key=min)


def _context(a: Area) -> str:
    if a.venue == "AAAI":
        return f"AAAI [{DOMAIN_NAMES[a.domain]}]: {a.keyword}"
    if a.venue in ("ACL", "CVPR", "ICCV"):
        return f"{a.venue} [{DOMAIN_NAMES[a.domain]} venue]: {a.area}"
    return f"{a.venue}: {a.area}"


def describe_unit(unit: list[int], areas: list[Area]) -> str:
    return " | ".join(_context(areas[i]) for i in unit)


SYNONYM_RULES = (
    "Rules:\n"
    "- Merge the same area listed at several venues, even when the wording or the "
    "parenthetical example lists differ: a venue's '(e.g., ...)' or '(..., etc.)' "
    "list only illustrates the heading (e.g. 'Optimization (e.g., convex, "
    "stochastic)' at one venue and 'optimization' at another are one area).\n"
    "- Otherwise merge only true synonyms/duplicates. If two different areas are "
    "merely related, do NOT merge.\n"
    "- Be exhaustive: compare every item with every other item; a missed "
    "duplicate is as bad as a wrong merge.\n"
    "- Test: two items are synonyms only if every paper of one clearly belongs to "
    "the other AND vice versa.\n"
    "- Never merge an umbrella area with a narrower area it contains (e.g. "
    "'Deep learning' stays separate from 'Graph neural networks'; 'Recognition: "
    "categorization, detection, retrieval' stays separate from 'Recognition: "
    "Detection').\n"
    "- Never merge merely related or overlapping areas (e.g. 'Privacy' vs "
    "'Fairness', 'Sentiment analysis' vs 'Text classification').\n"
    "- Areas of a domain venue (ACL = NLP; CVPR, ICCV = vision) or under an AAAI "
    "domain are scoped to that domain: merge them with a general or other-domain "
    "area only if both clearly denote the same field (e.g. 'Mechanism Design' "
    "under Game Theory and under Multiagent Systems), never e.g. 'Applications' "
    "of vision with 'Applications' of NLP.\n")


VIEWS = ("alphabetically", "in reverse alphabetical order", "in source order")


def synonym_prompt(units: list[list[int]], areas: list[Area], scope: str,
                   attempt: int = 1, feedback: str = "", cross: bool = False,
                   view: int = 0) -> str:
    """view 0/1 lists items (reverse) alphabetically by wording, so that
    near-identical areas from different venues sit next to each other; view
    2 keeps the unit order. Numbers always refer to the unit order."""
    order = sorted(range(len(units)), key=lambda k: min(
        normalize(areas[i].keyword) for i in units[k]), reverse=view == 1)
    if view == 2:
        order = list(range(len(units)))
    lines = "\n".join(f"{k}. {describe_unit(units[k], areas)}" for k in order)
    n = len(units)
    prompt = (
        f"Below are {n} official submission areas of top AI venues ({scope}), numbered "
        "0.." f"{n - 1} and listed {VIEWS[view]}. Each line shows '<venue> [<domain>]: "
        "<area>'; a line that already lists several areas separated by ' | ' is one "
        "item.\n\n"
        f"{lines}\n\n"
        "Task: find SYNONYMS ONLY — items that name the same research area with the "
        "same scope, just worded differently or listed at several venues.\n"
        + SYNONYM_RULES
        + ("- Items within one domain were already compared with each other; report "
           "only synonym sets whose items come from different domains.\n" if cross else "")
        + "First list the candidate synonym sets briefly with a one-line reason each "
        "(an item that already lists several areas counts as one item; do not report "
        "it on its own). Then give the final answer inside <answer></answer> as a JSON "
        "list of the synonym sets only, each a list of at least two item numbers; no "
        "number may appear twice; items not listed stay on their own. Example: "
        "<answer>[[1, 4], [2, 7, 9]]</answer>, or <answer>[]</answer> if there are none.")
    if attempt > 1:
        prompt += (f"\n\nAttempt {attempt}: your previous answer was rejected: "
                   f"{feedback}. Fix it.")
    return prompt


def naming_prompt(batch: list[tuple[int, list[int]]], areas: list[Area],
                  taken: list[str], attempt: int = 1, feedback: str = "") -> str:
    lines = "\n".join(f"{tid}. {describe_unit(u, areas)}" for tid, u in batch)
    prompt = (
        "Each numbered line below is one research topic, given by the official "
        "submission area(s) it stands for (merged synonyms are separated by ' | ').\n\n"
        f"{lines}\n\n"
        "For each topic write:\n"
        "- name: a clean, complete topic name (2-10 words) that never mentions venues, "
        "tracks or 'ML:'-style prefixes (to scope a domain area write the domain in "
        "plain words, e.g. 'Biometrics in Computer Vision'). Keep the area's exact scope: do not broaden or narrow "
        "it, and keep every component the source lists (e.g. all of 'self-, semi-, "
        "meta-, unsupervised'). A generic area scoped to a domain needs the domain in its name (e.g. "
        "AAAI [Computer Vision]: Applications -> 'Computer Vision Applications'). "
        "Some AAAI strings are cut off on the source page (e.g. '... & Retri', "
        "'... KB Completio'); complete them sensibly in the name.\n"
        "- definition: ONE sentence saying what is in the topic and what is out "
        "(name a neighbouring topic it excludes), so a tagger can apply it.\n"
        "Names must be unique (case-insensitive) within your answer"
        + (" and must not repeat any of these already-used names: "
           + "; ".join(taken) if taken else "") + ".\n"
        'Reply with ONLY a JSON list: [{"id": <topic number>, "name": "...", '
        '"definition": "..."}, ...] with exactly one object per topic number above.')
    if attempt > 1:
        prompt += (f"\n\nAttempt {attempt}: your previous answer was rejected: "
                   f"{feedback}. Fix it.")
    return prompt


def parse_names(reply: str, ids: list[int], taken: list[str]) -> dict[int, tuple[str, str]]:
    items = _json_span(reply, "[", "]")
    if not isinstance(items, list):
        raise ValueError("reply must be a JSON list")
    used = {t.lower() for t in taken}
    out: dict[int, tuple[str, str]] = {}
    for t in items:
        if not isinstance(t, dict):
            raise ValueError(f"expected objects, got {t!r}")
        tid, name = t.get("id"), str(t.get("name", "")).strip()
        definition = str(t.get("definition", "")).strip()
        if tid not in ids or tid in out:
            raise ValueError(f"unexpected or repeated id {tid!r}")
        if not name or name.lower() in used:
            raise ValueError(f"empty or duplicate name {name!r}")
        if _PREFIX.match(name):
            raise ValueError(f"name {name!r} keeps a venue prefix")
        if _VENUE_WORD.search(name):
            raise ValueError(f"name {name!r} mentions a venue")
        if not definition:
            raise ValueError(f"topic {name!r} has no definition")
        used.add(name.lower())
        out[tid] = (name, definition)
    missing = sorted(set(ids) - set(out))
    if missing:
        raise ValueError(f"missing ids: {missing}")
    return out


def fallback_name(unit: list[int], areas: list[Area], taken: list[str]) -> tuple[str, str]:
    """Deterministic name when the naming LLM fails: the first source's
    keyword, qualified by its domain if the bare keyword is already used."""
    a = areas[unit[0]]
    used = {t.lower() for t in taken}
    name = a.keyword
    if name.lower() in used:
        name = f"{a.keyword} ({DOMAIN_NAMES[a.domain]})"
    k = 2
    base = name
    while name.lower() in used:
        name, k = f"{base} {k}", k + 1
    return name, f"Research on {name} as listed by {a.venue}: {a.area}."


def build_topics(units: list[list[int]], areas: list[Area],
                 names: dict[int, tuple[str, str]]) -> list[dict]:
    """Topic dicts in stable order (first source's venue order, then source
    order); sources are the verbatim {venue, area} of every absorbed area."""
    order = sorted(range(len(units)), key=lambda k: min(units[k]))
    topics = []
    for k in order:
        name, definition = names[k]
        topics.append({"name": name, "definition": definition,
                       "sources": [areas[i].source for i in sorted(units[k])]})
    lowered = [t["name"].lower() for t in topics]
    if len(set(lowered)) != len(lowered):
        raise ValueError("duplicate topic names (case-insensitive)")
    return topics


def draft_yaml(topics: list[dict], dropped: list[Area]) -> str:
    drops = "".join(f"#   {a.venue}: {a.area}\n" for a in dropped)
    return (DRAFT_HEADER
            + "# Built from configs/p2_forum/venue_areas.yaml by\n"
              "# scripts/p2_forum/build_topics_v3.py.\n"
            + (f"# Dropped catch-all bins ({len(dropped)}):\n{drops}" if dropped else "")
            + yaml.safe_dump({"topics": topics}, allow_unicode=True,
                             sort_keys=False, width=100))
