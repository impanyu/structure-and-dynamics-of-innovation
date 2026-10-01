"""The independent labeling LLM (spec §4). One call per item; the frozen list is
a fixed system-prompt prefix so a provider-side prompt cache can reuse it.
Agents never label their own work: this runs inside the environment."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from innovation.p2_forum.topics import Topic


class UnlabeledError(Exception):
    pass


def parse_labels(reply: str, n_topics: int, max_labels: int = 5) -> list[int] | None:
    m = re.search(r"\[[^\[\]]*\]", reply)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: list[int] = []
    for x in raw:
        if not isinstance(x, int) or not 0 <= x < n_topics:
            return None
        if x not in out:
            out.append(x)
    return out if 1 <= len(out) <= max_labels else None


class TopicTagger:
    def __init__(self, *, llm, model: str, topics: list[Topic],
                 max_labels: int = 5, attempts: int = 3):
        self.llm, self.model, self.topics = llm, model, topics
        self.max_labels, self.attempts = max_labels, attempts
        listing = "\n".join(f"{t.id}: {t.name} — {t.definition}" for t in topics)
        self.system = (
            "You label AI research texts with topics from a fixed list.\n\n"
            f"TOPICS (id: name — definition):\n{listing}\n\n"
            f"Reply with ONLY a JSON list of 1 to {max_labels} topic ids, most "
            "relevant first. Include a topic only if the text genuinely belongs "
            f"to it; never pad the list to {max_labels}.")

    def label(self, text: str) -> list[int]:
        for attempt in range(self.attempts):
            # The attempt number is part of the prompt, so a disk cache keyed on
            # the prompt never replays a reply that already failed to parse.
            user = f"TEXT:\n{text}" + (f"\n\n(retry {attempt})" if attempt else "")
            reply = self.llm.complete(model=self.model, system=self.system,
                                      user=user, max_tokens=100)
            labels = parse_labels(reply, len(self.topics), self.max_labels)
            if labels is not None:
                return labels
        raise UnlabeledError(f"no valid labels after {self.attempts} attempts")

    def label_many(self, texts: list[str], workers: int = 8) -> list[list[int] | None]:
        def one(t):
            try:
                return self.label(t)
            except UnlabeledError:
                return None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, texts))
