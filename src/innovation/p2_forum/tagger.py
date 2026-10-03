"""The independent labeling LLM (spec §4). One call per item; the frozen list is
a fixed system-prompt prefix so a provider-side prompt cache can reuse it.
Agents never label their own work: this runs inside the environment."""
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from innovation.p2_forum.topics import Topic


class UnlabeledError(Exception):
    pass


def is_refusal(e: Exception) -> bool:
    """The LLM client's error for a reply without text because the model
    refused (core.llm.AnthropicLLM); other no-text stop reasons are not."""
    msg = str(e)
    return (isinstance(e, ValueError) and msg.startswith("no text in reply")
            and "stop_reason=refusal" in msg)


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
        if type(x) is bool:
            return None
        if isinstance(x, str) and x.isdecimal():
            x = int(x)
        if not isinstance(x, int) or not 0 <= x < n_topics:
            return None
        if x not in out:
            out.append(x)
    return out if 1 <= len(out) <= max_labels else None


class TopicTagger:
    def __init__(self, *, llm, model: str, topics: list[Topic],
                 max_labels: int = 5, attempts: int = 3,
                 refusal_log: Path | None = None):
        self.llm, self.model, self.topics = llm, model, topics
        self.max_labels, self.attempts = max_labels, attempts
        # A refusal is a failed attempt (retried; after the last one the item
        # is unlabeled), never a crash; each one is counted and logged.
        self.refusal_log = Path(refusal_log) if refusal_log else None
        self.refusals = 0
        self._lock = threading.Lock()
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
            try:
                reply = self.llm.complete(model=self.model, system=self.system,
                                          user=user, max_tokens=100)
            except ValueError as e:
                if not is_refusal(e):
                    raise
                self._record_refusal(text, attempt)
                continue
            labels = parse_labels(reply, len(self.topics), self.max_labels)
            if labels is not None:
                return labels
        raise UnlabeledError(f"no valid labels after {self.attempts} attempts")

    def _record_refusal(self, text: str, attempt: int) -> None:
        line = json.dumps({"ts": time.time(), "model": self.model,
                           "text_head": text[:200], "attempt": attempt}) + "\n"
        with self._lock:
            self.refusals += 1
            if self.refusal_log:
                self.refusal_log.parent.mkdir(parents=True, exist_ok=True)
                with self.refusal_log.open("a") as f:
                    f.write(line)

    def label_many(self, texts: list[str], workers: int = 8) -> list[list[int] | None]:
        def one(t):
            try:
                return self.label(t)
            except UnlabeledError:
                return None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, texts))
