"""Append-only JSONL event log; the action trace is primary research data (spec §3.5)."""
import json
from pathlib import Path


def load_events(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = len(load_events(self.path))

    def append(self, event: dict) -> dict:
        enriched = {"seq": self._seq, **event}
        with self.path.open("a") as f:
            f.write(json.dumps(enriched) + "\n")
        self._seq += 1
        return enriched

    def read_all(self) -> list[dict]:
        return load_events(self.path)
