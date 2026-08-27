"""LLM clients: a Protocol, a test fake, a disk cache, and the real Anthropic client."""
import hashlib
import json
from pathlib import Path
from typing import Protocol


class LLM(Protocol):
    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str: ...


class FakeLLM:
    """Returns canned responses in order, then `default`. Records every call."""

    def __init__(self, responses=None, default: str = "ok"):
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[dict] = []

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.responses.pop(0) if self.responses else self.default


class CachedLLM:
    """Disk cache keyed by sha256(model+system+user). Reproducibility + cost control (spec §3.2)."""

    def __init__(self, inner: LLM, cache_dir: Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model: str, system: str, user: str) -> Path:
        payload = json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(payload.encode()).hexdigest() + ".json")

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        path = self._path(model, system, user)
        if path.exists():
            return json.loads(path.read_text())["response"]
        response = self.inner.complete(model=model, system=system, user=user, max_tokens=max_tokens)
        path.write_text(json.dumps(
            {"model": model, "system": system, "user": user, "response": response}))
        return response


class AnthropicLLM:
    """Real client. Needs ANTHROPIC_API_KEY in the environment."""

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        msg = self.client.messages.create(
            model=model, system=system, max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
