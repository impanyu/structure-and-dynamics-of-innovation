"""LLM clients: a Protocol, a test fake, a disk cache, and the real Anthropic client."""
import hashlib
import json
import time
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
    """Real client. Needs ANTHROPIC_API_KEY in the environment.

    Claude 5 models think by default: replies start with a thinking block and
    thinking tokens count against max_tokens, so we add headroom and return
    only the text blocks."""

    # Same capped backoff as OpenAILLM (~30 minutes of outage tolerance).
    RETRY_DELAYS = [5, 10, 20, 40, 60, 120, 180, 300, 300, 300, 300]
    HEADROOM = 4000

    def __init__(self, client=None, sleep=time.sleep, thinking=None):
        import anthropic

        # thinking=None keeps the model default (adaptive thinking on Claude 5);
        # pass {"type": "disabled"} for long one-shot generations where
        # thinking would exhaust max_tokens before any text is written.
        self._extra = {"thinking": thinking} if thinking else {}
        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic()
        self._sleep = sleep

    def _transient(self, e: Exception) -> bool:
        a = self._anthropic
        return isinstance(e, (a.RateLimitError, a.InternalServerError,
                              a.APIConnectionError, a.APITimeoutError))

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        for delay in [*self.RETRY_DELAYS, None]:
            try:
                # Streaming: the SDK refuses non-streaming calls with large max_tokens.
                with self.client.messages.stream(
                        model=model, system=system, max_tokens=max_tokens + self.HEADROOM,
                        messages=[{"role": "user", "content": user}],
                        **self._extra) as stream:
                    msg = stream.get_final_message()
                break
            except Exception as e:
                if delay is None or not self._transient(e):
                    raise
                print(f"[anthropic] {type(e).__name__}; retrying in {delay}s", flush=True)
                self._sleep(delay)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        if not text:
            raise ValueError(f"no text in reply (stop_reason={msg.stop_reason})")
        return text


def parse_openai_model(model: str) -> tuple[str, str | None]:
    """'gpt-5-mini:minimal' -> ('gpt-5-mini', 'minimal'). GPT-5 models are
    reasoning models: reasoning tokens consume max_output_tokens, so callers
    pick an effort per role (summarizer: minimal; judge: low; agent: default)."""
    name, _, effort = model.partition(":")
    return name, (effort or None)


class OpenAILLM:
    """Real OpenAI client. Needs OPENAI_API_KEY in the environment."""

    # Server-side failures (5xx, 429, dropped connections) are retried with
    # capped exponential backoff for up to ~30 minutes, so a provider outage
    # stalls a long run instead of killing it. Client errors (4xx) still raise.
    RETRY_DELAYS = [5, 10, 20, 40, 60, 120, 180, 300, 300, 300, 300]

    def __init__(self, client=None, sleep=time.sleep):
        import openai

        self._openai = openai
        self.client = client or openai.OpenAI()
        self._sleep = sleep

    def _transient(self, e: Exception) -> bool:
        o = self._openai
        return isinstance(e, (o.InternalServerError, o.RateLimitError,
                              o.APIConnectionError, o.APITimeoutError))

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        name, effort = parse_openai_model(model)
        kwargs = {}
        if effort:
            kwargs["reasoning"] = {"effort": effort}
        # Headroom so reasoning tokens cannot starve the visible answer.
        headroom = 200 if effort == "minimal" else 2000
        for delay in [*self.RETRY_DELAYS, None]:
            try:
                resp = self.client.responses.create(
                    model=name, instructions=system, input=user,
                    max_output_tokens=max_tokens + headroom, **kwargs)
                return resp.output_text
            except Exception as e:
                if delay is None or not self._transient(e):
                    raise
                print(f"[openai] {type(e).__name__}; retrying in {delay}s", flush=True)
                self._sleep(delay)


class RoutedLLM:
    """Dispatch by model id: 'openai:<model>' -> OpenAI, anything else ->
    Anthropic. Clients are constructed lazily so only the providers actually
    used need credentials. Cache keys (CachedLLM) include the full prefixed
    model string, so providers never collide in the cache."""

    def __init__(self, anthropic_factory=AnthropicLLM, openai_factory=None):
        self._anthropic_factory = anthropic_factory
        self._openai_factory = openai_factory or OpenAILLM
        self._clients: dict[str, LLM] = {}

    def _client(self, provider: str) -> LLM:
        if provider not in self._clients:
            factory = (self._openai_factory if provider == "openai"
                       else self._anthropic_factory)
            self._clients[provider] = factory()
        return self._clients[provider]

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        if model.startswith("openai:"):
            return self._client("openai").complete(
                model=model.split(":", 1)[1], system=system, user=user,
                max_tokens=max_tokens)
        return self._client("anthropic").complete(
            model=model, system=system, user=user, max_tokens=max_tokens)
