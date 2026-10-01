from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest

from innovation.core.llm import AnthropicLLM


def _err(cls, status):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


THINK = NS(type="thinking", thinking="hmm")


def _reply(*blocks):
    return NS(content=list(blocks), stop_reason="end_turn")


class _Stream:
    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.reply


class _Client:
    def __init__(self, replies=(), failures=()):
        self.replies = list(replies)
        self.failures = list(failures)
        self.calls = []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        if self.failures:
            raise self.failures.pop(0)
        return _Stream(self.replies.pop(0))


def _llm(c, slept=None):
    return AnthropicLLM(client=c, sleep=(slept.append if slept is not None else lambda s: None))


def test_concatenates_text_blocks_and_skips_thinking():
    c = _Client([_reply(THINK, NS(type="text", text="a"), NS(type="text", text="b"))])
    assert _llm(c).complete(model="m", system="s", user="u", max_tokens=100) == "ab"
    assert c.calls[0]["max_tokens"] == 4100


def test_reply_without_text_raises():
    c = _Client([_reply(THINK)])
    with pytest.raises(ValueError, match="no text in reply"):
        _llm(c).complete(model="m", system="s", user="u")


def test_server_errors_are_retried_until_success():
    slept = []
    c = _Client([_reply(NS(type="text", text="fine"))],
                failures=[_err(anthropic.InternalServerError, 529),
                          _err(anthropic.InternalServerError, 500)])
    assert _llm(c, slept).complete(model="m", system="s", user="u") == "fine"
    assert len(c.calls) == 3 and slept == [5, 10]


def test_client_errors_raise_immediately():
    c = _Client(failures=[_err(anthropic.BadRequestError, 400)])
    with pytest.raises(anthropic.BadRequestError):
        _llm(c).complete(model="m", system="s", user="u")
    assert len(c.calls) == 1


def test_thinking_param_is_forwarded_only_when_set():
    c = _Client([_reply(NS(type="text", text="x")), _reply(NS(type="text", text="y"))])
    AnthropicLLM(client=c, sleep=lambda s: None).complete(model="m", system="s", user="u")
    AnthropicLLM(client=c, sleep=lambda s: None, thinking={"type": "disabled"}).complete(
        model="m", system="s", user="u")
    assert "thinking" not in c.calls[0] and c.calls[1]["thinking"] == {"type": "disabled"}
