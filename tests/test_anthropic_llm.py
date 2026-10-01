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
    return NS(content=list(blocks))


class _Client:
    def __init__(self, replies=(), failures=()):
        self.replies = list(replies)
        self.failures = list(failures)
        self.calls = []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.failures:
            raise self.failures.pop(0)
        return self.replies.pop(0)


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
