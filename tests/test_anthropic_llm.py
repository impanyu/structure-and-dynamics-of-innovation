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
    def __init__(self, reply, fail=None):
        self.reply, self.fail = reply, fail

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        if self.fail:
            raise self.fail
        return self.reply


class _Client:
    def __init__(self, replies=(), failures=(), midstream=()):
        self.midstream = list(midstream)
        self.replies = list(replies)
        self.failures = list(failures)
        self.calls = []
        self.messages = self

    def stream(self, **kw):
        self.calls.append(kw)
        if self.failures:
            raise self.failures.pop(0)
        fail = self.midstream.pop(0) if self.midstream else None
        return _Stream(self.replies[0] if fail else self.replies.pop(0), fail)


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


def test_midstream_error_event_is_retried():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    e = anthropic.APIStatusError("overloaded", response=httpx.Response(200, request=req),
                                 body={"type": "error"})
    slept = []
    c = _Client([_reply(NS(type="text", text="ok"))], midstream=[e])
    assert _llm(c, slept).complete(model="m", system="s", user="u") == "ok"
    assert slept == [5] and len(c.calls) == 2


_TRANSPORT = [httpx.RemoteProtocolError]
try:
    import httpx2
    _TRANSPORT.append(httpx2.RemoteProtocolError)
except ImportError:
    pass


@pytest.mark.parametrize("cls", _TRANSPORT)
def test_bare_transport_error_during_stream_is_retried(cls):
    slept = []
    c = _Client([_reply(NS(type="text", text="ok"))], midstream=[cls("cut")])
    assert _llm(c, slept).complete(model="m", system="s", user="u") == "ok"
    assert slept == [5]


def test_system_prompt_is_sent_as_cacheable_block():
    c = _Client(replies=[_reply(NS(type="text", text="ok"))])
    _llm(c).complete(model="m", system="big prompt", user="u")
    assert c.calls[0]["system"] == [
        {"type": "text", "text": "big prompt", "cache_control": {"type": "ephemeral"}}]


def test_effort_suffix_sets_output_config_and_strips_model():
    c = _Client([_reply(NS(type="text", text="ok"))])
    _llm(c).complete(model="claude-sonnet-5:medium", system="s", user="u")
    assert c.calls[0]["model"] == "claude-sonnet-5"
    assert c.calls[0]["output_config"] == {"effort": "medium"}


def test_plain_model_has_no_output_config():
    c = _Client([_reply(NS(type="text", text="ok"))])
    _llm(c).complete(model="claude-sonnet-5", system="s", user="u")
    assert c.calls[0]["model"] == "claude-sonnet-5"
    assert "output_config" not in c.calls[0]


def test_invalid_effort_raises_before_any_call():
    c = _Client([_reply(NS(type="text", text="ok"))])
    with pytest.raises(ValueError, match="effort"):
        _llm(c).complete(model="claude-sonnet-5:turbo", system="s", user="u")
    assert c.calls == []
