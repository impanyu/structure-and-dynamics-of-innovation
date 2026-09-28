import httpx
import openai
import pytest

from innovation.core.llm import OpenAILLM


def _err(cls, status):
    req = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return cls("boom", response=httpx.Response(status, request=req), body=None)


class _Resp:
    output_text = "fine"


class _Client:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0
        self.responses = self

    def create(self, **kw):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return _Resp()


def test_server_errors_are_retried_until_success():
    slept = []
    c = _Client([_err(openai.InternalServerError, 500), _err(openai.RateLimitError, 429)])
    llm = OpenAILLM(client=c, sleep=slept.append)
    assert llm.complete(model="gpt-5", system="s", user="u") == "fine"
    assert c.calls == 3 and slept == OpenAILLM.RETRY_DELAYS[:2]


def test_client_errors_raise_immediately():
    c = _Client([_err(openai.BadRequestError, 400)])
    with pytest.raises(openai.BadRequestError):
        OpenAILLM(client=c, sleep=lambda s: None).complete(model="gpt-5", system="s", user="u")
    assert c.calls == 1


def test_gives_up_after_the_retry_budget():
    n = len(OpenAILLM.RETRY_DELAYS) + 1
    c = _Client([_err(openai.InternalServerError, 500)] * n)
    with pytest.raises(openai.InternalServerError):
        OpenAILLM(client=c, sleep=lambda s: None).complete(model="gpt-5", system="s", user="u")
    assert c.calls == n
