import pytest

from innovation.core.llm import FakeLLM
from innovation.p2_forum.tagger import TopicTagger, UnlabeledError, parse_labels
from innovation.p2_forum.topics import Topic

TOPICS = [Topic(i, f"T{i}", f"def {i}") for i in range(10)]


@pytest.mark.parametrize("reply, want", [
    ("[3, 1]", [3, 1]),
    ("labels: [0]", [0]),
    ("[2, 2, 5]", [2, 5]),           # duplicates collapse, order kept
    ("[]", None),                    # at least one
    ("[1,2,3,4,5,6]", None),         # at most five
    ("[10]", None),                  # out of range
    ("no list", None),
    ('["0", "3"]', [0, 3]),          # ids given as digit strings
    ("[true]", None),                # booleans are not ids
])
def test_parse_labels(reply, want):
    assert parse_labels(reply, n_topics=10) == want


def test_system_prompt_lists_every_topic_with_id_and_definition():
    t = TopicTagger(llm=FakeLLM(), model="m", topics=TOPICS)
    assert "7: T7 — def 7" in t.system


def test_label_retries_then_succeeds():
    llm = FakeLLM(responses=["garbage", "[4]"])
    assert TopicTagger(llm=llm, model="m", topics=TOPICS).label("x") == [4]
    assert len(llm.calls) == 2


def test_label_gives_up_after_three_attempts():
    llm = FakeLLM(default="garbage")
    with pytest.raises(UnlabeledError):
        TopicTagger(llm=llm, model="m", topics=TOPICS).label("x")
    assert len(llm.calls) == 3


def test_retry_prompts_differ_so_a_cache_cannot_replay_the_bad_reply():
    llm = FakeLLM(responses=["garbage", "[4]"])
    TopicTagger(llm=llm, model="m", topics=TOPICS).label("x")
    assert llm.calls[0]["user"] != llm.calls[1]["user"]


def test_label_many_returns_none_for_failures():
    llm = FakeLLM(responses=["[1]"], default="garbage")
    out = TopicTagger(llm=llm, model="m", topics=TOPICS).label_many(["a", "b"], workers=1)
    assert out == [[1], None]


class ScriptedLLM:
    """Replies in order; an Exception instance is raised instead of returned."""

    def __init__(self, script):
        self.script, self.users = list(script), []

    def complete(self, *, model, system, user, max_tokens=1024):
        self.users.append(user)
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _refusal():
    return ValueError("no text in reply (stop_reason=refusal)")


def test_a_refusal_is_retried_then_succeeds(tmp_path):
    llm = ScriptedLLM([_refusal(), "[2]"])
    log = tmp_path / "refusals.jsonl"
    tagger = TopicTagger(llm=llm, model="m", topics=TOPICS, refusal_log=log)
    assert tagger.label("some text") == [2]
    assert "(retry 1)" in llm.users[1]
    assert tagger.refusals == 1
    assert len(log.read_text().splitlines()) == 1


def test_three_refusals_leave_the_item_unlabeled_and_are_logged(tmp_path):
    import json
    log = tmp_path / "sub" / "refusals.jsonl"
    tagger = TopicTagger(llm=ScriptedLLM([_refusal()] * 3), model="m", topics=TOPICS,
                         refusal_log=log)
    with pytest.raises(UnlabeledError):
        tagger.label("x" * 300)
    lines = [json.loads(ln) for ln in log.read_text().splitlines()]
    assert [ln["attempt"] for ln in lines] == [0, 1, 2]
    assert set(lines[0]) == {"ts", "model", "text_head", "attempt"}
    assert lines[0]["model"] == "m" and len(lines[0]["text_head"]) == 200
    assert tagger.refusals == 3
    many = TopicTagger(llm=ScriptedLLM([_refusal()] * 3 + ["[1]"]), model="m", topics=TOPICS)
    assert many.label_many(["a", "b"], workers=1) == [None, [1]]
    assert many.refusals == 3


def test_other_no_text_replies_and_errors_propagate():
    t = TopicTagger(llm=ScriptedLLM([ValueError("no text in reply (stop_reason=max_tokens)")]),
                    model="m", topics=TOPICS)
    with pytest.raises(ValueError, match="max_tokens"):
        t.label("x")
    t = TopicTagger(llm=ScriptedLLM([RuntimeError("boom")]), model="m", topics=TOPICS)
    with pytest.raises(RuntimeError):
        t.label_many(["x"])
    assert t.refusals == 0
