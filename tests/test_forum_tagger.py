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
