"""C1: resuming a paper-2 run must extend its event log, not corrupt it.

The failure mode this guards: a second pass over the same run_id with a fresh
Workspace re-issues `gen:<run_id>:0` while EventLog appends, so node ids repeat
and every structural metric — all of which replay the log — raises
`duplicate node_id`. The log is the primary research artifact and the damage
only surfaces at analysis time, after the LLM spend.
"""
import json
from collections import Counter

import pytest

from innovation.p2_forum.board_metrics import board_at, board_structure
from innovation.p2_forum.runner import ForumRunConfig, resume_forum, run_forum

AGENTS = [{"agent_id": "a0", "k_topics": 1}, {"agent_id": "a1", "k_topics": 1}]
POOL = ["alpha", "beta", "gamma"]


class PostingLLM:
    """Posts on every turn, so every step mints a new board id."""

    def complete(self, *, model, system, user, max_tokens):
        return json.dumps({"action": "generate",
                           "args": {"text": f"idea {len(user)}",
                                    "cited_ids": []}})


def _cfg(total_steps: int) -> ForumRunConfig:
    return ForumRunConfig(run_id="r", seed=0, total_steps=total_steps,
                          agents=AGENTS, topic_pool=POOL)


def _generated_ids(events) -> list[str]:
    return [e["result"]["node_id"] for e in events
            if e["action"] == "generate" and "node_id" in e.get("result", {})]


def _events(tmp_path):
    from innovation.core.events import load_events
    return load_events(tmp_path / "r" / "events.jsonl")


def test_resume_extends_the_log_without_repeating_node_ids(
        tmp_path, make_workspace, fake_embedder):
    ws = make_workspace()
    first = run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
                      embedder=fake_embedder, llm=PostingLLM(), model="m",
                      out_dir=tmp_path)
    assert len(first["generated"]) == 4

    fresh = make_workspace()
    out = resume_forum(_cfg(7), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index, embedder=fake_embedder,
                       llm=PostingLLM(), model="m", out_dir=tmp_path)

    assert out["resumed_from_step"] == 4
    events = _events(tmp_path)
    assert [e["step"] for e in events] == list(range(7))
    ids = _generated_ids(events)
    assert len(ids) == 7
    assert [i for i, n in Counter(ids).items() if n > 1] == []
    assert ids == [f"gen:r:{i}" for i in range(7)]


def test_the_resumed_log_still_replays(tmp_path, make_workspace, fake_embedder):
    ws = make_workspace()
    run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=PostingLLM(), model="m",
              out_dir=tmp_path)
    fresh = make_workspace()
    resume_forum(_cfg(7), corpus=fresh.corpus, corpus_index=fresh.corpus_index,
                 embedder=fake_embedder, llm=PostingLLM(), model="m",
                 out_dir=tmp_path)

    events = _events(tmp_path)
    replay = make_workspace()
    # would raise ValueError("duplicate node_id: gen:r:0") on a corrupted log
    board = board_at(events, len(events), corpus=replay.corpus,
                     corpus_index=replay.corpus_index, embedder=fake_embedder,
                     run_id="r")

    assert len(board.board_post_ids()) == 7
    assert board_structure(board, events)["n_posts"] == 7


def test_resume_reuses_the_recorded_topic_draws(tmp_path, make_workspace,
                                                fake_embedder):
    ws = make_workspace()
    first = run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
                      embedder=fake_embedder, llm=PostingLLM(), model="m",
                      out_dir=tmp_path)

    fresh = make_workspace()
    out = resume_forum(_cfg(7), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index, embedder=fake_embedder,
                       llm=PostingLLM(), model="m", out_dir=tmp_path)

    assert out["topic_assignments"] == first["topic_assignments"]
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["topic_assignments"] == first["topic_assignments"]
    assert meta["resumed_from"] == [4]
    assert meta["total_steps"] == 7


def test_resume_reconstructs_each_agents_memory(tmp_path, make_workspace,
                                                fake_embedder):
    """The agent must not resume amnesiac: its prompt carries its own history."""
    ws = make_workspace()
    run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=PostingLLM(), model="m",
              out_dir=tmp_path)

    class RecordingLLM(PostingLLM):
        def __init__(self):
            self.prompts = []

        def complete(self, *, model, system, user, max_tokens):
            self.prompts.append(user)
            return super().complete(model=model, system=system, user=user,
                                    max_tokens=max_tokens)

    llm = RecordingLLM()
    fresh = make_workspace()
    resume_forum(_cfg(6), corpus=fresh.corpus, corpus_index=fresh.corpus_index,
                 embedder=fake_embedder, llm=llm, model="m", out_dir=tmp_path)

    # a0 acted on steps 0 and 2, so its first resumed prompt recalls both posts
    assert "gen:r:0" in llm.prompts[0] and "gen:r:2" in llm.prompts[0]


def test_resume_refuses_when_there_is_nothing_to_resume(tmp_path, make_workspace,
                                                        fake_embedder):
    ws = make_workspace()
    with pytest.raises(SystemExit, match="nothing to resume"):
        resume_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
                     embedder=fake_embedder, llm=PostingLLM(), model="m",
                     out_dir=tmp_path)


def test_resume_refuses_without_a_raised_step_budget(tmp_path, make_workspace,
                                                     fake_embedder):
    ws = make_workspace()
    run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=PostingLLM(), model="m",
              out_dir=tmp_path)
    fresh = make_workspace()
    with pytest.raises(SystemExit, match="raise total_steps"):
        resume_forum(_cfg(4), corpus=fresh.corpus,
                     corpus_index=fresh.corpus_index, embedder=fake_embedder,
                     llm=PostingLLM(), model="m", out_dir=tmp_path)


def test_resume_refuses_when_the_recorded_topic_draws_are_gone(
        tmp_path, make_workspace, fake_embedder):
    ws = make_workspace()
    run_forum(_cfg(4), corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=PostingLLM(), model="m",
              out_dir=tmp_path)
    (tmp_path / "r" / "run_meta.json").unlink()

    fresh = make_workspace()
    with pytest.raises(SystemExit, match="run_meta.json is missing"):
        resume_forum(_cfg(7), corpus=fresh.corpus,
                     corpus_index=fresh.corpus_index, embedder=fake_embedder,
                     llm=PostingLLM(), model="m", out_dir=tmp_path)


def test_a_replayed_workspace_never_reissues_an_existing_board_id(
        make_workspace):
    """The counter is bumped past every replayed id — the root cause of C1."""
    ws = make_workspace()
    ws.post_idea("a", [], {}, node_id="gen:t:0")
    ws.post_idea("b", [], {}, node_id="gen:t:1")

    assert ws.post_idea("c", [], {}) == "gen:t:2"


def test_post_idea_rejects_a_replayed_id_outside_the_board(make_workspace):
    ws = make_workspace()
    with pytest.raises(ValueError, match="board post needs"):
        ws.post_idea("x", [], {}, node_id="p1")


def test_a_gap_in_replayed_ids_does_not_collide(make_workspace):
    """A log whose ids skip (a failed post consumes no id) must still advance
    the counter past the highest id seen, not to len(posts)."""
    ws = make_workspace()
    ws.post_idea("a", [], {}, node_id="gen:t:5")

    minted = ws.post_idea("b", [], {})

    assert minted == "gen:t:6"
