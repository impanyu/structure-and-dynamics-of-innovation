import json

import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.graph import FrozenGraphError
from innovation.p2_forum.env import Action, ForumEnvironment
from innovation.p2_forum.runner import ForumRunConfig, run_forum


class ScriptedLLM:
    """Posts once, then jumps forever."""

    def __init__(self):
        self.n = 0

    def complete(self, *, model, system, user, max_tokens):
        self.n += 1
        if self.n == 1:
            return json.dumps({"action": "generate",
                               "args": {"text": "an idea", "cited_ids": ["p1"]}})
        return json.dumps({"action": "sample_frontier", "args": {}})


def test_a_full_run_leaves_the_corpus_bit_identical(tmp_path, make_workspace,
                                                     fake_embedder):
    ws = make_workspace()
    corpus, index = ws.corpus, ws.corpus_index
    before = (sorted(corpus.node_ids()),
              sorted((s, d) for s in corpus.node_ids()
                     for d in corpus.citations_out(s)),
              index.vecs.tobytes(),
              list(index.ids))

    cfg = ForumRunConfig(run_id="inv", seed=0, total_steps=6,
                         agents=[{"agent_id": "a0", "k_topics": 1},
                                 {"agent_id": "a1", "k_topics": 1}],
                         topic_pool=["alpha", "beta"])
    out = run_forum(cfg, corpus=corpus, corpus_index=index,
                    embedder=fake_embedder, llm=ScriptedLLM(), model="m",
                    out_dir=tmp_path)

    after = (sorted(corpus.node_ids()),
             sorted((s, d) for s in corpus.node_ids()
                    for d in corpus.citations_out(s)),
             index.vecs.tobytes(),
             list(index.ids))
    assert before == after
    assert len(out["generated"]) >= 1


def test_the_corpus_index_never_gains_a_board_vector(tmp_path, make_workspace):
    ws = make_workspace()
    n_before = len(ws.corpus_index.ids)
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "e.jsonl"),
                           rng=np.random.default_rng(0))

    env.execute("a", 0, Action("generate", {"text": "x", "cited_ids": []}))

    assert len(ws.corpus_index.ids) == n_before
    assert len(ws.board_index.ids) == 1


def test_frozen_corpus_rejects_writes_even_directly(make_workspace):
    ws = make_workspace()
    try:
        ws.corpus.add_links("p2", ["p1"])
    except FrozenGraphError:
        return
    raise AssertionError("expected FrozenGraphError")
