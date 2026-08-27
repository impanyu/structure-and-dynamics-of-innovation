from innovation.experiments.events import EventLog, load_events


def test_append_assigns_seq_and_persists(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    e1 = log.append({"run_id": "r", "agent_id": "a0", "step": 0,
                     "action": "search", "args": {"query": "q"}, "result": {}})
    e2 = log.append({"run_id": "r", "agent_id": "a1", "step": 1,
                     "action": "generate", "args": {}, "result": {"node_id": "gen:r:0"}})
    assert (e1["seq"], e2["seq"]) == (0, 1)
    assert [e["action"] for e in load_events(path)] == ["search", "generate"]


def test_event_log_resumes_seq_from_existing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(path).append({"run_id": "r", "agent_id": "a", "step": 0,
                           "action": "search", "args": {}, "result": {}})
    log2 = EventLog(path)  # reopen, e.g. after a crash
    e = log2.append({"run_id": "r", "agent_id": "a", "step": 1,
                     "action": "search", "args": {}, "result": {}})
    assert e["seq"] == 1
    assert len(log2.read_all()) == 2
