from innovation.llm import CachedLLM, FakeLLM


def test_fake_llm_returns_canned_responses_and_records_calls():
    llm = FakeLLM(responses=["first", "second"])
    assert llm.complete(model="m", system="s", user="u1") == "first"
    assert llm.complete(model="m", system="s", user="u2") == "second"
    assert llm.complete(model="m", system="s", user="u3") == "ok"  # default
    assert [c["user"] for c in llm.calls] == ["u1", "u2", "u3"]


def test_cached_llm_hits_disk_cache(tmp_path):
    inner = FakeLLM(responses=["expensive"])
    llm = CachedLLM(inner, cache_dir=tmp_path)
    assert llm.complete(model="m", system="s", user="u") == "expensive"
    # Second identical call must come from cache, not the inner client.
    assert llm.complete(model="m", system="s", user="u") == "expensive"
    assert len(inner.calls) == 1
    # A different prompt is a cache miss.
    llm.complete(model="m", system="s", user="other")
    assert len(inner.calls) == 2
