from innovation.p2_forum.topics import load_topics


def test_frozen_topics_v3():
    t = load_topics("configs/p2_forum/topics-v3.yaml", expected=None)
    names = {x.name for x in t}
    assert len(t) == 324
    assert "Trustworthy Machine Learning" in names
    assert "Social and Economic Aspects of Machine Learning" in names
