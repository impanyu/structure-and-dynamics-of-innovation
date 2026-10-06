"""Apply the user-approved (2026-10-06) merges and split to the topics-v3
draft and write the frozen configs/p2_forum/topics-v3.yaml. Indices are
positions in the DRAFT; every expected name is asserted first."""
from pathlib import Path

import yaml

DRAFT = Path("configs/p2_forum/topics-v3.draft.yaml")
OUT = Path("configs/p2_forum/topics-v3.yaml")

# (keep_idx, keep_name, drop_idx, drop_name, new_name, new_definition)
MERGES = [
    (0, "Machine Learning Applications", 157, "ML Applications (General)",
     "Machine Learning Applications",
     "Applied machine learning in specific or general domains (e.g., vision, language, speech, biology, healthcare, sciences), excluding work whose main contribution is a new learning algorithm."),
    (36, "Affective Computing", 112, "Emotional Intelligence in AI",
     "Affective Computing and Emotional Intelligence",
     "Computational modeling, recognition and generation of human emotion and affect, and AI systems that respond to it."),
    (55, "Bias, Fairness and Privacy in Vision", 313, "Fairness, Privacy, Ethics and Transparency in Vision",
     "Fairness, Privacy and Ethics in Vision",
     "Bias, fairness, privacy, accountability, transparency and ethical issues specific to computer vision systems."),
    (62, "Multi-modal Computer Vision", 304, "Multimodal Learning in Vision",
     "Multimodal Learning in Vision",
     "Vision methods that learn from or fuse multiple modalities (e.g., images with text, audio, depth or other sensors)."),
    (65, "Segmentation in Computer Vision", 310, "Segmentation, Grouping and Shape Analysis",
     "Segmentation, Grouping and Shape Analysis",
     "Pixel- and region-level segmentation, perceptual grouping and shape analysis in images and video."),
    (260, "Discourse, Pragmatics and Argument Mining", 278, "Discourse and Pragmatics",
     "Discourse, Pragmatics and Argument Mining",
     "Discourse structure, pragmatic meaning and argumentation in language."),
    (269, "Machine Translation and Cross-Lingual NLP", 282, "Machine Translation in NLP",
     "Machine Translation and Multilinguality",
     "Translating text or speech between languages, and multilingual or cross-lingual NLP."),
    (270, "Lexical Semantics and Morphology", 286, "Lexical Semantics in NLP",
     "Lexical Semantics and Morphology",
     "Word-level meaning, lexical relations and morphological structure."),
    (289, "3D from Single Images", 317, "3D from a Single Image and Shape-from-X",
     "3D from a Single Image",
     "Recovering 3D shape or scene structure from a single image, including shape-from-X cues."),
    (296, "Deep Learning Architectures for Vision", 319, "Deep Learning Architectures in Vision",
     "Deep Learning Architectures for Vision",
     "Design of deep network architectures and training techniques for vision tasks."),
    (309, "Robotics in Computer Vision", 333, "Vision and Robotics",
     "Vision and Robotics",
     "Visual perception for robotic tasks such as manipulation, navigation and robot control."),
]

SPLIT_IDX = 10
SPLIT_NAME = "Social and Trustworthy Aspects of Machine Learning"

HEADER = ("FROZEN 2026-10-06 (user-reviewed). Synonym-only dedupe of configs/p2_forum/venue_areas.yaml; "
          "umbrellas and fine areas kept. Built by scripts/p2_forum/build_topics_v3.py, edited by "
          "scripts/p2_forum/freeze_topics_v3.py. ids = list positions; never edit.")


def main():
    text = DRAFT.read_text()
    topics = yaml.safe_load(text)["topics"]
    assert len(topics) == 334, len(topics)
    for ki, kn, di, dn, *_ in MERGES:
        assert topics[ki]["name"] == kn, (ki, topics[ki]["name"], kn)
        assert topics[di]["name"] == dn, (di, topics[di]["name"], dn)
    assert topics[SPLIT_IDX]["name"] == SPLIT_NAME, topics[SPLIT_IDX]["name"]

    drop = set()
    for ki, _, di, _, name, definition in MERGES:
        keep = topics[ki]
        keep["sources"] = list(keep["sources"]) + list(topics[di]["sources"])
        keep["name"], keep["definition"] = name, definition
        drop.add(di)

    t = topics[SPLIT_IDX]
    tw = [s for s in t["sources"] if s["venue"] == "ICML" and s["area"].startswith("Trustworthy Machine Learning")]
    soc = [s for s in t["sources"] if s not in tw]
    assert len(tw) == 1 and len(soc) == 2 and {s["venue"] for s in soc} == {"NeurIPS", "ICLR"}
    new = {"name": "Trustworthy Machine Learning",
           "definition": "Technical methods that make ML systems trustworthy: robustness, privacy, fairness, safety, accountability and causal reliability.",
           "sources": tw}
    t["name"] = "Social and Economic Aspects of Machine Learning"
    t["definition"] = "Societal, economic and ethical impacts of machine learning and their social context (fairness, safety and privacy as social questions)."
    t["sources"] = soc

    out = []
    for i, x in enumerate(topics):
        if i in drop:
            continue
        out.append(x)
        if i == SPLIT_IDX:
            out.append(new)

    dropped = [l for l in text.splitlines()[:40] if l.startswith("#") and "Dropped catch-all" in l or l.startswith("#   ")]
    header = "# " + HEADER + "\n" + "\n".join(dropped) + "\n"
    OUT.write_text(header + yaml.safe_dump({"topics": out}, sort_keys=False, allow_unicode=True, width=100))
    print(len(out))


if __name__ == "__main__":
    main()
