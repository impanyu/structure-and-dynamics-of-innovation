"""YAML experiment configuration (spec §3.5: configs are checked into the repo).

Supports single-level-or-deeper inheritance via an `extends: <relative path>`
key: the child is deep-merged over the base (dicts merge recursively; lists and
scalars replace), so per-condition experiment configs only state their diffs.
"""
import os
from pathlib import Path

import yaml


def load_env(path=".env") -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Real environment variables take precedence over the file; lines starting
    with '#' and blank lines are ignored; surrounding quotes are stripped.
    Missing file is fine (keys may come from the shell instead)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        cfg = _deep_merge(base, cfg)
    return cfg
