"""YAML experiment configuration (spec §3.5: configs are checked into the repo)."""
from pathlib import Path

import yaml


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())
