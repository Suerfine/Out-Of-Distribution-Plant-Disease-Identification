"""Versioned run configs. Every decision for a model lives in its config file (config/vlm/vN[.M]_<title>_config.json for the
vision-language versions, config/baseline/<name>.json for the CNN baselines); scripts only read it.

Keys starting with "_" (e.g. "_note", "_result") are documentation and are ignored by the code.
"""

import json
from pathlib import Path

from dpd.data import PROJECT_ROOT


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def output_dir(cfg: dict) -> Path:
    out = PROJECT_ROOT / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    return out
