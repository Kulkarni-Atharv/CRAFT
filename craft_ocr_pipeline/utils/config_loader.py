"""Load and validate the YAML config, resolve relative paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # resolve relative paths relative to the project root (parent of configs/)
    root = path.parent.parent
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(root / val)

    return cfg


def ensure_dirs(cfg: dict) -> None:
    """Create output/log directories declared in config."""
    for key in ("output_dir", "crops_dir", "viz_dir", "log_dir"):
        p = cfg["paths"].get(key)
        if p:
            os.makedirs(p, exist_ok=True)
