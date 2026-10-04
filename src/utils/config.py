"""Configuration loader."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration file.

    Args:
        path: Optional path to a YAML file. Defaults to ``config/config.yaml``.

    Returns:
        The parsed configuration dictionary.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p
