"""Shared runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def configure_genesis_cache(backend: str = "cpu") -> None:
    """Keep Genesis compiler caches inside this project."""
    os.environ.setdefault(
        "QD_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / f"quadrants-{backend}"),
    )
    os.environ.setdefault(
        "TI_OFFLINE_CACHE_FILE_PATH",
        str(PROJECT_ROOT / ".cache" / f"gstaichi-{backend}"),
    )

