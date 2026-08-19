"""Runtime path helpers for source and PyInstaller builds."""
from __future__ import annotations

import sys
from pathlib import Path


def get_base_dir() -> Path:
    """Return the application root in source and frozen modes."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[2]


def get_resource_path(*parts: str) -> Path:
    """Return an absolute path to a bundled application resource."""
    return get_base_dir().joinpath(*parts)
