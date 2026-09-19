"""Path helpers that avoid hard-coded machine-specific paths."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root based on this source file location."""
    return Path(__file__).resolve().parents[3]


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a possibly relative path against the repository root."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def ensure_parent(path: str | Path) -> Path:
    resolved = resolve_project_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

