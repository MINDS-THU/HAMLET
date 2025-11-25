"""Compatibility shim to keep the published distribution name in sync with uv.

This module re-exports the real ``hamlet`` package so users continue to
``import hamlet`` while the wheel is published as ``minds-hamlet``.
"""

from hamlet import *  # type: ignore # noqa: F401,F403

# Re-export hamlet's public API explicitly when available to keep introspection tidy.
try:
    from hamlet import __all__ as _hamlet_all
except ImportError:  # pragma: no cover - hamlet always defines __all__, but be safe.
    _hamlet_all = []

__all__ = list(_hamlet_all)
