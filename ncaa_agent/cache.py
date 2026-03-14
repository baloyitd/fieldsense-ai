"""
ncaa_agent.cache
================
Stage 07 — JSON-based result caching for NCAAReasoner outputs.

All reasoning traces, anomaly flags, and narratives are cached to disk so
that:
  - Re-running analysis with identical inputs returns the same result
    instantly without recomputing.
  - The full-bracket analysis budget is not wasted on repeated work.

Cache keys are derived from a deterministic hash of all inputs.
Cache files are stored as ``<cache_dir>/<md5_hex>.json``.

Usage
-----
::

    cache = ResultCache(cache_dir="/tmp/ncaa_cache")
    key = cache.make_key(X1.tolist(), X2.tolist(), ensemble_prob, meta)
    if not cache.exists(key):
        result = reasoner.reason(X1, X2, ensemble_prob, meta)
        cache.set(key, result)
    data = cache.get(key)
"""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: Default cache directory (inside system temp to avoid polluting repo)
_DEFAULT_CACHE_DIR: Optional[str] = None


def _default_dir() -> Path:
    """Return a stable temporary cache directory for this process."""
    global _DEFAULT_CACHE_DIR
    if _DEFAULT_CACHE_DIR is None:
        _DEFAULT_CACHE_DIR = str(Path(tempfile.gettempdir()) / "ncaa_agent_cache")
    return Path(_DEFAULT_CACHE_DIR)


class ResultCache:
    """
    File-based JSON result cache.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory for cache files.  Defaults to a temp-dir sub-folder.
    """

    def __init__(self, cache_dir: Optional[str | Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _default_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ResultCache initialised at %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Key construction
    # ------------------------------------------------------------------

    @staticmethod
    def make_key(*args: Any) -> str:
        """
        Build a deterministic cache key from arbitrary positional arguments.

        All arguments are serialised to JSON (sorted keys) and hashed with MD5.
        """
        payload = json.dumps(args, sort_keys=True, default=_json_default)
        return hashlib.md5(payload.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Dict]:
        """Return cached value or ``None`` if missing."""
        p = self._path(key)
        if p.exists():
            try:
                with p.open() as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache read error for key %s: %s", key, exc)
        return None

    def set(self, key: str, value: Dict) -> None:
        """Write ``value`` to cache under ``key``."""
        p = self._path(key)
        try:
            with p.open("w") as f:
                json.dump(value, f, sort_keys=True, default=_json_default)
        except OSError as exc:
            logger.warning("Cache write error for key %s: %s", key, exc)

    def exists(self, key: str) -> bool:
        """Return ``True`` iff the key is cached."""
        return self._path(key).exists()

    def invalidate(self, key: str) -> None:
        """Remove a single cache entry (no-op if missing)."""
        p = self._path(key)
        if p.exists():
            p.unlink()
            logger.debug("Invalidated cache key %s", key)

    def clear(self) -> None:
        """Remove all cache files in ``cache_dir``."""
        removed = 0
        for p in self.cache_dir.glob("*.json"):
            p.unlink()
            removed += 1
        logger.debug("Cleared %d cache entries from %s", removed, self.cache_dir)

    def size(self) -> int:
        """Return the number of cached entries."""
        return len(list(self.cache_dir.glob("*.json")))

    def __repr__(self) -> str:
        return f"ResultCache(dir={self.cache_dir}, size={self.size()})"


# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types not handled by default json module."""
    import numpy as np  # lazy import
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON-serialisable")
