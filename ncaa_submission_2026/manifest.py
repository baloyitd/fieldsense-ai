"""
ncaa_submission_2026.manifest
==============================
Stage 09 — SHA-256 manifest generation and verification for data and model files.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def sha256_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """Compute SHA-256 hex digest of a bytes object."""
    return hashlib.sha256(data).hexdigest()


def build_manifest(
    root: Union[str, Path],
    patterns: Optional[List[str]] = None,
    label: str = "manifest",
) -> Dict[str, str]:
    """
    Build a manifest dict mapping relative file paths to SHA-256 hashes.

    Parameters
    ----------
    root : path
        Root directory to scan.
    patterns : list of str, optional
        Glob patterns to include.  Default: all files.
    label : str
        Label included in log messages.

    Returns
    -------
    dict
        ``{relative_path: sha256_hex}``
    """
    root = Path(root)
    if not root.exists():
        logger.warning("%s root does not exist: %s", label, root)
        return {}

    patterns = patterns or ["**/*"]
    manifest: Dict[str, str] = {}
    for pattern in patterns:
        for p in sorted(root.glob(pattern)):
            if p.is_file():
                rel = str(p.relative_to(root))
                try:
                    manifest[rel] = sha256_file(p)
                except OSError as e:
                    logger.warning("Cannot hash %s: %s", p, e)
    logger.info("Built %s: %d files", label, len(manifest))
    return manifest


def verify_manifest(
    root: Union[str, Path],
    manifest: Dict[str, str],
) -> Dict[str, bool]:
    """
    Verify files against a manifest.

    Parameters
    ----------
    root : path
    manifest : dict
        ``{relative_path: expected_sha256}``

    Returns
    -------
    dict
        ``{relative_path: True/False}`` (True = hash matches)
    """
    root = Path(root)
    results: Dict[str, bool] = {}
    for rel, expected in manifest.items():
        p = root / rel
        if not p.exists():
            logger.warning("Missing file: %s", p)
            results[rel] = False
        else:
            actual = sha256_file(p)
            results[rel] = actual == expected
            if actual != expected:
                logger.warning("Hash mismatch: %s (expected %s, got %s)",
                               rel, expected[:12], actual[:12])
    return results


def save_manifest(manifest: Dict[str, str], path: Union[str, Path]) -> None:
    """Write manifest to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s (%d files)", path, len(manifest))


def load_manifest(path: Union[str, Path]) -> Dict[str, str]:
    """Load a manifest from a JSON file."""
    with open(path) as f:
        return json.load(f)
