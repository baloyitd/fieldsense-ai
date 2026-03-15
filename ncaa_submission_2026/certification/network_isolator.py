"""
ncaa_submission_2026.certification.network_isolator
====================================================
Stage 09 — Network isolation for offline inference certification.

Provides a context manager that blocks all outbound network connections
(socket, urllib, requests) during prediction.  Any attempt to open a
network connection raises :class:`NetworkAccessBlocked`.

Usage
-----
::

    isolator = NetworkIsolator()
    with isolator:
        probs = model.predict_proba(X1, X2)   # must not touch network

    # Or via run_isolated:
    report = isolator.run_isolated(model.predict_proba, X1, X2)
    assert report.is_isolated
"""

from __future__ import annotations

import logging
import socket
import unittest.mock as mock
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NetworkAccessBlocked(Exception):
    """Raised when network access is attempted in isolation mode."""


def _blocked_socket(*args, **kwargs):
    raise NetworkAccessBlocked(
        "Network access is blocked during offline inference certification. "
        "No external connections are permitted."
    )


@dataclass
class NetworkIsolationReport:
    """Result of a network isolation check."""

    is_isolated: bool
    network_calls_attempted: int
    errors: List[str]
    details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_isolated": self.is_isolated,
            "network_calls_attempted": self.network_calls_attempted,
            "errors": self.errors,
            "details": self.details,
        }


class NetworkIsolator:
    """
    Context manager that blocks all outbound network connections.

    Records any attempted network calls so the isolation report can be
    populated after the run.
    """

    def __init__(self) -> None:
        self._call_count = 0
        self._errors: List[str] = []
        self._patches: List[mock._patch] = []

    def __enter__(self) -> "NetworkIsolator":
        self._call_count = 0
        self._errors = []
        self._patches = self._build_patches()
        for p in self._patches:
            p.start()
        logger.debug("NetworkIsolator: network access blocked")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass
        logger.debug("NetworkIsolator: network access restored")
        # Suppress NetworkAccessBlocked so it can be recorded, not raised
        if exc_type is NetworkAccessBlocked:
            self._call_count += 1
            self._errors.append(str(exc_val))
            return True
        return False

    def run_isolated(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> NetworkIsolationReport:
        """
        Run *func* with all network access blocked.

        The function must complete without triggering any network calls.

        Parameters
        ----------
        func : callable
        *args, **kwargs : forwarded to func.

        Returns
        -------
        NetworkIsolationReport
            ``is_isolated=True`` iff no network calls were attempted.
        """
        attempted = 0
        errors: List[str] = []
        run_errors: List[str] = []

        patches = self._build_patches()
        call_tracker = {"count": 0, "msgs": []}

        def _tracking_blocker(tracker, *a, **kw):
            tracker["count"] += 1
            msg = f"Blocked network call: args={a}"
            tracker["msgs"].append(msg)
            raise NetworkAccessBlocked(msg)

        # Re-build patches with call tracking
        tracked_patches = [
            mock.patch("socket.socket", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
            mock.patch("socket.create_connection", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
            mock.patch("urllib.request.urlopen", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
            mock.patch("urllib.request.Request", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
        ]

        # Add requests if installed
        try:
            import requests
            tracked_patches.extend([
                mock.patch("requests.get", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
                mock.patch("requests.post", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
                mock.patch("requests.Session.send", side_effect=lambda *a, **kw: _tracking_blocker(call_tracker, *a, **kw)),
            ])
        except ImportError:
            pass

        for p in tracked_patches:
            p.start()

        try:
            func(*args, **kwargs)
        except NetworkAccessBlocked as e:
            call_tracker["count"] += 1
            call_tracker["msgs"].append(str(e))
        except Exception as e:
            # Non-network exception: record but don't count as network call
            run_errors.append(f"Non-network error: {type(e).__name__}: {e}")
        finally:
            for p in tracked_patches:
                try:
                    p.stop()
                except RuntimeError:
                    pass

        is_isolated = call_tracker["count"] == 0
        details = (
            "Inference completed with zero network calls. ISOLATED."
            if is_isolated
            else f"{call_tracker['count']} network call(s) attempted. NOT ISOLATED."
        )
        if run_errors:
            details += " Run errors: " + "; ".join(run_errors)

        return NetworkIsolationReport(
            is_isolated=is_isolated,
            network_calls_attempted=call_tracker["count"],
            errors=call_tracker["msgs"] + run_errors,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_patches(self) -> list:
        """Build the list of patches to apply."""
        patches = [
            mock.patch("socket.socket", side_effect=_blocked_socket),
            mock.patch("socket.create_connection", side_effect=_blocked_socket),
            mock.patch("urllib.request.urlopen", side_effect=_blocked_socket),
        ]
        try:
            import requests
            patches.extend([
                mock.patch("requests.get", side_effect=_blocked_socket),
                mock.patch("requests.post", side_effect=_blocked_socket),
            ])
        except ImportError:
            pass
        return patches
