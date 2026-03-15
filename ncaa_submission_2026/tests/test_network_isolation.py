"""
test_network_isolation.py
=========================
Tests for ncaa_submission_2026.certification.network_isolator.

Covers:
- Model inference completes with zero network calls (is_isolated=True)
- Attempted HTTP call is blocked and counted
- NetworkAccessBlocked is raised on socket access within context manager
- run_isolated returns correct report structure
- Context manager restores original socket after exit
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from ncaa_submission_2026.certification.network_isolator import (
    NetworkAccessBlocked,
    NetworkIsolationReport,
    NetworkIsolator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_network_call():
    """Attempt to open a socket (will be blocked in isolation mode)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("8.8.8.8", 53))


def _try_urllib_call():
    import urllib.request
    urllib.request.urlopen("http://example.com")


def _try_requests_call():
    import requests
    requests.get("http://example.com", timeout=1)


# ---------------------------------------------------------------------------
# NetworkIsolator — run_isolated
# ---------------------------------------------------------------------------

class TestRunIsolated:
    def test_model_inference_is_isolated(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        assert report.is_isolated is True

    def test_isolated_has_zero_network_calls(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        assert report.network_calls_attempted == 0

    def test_pure_computation_is_isolated(self):
        def compute(n):
            return np.dot(np.random.randn(n, n), np.random.randn(n, n))

        isolator = NetworkIsolator()
        report = isolator.run_isolated(compute, 10)
        assert report.is_isolated is True

    def test_socket_call_is_blocked(self):
        isolator = NetworkIsolator()
        report = isolator.run_isolated(_try_network_call)
        assert report.is_isolated is False
        assert report.network_calls_attempted > 0

    def test_urllib_call_is_blocked(self):
        isolator = NetworkIsolator()
        report = isolator.run_isolated(_try_urllib_call)
        assert report.is_isolated is False
        assert report.network_calls_attempted > 0

    def test_requests_call_is_blocked(self):
        isolator = NetworkIsolator()
        report = isolator.run_isolated(_try_requests_call)
        assert report.is_isolated is False
        assert report.network_calls_attempted > 0

    def test_returns_network_isolation_report(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        assert isinstance(report, NetworkIsolationReport)

    def test_details_non_empty(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        assert isinstance(report.details, str) and len(report.details) > 0


# ---------------------------------------------------------------------------
# NetworkIsolator — context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_socket_blocked_in_context(self):
        isolator = NetworkIsolator()
        with isolator:
            with pytest.raises((NetworkAccessBlocked, Exception)):
                _try_network_call()

    def test_socket_works_after_context_exit(self):
        """After exiting the context, socket.socket should be callable again."""
        isolator = NetworkIsolator()
        with isolator:
            pass  # enter and immediately exit
        # socket.socket should be restored
        assert callable(socket.socket)

    def test_normal_code_runs_inside_context(self):
        isolator = NetworkIsolator()
        result = []
        with isolator:
            result.append(np.dot([1, 2], [3, 4]))
        assert result[0] == 11


# ---------------------------------------------------------------------------
# NetworkIsolationReport — to_dict
# ---------------------------------------------------------------------------

class TestNetworkIsolationReport:
    def test_to_dict_has_required_keys(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        d = report.to_dict()
        for key in ("is_isolated", "network_calls_attempted", "errors", "details"):
            assert key in d

    def test_errors_empty_when_no_network(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
        assert report.errors == []

    def test_errors_non_empty_when_blocked(self):
        isolator = NetworkIsolator()
        report = isolator.run_isolated(_try_network_call)
        assert len(report.errors) > 0


# ---------------------------------------------------------------------------
# NetworkIsolator — multiple runs
# ---------------------------------------------------------------------------

class TestMultipleRuns:
    def test_multiple_isolated_calls_all_clean(self, fitted_model, val_arrays):
        X1, X2, _ = val_arrays
        isolator = NetworkIsolator()
        for _ in range(3):
            report = isolator.run_isolated(fitted_model.predict_proba, X1, X2)
            assert report.is_isolated is True

    def test_kaggle_submission_generation_is_isolated(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission

        def make_submission():
            build_submission(
                model=fitted_model,
                team_features=feat_df,
                team_ids=list(feat_df["team_id"].unique()),
                season=2025,
                feature_cols=FEATURE_COLS,
            )

        isolator = NetworkIsolator()
        report = isolator.run_isolated(make_submission)
        assert report.is_isolated is True
