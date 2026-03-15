"""
ncaa_live.monitor
=================
Stage 08 — Performance monitoring and automatic rollback management.

PerformanceMonitor tracks per-round Brier scores and decides whether a
newly recalibrated model should be accepted or rolled back.

FallbackManager maintains a versioned model stack that enables instant
reversion to any prior adapter state.

Rollback rule (from spec):
    ``should_rollback(old, new, threshold=0.005) → True`` when
    ``new_brier - old_brier > threshold``.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level rollback function (matches spec signature)
# ---------------------------------------------------------------------------

def should_rollback(
    old_brier: float,
    new_brier: float,
    threshold: float = 0.005,
) -> bool:
    """
    Return True if the recalibrated model is meaningfully worse.

    Parameters
    ----------
    old_brier : float
        Brier score of the currently deployed model.
    new_brier : float
        Brier score of the newly recalibrated model.
    threshold : float
        Tolerated degradation.  Default 0.005.
    """
    return (new_brier - old_brier) > threshold


# ---------------------------------------------------------------------------
# MonitorRecord
# ---------------------------------------------------------------------------

@dataclass
class MonitorRecord:
    """One recorded Brier evaluation."""

    round_num: int
    brier: float
    label: str
    triggered_rollback: bool = False
    timestamp: str = field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num": self.round_num,
            "brier": self.brier,
            "label": self.label,
            "triggered_rollback": self.triggered_rollback,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Tracks Brier scores over tournament rounds and triggers rollback logic.

    Parameters
    ----------
    threshold : float
        Brier degradation threshold; default 0.005.
    """

    def __init__(self, threshold: float = 0.005) -> None:
        self.threshold = threshold
        self._records: List[MonitorRecord] = []

    def record(
        self,
        brier: float,
        round_num: int,
        label: str = "",
        triggered_rollback: bool = False,
    ) -> MonitorRecord:
        """Record a Brier score evaluation."""
        rec = MonitorRecord(
            round_num=round_num,
            brier=brier,
            label=label,
            triggered_rollback=triggered_rollback,
        )
        self._records.append(rec)
        logger.info("Monitor record: round=%d brier=%.4f label=%s rollback=%s",
                    round_num, brier, label, triggered_rollback)
        return rec

    def should_rollback(self, old_brier: float, new_brier: float) -> bool:
        """Delegate to module-level :func:`should_rollback`."""
        return should_rollback(old_brier, new_brier, self.threshold)

    def history(self) -> List[MonitorRecord]:
        return list(self._records)

    def best_brier(self) -> Optional[float]:
        if not self._records:
            return None
        return min(r.brier for r in self._records)

    def last_brier(self) -> Optional[float]:
        return self._records[-1].brier if self._records else None

    def n_rollbacks(self) -> int:
        return sum(1 for r in self._records if r.triggered_rollback)

    def __repr__(self) -> str:
        return (
            f"PerformanceMonitor(threshold={self.threshold}, "
            f"n_records={len(self._records)}, "
            f"best_brier={self.best_brier()})"
        )


# ---------------------------------------------------------------------------
# FallbackManager
# ---------------------------------------------------------------------------

class FallbackManager:
    """
    Versioned model stack enabling instant rollback.

    Each call to :meth:`save_version` pushes a deep copy of the model.
    :meth:`rollback` pops the most recent (bad) version and returns the
    previous one.

    Parameters
    ----------
    max_versions : int
        Maximum number of versions retained; oldest discarded first.
    """

    def __init__(self, max_versions: int = 10) -> None:
        self.max_versions = max_versions
        self._stack: List[Dict[str, Any]] = []

    def save_version(
        self,
        model: Any,
        brier: float,
        round_num: int,
        label: str = "",
    ) -> None:
        """
        Push a model version onto the stack.

        Parameters
        ----------
        model : MatchupPredictor
            Deep-copied before storing.
        brier : float
        round_num : int
        label : str
        """
        entry = {
            "model": copy.deepcopy(model),
            "brier": brier,
            "round_num": round_num,
            "label": label,
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        self._stack.append(entry)
        if len(self._stack) > self.max_versions:
            self._stack.pop(0)
        logger.info(
            "Saved version round=%d brier=%.4f label=%s (stack depth=%d)",
            round_num, brier, label, len(self._stack),
        )

    def rollback(self) -> Optional[Dict[str, Any]]:
        """
        Remove the current version and return the previous one.

        Returns
        -------
        dict with keys ``model``, ``brier``, ``round_num``, ``label``
        or ``None`` if there is nothing to roll back to.
        """
        if len(self._stack) < 2:
            logger.warning("Cannot rollback: fewer than 2 versions saved")
            return None
        discarded = self._stack.pop()
        logger.info(
            "Rolled back from round=%d brier=%.4f → round=%d brier=%.4f",
            discarded["round_num"], discarded["brier"],
            self._stack[-1]["round_num"], self._stack[-1]["brier"],
        )
        return self._stack[-1]

    def current(self) -> Optional[Dict[str, Any]]:
        """Return the top-of-stack version without removing it."""
        return self._stack[-1] if self._stack else None

    def n_versions(self) -> int:
        return len(self._stack)

    def version_history(self) -> List[Dict[str, Any]]:
        return list(self._stack)

    def clear(self) -> None:
        self._stack.clear()

    def __repr__(self) -> str:
        cur = self._stack[-1] if self._stack else None
        cur_brier = cur["brier"] if cur else None
        return (
            f"FallbackManager(n_versions={len(self._stack)}, "
            f"current_brier={cur_brier})"
        )
