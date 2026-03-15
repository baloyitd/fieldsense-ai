"""
ncaa_live
=========
Stage 08/10 — Live tournament adaptation pipeline.

Ingests real-time box score results, recalibrates the prediction model
after each round, regenerates Kaggle submission CSVs, and automatically
rolls back degraded adapters.
"""

__stage__ = "08/10"

from ncaa_live.ingester import BoxScoreIngester, GameResult
from ncaa_live.recalibrator import RoundRecalibrator, AdapterVersion, brier_score
from ncaa_live.monitor import (
    PerformanceMonitor,
    FallbackManager,
    MonitorRecord,
    should_rollback,
)
from ncaa_live.regenerator import SubmissionRegenerator
from ncaa_live.simulator import (
    TournamentSimulator,
    BracketTeam,
    make_simulator_from_feature_df,
)

__all__ = [
    "BoxScoreIngester",
    "GameResult",
    "RoundRecalibrator",
    "AdapterVersion",
    "brier_score",
    "PerformanceMonitor",
    "FallbackManager",
    "MonitorRecord",
    "should_rollback",
    "SubmissionRegenerator",
    "TournamentSimulator",
    "BracketTeam",
    "make_simulator_from_feature_df",
]
