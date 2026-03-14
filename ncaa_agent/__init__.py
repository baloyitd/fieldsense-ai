"""
ncaa_agent
==========
Stage 07/10 — Explainability, anomaly detection, and narrative generation.

Public API
----------
NCAAReasoner
    4-step structured reasoning engine for NCAA tournament matchup analysis.

ReasoningTrace, ReasoningStep
    Dataclasses that carry the full reasoning output.

AnomalyDetector
    Flags predictions that deviate significantly from seed-based baselines.

AnomalyFlag
    Dataclass returned by AnomalyDetector.

BracketNarrativeGenerator
    Converts a ReasoningTrace into 2–3 natural-language sentences.

ResultCache
    JSON-based disk cache for reasoning traces (MD5-keyed).

seed_baseline_win_rate
    Historical win probability for a given seed matchup.
"""

__stage__ = "07/10"

from ncaa_agent.cache import ResultCache
from ncaa_agent.reasoner import NCAAReasoner, ReasoningStep, ReasoningTrace
from ncaa_agent.anomaly import (
    AnomalyDetector,
    AnomalyFlag,
    seed_baseline_win_rate,
    SEED_WIN_RATES,
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
)
from ncaa_agent.narrative import BracketNarrativeGenerator

__all__ = [
    # Reasoner
    "NCAAReasoner",
    "ReasoningStep",
    "ReasoningTrace",
    # Anomaly detection
    "AnomalyDetector",
    "AnomalyFlag",
    "seed_baseline_win_rate",
    "SEED_WIN_RATES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOP_K",
    # Narrative
    "BracketNarrativeGenerator",
    # Cache
    "ResultCache",
]
