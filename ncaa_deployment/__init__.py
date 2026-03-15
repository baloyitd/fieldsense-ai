"""
ncaa_deployment
===============
Stage 10/10 — Final optimization, competition submissions, and retrospective.

Modules
-------
optimize     : BayesianEnsembleOptimizer — scipy differential_evolution search
               over ensemble weights + temperature scaling.
submission   : SubmissionBuilder — conservative and aggressive Kaggle CSVs.
post_round   : PostRoundAnalyzer — per-round Brier tracking + variant comparison.
retrospective: TechnicalRetrospective — full 10-stage knowledge capture.
"""

from __future__ import annotations

__version__ = "2026.1.0"
__stage__ = "10/10"

from ncaa_deployment.optimize import (
    BayesianEnsembleOptimizer,
    OptimizationResult,
    WeightedEnsemble,
    apply_temperature,
)
from ncaa_deployment.submission import (
    SubmissionBuilder,
    SubmissionVariant,
    kl_divergence,
)
from ncaa_deployment.post_round import (
    PostRoundAnalyzer,
    RoundReport,
    VariantComparison,
)
from ncaa_deployment.retrospective import (
    TechnicalRetrospective,
    StageMetrics,
    generate_retrospective,
)

__all__ = [
    # optimize
    "BayesianEnsembleOptimizer",
    "OptimizationResult",
    "WeightedEnsemble",
    "apply_temperature",
    # submission
    "SubmissionBuilder",
    "SubmissionVariant",
    "kl_divergence",
    # post_round
    "PostRoundAnalyzer",
    "RoundReport",
    "VariantComparison",
    # retrospective
    "TechnicalRetrospective",
    "StageMetrics",
    "generate_retrospective",
]
