"""
ncaa_models — NCAA Tournament Prediction Models
================================================
Stage 06/10: Ensemble stacking, calibration, and meta-learning.

Exports
-------
MatchupPredictor    : Abstract base class all models must implement.
LogisticBaseline    : Sklearn logistic regression baseline.
NCAAPredictor       : PyTorch backbone (shared encoder + matchup head).
NCAANeuralModel     : MatchupPredictor wrapper (torch/sklearn dual backend).
LoRALinear          : Low-rank adapter layer wrapping nn.Linear.
AdapterRegistry     : Hot-swap LoRA adapter manager.
CalibrationPipeline : Create/train adapters within a time budget.
compute_brier_score : Standalone Brier score metric.
calibration_report  : Reliability diagram data + ECE.
per_seed_analysis   : Accuracy breakdown by seed matchup type.
per_round_analysis  : Brier score breakdown by tournament round.
build_submission    : Generate Kaggle submission CSV.
temporal_cross_validate : Leave-one-season-out CV with expanding window.
CounterfactualGenerator  : Generate what-if matchup scenarios.
CounterfactualScenario   : Single counterfactual scenario dataclass.
CVSComputer         : Counterfactual Validity Score computation.
TrainingDistribution : Per-feature statistics for CVS.
UpsetDetector       : Upset plausibility classification pipeline.
classify_matchup    : Standalone matchup classification function.
ALL_PERTURBATION_TYPES : All 5 perturbation type constants.
SimpleAverageEnsemble    : Equal-weight ensemble.
BrierWeightedEnsemble    : Inverse-Brier weighted ensemble.
ContextualEnsemble       : Context-dependent weighted ensemble.
MetaLearnerEnsemble      : NNLS stacking meta-learner ensemble.
PlattScaler         : Platt scaling calibrator.
IsotonicCalibrator  : Isotonic regression calibrator.
PostHocCalibrator   : MatchupPredictor wrapper with calibration.
compare_calibration : Utility to compare calibration methods.
"""

from .base import MatchupPredictor
from .baseline import LogisticBaseline, FEATURE_COLS
from .neural import NCAAPredictor, NCAANeuralModel, TORCH_AVAILABLE
from .lora import (
    LoRALinear,
    apply_lora_to_model,
    remove_lora,
    get_lora_state_dict,
    load_lora_state_dict,
    zero_lora_weights,
    count_trainable_params,
    count_total_params,
    count_lora_params,
    DEFAULT_TARGET_MODULES,
    ALL_TARGET_MODULES,
)
from .adapter import AdapterRegistry
from .calibrate import CalibrationPipeline, VALID_ADAPTER_TYPES
from .evaluate import (
    compute_brier_score,
    calibration_report,
    per_seed_analysis,
    per_round_analysis,
)
from .submit import build_submission, validate_submission, make_submission_id
from .cv import temporal_cross_validate, build_matchup_df, CVResult, CVFold
from .counterfactual import (
    CounterfactualGenerator,
    CounterfactualScenario,
    ALL_PERTURBATION_TYPES,
    INJURY,
    FOUL_TROUBLE,
    HOT_SHOOTING,
    COLD_SHOOTING,
    REST_ADVANTAGE,
    DEFAULT_SEVERITIES,
    DEFAULT_CVS_THRESHOLD,
    BOTH_TEAMS,
)
from .cvs import CVSComputer, TrainingDistribution
from .upset_detector import (
    UpsetDetector,
    classify_matchup,
    BLOWOUT_LIKELY,
    COMPETITIVE,
    UPSET_PLAUSIBLE,
    BLOWOUT_THRESHOLD,
    COMPETITIVE_LO,
    COMPETITIVE_HI,
    UPSET_THRESHOLD,
)
from .ensemble import (
    SimpleAverageEnsemble,
    BrierWeightedEnsemble,
    ContextualEnsemble,
    build_simple_ensemble,
    build_weighted_ensemble,
)
from .meta_learner import MetaLearnerEnsemble
from .calibration import (
    PlattScaler,
    IsotonicCalibrator,
    PostHocCalibrator,
    compare_calibration,
)

__version__ = "1.0.0"
__stage__ = "06/10"

__all__ = [
    # Base
    "MatchupPredictor",
    # Models
    "LogisticBaseline",
    "FEATURE_COLS",
    "NCAAPredictor",
    "NCAANeuralModel",
    "TORCH_AVAILABLE",
    # Stage 04: LoRA
    "LoRALinear",
    "apply_lora_to_model",
    "remove_lora",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "zero_lora_weights",
    "count_trainable_params",
    "count_total_params",
    "count_lora_params",
    "DEFAULT_TARGET_MODULES",
    "ALL_TARGET_MODULES",
    "AdapterRegistry",
    "CalibrationPipeline",
    "VALID_ADAPTER_TYPES",
    # Evaluation
    "compute_brier_score",
    "calibration_report",
    "per_seed_analysis",
    "per_round_analysis",
    # Submission
    "build_submission",
    "validate_submission",
    "make_submission_id",
    # CV
    "temporal_cross_validate",
    "build_matchup_df",
    "CVResult",
    "CVFold",
    # Stage 05: Counterfactual engine
    "CounterfactualGenerator",
    "CounterfactualScenario",
    "ALL_PERTURBATION_TYPES",
    "INJURY",
    "FOUL_TROUBLE",
    "HOT_SHOOTING",
    "COLD_SHOOTING",
    "REST_ADVANTAGE",
    "DEFAULT_SEVERITIES",
    "DEFAULT_CVS_THRESHOLD",
    "BOTH_TEAMS",
    "CVSComputer",
    "TrainingDistribution",
    "UpsetDetector",
    "classify_matchup",
    "BLOWOUT_LIKELY",
    "COMPETITIVE",
    "UPSET_PLAUSIBLE",
    "BLOWOUT_THRESHOLD",
    "COMPETITIVE_LO",
    "COMPETITIVE_HI",
    "UPSET_THRESHOLD",
    # Stage 06: Ensemble
    "SimpleAverageEnsemble",
    "BrierWeightedEnsemble",
    "ContextualEnsemble",
    "build_simple_ensemble",
    "build_weighted_ensemble",
    "MetaLearnerEnsemble",
    # Stage 06: Calibration
    "PlattScaler",
    "IsotonicCalibrator",
    "PostHocCalibrator",
    "compare_calibration",
]
