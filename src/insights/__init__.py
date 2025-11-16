"""Insights and decision-making modules"""

from .heatmap import (
    compute_opportunity_heatmap,
    generate_insight,
    save_opportunity_heatmap,
    find_top_decision
)
from .active_learning import (
    ActiveLearningManager,
    create_ui_label_payload
)

__all__ = [
    'compute_opportunity_heatmap',
    'generate_insight',
    'save_opportunity_heatmap',
    'find_top_decision',
    'ActiveLearningManager',
    'create_ui_label_payload'
]
