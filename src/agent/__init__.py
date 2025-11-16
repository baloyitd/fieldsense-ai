"""
FieldSense AI v3.1 - MiroThinker Agent Module

Meta-reasoning agent for multi-step verification and enhanced decision support.

Model: MiroThinker-v1.0-30B (Qwen3-based)
- 31B parameters, 4-bit quantized (~8GB VRAM)
- 256K context window
- Offline/privacy-compliant (no external tools)
- Benchmarks: HLE-Text 37.7%, GAIA 81.9%

Usage:
    from src.agent import MiroThinkerAgent

    agent = MiroThinkerAgent()
    result = agent.reason(xT_grid, play_data)
"""

from .load import (
    MiroThinkerAgent,
    load_model,
    load_tokenizer,
    check_model_availability
)
from .calibrate_reason import (
    CalibrationReasoner,
    reason_calibration
)

__all__ = [
    'MiroThinkerAgent',
    'load_model',
    'load_tokenizer',
    'check_model_availability',
    'CalibrationReasoner',
    'reason_calibration'
]

__version__ = '3.1.0'
__model__ = 'MiroThinker-v1.0-30B'
