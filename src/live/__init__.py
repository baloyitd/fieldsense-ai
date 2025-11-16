"""
FieldSense AI v3.0 - Live Processing Module

Real-time video ingestion, inference, and clip generation.
"""

from .ingest import VideoIngestor, DummyIngestor, FrameMetadata
from .live_engine import LiveEngine, OverlayRenderer, InferenceResult
from .clips import ClipGenerator, LiveClipEngine, Clip, ClipTrigger

__all__ = [
    'VideoIngestor',
    'DummyIngestor',
    'FrameMetadata',
    'LiveEngine',
    'OverlayRenderer',
    'InferenceResult',
    'ClipGenerator',
    'LiveClipEngine',
    'Clip',
    'ClipTrigger',
]
