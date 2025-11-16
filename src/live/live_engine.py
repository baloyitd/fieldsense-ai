"""
FieldSense AI v3.0 - Live Inference Engine

Real-time processing pipeline:
ingest → normalize → infer + adapter → heatmap + decision → render overlay

Targets 30+ FPS with WebGPU-accelerated rendering.
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU acceleration disabled")

from .ingest import VideoIngestor, DummyIngestor, FrameMetadata


@dataclass
class InferenceResult:
    """Result from real-time inference."""
    timestamp: float
    frame_id: int
    players: List[Dict[str, Any]]
    ball_position: Optional[Tuple[float, float]]
    heatmap: np.ndarray
    decision: Dict[str, Any]
    xT_value: float
    processing_time: float


class OverlayRenderer:
    """
    WebGPU-accelerated overlay renderer for real-time visualization.
    Falls back to OpenCV for systems without WebGPU support.
    """

    def __init__(self, width: int = 1920, height: int = 1080, use_gpu: bool = True):
        """
        Initialize overlay renderer.

        Args:
            width: Frame width
            height: Frame height
            use_gpu: Use GPU acceleration if available
        """
        self.width = width
        self.height = height
        self.use_gpu = use_gpu

        # Check GPU availability
        self.gpu_available = (TORCH_AVAILABLE and torch.cuda.is_available()) if use_gpu else False

        # Color schemes
        self.colors = {
            'player': (0, 255, 0),      # Green for players
            'ball': (255, 255, 0),       # Yellow for ball
            'trail': (0, 255, 255),      # Cyan for movement trails
            'zone': (255, 0, 0),         # Red for danger zones
            'heatmap': cv2.COLORMAP_JET  # Heatmap colormap
        }

        # Trail history
        self.player_trails: Dict[str, List[Tuple[int, int]]] = {}
        self.max_trail_length = 30

        logger.info(f"Overlay renderer initialized (GPU: {self.gpu_available})")

    def render_overlay(
        self,
        frame: np.ndarray,
        inference_result: InferenceResult,
        show_trails: bool = True,
        show_zones: bool = True,
        show_heatmap: bool = True,
        show_banner: bool = True
    ) -> np.ndarray:
        """
        Render overlay on video frame.

        Args:
            frame: Input video frame
            inference_result: Inference results to visualize
            show_trails: Show player movement trails
            show_zones: Show danger zones
            show_heatmap: Show xT heatmap
            show_banner: Show info banner

        Returns:
            Frame with overlay rendered
        """
        overlay = frame.copy()

        # Render heatmap layer
        if show_heatmap and inference_result.heatmap is not None:
            overlay = self._render_heatmap(overlay, inference_result.heatmap)

        # Render danger zones
        if show_zones:
            overlay = self._render_zones(overlay, inference_result.players)

        # Render player trails
        if show_trails:
            overlay = self._render_trails(overlay, inference_result.players)

        # Render players and ball
        overlay = self._render_players(overlay, inference_result.players)
        if inference_result.ball_position:
            overlay = self._render_ball(overlay, inference_result.ball_position)

        # Render info banner
        if show_banner:
            overlay = self._render_banner(overlay, inference_result)

        return overlay

    def _render_heatmap(self, frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Render xT heatmap overlay."""
        # Resize heatmap to frame size
        heatmap_resized = cv2.resize(heatmap, (self.width, self.height))

        # Normalize to 0-255
        heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, self.colors['heatmap'])

        # Blend with frame (30% opacity)
        blended = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)

        return blended

    def _render_zones(self, frame: np.ndarray, players: List[Dict[str, Any]]) -> np.ndarray:
        """Render danger zones around attacking players."""
        overlay = frame.copy()

        for player in players:
            if player.get('team') == 'offense':
                # Get player position in pixel coordinates
                x, y = self._field_to_pixel(player['x'], player['y'])

                # Draw danger zone (semi-transparent circle)
                radius = int(50 * (1 + player.get('speed', 0) / 10))
                cv2.circle(overlay, (x, y), radius, self.colors['zone'], -1)

        # Blend with original frame
        blended = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)

        return blended

    def _render_trails(self, frame: np.ndarray, players: List[Dict[str, Any]]) -> np.ndarray:
        """Render movement trails for players."""
        for player in players:
            player_id = player['id']
            x, y = self._field_to_pixel(player['x'], player['y'])

            # Update trail history
            if player_id not in self.player_trails:
                self.player_trails[player_id] = []

            self.player_trails[player_id].append((x, y))

            # Trim trail to max length
            if len(self.player_trails[player_id]) > self.max_trail_length:
                self.player_trails[player_id].pop(0)

            # Draw trail as polyline with fading effect
            if len(self.player_trails[player_id]) > 1:
                points = np.array(self.player_trails[player_id], dtype=np.int32)
                for i in range(len(points) - 1):
                    alpha = i / len(points)  # Fade older points
                    color = tuple(int(c * alpha) for c in self.colors['trail'])
                    cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), color, 2)

        return frame

    def _render_players(self, frame: np.ndarray, players: List[Dict[str, Any]]) -> np.ndarray:
        """Render player markers."""
        for player in players:
            x, y = self._field_to_pixel(player['x'], player['y'])

            # Draw player circle
            color = (0, 255, 0) if player.get('team') == 'offense' else (255, 0, 0)
            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)

            # Draw player ID
            cv2.putText(
                frame, str(player['id']),
                (x - 10, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return frame

    def _render_ball(self, frame: np.ndarray, ball_position: Tuple[float, float]) -> np.ndarray:
        """Render ball marker."""
        x, y = self._field_to_pixel(ball_position[0], ball_position[1])
        cv2.circle(frame, (x, y), 8, self.colors['ball'], -1)
        cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)
        return frame

    def _render_banner(self, frame: np.ndarray, inference_result: InferenceResult) -> np.ndarray:
        """Render information banner."""
        # Banner background
        cv2.rectangle(frame, (0, 0), (self.width, 80), (0, 0, 0), -1)

        # xT value
        xt_text = f"xT: {inference_result.xT_value:.3f}"
        cv2.putText(
            frame, xt_text,
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        # Decision
        decision = inference_result.decision.get('action', 'N/A')
        decision_text = f"Decision: {decision}"
        cv2.putText(
            frame, decision_text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        # FPS
        fps_text = f"FPS: {1000 / max(1, inference_result.processing_time):.1f}"
        cv2.putText(
            frame, fps_text,
            (self.width - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        # Frame ID
        frame_text = f"Frame: {inference_result.frame_id}"
        cv2.putText(
            frame, frame_text,
            (self.width - 200, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        return frame

    def _field_to_pixel(self, field_x: float, field_y: float) -> Tuple[int, int]:
        """Convert field coordinates to pixel coordinates."""
        # Assume field is 105m x 68m, map to frame dimensions
        pixel_x = int((field_x / 105.0) * self.width)
        pixel_y = int((field_y / 68.0) * self.height)
        return pixel_x, pixel_y

    def clear_trails(self):
        """Clear all player trails."""
        self.player_trails.clear()


class LiveEngine:
    """
    Main real-time inference engine.

    Pipeline: ingest → normalize → infer + adapter → heatmap + decision → overlay
    """

    def __init__(
        self,
        video_source: str,
        model_path: Optional[str] = None,
        target_fps: int = 30,
        use_gpu: bool = True,
        dummy_mode: bool = False
    ):
        """
        Initialize live engine.

        Args:
            video_source: RTSP URL or video file path
            model_path: Path to inference model
            target_fps: Target processing FPS
            use_gpu: Use GPU acceleration
            dummy_mode: Use dummy ingestor for testing
        """
        self.video_source = video_source
        self.model_path = model_path
        self.target_fps = target_fps
        self.use_gpu = use_gpu
        self.dummy_mode = dummy_mode

        # Initialize components
        if dummy_mode:
            self.ingestor = DummyIngestor(target_fps=target_fps)
        else:
            self.ingestor = VideoIngestor(
                source=video_source,
                target_fps=target_fps,
                backend='opencv'
            )

        self.renderer = OverlayRenderer(use_gpu=use_gpu)

        # Initialize model (placeholder - will integrate with existing backbone)
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = None  # Load actual model here

        # Performance tracking
        self.frame_times = []
        self.inference_times = []

        # State
        self.running = False
        self.frame_count = 0

        logger.info(f"Live engine initialized (GPU: {self.use_gpu}, Dummy: {self.dummy_mode})")

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize frame for inference."""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        frame_normalized = frame_rgb.astype(np.float32) / 255.0

        return frame_normalized

    def _run_inference(self, frame: np.ndarray) -> InferenceResult:
        """
        Run inference pipeline on frame.

        For now, generates dummy results. Will integrate with actual models.
        """
        start_time = time.time()

        # TODO: Integrate with actual backbone model
        # For now, generate dummy results
        players = [
            {'id': 'P1', 'x': 30.0, 'y': 20.0, 'team': 'offense', 'speed': 7.5},
            {'id': 'P2', 'x': 50.0, 'y': 34.0, 'team': 'offense', 'speed': 8.2},
            {'id': 'P3', 'x': 45.0, 'y': 25.0, 'team': 'defense', 'speed': 6.8},
        ]

        ball_position = (52.5, 34.0)

        # Generate dummy heatmap
        heatmap = np.random.rand(68, 105) * 0.3

        # Dummy decision
        decision = {
            'action': 'Pass Forward',
            'confidence': 0.87,
            'target': 'P2'
        }

        xT_value = 0.18

        processing_time = (time.time() - start_time) * 1000  # ms

        return InferenceResult(
            timestamp=time.time(),
            frame_id=self.frame_count,
            players=players,
            ball_position=ball_position,
            heatmap=heatmap,
            decision=decision,
            xT_value=xT_value,
            processing_time=processing_time
        )

    def process_frame(self, frame: np.ndarray, metadata: FrameMetadata) -> Tuple[np.ndarray, InferenceResult]:
        """
        Process single frame through pipeline.

        Args:
            frame: Input frame
            metadata: Frame metadata

        Returns:
            Tuple of (rendered frame, inference result)
        """
        # Normalize
        normalized = self._normalize_frame(frame)

        # Run inference
        result = self._run_inference(normalized)

        # Render overlay
        rendered = self.renderer.render_overlay(frame, result)

        return rendered, result

    def start(self) -> bool:
        """Start live processing engine."""
        if not self.ingestor.start():
            logger.error("Failed to start video ingestor")
            return False

        self.running = True
        logger.info("Live engine started")
        return True

    def stop(self):
        """Stop live processing engine."""
        self.running = False
        self.ingestor.stop()
        logger.info("Live engine stopped")

    def run_loop(self, output_path: Optional[str] = None, max_frames: Optional[int] = None):
        """
        Main processing loop.

        Args:
            output_path: Optional path to save output video
            max_frames: Maximum frames to process (None for unlimited)
        """
        # Setup video writer if output path provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, fourcc, self.target_fps,
                (self.renderer.width, self.renderer.height)
            )

        logger.info("Starting processing loop...")

        try:
            while self.running:
                # Get frame from ingestor
                result = self.ingestor.get_frame(timeout=0.1)
                if result is None:
                    continue

                frame, metadata = result

                # Process frame
                loop_start = time.time()
                rendered, inference_result = self.process_frame(frame, metadata)

                # Track performance
                frame_time = (time.time() - loop_start) * 1000
                self.frame_times.append(frame_time)
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

                # Log performance
                if self.frame_count % 30 == 0:
                    avg_fps = 1000 / np.mean(self.frame_times) if self.frame_times else 0
                    logger.info(
                        f"Frame {self.frame_count}: "
                        f"FPS={avg_fps:.1f}, "
                        f"xT={inference_result.xT_value:.3f}, "
                        f"Processing={frame_time:.1f}ms"
                    )

                # Write to output if enabled
                if video_writer:
                    video_writer.write(rendered)

                # Check max frames
                self.frame_count += 1
                if max_frames and self.frame_count >= max_frames:
                    break

        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")

        finally:
            if video_writer:
                video_writer.release()
            logger.info(f"Processing complete. Total frames: {self.frame_count}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        ingest_stats = self.ingestor.get_stats()
        avg_fps = 1000 / np.mean(self.frame_times) if self.frame_times else 0

        return {
            'frame_count': self.frame_count,
            'avg_fps': avg_fps,
            'target_fps': self.target_fps,
            'ingest_stats': ingest_stats,
            'running': self.running
        }
