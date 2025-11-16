"""
FieldSense AI v3.0 - Live Video Ingestion Module

Handles RTSP/UDP video stream ingestion with frame buffering and synchronization.
Supports GStreamer and FFmpeg backends for maximum compatibility.
"""

import cv2
import numpy as np
import subprocess
import threading
import time
from queue import Queue, Full, Empty
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FrameMetadata:
    """Metadata for ingested frames."""
    timestamp: float
    frame_id: int
    width: int
    height: int
    fps: float


class VideoIngestor:
    """
    Video stream ingestion with double buffering for real-time processing.

    Supports:
    - RTSP streams (e.g., rtsp://camera:554/stream)
    - UDP streams (e.g., udp://@:5000)
    - File playback for testing
    - Double buffering for sync
    - 30+ FPS target rate
    """

    def __init__(
        self,
        source: str,
        target_fps: int = 30,
        buffer_size: int = 2,
        backend: str = 'opencv',  # 'opencv', 'gstreamer', 'ffmpeg'
        width: int = 1920,
        height: int = 1080
    ):
        """
        Initialize video ingestor.

        Args:
            source: Video source (RTSP URL, UDP address, or file path)
            target_fps: Target frame rate for processing
            buffer_size: Number of frames to buffer (2 for double buffering)
            backend: Video backend to use
            width: Target frame width
            height: Target frame height
        """
        self.source = source
        self.target_fps = target_fps
        self.buffer_size = buffer_size
        self.backend = backend
        self.width = width
        self.height = height

        # Double buffer for frame synchronization
        self.frame_buffer = Queue(maxsize=buffer_size)
        self.current_frame: Optional[np.ndarray] = None
        self.current_metadata: Optional[FrameMetadata] = None

        # Stream control
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.frame_count = 0
        self.dropped_frames = 0

        # Performance tracking
        self.fps_tracker = []
        self.last_frame_time = 0.0

        # Video capture object
        self.cap: Optional[cv2.VideoCapture] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None

    def _init_opencv_capture(self) -> bool:
        """Initialize OpenCV video capture."""
        try:
            self.cap = cv2.VideoCapture(self.source)

            # Configure for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False

            # Get actual stream properties
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"OpenCV capture initialized: {actual_width}x{actual_height} @ {actual_fps} FPS")
            return True

        except Exception as e:
            logger.error(f"OpenCV capture initialization failed: {e}")
            return False

    def _init_ffmpeg_capture(self) -> bool:
        """Initialize FFmpeg subprocess for video capture."""
        try:
            # FFmpeg command for RTSP/UDP with low latency
            cmd = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',  # Use TCP for RTSP
                '-fflags', 'nobuffer',      # Minimize buffering
                '-flags', 'low_delay',      # Low latency mode
                '-i', self.source,
                '-f', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.target_fps),
                'pipe:1'
            ]

            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.width * self.height * 3
            )

            logger.info(f"FFmpeg capture initialized for {self.source}")
            return True

        except Exception as e:
            logger.error(f"FFmpeg capture initialization failed: {e}")
            return False

    def _capture_loop_opencv(self):
        """Capture loop using OpenCV backend."""
        frame_interval = 1.0 / self.target_fps

        while self.running:
            loop_start = time.time()

            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                continue

            # Resize if needed
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            # Create metadata
            current_time = time.time()
            metadata = FrameMetadata(
                timestamp=current_time,
                frame_id=self.frame_count,
                width=self.width,
                height=self.height,
                fps=self._calculate_fps(current_time)
            )

            # Add to buffer (drop if full - prefer fresh frames)
            try:
                self.frame_buffer.put_nowait((frame, metadata))
                self.frame_count += 1
            except Full:
                self.dropped_frames += 1
                # Drop oldest frame and add new one
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait((frame, metadata))
                except:
                    pass

            # Maintain target FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _capture_loop_ffmpeg(self):
        """Capture loop using FFmpeg backend."""
        frame_size = self.width * self.height * 3
        frame_interval = 1.0 / self.target_fps

        while self.running:
            loop_start = time.time()

            try:
                # Read raw frame from FFmpeg stdout
                raw_frame = self.ffmpeg_process.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    logger.warning("Incomplete frame received")
                    continue

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))

                # Create metadata
                current_time = time.time()
                metadata = FrameMetadata(
                    timestamp=current_time,
                    frame_id=self.frame_count,
                    width=self.width,
                    height=self.height,
                    fps=self._calculate_fps(current_time)
                )

                # Add to buffer
                try:
                    self.frame_buffer.put_nowait((frame, metadata))
                    self.frame_count += 1
                except Full:
                    self.dropped_frames += 1
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put_nowait((frame, metadata))
                    except:
                        pass

                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"FFmpeg capture error: {e}")
                break

    def _calculate_fps(self, current_time: float) -> float:
        """Calculate actual FPS."""
        if self.last_frame_time > 0:
            fps = 1.0 / (current_time - self.last_frame_time)
            self.fps_tracker.append(fps)
            if len(self.fps_tracker) > 30:
                self.fps_tracker.pop(0)
            self.last_frame_time = current_time
            return np.mean(self.fps_tracker)
        else:
            self.last_frame_time = current_time
            return 0.0

    def start(self) -> bool:
        """Start video ingestion."""
        if self.running:
            logger.warning("Ingestor already running")
            return False

        # Initialize capture backend
        if self.backend == 'opencv':
            if not self._init_opencv_capture():
                return False
            capture_func = self._capture_loop_opencv
        elif self.backend == 'ffmpeg':
            if not self._init_ffmpeg_capture():
                return False
            capture_func = self._capture_loop_ffmpeg
        else:
            logger.error(f"Unsupported backend: {self.backend}")
            return False

        # Start capture thread
        self.running = True
        self.capture_thread = threading.Thread(target=capture_func, daemon=True)
        self.capture_thread.start()

        logger.info(f"Video ingestion started: {self.source}")
        return True

    def stop(self):
        """Stop video ingestion."""
        self.running = False

        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()

        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait(timeout=2.0)

        logger.info("Video ingestion stopped")

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """
        Get latest frame from buffer.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (frame, metadata) or None if no frame available
        """
        try:
            frame, metadata = self.frame_buffer.get(timeout=timeout)
            self.current_frame = frame
            self.current_metadata = metadata
            return frame, metadata
        except Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        avg_fps = np.mean(self.fps_tracker) if self.fps_tracker else 0.0

        return {
            'frame_count': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'drop_rate': self.dropped_frames / max(1, self.frame_count),
            'avg_fps': avg_fps,
            'buffer_size': self.frame_buffer.qsize(),
            'running': self.running
        }


class DummyIngestor:
    """
    Dummy video ingestor for testing without real camera.
    Generates synthetic frames at target FPS.
    """

    def __init__(self, target_fps: int = 30, width: int = 1920, height: int = 1080):
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.frame_buffer = Queue(maxsize=2)
        self.running = False
        self.frame_count = 0
        self.capture_thread: Optional[threading.Thread] = None

    def _generate_frame(self) -> np.ndarray:
        """Generate synthetic frame with moving pattern."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Add moving pattern (simulates field)
        offset = (self.frame_count * 5) % self.width
        cv2.rectangle(frame, (offset, 200), (offset + 200, 400), (0, 255, 0), -1)

        # Add frame counter
        cv2.putText(
            frame, f"Frame: {self.frame_count}",
            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2
        )

        return frame

    def _capture_loop(self):
        """Generate frames at target FPS."""
        frame_interval = 1.0 / self.target_fps

        while self.running:
            loop_start = time.time()

            frame = self._generate_frame()
            metadata = FrameMetadata(
                timestamp=time.time(),
                frame_id=self.frame_count,
                width=self.width,
                height=self.height,
                fps=self.target_fps
            )

            try:
                self.frame_buffer.put_nowait((frame, metadata))
                self.frame_count += 1
            except Full:
                try:
                    self.frame_buffer.get_nowait()
                    self.frame_buffer.put_nowait((frame, metadata))
                except:
                    pass

            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def start(self) -> bool:
        """Start dummy ingestion."""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        logger.info(f"Dummy ingestion started: {self.width}x{self.height} @ {self.target_fps} FPS")
        return True

    def stop(self):
        """Stop dummy ingestion."""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        logger.info("Dummy ingestion stopped")

    def get_frame(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, FrameMetadata]]:
        """Get latest frame."""
        try:
            return self.frame_buffer.get(timeout=timeout)
        except Empty:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get ingestion statistics."""
        return {
            'frame_count': self.frame_count,
            'dropped_frames': 0,
            'drop_rate': 0.0,
            'avg_fps': self.target_fps,
            'buffer_size': self.frame_buffer.qsize(),
            'running': self.running
        }
