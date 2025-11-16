"""
FieldSense AI v3.0 - Auto-Clip Generation

Automatically saves video clips when high-value events occur (xT > 0.25).
Includes overlay rendering and optional voice commentary.
"""

import cv2
import numpy as np
import subprocess
import time
from collections import deque
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClipTrigger:
    """Trigger condition for auto-clip generation."""
    timestamp: float
    frame_id: int
    xT_value: float
    event_type: str
    description: str


@dataclass
class Clip:
    """Auto-generated video clip."""
    clip_id: str
    start_time: float
    end_time: float
    duration: float
    trigger: ClipTrigger
    output_path: str
    frames_saved: int


class FrameBuffer:
    """
    Circular buffer for storing recent frames.
    Enables retrospective clip generation.
    """

    def __init__(self, max_duration: float = 10.0, fps: int = 30):
        """
        Initialize frame buffer.

        Args:
            max_duration: Maximum buffer duration in seconds
            fps: Frames per second
        """
        self.max_duration = max_duration
        self.fps = fps
        self.max_frames = int(max_duration * fps)

        # Deque for efficient FIFO operations
        self.frames = deque(maxlen=self.max_frames)
        self.timestamps = deque(maxlen=self.max_frames)
        self.metadata = deque(maxlen=self.max_frames)

    def add_frame(self, frame: np.ndarray, timestamp: float, metadata: Dict[str, Any]):
        """Add frame to buffer."""
        self.frames.append(frame.copy())
        self.timestamps.append(timestamp)
        self.metadata.append(metadata)

    def get_frames(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        duration: Optional[float] = None
    ) -> List[np.ndarray]:
        """
        Get frames from buffer.

        Args:
            start_time: Start timestamp (None for beginning)
            end_time: End timestamp (None for current)
            duration: Duration in seconds (overrides start_time if provided)

        Returns:
            List of frames
        """
        if len(self.frames) == 0:
            return []

        # Convert to lists for easier indexing
        frames_list = list(self.frames)
        timestamps_list = list(self.timestamps)

        # Determine time range
        if end_time is None:
            end_time = timestamps_list[-1]

        if duration is not None:
            start_time = end_time - duration
        elif start_time is None:
            start_time = timestamps_list[0]

        # Find frames in time range
        selected_frames = []
        for i, ts in enumerate(timestamps_list):
            if start_time <= ts <= end_time:
                selected_frames.append(frames_list[i])

        return selected_frames

    def get_duration(self) -> float:
        """Get current buffer duration."""
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def clear(self):
        """Clear buffer."""
        self.frames.clear()
        self.timestamps.clear()
        self.metadata.clear()


class VoiceCommentary:
    """
    Text-to-speech commentary generator for clips.
    Uses system TTS or external service.
    """

    def __init__(self, enabled: bool = True, voice: str = 'default'):
        """
        Initialize voice commentary.

        Args:
            enabled: Enable voice commentary
            voice: Voice profile to use
        """
        self.enabled = enabled
        self.voice = voice

    def generate_commentary(self, trigger: ClipTrigger) -> str:
        """
        Generate commentary text for trigger event.

        Args:
            trigger: Clip trigger event

        Returns:
            Commentary text
        """
        if not self.enabled:
            return ""

        # Generate contextual commentary based on xT value and event
        if trigger.xT_value > 0.4:
            intensity = "Excellent"
        elif trigger.xT_value > 0.3:
            intensity = "Great"
        elif trigger.xT_value > 0.25:
            intensity = "High value"
        else:
            intensity = "Notable"

        commentary = f"{intensity} attack! Expected threat of {trigger.xT_value:.2f}. {trigger.description}"

        return commentary

    def synthesize_audio(self, text: str, output_path: str) -> bool:
        """
        Synthesize speech audio from text.

        Args:
            text: Commentary text
            output_path: Output audio file path

        Returns:
            True if successful
        """
        if not self.enabled or not text:
            return False

        try:
            # Use espeak for TTS (cross-platform)
            # Alternative: pyttsx3, gTTS, cloud TTS services
            subprocess.run(
                ['espeak', '-w', output_path, text],
                check=True,
                capture_output=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("TTS synthesis failed (espeak not found)")
            return False


class ClipGenerator:
    """
    Automatic clip generation on trigger events.

    Features:
    - Circular buffer for retrospective capture
    - xT threshold triggers (default: 0.25)
    - FFmpeg encoding with overlay
    - Optional voice commentary
    """

    def __init__(
        self,
        output_dir: str = './clips',
        clip_duration: float = 10.0,
        fps: int = 30,
        xT_threshold: float = 0.25,
        enable_voice: bool = True,
        codec: str = 'mp4v'
    ):
        """
        Initialize clip generator.

        Args:
            output_dir: Directory for saving clips
            clip_duration: Clip duration in seconds
            fps: Frames per second
            xT_threshold: xT value threshold for triggering clips
            enable_voice: Enable voice commentary
            codec: Video codec (mp4v, h264, etc.)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.clip_duration = clip_duration
        self.fps = fps
        self.xT_threshold = xT_threshold
        self.codec = codec

        # Frame buffer for retrospective capture
        self.buffer = FrameBuffer(max_duration=clip_duration, fps=fps)

        # Voice commentary
        self.voice = VoiceCommentary(enabled=enable_voice)

        # Clip tracking
        self.clips: List[Clip] = []
        self.last_trigger_time = 0.0
        self.min_trigger_interval = 5.0  # Minimum seconds between triggers

        logger.info(f"Clip generator initialized: {output_dir}")

    def add_frame(self, frame: np.ndarray, timestamp: float, xT_value: float, metadata: Dict[str, Any]):
        """
        Add frame to buffer and check for triggers.

        Args:
            frame: Video frame with overlay
            timestamp: Frame timestamp
            xT_value: Current xT value
            metadata: Frame metadata
        """
        # Add to buffer
        self.buffer.add_frame(frame, timestamp, metadata)

        # Check trigger condition
        if self._should_trigger(timestamp, xT_value):
            self._trigger_clip(timestamp, xT_value, metadata)

    def _should_trigger(self, timestamp: float, xT_value: float) -> bool:
        """Check if clip should be triggered."""
        # Check threshold
        if xT_value < self.xT_threshold:
            return False

        # Check minimum interval
        if timestamp - self.last_trigger_time < self.min_trigger_interval:
            return False

        return True

    def _trigger_clip(self, timestamp: float, xT_value: float, metadata: Dict[str, Any]):
        """
        Trigger clip generation.

        Args:
            timestamp: Trigger timestamp
            xT_value: xT value at trigger
            metadata: Frame metadata
        """
        # Create trigger
        trigger = ClipTrigger(
            timestamp=timestamp,
            frame_id=metadata.get('frame_id', 0),
            xT_value=xT_value,
            event_type='high_xT',
            description=f"High value attack detected"
        )

        logger.info(f"Clip triggered: xT={xT_value:.3f} at {timestamp:.2f}s")

        # Generate clip
        clip = self._generate_clip(trigger)

        if clip:
            self.clips.append(clip)
            self.last_trigger_time = timestamp
            logger.info(f"Clip saved: {clip.output_path}")

    def _generate_clip(self, trigger: ClipTrigger) -> Optional[Clip]:
        """
        Generate video clip from buffered frames.

        Args:
            trigger: Clip trigger

        Returns:
            Generated clip or None if failed
        """
        # Get frames from buffer (10s before trigger)
        end_time = trigger.timestamp
        start_time = end_time - self.clip_duration
        frames = self.buffer.get_frames(start_time=start_time, end_time=end_time)

        if len(frames) == 0:
            logger.warning("No frames available for clip")
            return None

        # Generate clip filename
        clip_id = f"clip_{int(trigger.timestamp)}_{trigger.frame_id}"
        output_filename = f"{clip_id}.mp4"
        output_path = self.output_dir / output_filename

        # Write video with OpenCV
        success = self._write_video(frames, str(output_path))

        if not success:
            return None

        # Generate voice commentary
        commentary = self.voice.generate_commentary(trigger)
        if commentary:
            audio_path = self.output_dir / f"{clip_id}_commentary.wav"
            self.voice.synthesize_audio(commentary, str(audio_path))

            # Merge audio with video using FFmpeg
            self._merge_audio(str(output_path), str(audio_path))

        # Create clip object
        clip = Clip(
            clip_id=clip_id,
            start_time=start_time,
            end_time=end_time,
            duration=self.clip_duration,
            trigger=trigger,
            output_path=str(output_path),
            frames_saved=len(frames)
        )

        return clip

    def _write_video(self, frames: List[np.ndarray], output_path: str) -> bool:
        """
        Write frames to video file.

        Args:
            frames: List of frames
            output_path: Output file path

        Returns:
            True if successful
        """
        if len(frames) == 0:
            return False

        try:
            # Get frame dimensions
            height, width = frames[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

            # Write frames
            for frame in frames:
                writer.write(frame)

            writer.release()
            return True

        except Exception as e:
            logger.error(f"Video writing failed: {e}")
            return False

    def _merge_audio(self, video_path: str, audio_path: str):
        """
        Merge audio commentary with video using FFmpeg.

        Args:
            video_path: Video file path
            audio_path: Audio file path
        """
        try:
            output_path = video_path.replace('.mp4', '_with_audio.mp4')

            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-strict', 'experimental',
                '-y',
                output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)

            # Replace original with merged version
            Path(video_path).unlink()
            Path(output_path).rename(video_path)
            Path(audio_path).unlink()

        except Exception as e:
            logger.warning(f"Audio merge failed: {e}")

    def get_clips(self) -> List[Clip]:
        """Get all generated clips."""
        return self.clips

    def get_stats(self) -> Dict[str, Any]:
        """Get clip generation statistics."""
        return {
            'total_clips': len(self.clips),
            'threshold': self.xT_threshold,
            'clip_duration': self.clip_duration,
            'buffer_duration': self.buffer.get_duration()
        }


class LiveClipEngine:
    """
    Integrated live clip generation engine.
    Works with LiveEngine to automatically save highlights.
    """

    def __init__(
        self,
        output_dir: str = './clips',
        xT_threshold: float = 0.25,
        enable_voice: bool = True
    ):
        """
        Initialize live clip engine.

        Args:
            output_dir: Output directory for clips
            xT_threshold: xT threshold for triggers
            enable_voice: Enable voice commentary
        """
        self.generator = ClipGenerator(
            output_dir=output_dir,
            xT_threshold=xT_threshold,
            enable_voice=enable_voice
        )

        logger.info("Live clip engine initialized")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        xT_value: float,
        metadata: Dict[str, Any]
    ):
        """
        Process frame and trigger clips if needed.

        Args:
            frame: Rendered frame with overlay
            timestamp: Frame timestamp
            xT_value: Current xT value
            metadata: Frame metadata
        """
        self.generator.add_frame(frame, timestamp, xT_value, metadata)

    def get_clips(self) -> List[Clip]:
        """Get all generated clips."""
        return self.generator.get_clips()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.generator.get_stats()
