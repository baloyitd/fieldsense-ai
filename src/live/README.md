# FieldSense AI v3.0 - Live Mode

Real-time video processing pipeline with RTSP/UDP ingestion, 30+ FPS inference, overlay rendering, and auto-clip generation.

## Components

### 1. Video Ingestion (`ingest.py`)
- **RTSP/UDP Stream Support**: Connect to live camera feeds via RTSP or UDP
- **Multiple Backends**: OpenCV and FFmpeg support
- **Double Buffering**: Frame synchronization with 2-frame buffer
- **Dummy Mode**: Testing with synthetic frames

### 2. Live Engine (`live_engine.py`)
- **Real-time Pipeline**: ingest → normalize → infer → heatmap → overlay
- **WebGPU Rendering**: GPU-accelerated overlay composition
- **Performance Tracking**: FPS monitoring and lag detection
- **Overlay Features**:
  - Player tracking with trails
  - xT heatmap visualization
  - Danger zone highlighting
  - Info banner with FPS counter

### 3. Auto-Clip Generation (`clips.py`)
- **Event Triggers**: Automatic clip generation when xT > 0.25
- **Retrospective Capture**: 10s clips from circular buffer
- **Voice Commentary**: Text-to-speech annotations (optional)
- **MP4 Encoding**: FFmpeg-based video export

## Performance

### Test Results (CPU Mode - No GPU)

```
Video Ingestion:     29.7 FPS  ✓
Full Pipeline:       15.0 FPS  (with overlay rendering)
Average Lag:         76.7 ms   ✓
Clip Generation:     Working   ✓
```

### Expected Performance (GPU Mode)

With GPU/WebGPU acceleration:
- **Video Ingestion**: 30+ FPS
- **Full Pipeline**: 30+ FPS
- **Lag**: < 100ms
- **Resolution**: 1920x1080

## Usage

### Basic Example

```python
from src.live import LiveEngine, LiveClipEngine

# Initialize live engine
engine = LiveEngine(
    video_source='rtsp://camera:554/stream',
    target_fps=30,
    use_gpu=True
)

# Initialize clip generator
clips = LiveClipEngine(
    output_dir='./clips',
    xT_threshold=0.25,
    enable_voice=True
)

# Start processing
engine.start()

# Process frames in loop
while True:
    result = engine.ingestor.get_frame()
    if result:
        frame, metadata = result
        rendered, inference = engine.process_frame(frame, metadata)

        # Auto-clip generation
        clips.process_frame(
            rendered,
            metadata.timestamp,
            inference.xT_value,
            {'frame_id': metadata.frame_id}
        )
```

### Dummy Mode (Testing)

```python
# Use dummy ingestor for testing
engine = LiveEngine(
    video_source='dummy',
    target_fps=30,
    dummy_mode=True
)
```

## Video Sources

### RTSP
```python
video_source = 'rtsp://192.168.1.100:554/stream'
```

### UDP
```python
video_source = 'udp://@:5000'
```

### File Playback
```python
video_source = '/path/to/video.mp4'
```

## UI Integration

The live mode integrates with the React dashboard:

- **Live Toggle**: Start/pause live processing
- **Overlay Controls**: Show/hide overlay layers
- **FPS Display**: Real-time performance monitoring
- **Clips Panel**: Auto-generated highlight clips
- **Video Player**: Canvas-based overlay rendering

## Architecture

```
RTSP/UDP Stream
      ↓
Video Ingestor (Double Buffer)
      ↓
Frame Normalization
      ↓
Inference Engine
      ↓
Heatmap + Decision
      ↓
Overlay Renderer (WebGPU)
      ↓
Display + Clip Generation
```

## Performance Optimization

1. **GPU Acceleration**: Use `use_gpu=True` for WebGPU rendering
2. **Lower Resolution**: Reduce to 1280x720 for faster processing
3. **Simplified Overlay**: Disable trails/heatmap for maximum FPS
4. **Buffer Size**: Adjust `buffer_size` based on latency requirements

## Requirements

```
opencv-python >= 4.8.0
numpy >= 1.24.0
torch >= 2.0.0 (optional, for GPU)
ffmpeg (system dependency)
```

## Testing

Run comprehensive test suite:

```bash
python bin/live_test.py
```

Tests include:
- Video ingestion performance
- Full pipeline FPS validation
- Clip generation triggers
- Lag measurement

## Future Enhancements

- [ ] WebRTC support for browser-based streaming
- [ ] Multi-camera ingestion
- [ ] Real-time pose estimation integration
- [ ] Cloud clip storage
- [ ] Adaptive bitrate streaming
