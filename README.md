# FieldSense AI v3.0

Advanced sports analytics platform using xT-Grid and Temporal ConvNet for field threat prediction.

## Week 1: Data Pipeline + Backbone Inference

This implementation includes:
- Data ingestion for multiple formats (STATSBomb JSON, Wyscout CSV, Custom NFL-style CSV)
- Data normalization to unified schema
- Backbone model (xT-Grid + Temporal ConvNet) inference
- Heatmap visualization

## Week 2: LoRA Calibration (NEW!)

Team-specific adaptation with LoRA (Low-Rank Adaptation):
- **Fast calibration**: <42 seconds on 20-100 user plays
- **Lightweight adapter**: ~0.8MB (only 4-rank LoRA parameters)
- **PyTorch backbone** with frozen base + trainable LoRA layers
- **Automatic improvement** in team-specific zones (>0.05 xT delta)
- **No catastrophic forgetting**: validation loss remains stable

## Installation

```bash
pip install -r requirements.txt
```

**Note**: PyTorch installation may take several minutes. For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### Week 1: Data Pipeline & Backbone

Run the test script to verify the pipeline:

```bash
python bin/test_backbone.py
```

This will:
1. Load sample data from 3 different formats
2. Normalize to unified schema
3. Run backbone inference
4. Generate heatmap visualizations

### Week 2: Team Calibration

Run the calibration test to see team-specific adaptation:

```bash
python bin/test_calibrate.py
```

This will:
1. Generate 50 synthetic user plays (with left-side attack bias)
2. Calibrate LoRA adapter in <42 seconds
3. Compare before/after heatmaps
4. Show xT improvement in team-specific zones
5. Verify adapter size (~0.8MB)

## Project Structure

```
fieldsense-ai/
├── src/
│   ├── data/
│   │   ├── ingest.py           # Data loading for multiple formats
│   │   └── normalize.py        # Normalization to unified schema
│   └── model/
│       ├── backbone.py         # ONNX model inference and visualization
│       ├── pytorch_backbone.py # PyTorch backbone (trainable)
│       ├── adapter.py          # LoRA adapter module
│       └── calibrate.py        # Team calibration engine
├── assets/
│   ├── backbone.onnx       # xT-Grid model (ONNX format, 5.3KB)
│   └── sample_data/        # Sample data files
├── bin/
│   ├── test_backbone.py    # Week 1 test script
│   └── test_calibrate.py   # Week 2 calibration test
├── tests/
│   ├── test_data.py        # Data pipeline unit tests
│   └── test_model.py       # Model/adapter unit tests
└── outputs/                # Generated heatmaps & adapters
```

## Data Formats

### Unified Schema

All input formats are normalized to:

```json
{
  "frames": [
    {
      "time": 0.0,
      "players": [
        {
          "id": "101",
          "x": 52.5,
          "y": 34.0,
          "vel": 5.2,
          "dir": 45.0,
          "acc": 1.2,
          "role": "QB",
          "possession": true
        }
      ],
      "ball": {"x": 52.5, "y": 34.0}
    }
  ]
}
```

Field dimensions: 105m x 68m (standard soccer pitch)

## Testing

### Week 1 Tests

Run data pipeline unit tests:

```bash
python tests/test_data.py
```

Run backbone integration test:

```bash
python bin/test_backbone.py
```

### Week 2 Tests

Run model/adapter unit tests:

```bash
python tests/test_model.py
```

Run calibration integration test:

```bash
python bin/test_calibrate.py
```

## Performance

### Week 1 (Baseline)
- Processing: <10s per 10-play sequence
- Inference: ~0.1s per frame
- Output: PNG heatmaps showing xT zones

### Week 2 (Calibration)
- Calibration time: <42s for 20-100 plays
- Adapter size: ~0.8MB (rank-4 LoRA)
- Training: 5 epochs, batch=8, AdamW optimizer
- Improvement: >0.05 xT delta in team zones
- Parameters: <1% trainable (LoRA only)

## API Usage

### Basic Inference (Week 1)

```python
from src.model.backbone import BackboneModel

# Load and infer
model = BackboneModel('assets/backbone.onnx')
xt_grid = model.infer_from_json(normalized_data)
```

### Team Calibration (Week 2)

```python
from src.model.adapter import create_adapter_model
from src.model.calibrate import calibrate_model

# Create model with LoRA adapter
model = create_adapter_model(
    onnx_path='assets/backbone.onnx',
    lora_rank=4
)

# Calibrate on user plays
results = calibrate_model(
    model=model,
    user_plays=my_team_plays,  # List of normalized frames
    output_path='outputs/my_team_adapter.bin',
    epochs=5,
    batch_size=8,
    max_time=42.0
)

# Load calibrated model
from src.model.calibrate import load_calibrated_model
calibrated = load_calibrated_model(
    'assets/backbone.onnx',
    'outputs/my_team_adapter.bin'
)
```

## Next Steps

- Week 3: Multi-agent trajectory prediction
- Week 4: Real-time inference optimization
- Week 5: Production deployment & API
