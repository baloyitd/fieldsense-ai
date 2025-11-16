# FieldSense AI v3.0

Advanced sports analytics platform using xT-Grid and Temporal ConvNet for field threat prediction.

## Week 1: Data Pipeline + Backbone Inference

This implementation includes:
- Data ingestion for multiple formats (STATSBomb JSON, Wyscout CSV, Custom NFL-style CSV)
- Data normalization to unified schema
- Backbone model (xT-Grid + Temporal ConvNet) inference
- Heatmap visualization

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

Run the test script to verify the pipeline:

```bash
python bin/test_backbone.py
```

This will:
1. Load sample data from 3 different formats
2. Normalize to unified schema
3. Run backbone inference
4. Generate heatmap visualizations

## Project Structure

```
fieldsense-ai/
├── src/
│   ├── data/
│   │   ├── ingest.py       # Data loading for multiple formats
│   │   └── normalize.py    # Normalization to unified schema
│   └── model/
│       └── backbone.py     # ONNX model inference and visualization
├── assets/
│   ├── backbone.onnx       # xT-Grid model (ONNX format)
│   └── sample_data/        # Sample data files
├── bin/
│   └── test_backbone.py    # Test script
├── tests/
│   └── test_data.py        # Unit tests
└── outputs/                # Generated heatmaps (created on run)
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

Run unit tests:

```bash
python tests/test_data.py
```

Run integration test:

```bash
python bin/test_backbone.py
```

## Performance

- Processing: <10s per 10-play sequence
- Inference: ~0.1s per frame
- Output: PNG heatmaps showing xT zones

## Next Steps

- Week 2: Enhanced spatiotemporal features
- Week 3: Player trajectory prediction
- Week 4: Real-time inference optimization
