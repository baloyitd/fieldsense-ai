# FieldSense NCAA 2026 Prediction System

Stage 09/10 — Privacy-certified, deterministic, fully offline submission package.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate submission using synthetic data (for testing)
python run_prediction.py --synthetic --output submission.csv

# Generate submission with Kaggle data + full certification
python run_prediction.py \
    --data-dir ./kaggle_data \
    --output submission.csv \
    --season 2026 \
    --seed 42 \
    --certify
```

## Reproduction Instructions

1. **Clone the repository** and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Kaggle data** into `./kaggle_data/`:
   - `MTeams.csv`, `WTeams.csv`
   - `MRegularSeasonDetailedResults.csv`

3. **Run prediction** (deterministic, offline):
   ```bash
   python run_prediction.py --data-dir ./kaggle_data --output submission.csv --seed 42
   ```

4. **Run certification** (optional but recommended):
   ```bash
   python run_prediction.py --synthetic --certify --cert-dir certification/
   cat certification/privacy_cert.md
   ```

5. **Run tests**:
   ```bash
   pytest ncaa_submission_2026/tests/ -v
   ```

## Architecture

```
run_prediction.py          ← Single entry point
├── ncaa_models/           ← Stages 02-06: Baseline, LoRA, Ensemble, Calibration
├── ncaa_agent/            ← Stage 07: Reasoner, Anomaly, Narrative
├── ncaa_live/             ← Stage 08: Live adaptation pipeline
└── certification/         ← Stage 09: Privacy & reproducibility certification
    ├── leakage_detector.py
    ├── determinism_checker.py
    ├── network_isolator.py
    ├── dependency_auditor.py
    └── certifier.py
```

## Certification Checks

| Check | Description | Pass Condition |
|-------|-------------|----------------|
| Data Leakage | No 2026 data in training set | Zero rows with season ≥ 2026 |
| Determinism | Identical outputs across runs | Bitwise-identical predictions |
| Network Isolation | No outbound HTTP/socket calls | Zero network attempts during inference |
| Dependency Audit | All required packages present | numpy, pandas, scikit-learn, scipy |

## Random Seed

All predictions use `--seed 42` by default. This seed is applied to:
- Python `random`
- NumPy `np.random`
- PyTorch (if installed)

## Privacy Guarantees

- No personal data (PII) is processed
- Training data: seasons 2021–2024 only
- Validation data: season 2025 (held out from training)
- Test prediction: season 2026 (features only, no outcomes used)
- All computation is offline; no API calls during inference
