# Samples Directory

This directory contains representative samples of logs and MLflow experiment artifacts for reference and future reconstruction.

## Contents

### logs/
- `experiment_manager_sample.log` - Sample experiment manager log showing CLIP model training workflow
  - Demonstrates dataset splits, training initialization, and epoch tracking
  - Typical size: 98MB (full logs directory)

### mlruns/
- `run_metadata_sample.json` - Sample MLflow run metadata containing experiment parameters
  - Includes model config, hyperparameters, device info, and timestamps
  - Typical size: 6.5MB (full mlruns directory with 371 files)

## Original Directories

The full `logs/` and `mlruns/` directories are not committed to git due to their size (104.5MB total) as they are:
1. Auto-generated during training/experimentation
2. Not critical for code functionality
3. Better stored in cloud storage (S3, GCS) for ML artifacts

To restore full experiment logs:
- Use MLflow UI: `mlflow ui`
- Check cloud storage backups or previous experiment databases
- Re-run experiments to regenerate logs

## Models Tracked

- **CLIP** - Vision-text alignment model
- **ViLT** - Vision-Language Transformer
- **FLAVA** - Foundational Language And Vision Alignment

Dataset splits used: Train=300, Validation=100, Test=100 samples
