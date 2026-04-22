# Project Handover Guide

This document is for the next student who will continue this repository.

## 1) Repository State

Current experiment branches:

- `main`: baseline/stable project history
- `exp/negative-testing-route`: negative-test + ensemble/calibration route
- `exp/ratio-1-1-route`: strict 1:1 sampling route and comparison utility

Important note:

- Large model checkpoints and generated plots are intentionally ignored by git.
- Committed outputs are lightweight summaries (`results.json`, `results.txt`, selected reports).

## 2) Recommended Environment

Use the GPU-capable environment:

```bash
conda activate efficientnet_env
```

Verify GPU is visible to TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Expected:

- At least one GPU listed, for example `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`

## 3) Fast Start Commands

From repository root:

### Baseline training

```bash
conda run -n efficientnet_env --no-capture-output python original_dataset_unbalanced_optimized.py
```

### Negative-testing route (branch: `exp/negative-testing-route`)

```bash
git switch exp/negative-testing-route
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious_negative.py
conda run -n efficientnet_env --no-capture-output python train_gdc_risk_negative.py
conda run -n efficientnet_env --no-capture-output python optimize_ensemble_calibrated.py
```

### Strict 1:1 route (branch: `exp/ratio-1-1-route`)

```bash
git switch exp/ratio-1-1-route
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious_ratio_1_1.py
conda run -n efficientnet_env --no-capture-output python train_gdc_risk_ratio_1_1.py
conda run -n efficientnet_env --no-capture-output python compare_base_vs_ratio_1_1.py
```

## 4) Where Results Are Saved

Typical summary files:

- `gdc_suspicious_results/results.json`
- `gdc_risk_results/results.json`
- `gdc_suspicious_results_ensemble/*.json`
- `gdc_suspicious_results_ratio_1_1/results.json`
- `gdc_risk_results_ratio_1_1/results.json`

## 5) What Was Learned So Far

- GPU was available but only in `efficientnet_env`; base env was CPU-only.
- Strict 1:1 sampling did **not** improve metrics in recent runs.
- Negative-testing/calibration route produced stronger improvements than strict 1:1.

## 6) Practical Next Steps for Successor

1. Start from baseline and rerun one clean evaluation to verify local setup.
2. Continue with moderate ratios instead of strict 1:1:
   - suspicious: try 1:1.5 or 1:2
   - risk: try 1:4 to 1:8
3. Keep patient-level splits fixed for comparability.
4. Commit only scripts + lightweight summaries; keep heavy artifacts local.

## 7) Git Workflow to Keep Repo Healthy

- One branch per hypothesis:
  - `exp/<short-hypothesis-name>`
- Commit pattern:
  - scripts/config changes
  - summary json/txt
- Avoid committing:
  - `*.h5`, `*.weights.h5`, `*.png`, cache files

Example:

```bash
git switch -c exp/new-ratio-sweep
git add train_*.py results/*.json results/*.txt
git commit -m "Add ratio sweep for risk task"
git push -u origin exp/new-ratio-sweep
```
