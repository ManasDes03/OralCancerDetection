# Experiments Index

Quick index of major experiment routes and entry scripts.

## Route A: Baseline 3-Class Training

- `original_dataset_unbalanced_optimized.py`
- `balanced_3class_optimized.py`
- `balanced_3class_ratio_1_2_2_optimized.py`

Outputs:

- `original_results_optimized/`
- `balanced_results_optimized/`
- `ratio_1_2_2_results_optimized/`

## Route B: Negative-Testing + Ensemble

Branch:

- `exp/negative-testing-route`

Training scripts:

- `train_gdc_suspicious_negative.py`
- `train_gdc_risk_negative.py`

Inference/optimization:

- `predict_gdc_pipeline_ensemble.py`
- `generate_results_report_ensemble.py`
- `optimize_ensemble_significance.py`
- `optimize_ensemble_calibrated.py`
- `optimize_ensemble_oof_stacking.py`
- `optimize_ensemble_two_stage.py`

Core helper:

- `ensemble_fusion.py`

## Route C: Strict 1:1 Sampling

Branch:

- `exp/ratio-1-1-route`

Training scripts:

- `train_gdc_suspicious_ratio_1_1.py`
- `train_gdc_risk_ratio_1_1.py`

Comparison:

- `compare_base_vs_ratio_1_1.py`

## Recommended Continuation

- Keep Route B as primary reference for strongest recent results.
- Use Route C only as strict-balancing baseline and compare against moderate-ratio sweeps.
