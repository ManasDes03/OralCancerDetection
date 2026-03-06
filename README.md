# Oral Cancer Detection using Deep Learning

A comprehensive deep learning system for automated oral cancer detection across **two datasets** and **two model architectures**:

1. **GDC Two-Stage Pipeline** — EfficientNetB0, binary classification, real-world clinical dataset (2,964 images, 518 patients)
2. **Sri Lankan 3-Class Classifier** — MobileNetV2, OCA / Healthy / Mutations classification

---

## 🎯 Project Overview

### GDC Two-Stage Pipeline (Primary — Production Ready)

A recall-biased, two-stage screening system trained on the GDC Oral Cancer Dataset:

```
Any image → Stage 1: Suspicious?  ──YES──→ Stage 2: High-risk?  → Risk result
                                   ──NO───→ Not suspicious (safe)
```

- **Stage 1 — Suspicious vs Non-suspicious**: Flags any potentially abnormal tissue. Optimised for near-perfect recall so nothing dangerous is missed.
- **Stage 2 — High-risk vs Low-risk**: Among suspicious cases, identifies high-risk (malignant) lesions. Achieves **100% recall** (zero missed high-risk cases) on test set.

### Sri Lankan 3-Class Classifier (Baseline)

Classifies oral cavity images into three categories:
- **OCA (Oral Cancer)** — Malignant lesions
- **Healthy** — Normal tissue
- **Mutations (OPMD + Benign)** — Pre-malignant and benign lesions

---

## 🔬 GDC Two-Stage Pipeline — Key Features

### Architecture & Training
- **Backbone**: EfficientNetB0 (ImageNet pretrained), 224×224 input
- **Two-phase training**: Phase 1 — frozen backbone; Phase 2 — top N layers unfrozen
- **Save strategy**: `model.save_weights()` only (avoids TF 2.10 mixed-precision JSON crash)
- **Mixed precision** (`float16`) + XLA JIT compilation for GPU throughput

### Handling Class Imbalance
| Problem | Solution |
|---|---|
| Stage 2 is 24:1 imbalanced (116 high-risk / 2,964 total) | Balanced batch interleaving via `tf.data.Dataset.zip` |
| Minority class underrepresented each epoch | `.repeat()` + `steps_per_epoch` keeps minority visible every step |
| Standard weights biased post-oversampling | Class weights computed **pre-oversampling** from original split |
| Missing rare examples from training signal | 1.5× recall boost applied to minority class weight |

### Balanced Batch Interleaving
Every training batch is exactly 50% positive (minority) + 50% negative (majority):
- **Minority stream**: heavy augmentation (`augment_image_heavy`) — rot90, ±20% brightness, 0.7–1.4× contrast, coarse dropout patch
- **Majority stream**: standard augmentation — flips, mild brightness/contrast, noise
- Guaranteed minority gradient signal regardless of how rare the class is

### Recall-Biased Threshold Selection
Uses **F2-score** (β=2, recall weighted 4× over precision), capped at 0.45:

$$F_2 = \frac{(1 + 2^2) \cdot \text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}}$$

In medical screening, false negatives (missed cancer) are clinically far worse than false positives (unnecessary follow-up). F2 directly encodes this priority.

### Test-Time Augmentation (TTA)
Final predictions are the average of the original image and its horizontal flip, reducing prediction variance.

---

## 📊 GDC Results (Test Set)

### Stage 1 — Suspicious Classifier
| Metric | Value |
|---|---|
| ROC-AUC | **0.607** |
| PR-AUC | **0.438** |
| Suspicious Recall | **0.937** (134/143 caught) |
| Suspicious Precision | 0.307 |
| Classification Threshold | 0.050 (recall-biased F2) |
| Fine-tuned layers | Top 60 of EfficientNetB0 |
| Focal loss | α=0.78, γ=2.0 |

**Confusion Matrix (Stage 1)**:
```
                  Predicted Non-susp   Predicted Suspicious
Actual Non-susp         35                  302
Actual Suspicious        9                  134
```
*9 missed suspicious cases out of 143 (6.3% miss rate)*

### Stage 2 — Risk Classifier
| Metric | Value |
|---|---|
| ROC-AUC | **0.750** |
| PR-AUC | **0.138** |
| High-risk Recall | **1.000** (19/19 caught — zero missed) ✅ |
| High-risk Precision | 0.064 |
| Classification Threshold | 0.450 (F2-biased, capped) |
| Fine-tuned layers | Top 30 of EfficientNetB0 |
| Focal loss | α=0.92, γ=2.5 |

**Confusion Matrix (Stage 2)**:
```
              Predicted Low-risk   Predicted High-risk
Actual Low         140                  277
Actual High          0                   19
```
*Zero high-risk cases missed — all 19 correctly flagged*

> **Why is precision low?** Both models use recall-biased thresholds by design. For cancer screening, a false positive (unnecessary follow-up) is vastly preferable to a false negative (missed cancer). Accuracy figures are misleading here — recall on the positive class is the clinical target.

---

## 🏗️ GDC Model Architecture

### Stage 1 — Suspicious Head
```
EfficientNetB0 (frozen/partially unfrozen)
  → GlobalAveragePooling2D
  → BatchNorm → Dropout(0.35)
  → Dense(256, relu, L2=1e-3)
  → BatchNorm → Dropout(0.30)
  → Dense(96, relu, L2=1e-3) → Dropout(0.20)
  → Dense(1, sigmoid, float32)
```

### Stage 2 — Risk Head
```
EfficientNetB0 (frozen/partially unfrozen)
  → GlobalAveragePooling2D
  → BatchNorm → Dropout(0.40)
  → Dense(256, relu, L2=1e-3)
  → BatchNorm → Dropout(0.30)
  → Dense(128, relu, L2=1e-3) → Dropout(0.25)
  → Dense(1, sigmoid, float32)
```

---

## 🚀 GDC Quick Start

### Environment Setup
```bash
conda create -n efficientnet_env python=3.10
conda activate efficientnet_env
pip install -r requirements_gdc_fresh.txt
```

### Prepare Data
```bash
# Downloads and processes GDC labels
conda run -n efficientnet_env python prepare_gdc_data.py
```

### Train Models
```bash
# Stage 1 — Suspicious classifier (~45 min on RTX 3050)
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious.py

# Stage 2 — Risk classifier (~40 min on RTX 3050)
conda run -n efficientnet_env --no-capture-output python train_gdc_risk.py
```

### Evaluate Only (no retraining)
```bash
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious.py --eval-only
conda run -n efficientnet_env --no-capture-output python train_gdc_risk.py --eval-only
```

### Run Inference on a Single Image
```bash
conda run -n efficientnet_env --no-capture-output python predict_gdc_pipeline.py --image path/to/image.jpg
```

### Generate Full Presentation Report
```bash
# Produces results_presentation/ with 8 charts + text summary
conda run -n efficientnet_env --no-capture-output python generate_results_report.py
```

---

## 📁 GDC Results Files

### `gdc_suspicious_results/`
| File | Description |
|---|---|
| `best_phase2_weights.h5` | Best checkpoint (Phase 2 fine-tuning) |
| `best_phase1_weights.h5` | Best Phase 1 checkpoint (fallback) |
| `suspicious_classifier_final.h5` | Final deployed weights |
| `results.json` | Full metrics, config, confusion matrix |
| `results.txt` | Human-readable evaluation report |
| `confusion_matrix.png` | Test set confusion matrix heatmap |
| `roc_curve.png` | ROC curve |
| `combined_training_history.png` | Loss/AUC across both training phases |

### `gdc_risk_results/`
Same structure as above (`risk_classifier_final.h5`, etc.)

### `results_presentation/`
| File | Description |
|---|---|
| `01_confusion_matrices.png` | Side-by-side CMs for both stages |
| `02_roc_curves.png` | Overlaid ROC curves (Stage 1 + Stage 2) |
| `03_pr_curves.png` | Overlaid Precision-Recall curves |
| `04_summary_table.png` | Metrics table (AUC, recall, precision, F1, threshold) |
| `05_pipeline_overview.png` | Mini CMs + bar chart of 6 key metrics |
| `06_probability_distributions.png` | Score histograms by class + threshold line |
| `07/08_*_training_history.png` | Combined Phase 1+2 training curves |
| `report_summary.txt` | Plain-text full results summary |

---

## 🛠️ GDC Training Configuration

### Stage 1 — Suspicious
| Parameter | Value |
|---|---|
| Backbone | EfficientNetB0 |
| Input size | 224×224 |
| Batch size | 8 |
| Phase 1 LR | 2e-4 (cosine decay, 2-epoch warmup) |
| Phase 2 LR | 2e-5 |
| Phase 1 epochs | 22 (early stopping, patience=10) |
| Phase 2 epochs | 20 |
| Fine-tune layers | Top 60 of EfficientNetB0 |
| Focal loss | α=0.78, γ=2.0 |
| Monitor metric | `val_auc` |
| Target minority ratio | 0.45 (after patient-level oversampling) |
| Class weight boost | 1.5× on suspicious class |

### Stage 2 — Risk
| Parameter | Value |
|---|---|
| Backbone | EfficientNetB0 |
| Batch size | 8 |
| Phase 1 LR | 2e-4 |
| Phase 2 LR | 2e-5 |
| Phase 1 epochs | 25 (early stopping, patience=8) |
| Phase 2 epochs | 22 |
| Fine-tune layers | Top 30 of EfficientNetB0 |
| Focal loss | α=0.92, γ=2.5 |
| Monitor metric | `val_auc` |
| Target minority ratio | 0.40 |
| Class weight boost | 1.5× on high-risk class |

---

## 🔬 Sri Lankan 3-Class Classifier — Key Features

### Advanced Training Techniques
- **Two-Phase Fine-Tuning**: Freeze-then-unfreeze strategy for optimal transfer learning
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples
- **Class Weights**: Computed from training distribution to balance minority class learning
- **Patient-Aware Splitting**: 70/15/15 train/val/test split at patient level to prevent data leakage
- **Heavy Data Augmentation**: GPU-accelerated random flips, rotations, brightness, contrast, zoom

### Performance Optimizations
- **tf.data Pipeline**: 40-60% faster than ImageDataGenerator
- **Mixed Precision Training**: Leverages Tensor Cores (float16/float32)
- **Prefetching & Caching**: Overlaps data loading with GPU computation
- **XLA JIT Compilation**: Reduces kernel launch overhead
- **Parallel Image Loading**: Multi-threaded decode/augment with AUTOTUNE

### Model Architecture
- **Base**: MobileNetV2 (ImageNet pretrained)
- **Input**: 224×224×3 RGB images
- **Head**: Global Average Pooling → Dense layers with Dropout & L2 regularization
- **Output**: 3-class softmax with float32 dtype for stability

## 📊 Sri Lankan Training Variants

Three optimized training scripts explore different class balancing strategies:

### 1. Unbalanced (Full Dataset)
**Script**: `original_dataset_unbalanced_optimized.py`

Uses the complete dataset with natural class distribution.
- **Purpose**: Baseline with maximum data
- **Advantage**: Highest overall accuracy
- **Trade-off**: Potential bias toward majority classes
- **Results**: `original_results_optimized/`

```bash
conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py
conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py --eval-only
```

### 2. Balanced 1:1:1
**Script**: `balanced_3class_optimized.py`

Equal samples per class (limited by minority class size).
- **Purpose**: Demonstrate fairness vs accuracy trade-off
- **Advantage**: Improved minority class (OCA) sensitivity
- **Results**: `balanced_results_optimized/`

```bash
conda run -n efficientnet_env python balanced_3class_optimized.py
```

### 3. Ratio 1:2:2
**Script**: `balanced_3class_ratio_1_2_2_optimized.py`

All OCA images + 2× OCA count for Healthy and Mutations.
- **Purpose**: Pragmatic compromise between fairness and performance
- **Results**: `ratio_1_2_2_results_optimized/`

```bash
conda run -n efficientnet_env python balanced_3class_ratio_1_2_2_optimized.py
```

### Sri Lankan Performance Comparison

| Variant | Dataset Size | Test Accuracy | OCA Recall | Strengths |
|---|---|---|---|---|
| **Unbalanced** | ~3000 images | ~0.72 | Lower | Maximum data, high overall accuracy |
| **Balanced 1:1:1** | ~400 images | ~0.41 | ~0.67 | Fair across classes, good minority sensitivity |
| **Ratio 1:2:2** | ~600 images | Mid-range | Mid-range | Balanced trade-off, practical compromise |

---

## 📁 Full Project Structure

```
cancer-detection-ml/
│
├── ── GDC Two-Stage Pipeline ──────────────────────────────────────────
├── train_gdc_suspicious.py          # Stage 1 training (Suspicious vs Non)
├── train_gdc_risk.py                # Stage 2 training (High-risk vs Low)
├── predict_gdc_pipeline.py          # Two-stage inference on a single image
├── generate_results_report.py       # Presentation charts + summary report
├── prepare_gdc_data.py              # GDC data preparation and label processing
├── evaluate_all_models.py           # Batch evaluation of all saved models
├── evaluate_external_dataset.py     # Evaluate on out-of-distribution data
├── requirements_gdc_fresh.txt       # Dependencies for GDC environment
│
├── gdc_suspicious_results/
│   ├── best_phase2_weights.h5       # Best checkpoint (Phase 2 fine-tuning)
│   ├── best_phase1_weights.h5       # Best Phase 1 checkpoint (fallback)
│   ├── suspicious_classifier_final.h5
│   ├── results.json / results.txt   # Metrics, config, confusion matrix
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── combined_training_history.png
│
├── gdc_risk_results/
│   ├── best_phase2_weights.h5
│   ├── best_phase1_weights.h5
│   ├── risk_classifier_final.h5
│   ├── results.json / results.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── combined_training_history.png
│
├── results_presentation/            # Presentation-ready charts (8 files)
│   ├── 01_confusion_matrices.png    # Side-by-side CMs for both stages
│   ├── 02_roc_curves.png            # Overlaid ROC curves
│   ├── 03_pr_curves.png             # Overlaid Precision-Recall curves
│   ├── 04_summary_table.png         # Metrics table
│   ├── 05_pipeline_overview.png     # Pipeline mini-CMs + bar chart
│   ├── 06_probability_distributions.png
│   ├── 07_suspicious_training_history.png
│   ├── 08_risk_training_history.png
│   └── report_summary.txt
│
├── data/
│   ├── gdc_oral_cancer_dataset/
│   │   ├── images/                  # GDC images (by case number)
│   │   ├── labels.csv
│   │   └── processed_labels.csv
│   └── Sri Lankan Dataset/
│       ├── Images/
│       ├── Imagewise_Data.csv
│       ├── Patientwise_Data.csv
│       └── Annotation.json
│
├── ── Sri Lankan 3-Class Classifier ──────────────────────────────────
├── original_dataset_unbalanced_optimized.py
├── balanced_3class_optimized.py
├── balanced_3class_ratio_1_2_2_optimized.py
├── original_results_optimized/
├── balanced_results_optimized/
├── ratio_1_2_2_results_optimized/
│
├── ── Experimental / Ensemble ─────────────────────────────────────────
├── train_ensemble_risk.py           # Ensemble of multiple risk models
├── train_ensemble_suspicious.py
├── train_perfected_risk.py          # Perfected single model variants
├── train_perfected_suspicious.py
├── train_cross_validation.py        # K-fold cross validation
├── test_time_augmentation.py        # TTA utilities
│
├── dataloaders/                     # Shared data loading utilities
├── model/                           # Shared model architecture modules
├── statistics/                      # Dataset statistics scripts
├── config.yaml
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

📖 **For detailed setup, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### GDC Pipeline (Recommended)

```bash
git clone https://github.com/ManasDes03/OralCancerDetection.git
cd OralCancerDetection

conda create -n efficientnet_env python=3.10
conda activate efficientnet_env
pip install -r requirements_gdc_fresh.txt

# Train both stages
conda run -n efficientnet_env --no-capture-output python train_gdc_suspicious.py
conda run -n efficientnet_env --no-capture-output python train_gdc_risk.py

# Run inference
conda run -n efficientnet_env --no-capture-output python predict_gdc_pipeline.py --image your_image.jpg

# Generate presentation report
conda run -n efficientnet_env --no-capture-output python generate_results_report.py
```

### Sri Lankan Classifier

```bash
conda create -n efficientnet_env python=3.10
conda activate efficientnet_env
pip install -r requirements.txt

conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py
```

---

## 🎓 Shared Methodology

### Patient-Aware Splitting (Both Pipelines)
All splits are done at **patient level** — never at image level — to prevent data leakage. No patient appears in more than one split. Split is 70/15/15 (train/val/test), stratified by label.

### Why `model.save_weights()` not `model.save()`
TensorFlow 2.10 with `mixed_float16` precision causes a `TypeError: Unable to serialize EagerTensor to JSON` crash when calling `model.save()`. All checkpoints use `save_weights_only=True` and weights are loaded by rebuilding the identical architecture first, then calling `model.load_weights()`.

### Weight Loading Rule
Weights must be loaded with the **same backbone trainability state** as when they were saved:
- Phase 1 weights: load with `base_model.trainable = False`
- Phase 2 weights: load with top-N layers unfrozen (matching the fine-tune config)

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| GPU OOM | Reduce `batch_size` in CONFIG (8 → 4) |
| `axes don't match array` on `load_weights` | Backbone trainability state doesn't match. See weight loading rule above |
| `No model config found` | Use `model.load_weights()`, not `keras.models.load_model()` |
| cuDNN frontend warnings | Normal — identity activation convolutions, does not affect results |
| val_auc very noisy | Expected with <20 validation positives (Stage 2). Monitor `val_auc` not `val_pr_auc` |
| Slow training | Ensure `Created device GPU:0` appears in logs. Use `--no-capture-output` |

---

## 📚 Dependencies

- **GDC pipeline**: `requirements_gdc_fresh.txt` — TF 2.10, numpy, pandas, scikit-learn, matplotlib, seaborn
- **Sri Lankan classifier**: `requirements.txt`

---

## 🤝 Contributing

### For Team Members
```bash
git clone https://github.com/ManasDes03/OralCancerDetection.git
cd OralCancerDetection
git checkout -b feature/your-feature-name
# ... make changes ...
git push origin feature/your-feature-name
# Open Pull Request on GitHub
```

### For External Contributors
Fork the repo, create a branch, and open a Pull Request.

**Contact**: f20220728@goa.bits-pilani.ac.in

---

## 📄 License

All Rights Reserved

## 🙏 Acknowledgments

- GDC Oral Cancer Dataset
- Sri Lankan Oral Cancer Dataset
- EfficientNetB0 / MobileNetV2 architectures (Google)
- TensorFlow/Keras framework
- Research conducted at BITS Pilani, K K Birla Goa Campus

## 📊 Citation

```bibtex
@misc{oralcancer2026,
  author = {ManasDes03},
  title  = {Oral Cancer Detection — GDC Two-Stage EfficientNetB0 Pipeline},
  year   = {2026},
  url    = {https://github.com/ManasDes03/OralCancerDetection}
}
```

---

**Last Updated**: March 2026 | **Branch**: `gdc-two-stage-classifier` | **Maintainer**: ManasDes03
