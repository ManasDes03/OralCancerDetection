# Oral Cancer Detection using Deep Learning

A comprehensive deep learning system for automated 3-class oral cancer classification using MobileNetV2 with optimized training pipelines and patient-aware evaluation.

## 🎯 Project Overview

This project implements an advanced oral cancer detection system that classifies oral cavity images into three categories:
- **OCA (Oral Cancer)** - Malignant lesions
- **Healthy** - Normal tissue
- **Mutations (OPMD + Benign)** - Pre-malignant and benign lesions

The system uses transfer learning with MobileNetV2 as the backbone, enhanced with focal loss, class weighting, heavy data augmentation, and GPU-accelerated training pipelines.

## 🔬 Key Features

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

## 📊 Training Variants

Three optimized training scripts explore different class balancing strategies:

### 1. Unbalanced (Full Dataset)
**Script**: `original_dataset_unbalanced_optimized.py`

Uses the complete dataset with natural class distribution.
- **Purpose**: Baseline with maximum data
- **Advantage**: Highest overall accuracy
- **Trade-off**: Potential bias toward majority classes
- **Results**: `original_results_optimized/`

**Usage**:
```bash
# Train from scratch
conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py

# Evaluate only (load checkpoint, regenerate metrics/plots)
conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py --eval-only
```

### 2. Balanced 1:1:1
**Script**: `balanced_3class_optimized.py`

Equal samples per class (limited by minority class size).
- **Purpose**: Demonstrate fairness vs accuracy trade-off
- **Advantage**: Improved minority class (OCA) sensitivity
- **Trade-off**: Reduced overall accuracy due to smaller dataset
- **Results**: `balanced_results_optimized/`

**Usage**:
```bash
conda run -n efficientnet_env python balanced_3class_optimized.py
```

### 3. Ratio 1:2:2
**Script**: `balanced_3class_ratio_1_2_2_optimized.py`

All OCA images + 2× OCA count for Healthy and Mutations.
- **Purpose**: Pragmatic compromise between fairness and performance
- **Advantage**: Better minority recall than unbalanced, better accuracy than 1:1:1
- **Results**: `ratio_1_2_2_results_optimized/`

**Usage**:
```bash
conda run -n efficientnet_env python balanced_3class_ratio_1_2_2_optimized.py
```

## 📁 Project Structure

```
cancer-detection-ml/
├── data/                                    # Dataset (images + CSV)
│   └── Sri Lankan Dataset/
│       ├── Images/                          # Oral cavity images
│       ├── Imagewise_Data.csv              # Image-level labels
│       ├── Patientwise_Data.csv            # Patient metadata
│       └── Annotation.json                 # Annotations
│
├── dataloaders/                            # Data loading utilities
├── model/                                  # Model architecture modules
├── statistics/                             # Dataset statistics scripts
│
├── original_dataset_unbalanced_optimized.py    # Full dataset training
├── balanced_3class_optimized.py                # 1:1:1 balanced training
├── balanced_3class_ratio_1_2_2_optimized.py    # 1:2:2 ratio training
│
├── original_results_optimized/             # Unbalanced results
│   ├── results.txt                         # Metrics & confusion matrix
│   ├── training_history.png               # Loss/accuracy curves
│   ├── confusion_matrix.png               # Heatmap visualization
│   ├── per_class_accuracy.png             # Bar chart
│   └── mobilenet_original_best_*.h5       # Best model checkpoint
│
├── balanced_results_optimized/             # 1:1:1 results
├── ratio_1_2_2_results_optimized/          # 1:2:2 results
│
├── config.yaml                             # Configuration
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🚀 Quick Start

📖 **For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### Basic Installation

```bash
# Clone repository
git clone https://github.com/ManasDes03/OralCancerDetection.git
cd OralCancerDetection

# Create environment and install dependencies
conda create -n efficientnet_env python=3.10
conda activate efficientnet_env
pip install -r requirements.txt

# Run training
conda run -n efficientnet_env python original_dataset_unbalanced_optimized.py
```

For complete setup including GPU configuration, troubleshooting, and verification steps, refer to **[SETUP_GUIDE.md](SETUP_GUIDE.md)**

## 🔧 Training Configuration

### Hyperparameters (Default)

| Parameter | Phase 1 (Frozen) | Phase 2 (Fine-tune) |
|-----------|-----------------|---------------------|
| Learning Rate | 1e-3 | 1e-4 |
| Batch Size | 32 | 32 |
| Max Epochs | 15-20 | 15-20 |
| Optimizer | Adam | Adam |
| Loss | Focal (α=0.25, γ=2.0) | Focal (α=0.25, γ=2.0) |
| Early Stopping | Patience 7-8 | Patience 8 |
| LR Reduction | Factor 0.5, Patience 3-4 | Factor 0.5, Patience 3-4 |

### Data Augmentation (Training Only)
- Random horizontal & vertical flips
- Random brightness (±20%)
- Random contrast (0.8-1.2)
- Random saturation (0.8-1.2)
- Random hue shift (±10%)
- Random rotation (0°, 90°, 180°, 270°)
- Random zoom (0.7-1.0 crop then resize)

## 📈 Results & Evaluation

Each training run generates:

### Metrics File (`results.txt`)
- Train/Val/Test split sizes
- Class weights
- Test accuracy & loss
- Per-class precision, recall, F1-score
- Confusion matrix (numerical)

### Visualizations
1. **training_history.png**: Train/val accuracy and loss curves with phase boundary
2. **confusion_matrix.png**: Labeled heatmap showing prediction patterns
3. **per_class_accuracy.png**: Bar chart of per-class accuracies

### Model Checkpoints
- **Phase 1**: `mobilenet_*_initial.h5`
- **Phase 2**: `mobilenet_*_final.h5` (best model based on validation)

## 🎓 Methodology Highlights

### Patient-Aware Splitting
Prevents data leakage by splitting at patient level:
- Patient ID derived from image name (e.g., `R-01-012.jpg` → Patient `R-01`)
- Ensures no patient appears in multiple splits
- 70% train, 15% validation, 15% test

### Label Mapping
- OCA → Class 0 (Cancer)
- Healthy → Class 1
- OPMD & Benign → Class 2 (Mutations, combined)

### Class Imbalance Handling
1. **Focal Loss**: Down-weights easy examples, focuses on hard cases
2. **Class Weights**: Inversely proportional to class frequency
3. **Balanced Sampling** (variants 2 & 3): Equalize or adjust class ratios

### Reproducibility
- Fixed seeds: `np.random.seed(42)`, `tf.random.set_seed(42)`
- Patient-aware splitting ensures consistent evaluation
- Note: Some GPU/tf.data randomness may cause minor variations

## 🔬 Performance Comparison

| Variant | Dataset Size | Test Accuracy | OCA Recall | Strengths |
|---------|-------------|---------------|------------|-----------|
| **Unbalanced** | ~3000 images | ~0.72 | Lower | Maximum data, high overall accuracy |
| **Balanced 1:1:1** | ~400 images | ~0.41 | ~0.67 | Fair across classes, good minority sensitivity |
| **Ratio 1:2:2** | ~600 images | Mid-range | Mid-range | Balanced trade-off, practical compromise |

*Note: Exact numbers depend on random split and training convergence*

## 🛠️ Troubleshooting

### GPU Out of Memory
- Reduce batch size in CONFIG (e.g., 32 → 24 or 16)
- Close other GPU-intensive applications

### Missing Images Warning
- Some filenames in CSV may not match files in `Images/`
- Scripts automatically filter out missing files
- Check CSV for incorrect filenames or add missing .jpg extensions

### Mixed Precision Warnings
- Normal cuDNN frontend warnings during phase 2 fine-tuning
- Does not affect training; indicates automatic optimization

### Slow Training
- Ensure GPU is detected: TensorFlow should log "Created device GPU:0"
- Verify CUDA/cuDNN versions match TensorFlow requirements
- Use `--no-capture-output` with conda run to see detailed logs

## 📚 Dependencies

Key packages (see `requirements.txt` for full list):
- TensorFlow 2.10+ (with GPU support)
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue on GitHub
- Contact: f20220728@goa.bits-pilani.ac.in

## 📄 License

All Rights Reserved

## 🙏 Acknowledgments

- Sri Lankan Oral Cancer Dataset
- MobileNetV2 architecture (Google)
- TensorFlow/Keras framework
- Research conducted at BITS Pilani, K K Birla Goa Campus

## 📊 Citation

If you use this code in your research, please cite:
```
@misc{oralcancer2025,
  author = {ManasDes03},
  title = {Oral Cancer Detection using Deep Learning with Optimized Training Pipelines},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ManasDes03/OralCancerDetection}
}
```

---

**Last Updated**: November 2025  
**Maintainer**: ManasDes03  
**Repository**: [https://github.com/ManasDes03/OralCancerDetection](https://github.com/ManasDes03/OralCancerDetection)
