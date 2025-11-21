# Complete Setup Guide for Oral Cancer Detection Project

This guide provides step-by-step instructions to replicate this project on your device, from system requirements to running your first training.

---

## 📋 Table of Contents
1. [System Requirements](#system-requirements)
2. [Software Installation](#software-installation)
3. [Environment Setup](#environment-setup)
4. [Dataset Configuration](#dataset-configuration)
5. [GPU Configuration](#gpu-configuration)
6. [Verification & Testing](#verification--testing)
7. [Running Training](#running-training)
8. [Common Issues & Solutions](#common-issues--solutions)

---

## 💻 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, Ubuntu 18.04+, or macOS 10.14+
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 10 GB free space
- **Python**: 3.8 - 3.10 (Python 3.10 recommended)

### Recommended Requirements (for GPU training)
- **GPU**: NVIDIA GPU with CUDA Compute Capability ≥ 3.5
  - RTX 30/40 series (recommended)
  - GTX 16 series
  - Minimum 4 GB VRAM (8 GB+ recommended)
- **CUDA**: 11.2 or higher
- **cuDNN**: 8.1 or higher

### CPU-Only Training
- Possible but **~10-20x slower**
- Requires patience (hours instead of minutes per epoch)
- Not recommended for production use

---

## 🔧 Software Installation

### Step 1: Install Python

#### Option A: Anaconda (Recommended)
**Why Anaconda?** Simplifies package management, handles CUDA/cuDNN automatically

1. Download Anaconda/Miniconda:
   - **Anaconda**: [https://www.anaconda.com/download](https://www.anaconda.com/download)
   - **Miniconda** (lightweight): [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

2. Install for your OS:
   - **Windows**: Run `.exe` installer, check "Add to PATH"
   - **Linux/macOS**: Run `bash Miniconda3-latest-*.sh` in terminal

3. Verify installation:
   ```bash
   conda --version
   # Should output: conda 23.x.x or similar
   ```

#### Option B: Standalone Python
1. Download Python 3.10 from [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. During installation:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"
3. Verify:
   ```bash
   python --version  # Should show Python 3.10.x
   pip --version
   ```

### Step 2: Install Git (for cloning repository)

1. Download Git: [https://git-scm.com/downloads](https://git-scm.com/downloads)
2. Install with default settings
3. Verify:
   ```bash
   git --version
   ```

### Step 3: Install CUDA & cuDNN (GPU Users Only)

#### Automatic Installation (via Conda - Easiest)
Conda will automatically install CUDA/cuDNN when you install TensorFlow-GPU. **Skip manual installation if using conda.**

#### Manual Installation (if not using Conda)

**For NVIDIA GPUs:**

1. **Check GPU compatibility**:
   - Windows: Open "Device Manager" → "Display adapters"
   - Linux: Run `lspci | grep -i nvidia`
   - Verify your GPU model supports CUDA: [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)

2. **Install NVIDIA Driver**:
   - Download latest driver: [https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)
   - Restart after installation

3. **Install CUDA Toolkit 11.8**:
   - Download: [https://developer.nvidia.com/cuda-11-8-0-download-archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)
   - Follow installer prompts (use Express Installation)
   - Verify:
     ```bash
     nvcc --version
     ```

4. **Install cuDNN 8.6**:
   - Register/login at [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
   - Download cuDNN 8.6 for CUDA 11.x
   - Extract and copy files:
     - Windows: Copy `bin`, `include`, `lib` folders to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\`
     - Linux: Copy to `/usr/local/cuda-11.8/`

---

## 🐍 Environment Setup

### Step 1: Clone the Repository

```bash
# Navigate to your desired directory
cd C:\Users\YourName\Projects  # Windows
# or
cd ~/Projects  # Linux/macOS

# Clone the repository
git clone https://github.com/ManasDes03/OralCancerDetection.git
cd OralCancerDetection
```

### Step 2: Create Virtual Environment

#### Using Conda (Recommended)

```bash
# Create environment with Python 3.10
conda create -n efficientnet_env python=3.10 -y

# Activate environment
conda activate efficientnet_env

# Verify Python version
python --version  # Should show Python 3.10.x
```

#### Using venv (Alternative)

```bash
# Create virtual environment
python -m venv cancer_env

# Activate environment
# Windows:
cancer_env\Scripts\activate
# Linux/macOS:
source cancer_env/bin/activate

# Verify activation (prompt should change)
```

### Step 3: Install Python Dependencies

```bash
# Ensure environment is activated (you should see env name in prompt)

# Upgrade pip first
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Expected installation time**: 5-15 minutes depending on internet speed

#### Verify Critical Packages

```bash
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
# Expected: TensorFlow Version: 2.10.x or 2.11.x

python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
# Expected (GPU): GPU Available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# Expected (CPU): GPU Available: []
```

### Step 4: Install Additional System Dependencies (Linux Only)

For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libsm6 libxext6 libxrender-dev libgomp1
```

For CentOS/RHEL:
```bash
sudo yum install -y libSM libXext libXrender
```

---

## 📂 Dataset Configuration

### Step 1: Prepare Dataset Structure

Your dataset should follow this structure:

```
OralCancerDetection/
└── data/
    └── Sri Lankan Dataset/
        ├── Images/
        │   ├── R-01-001.jpg
        │   ├── R-01-002.jpg
        │   ├── ...
        ├── Imagewise_Data.csv
        ├── Patientwise_Data.csv (optional)
        └── Annotation.json (optional)
```

### Step 2: CSV Format Requirements

Your `Imagewise_Data.csv` must contain these columns:

| Image Name | Category |
|-----------|----------|
| R-01-001.jpg | OCA |
| R-01-002.jpg | Healthy |
| R-02-001.jpg | OPMD |
| R-02-002.jpg | Benign |

**Requirements**:
- `Image Name`: Exact filename including extension (`.jpg`, `.jpeg`, `.png`)
- `Category`: One of {`OCA`, `Healthy`, `OPMD`, `Benign`}
- Patient ID will be auto-extracted from filename pattern (e.g., `R-01-001` → Patient `R-01`)

### Step 3: Image Requirements

- **Format**: JPEG or PNG
- **Size**: Any size (will be resized to 224×224 automatically)
- **Color**: RGB (grayscale will be converted automatically)
- **Naming**: Consistent pattern with patient ID prefix
  - ✅ Good: `R-01-001.jpg`, `P-123-005.png`, `A_05_002.jpg`
  - ❌ Bad: Random UUIDs without patient grouping

### Step 4: Verify Dataset Integrity

Create this quick verification script (`verify_dataset.py`):

```python
import os
import pandas as pd

# Paths
data_dir = "data/Sri Lankan Dataset"
csv_path = os.path.join(data_dir, "Imagewise_Data.csv")
images_dir = os.path.join(data_dir, "Images")

# Load CSV
df = pd.read_csv(csv_path)
print(f"✓ CSV loaded: {len(df)} entries")

# Check columns
assert "Image Name" in df.columns, "Missing 'Image Name' column"
assert "Category" in df.columns, "Missing 'Category' column"
print("✓ Required columns present")

# Check categories
valid_categories = {"OCA", "Healthy", "OPMD", "Benign"}
categories = set(df["Category"].unique())
assert categories.issubset(valid_categories), f"Invalid categories found: {categories - valid_categories}"
print(f"✓ Categories valid: {categories}")

# Check image files exist
missing = []
for img_name in df["Image Name"]:
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        missing.append(img_name)

if missing:
    print(f"⚠ Warning: {len(missing)} images not found (will be auto-filtered)")
    print(f"  First 5: {missing[:5]}")
else:
    print(f"✓ All {len(df)} images found")

print("\n✅ Dataset verification complete!")
```

Run verification:
```bash
conda activate efficientnet_env
python verify_dataset.py
```

---

## 🎮 GPU Configuration

### Step 1: Verify GPU Detection

```bash
# Activate environment
conda activate efficientnet_env

# Check TensorFlow GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Expected outputs**:
- ✅ **GPU detected**: `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`
- ❌ **No GPU**: `[]` (CPU-only mode, much slower)

### Step 2: Test GPU Training

Create quick GPU test (`test_gpu.py`):

```python
import tensorflow as tf
import time

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Enable mixed precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax', dtype='float32')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Dummy data
x = tf.random.normal((1000, 784))
y = tf.random.uniform((1000,), maxval=10, dtype=tf.int32)

# Time training
print("\nTraining for 5 steps...")
start = time.time()
model.fit(x, y, batch_size=32, epochs=5, verbose=0)
duration = time.time() - start

print(f"✓ Training completed in {duration:.2f}s")
if duration < 5:
    print("✅ GPU is working efficiently!")
else:
    print("⚠ Training seems slow, GPU may not be utilized")
```

Run test:
```bash
python test_gpu.py
```

### Step 3: Configure GPU Memory Growth (Optional)

If you encounter GPU memory errors, add this to the start of training scripts:

```python
import tensorflow as tf

# Allow dynamic memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

---

## ✅ Verification & Testing

### Complete System Check

Create `system_check.py`:

```python
import sys
import os

print("=" * 60)
print("ORAL CANCER DETECTION - SYSTEM CHECK")
print("=" * 60)

# Python version
print(f"\n1. Python Version: {sys.version.split()[0]}")
assert sys.version_info >= (3, 8), "❌ Python 3.8+ required"
print("   ✓ Version OK")

# Import critical packages
print("\n2. Package Imports:")
try:
    import tensorflow as tf
    print(f"   ✓ TensorFlow {tf.__version__}")
except ImportError:
    print("   ❌ TensorFlow not installed")
    sys.exit(1)

try:
    import numpy as np
    print(f"   ✓ NumPy {np.__version__}")
except ImportError:
    print("   ❌ NumPy not installed")

try:
    import pandas as pd
    print(f"   ✓ Pandas {pd.__version__}")
except ImportError:
    print("   ❌ Pandas not installed")

try:
    import sklearn
    print(f"   ✓ scikit-learn {sklearn.__version__}")
except ImportError:
    print("   ❌ scikit-learn not installed")

try:
    import matplotlib
    print(f"   ✓ Matplotlib {matplotlib.__version__}")
except ImportError:
    print("   ❌ Matplotlib not installed")

# GPU check
print("\n3. GPU Configuration:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"   ✓ GPU detected: {len(gpus)} device(s)")
    for i, gpu in enumerate(gpus):
        print(f"     - {gpu.name}")
else:
    print("   ⚠ No GPU detected (CPU-only mode)")

# Dataset check
print("\n4. Dataset Check:")
data_dir = "data/Sri Lankan Dataset"
csv_path = os.path.join(data_dir, "Imagewise_Data.csv")
images_dir = os.path.join(data_dir, "Images")

if os.path.exists(csv_path):
    print(f"   ✓ CSV found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"     - Total entries: {len(df)}")
else:
    print(f"   ❌ CSV not found: {csv_path}")

if os.path.exists(images_dir):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"   ✓ Images folder found: {len(image_files)} images")
else:
    print(f"   ❌ Images folder not found: {images_dir}")

print("\n" + "=" * 60)
print("✅ SYSTEM CHECK COMPLETE")
print("=" * 60)
```

Run check:
```bash
conda activate efficientnet_env
python system_check.py
```

---

## 🚀 Running Training

### First Training Run (Recommended: Unbalanced Dataset)

```bash
# Ensure environment is activated
conda activate efficientnet_env

# Run training
python original_dataset_unbalanced_optimized.py
```

**What to expect**:
- Initial output shows dataset loading and split sizes
- Phase 1 training (frozen backbone): ~5-10 minutes
- Phase 2 fine-tuning: ~5-10 minutes
- Total time: **15-30 minutes** (with GPU), **2-4 hours** (CPU only)
- Final accuracy around 70-75%

### Monitor Training Progress

Training will display:
```
Epoch 1/15
56/56 [==============================] - 45s 800ms/step - loss: 0.8234 - accuracy: 0.6123 - val_loss: 0.7145 - val_accuracy: 0.6543
```

**Key metrics to watch**:
- `accuracy` increasing (train accuracy)
- `val_accuracy` increasing (validation accuracy)
- `loss` decreasing
- If `val_accuracy` stops improving → early stopping will trigger

### After Training Completes

Check results in `original_results_optimized/`:
```
original_results_optimized/
├── results.txt              # Detailed metrics & confusion matrix
├── training_history.png     # Loss/accuracy curves
├── confusion_matrix.png     # Prediction heatmap
├── per_class_accuracy.png   # Per-class performance
└── mobilenet_original_best_*.h5  # Trained model
```

### Running Other Variants

**Balanced 1:1:1**:
```bash
python balanced_3class_optimized.py
```
- Smaller dataset (~400 images)
- Faster training (~10-15 minutes)
- Lower overall accuracy but better OCA recall

**Ratio 1:2:2**:
```bash
python balanced_3class_ratio_1_2_2_optimized.py
```
- Medium dataset (~600 images)
- Balanced trade-off between variants

### Evaluation Only (Skip Training)

If you already have a trained model:
```bash
python original_dataset_unbalanced_optimized.py --eval-only
```
- Loads best checkpoint
- Re-generates metrics and plots
- Useful for regenerating visualizations

---

## 🔧 Common Issues & Solutions

### Issue 1: "No module named 'tensorflow'"

**Cause**: TensorFlow not installed or wrong environment

**Solution**:
```bash
# Check if environment is activated (should see env name in prompt)
conda activate efficientnet_env

# Reinstall TensorFlow
pip install tensorflow==2.10.1
```

### Issue 2: "Could not load dynamic library 'cudart64_110.dll'"

**Cause**: CUDA version mismatch or not installed

**Solution (Conda)**:
```bash
# Let conda handle CUDA
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

**Solution (Manual)**:
- Download CUDA 11.x from NVIDIA website
- Ensure CUDA is in system PATH

### Issue 3: "ResourceExhaustedError: OOM when allocating tensor"

**Cause**: GPU out of memory

**Solutions**:
1. Reduce batch size (edit script):
   ```python
   BATCH_SIZE = 16  # Change from 32
   ```

2. Enable memory growth (add to script start):
   ```python
   gpus = tf.config.list_physical_devices('GPU')
   if gpus:
       tf.config.experimental.set_memory_growth(gpus[0], True)
   ```

3. Close other GPU applications (browsers, games)

### Issue 4: "FileNotFoundError: [Errno 2] No such file or directory: 'data/...'"

**Cause**: Running script from wrong directory or dataset not placed correctly

**Solution**:
```bash
# Ensure you're in project root
cd OralCancerDetection

# Verify dataset structure
ls data/Sri\ Lankan\ Dataset/
# Should show: Images/, Imagewise_Data.csv
```

### Issue 5: Very Slow Training (CPU mode when GPU expected)

**Diagnosis**:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**If returns `[]`**:
- GPU not detected
- Check NVIDIA driver installation
- Reinstall CUDA/cuDNN
- Verify GPU compatibility

### Issue 6: "ValueError: columns overlap but no suffix specified"

**Cause**: CSV has duplicate column names

**Solution**: Open CSV and ensure no duplicate column names

### Issue 7: "MemoryError" during data loading

**Cause**: Insufficient RAM

**Solution**:
- Close other applications
- Reduce dataset size temporarily
- Use balanced variants (smaller datasets)

### Issue 8: SSL Certificate Error during pip install

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

---

## 📊 Performance Expectations

### Training Times (approximate)

| Hardware | Unbalanced | Balanced 1:1:1 | Ratio 1:2:2 |
|----------|-----------|---------------|-------------|
| RTX 3090 | 10-15 min | 5-8 min | 8-12 min |
| RTX 3060 | 15-25 min | 8-12 min | 12-18 min |
| GTX 1660 | 25-40 min | 12-20 min | 18-30 min |
| CPU (16 cores) | 2-4 hours | 1-2 hours | 1.5-3 hours |

### Expected Accuracy Ranges

| Variant | Test Accuracy | OCA Recall |
|---------|--------------|------------|
| Unbalanced | 0.65 - 0.75 | 0.30 - 0.50 |
| Balanced 1:1:1 | 0.35 - 0.50 | 0.60 - 0.75 |
| Ratio 1:2:2 | 0.50 - 0.65 | 0.45 - 0.65 |

*Note: Varies based on dataset quality and random split*

---

## 📞 Getting Help

If you encounter issues not covered here:

1. **Check error message carefully** - often contains solution
2. **Search GitHub Issues**: [https://github.com/ManasDes03/OralCancerDetection/issues](https://github.com/ManasDes03/OralCancerDetection/issues)
3. **Create new issue** with:
   - Full error message
   - Output of `system_check.py`
   - OS and Python version
   - Steps to reproduce

---

## ✨ Quick Start Checklist

- [ ] Python 3.8-3.10 installed
- [ ] Conda/virtualenv created and activated
- [ ] Repository cloned
- [ ] `requirements.txt` installed
- [ ] CUDA/cuDNN installed (GPU users)
- [ ] Dataset placed in `data/Sri Lankan Dataset/`
- [ ] CSV format verified
- [ ] `system_check.py` passes all checks
- [ ] First training run completed successfully

---

## 🎉 Success!

If your first training completes without errors:

✅ **Your setup is complete!**

You can now:
- Train on all 3 variants
- Experiment with hyperparameters
- Use trained models for inference
- Contribute to the project

---

**Setup Guide Version**: 1.0  
**Last Updated**: November 2025  
**Maintainer**: ManasDes03
