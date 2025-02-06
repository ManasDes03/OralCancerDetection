# cancer-detection-ml

# 🦷 Oral Cancer Detection Using TensorFlow

This project aims to detect oral cancer from medical images using deep learning models built with TensorFlow. It supports multiple datasets, including **Oral Cancer**, **Oral Cancer 2.0**, and the **Sri Lankan Dataset**, providing flexibility for experimentation and real-world applications.

---

## 📁 Project Structure
- **`data/`** – Stores datasets (Oral Cancer, Oral, Sri Lankan). Update `dataloader.py` to add new datasets.  
- **`dataloader/`** – Contains `dataloader.py` for loading datasets.  
- **`experiments/`** – Stores timestamp-based experiment results (config, loss plots, metrics).  
- **`legacy_code/`** – Contains older code (e.g., Anshul’s work).  
- **`models/`** – Includes `model.py` defining Inception, MobileNet, ResNet, EfficientNet models.  
- **`config.py`** – Configuration file for hyperparameters.  
- **`main.py`** – Main script to run models (`python main.py`).  
- **`requirements.txt`** – Lists dependencies (`pip install -r requirements.txt`).  

⚠ **Note:** Do not share private datasets.

## ⚙️ Configuration

All project settings are managed via `config.yaml`:

```yaml
# General Configuration
dataset:
  name: "Oral_Cancer"    # Options: "Oral_Cancer", "Oral_Cancer_2.0", "Sri_Lankan"
  image_size: [224, 224] # Specify image size (height, width)
  batch_size: 32
  num_classes: 2         # Assuming binary classification: cancer vs non-cancer
  
model:
  name: "EfficientNetB0"      # Options: "EfficientNetB0", "MobileNetV2", "ResNet50", "InceptionV3"
  weights: "imagenet"         # Use "imagenet" for pre-trained weights, or "None" for random initialization
  trainable: False            # Set to True for fine-tuning the base model

# Dataset Paths
paths:
  Oral_Cancer:
    images_dir: "data/OralCancer"

  Oral_Cancer_2.0:
    images_dir: "data/Oral_Cancer_Dataset_2.0"

  Sri_Lankan:
    images_dir: "data/Sri_Lankan_Dataset/Images"
    annotations_file: "data/Sri_Lankan_Dataset/Imagewise_Data.csv"

# Training Configuration
training:
  learning_rate: 0.001
  epochs: 20
  optimizer: "adam"       # Options: "adam", "sgd", "rmsprop"
  loss_function: "categorical_crossentropy"
  metrics: ["accuracy"]

```

---

## 🗂️ Dataset Formats

### 1️⃣ **Oral Cancer & Oral Cancer 2.0**
```
data/
└── Oral_Cancer/
    |
    ├── cancer/
    |   ├── image1.jpg
    │   └── image2.jpg
    └── non-cancer/
        ├── image1.jpg
        └── image2.jpg
```

### 2️⃣ **Sri Lankan Dataset**
```
data/
└── Sri_Lankan_Dataset/
    ├── Images
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Annotation.json
    ├── Patientwise_Data.csv
    └── Imagewise_Data.csv
    
```

**CSV Format Example:**
```csv
image_name,label
image1.jpg,0
image2.jpg,1
```
- `label`: `0` = non-cancer, `1` = cancer.

---

## 🖥️ Installation & Setup

### 1️⃣ Install Anaconda (Recommended)
- Download Anaconda: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

### 2️⃣ Create a Virtual Environment
```bash
conda create -n oral_cancer_detection python=3.10
conda init
source ~/.bashrc
conda activate oral_cancer_detection
```

### 3️⃣ Install Dependencies
```bash
pip install tensorflow pyyaml pandas
```

Or if using `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Project

Simply run the main script:
```bash
python main.py
```

---

## 📊 Model Training

- The model is a simple **Convolutional Neural Network (CNN)** optimized for binary classification.
- You can tweak the architecture in `models/models.py` and adjust hyperparameters via `config.yaml`.

---

## ✅ Evaluation Metrics

- **Accuracy** (default metric)
- Optionally, add **Precision**, **Recall**, and **F1-score** for more detailed performance analysis.

---

## 📌 Future Improvements
- Add data augmentation techniques.
- Experiment with transfer learning using pre-trained models (e.g., ResNet, VGG).
- Implement advanced evaluation metrics and visualization tools (e.g., confusion matrix).

---

## 🔗 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Anaconda Documentation](https://docs.anaconda.com/)
- Relevant medical image datasets and publications.

