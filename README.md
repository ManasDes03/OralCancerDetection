# cancer-detection-ml

# 🦷 Oral Cancer Detection Using TensorFlow

This project aims to detect oral cancer from medical images using deep learning models built with TensorFlow. It supports multiple datasets, including **Oral Cancer**, **Oral Cancer 2.0**, and the **Sri Lankan Dataset**, providing flexibility for experimentation and real-world applications.

---

## 🚀 Features
- **Multi-Dataset Support:** Handles different dataset structures seamlessly.
- **Configurable Pipeline:** Adjust dataset, image size, batch size, and more through a simple `config.yaml` file.
- **Modular Architecture:** Clean separation between data loading, model architecture, and training scripts.
- **Binary Classification:** Classifies images into `cancer` or `non-cancer` categories.

---

## 📁 Project Structure

```
oral-cancer-detection/
├── main.py                 # Entry point to train the model
├── config.yaml             # Configuration file for easy adjustments
├── dataloaders/
│   ├── __init__.py
│   └── dataloaders.py      # Handles data loading for all datasets
├── models/
│   ├── __init__.py
│   └── models.py           # Contains the model architecture
└── README.md               # Project documentation
```

---

## ⚙️ Configuration

All project settings are managed via `config.yaml`:

```yaml
dataset:
  name: "Oral_Cancer"        # Options: "Oral_Cancer", "Oral_Cancer_2.0", "Sri_Lankan"
  image_size: [224, 224]     # Image dimensions (height, width)
  batch_size: 32
  num_classes: 2             # Binary classification: cancer vs non-cancer

paths:
  Oral_Cancer:
    train_dir: "data/Oral_Cancer/train"
    val_dir: "data/Oral_Cancer/val"

  Oral_Cancer_2.0:
    train_dir: "data/Oral_Cancer_2.0/train"
    val_dir: "data/Oral_Cancer_2.0/val"

  Sri_Lankan:
    images_dir: "data/Sri_Lankan_Dataset/images"
    annotations_file: "data/Sri_Lankan_Dataset/labels.csv"

training:
  learning_rate: 0.001
  epochs: 20
  optimizer: "adam"          # Options: "adam", "sgd", "rmsprop"
  loss_function: "categorical_crossentropy"
  metrics: ["accuracy"]
```

---

## 🗂️ Dataset Formats

### 1️⃣ **Oral Cancer & Oral Cancer 2.0**
```
data/
└── Oral_Cancer/
    ├── train/
    │   ├── cancer/
    │   └── non-cancer/
    └── val/
        ├── cancer/
        └── non-cancer/
```

### 2️⃣ **Sri Lankan Dataset**
```
data/
└── Sri_Lankan_Dataset/
    ├── images/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── labels.csv
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
conda create -n oral_cancer_detection python=3.9
conda activate oral_cancer_detection
```

### 3️⃣ Install Dependencies
```bash
pip install tensorflow pyyaml pandas
```

Or if using a `requirements.txt`:
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

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🔗 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Anaconda Documentation](https://docs.anaconda.com/)
- Relevant medical image datasets and publications.

