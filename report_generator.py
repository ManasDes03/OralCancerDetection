"""
report_generator.py

Generates a professional PDF report for the oral cancer detection project, including:
- Title page
- Dataset statistics and split summary
- Class distribution
- Multiple sample images per class
- Model/config summary
- Training/validation curves
- Test set evaluation (loss, accuracy, overall metrics)
- Per-class precision/recall/F1 table
- Confusion matrix (with class names)
- Classification report (full text)

Usage:
    python report_generator.py

Dependencies:
    pip install matplotlib seaborn pandas scikit-learn fpdf pillow
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report


# --- CONFIG ---
DATASET_CSV = 'data/Sri_Lankan_Dataset/Imagewise_Data (2).csv'
IMAGES_DIR = 'data/Sri_Lankan_Dataset/Images'
TRAINING_CURVES = 'training_curves.png'
CONFUSION_MATRIX = 'confusion_matrix.png'
MODEL_NAME = 'MobileNetV2_oral_cancer_model.h5'
CONFIG_YAML = 'config.yaml'
SAMPLE_IMAGES_PER_CLASS = 3


# --- LOAD DATASET ---
df = pd.read_csv(DATASET_CSV)


# --- DATASET STATS ---
total_images = len(df)
class_counts = df['Category'].value_counts().sort_index()
class_names = class_counts.index.tolist()

# --- DATASET SPLIT SUMMARY (estimate from file counts if possible) ---
split_summary = ""
try:
    from dataloaders.dataloaders import get_classification_data_loader
    import yaml
    with open(CONFIG_YAML, 'r') as f:
        config = yaml.safe_load(f)
    train_ds, val_ds, test_ds, test_files, test_labels, label_mapping = get_classification_data_loader(config)
    split_summary = f"Train: {len(train_ds)*train_ds._batch_size} | Val: {len(val_ds)*val_ds._batch_size} | Test: {len(test_ds)*test_ds._batch_size} (approximate)"
except Exception as e:
    split_summary = "(Split summary unavailable)"


# --- PLOT CLASS DISTRIBUTION ---
plt.figure(figsize=(6,4))
palette = sns.color_palette('Set2', n_colors=len(class_names))
sns.barplot(x=class_names, y=class_counts.values, palette=palette)
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()


# --- MULTIPLE SAMPLE IMAGES PER CLASS ---
sample_imgs = {cls: [] for cls in class_names}
for cls in class_names:
    samples = df[df['Category'] == cls]['image_name'].head(SAMPLE_IMAGES_PER_CLASS)
    for s in samples:
        img_path = os.path.join(IMAGES_DIR, s)
        if os.path.exists(img_path):
            sample_imgs[cls].append(img_path)


# --- LOAD TRAINING/TEST RESULTS & METRICS ---
def load_text_results():
    results = []
    if os.path.exists('test_results.txt'):
        with open('test_results.txt') as f:
            results = f.readlines()
    return results

def extract_metrics_from_report():
    # Parse test_results.txt for metrics and classification report
    import re
    metrics = {}
    class_report = ""
    if os.path.exists('test_results.txt'):
        with open('test_results.txt') as f:
            lines = f.readlines()
        for line in lines:
            if 'Test Loss' in line:
                metrics['Test Loss'] = float(re.findall(r"[\d\.]+", line)[0])
            if 'Test Accuracy' in line:
                metrics['Test Accuracy'] = float(re.findall(r"[\d\.]+", line)[0])
            if 'Classification Report' in line:
                idx = lines.index(line)
                class_report = ''.join(lines[idx+1:])
    return metrics, class_report


# --- PDF REPORT ---
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# Title Page
pdf.add_page()
pdf.set_font('Arial', 'B', 22)
pdf.cell(0, 20, 'Oral Cancer Detection', ln=True, align='C')
pdf.set_font('Arial', '', 16)
pdf.cell(0, 12, 'Comprehensive Model Evaluation Report', ln=True, align='C')
pdf.ln(20)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Date: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', ln=True, align='C')
pdf.ln(10)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 8, "This report summarizes the dataset, model configuration, training process, and evaluation metrics for the oral cancer detection project. All results are generated automatically from the latest experiment run.", align='C')

# Dataset Overview
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 12, 'Dataset Overview', ln=True)
pdf.set_font('Arial', '', 12)
pdf.cell(0, 10, f'Total Images: {total_images}', ln=True)
pdf.cell(0, 10, f'Classes: {", ".join(map(str, class_names))}', ln=True)
pdf.cell(0, 10, f'Class Counts: {class_counts.to_dict()}', ln=True)
pdf.cell(0, 10, f'Dataset Split: {split_summary}', ln=True)
pdf.ln(5)


# Class distribution plot
if os.path.exists('class_distribution.png'):
    pdf.image('class_distribution.png', w=120)
    pdf.ln(5)


# Sample images (multiple per class)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Sample Images by Class:', ln=True)
pdf.set_font('Arial', '', 11)
for cls in class_names:
    pdf.cell(0, 8, f'Class: {cls}', ln=True)
    for img_path in sample_imgs[cls]:
        pdf.cell(0, 7, os.path.basename(img_path), ln=True)
        try:
            pdf.image(img_path, w=35)
        except:
            pass
        pdf.ln(1)
    pdf.ln(2)
pdf.ln(5)


# Model/Config Summary
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Model & Training Configuration:', ln=True)
pdf.set_font('Arial', '', 11)
try:
    import yaml
    with open(CONFIG_YAML, 'r') as f:
        config = yaml.safe_load(f)
    for section in ['model', 'training']:
        pdf.cell(0, 8, f'{section.capitalize()}:', ln=True)
        for k, v in config[section].items():
            pdf.cell(0, 7, f'  {k}: {v}', ln=True)
except Exception as e:
    pdf.cell(0, 8, f'(Config unavailable: {e})', ln=True)
pdf.ln(5)

# Training curves
if os.path.exists(TRAINING_CURVES):
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Training/Validation Curves:', ln=True)
    pdf.image(TRAINING_CURVES, w=120)
    pdf.ln(5)


# Test results (loss, accuracy, classification report)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Test Set Evaluation:', ln=True)
pdf.set_font('Arial', '', 11)
metrics, class_report = extract_metrics_from_report()
for k, v in metrics.items():
    pdf.cell(0, 8, f'{k}: {v}', ln=True)
pdf.ln(2)

# Per-class metrics table (from classification report)
import re
def parse_classification_report(report):
    lines = report.split('\n')
    rows = []
    for line in lines:
        if re.match(r'\s*\w', line) and ('precision' not in line and 'accuracy' not in line and 'macro avg' not in line and 'weighted avg' not in line):
            parts = line.split()
            if len(parts) >= 4:
                rows.append(parts[:5])
    return rows
table_rows = parse_classification_report(class_report)
if table_rows:
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 8, 'Per-Class Precision/Recall/F1:', ln=True)
    pdf.set_font('Arial', '', 10)
    col_widths = [30, 25, 25, 25, 25]
    headers = ['Class', 'Precision', 'Recall', 'F1-score', 'Support']
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1)
    pdf.ln()
    for row in table_rows:
        for i, val in enumerate(row):
            pdf.cell(col_widths[i], 7, val, border=1)
        pdf.ln()
    pdf.ln(3)

# Confusion matrix
if os.path.exists(CONFUSION_MATRIX):
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Confusion Matrix:', ln=True)
    pdf.image(CONFUSION_MATRIX, w=90)
    pdf.ln(5)


# Full classification report (text)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Full Classification Report:', ln=True)
pdf.set_font('Arial', '', 8)
pdf.multi_cell(0, 5, class_report)


# Save PDF
pdf.output('oral_cancer_detection_report.pdf')
print('Report saved as oral_cancer_detection_report.pdf')
