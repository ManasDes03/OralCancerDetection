"""
Advanced Ensemble Model for High-Risk vs Low-Risk Classification
Same architecture as suspicious but optimized for extreme imbalance
Uses focal loss with α=0.97, γ=2.7 for strong minority focus
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks, mixed_precision
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision and XLA
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

print("=" * 80)
print("ADVANCED ENSEMBLE MODEL - HIGH-RISK vs LOW-RISK")
print("=" * 80)

# =============================================================================
# ADVANCED LOSS FUNCTIONS
# =============================================================================

class FocalLoss(keras.losses.Loss):
    """Focal Loss for extreme class imbalance"""
    def __init__(self, alpha=0.97, gamma=2.7, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        ce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        focal_weight = y_true * self.alpha * tf.pow(1 - y_pred, self.gamma) + \
                      (1 - y_true) * (1 - self.alpha) * tf.pow(y_pred, self.gamma)
        focal_loss = focal_weight * ce
        return tf.reduce_mean(focal_loss)


class CombinedLoss(keras.losses.Loss):
    """Focal Loss + Weighted BCE for extreme imbalance"""
    def __init__(self, focal_alpha=0.97, focal_gamma=2.7, bce_weight=0.2, **kwargs):
        super().__init__(**kwargs)
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.bce = keras.losses.BinaryCrossentropy(from_logits=False)
        self.bce_weight = bce_weight

    def call(self, y_true, y_pred):
        focal_loss = self.focal(y_true, y_pred)
        bce_loss = self.bce(y_true, y_pred)
        return (1 - self.bce_weight) * focal_loss + self.bce_weight * bce_loss


# =============================================================================
# AUGMENTATION & PREPROCESSING
# =============================================================================

def augment_image(image, label):
    """Advanced GPU augmentation"""
    # Random flip
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_up_down(image)
    
    # Random 90 degree rotation
    if tf.random.uniform(()) > 0.5:
        k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=k)
    
    # Random zoom
    zoom_range = tf.random.uniform((), minval=0.75, maxval=1.0)
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    crop_h = tf.cast(tf.cast(h, tf.float32) * zoom_range, tf.int32)
    crop_w = tf.cast(tf.cast(w, tf.float32) * zoom_range, tf.int32)
    
    image = tf.image.random_crop(image, [crop_h, crop_w, 3])
    image = tf.image.resize(image, [224, 224])
    
    # Color augmentation
    image = tf.image.random_brightness(image, 0.25)
    image = tf.image.random_contrast(image, 0.75, 1.25)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.12)
    
    # Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), stddev=0.02)
    image = image + noise
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


def create_dataset(df, batch_size=32, augment=True):
    """Create tf.data pipeline"""
    images = []
    labels = df['risk_label'].values.astype(np.float32)
    
    for img_path in df['image_path'].values:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
        except:
            images.append(np.zeros((224, 224, 3)))
    
    images = np.array(images, dtype=np.float32)
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size)
    ds = ds.cache()
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def create_mobilenet_model(input_shape=(224, 224, 3)):
    """MobileNetV2"""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base_model


def create_efficientnet_model(input_shape=(224, 224, 3)):
    """EfficientNetB3"""
    base_model = keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base_model


def create_densenet_model(input_shape=(224, 224, 3)):
    """DenseNet201 - Better feature reuse"""
    base_model = keras.applications.DenseNet201(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base_model


# =============================================================================
# LEARNING RATE SCHEDULING
# =============================================================================

def make_epoch_lr_fn(base_lr, total_epochs, warmup_epochs=2, final_ratio=0.1):
    """Warmup + Cosine decay"""
    def epoch_lr(epoch):
        if epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return base_lr * (final_ratio + (1 - final_ratio) * 
                            (1 + np.cos(np.pi * progress)) / 2)
    return epoch_lr


# =============================================================================
# DATA LOADING & SPLITTING
# =============================================================================

print("\n[1/8] Loading GDC dataset (Risk labels)...")
df = pd.read_csv('data/gdc_oral_cancer_dataset/processed_labels.csv')

# Filter to only cases with risk labels
df = df[df['risk_label'].notna()].reset_index(drop=True)

print(f"Total images with risk labels: {len(df)}")
print(f"High-risk: {(df['risk_label'] == 1).sum()}")
print(f"Low-risk: {(df['risk_label'] == 0).sum()}")

# Stratified patient-aware split
patients = df['case_no'].unique()
patient_labels = df.groupby('case_no')['risk_label'].first().values

train_patients, temp_patients = train_test_split(
    patients, test_size=0.3, stratify=patient_labels, random_state=42
)
temp_labels = df[df['case_no'].isin(temp_patients)].groupby('case_no')['risk_label'].first().values
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, stratify=temp_labels, random_state=42
)

train_df = df[df['case_no'].isin(train_patients)].reset_index(drop=True)
val_df = df[df['case_no'].isin(val_patients)].reset_index(drop=True)
test_df = df[df['case_no'].isin(test_patients)].reset_index(drop=True)

# Aggressive oversampling for extreme imbalance
from collections import Counter
train_counts = Counter(train_df['risk_label'])
print(f"Train before oversample: {dict(train_counts)}")

# Oversample minority to match majority
max_class_count = max(train_counts.values())
for label in [0, 1]:
    class_df = train_df[train_df['risk_label'] == label]
    if len(class_df) < max_class_count:
        oversample_count = max_class_count - len(class_df)
        oversample_indices = np.random.choice(class_df.index, oversample_count, replace=True)
        train_df = pd.concat([train_df, df.loc[oversample_indices]], ignore_index=True)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

train_counts = Counter(train_df['risk_label'])
print(f"Train after oversample: {dict(train_counts)}")
print(f"\nTrain split: {len(train_df)} images")
print(f"Val split: {len(val_df)} images")
print(f"Test split: {len(test_df)} images")

# Create datasets
print("\n[2/8] Creating tf.data pipelines...")
train_ds = create_dataset(train_df, batch_size=32, augment=True)
val_ds = create_dataset(val_df, batch_size=32, augment=False)
test_ds = create_dataset(test_df, batch_size=32, augment=False)

# =============================================================================
# TRAIN ENSEMBLE MODELS
# =============================================================================

models_dict = {}
architectures = [
    ('MobileNetV2', create_mobilenet_model),
    ('EfficientNetB3', create_efficientnet_model),
    ('DenseNet201', create_densenet_model)
]

for arch_name, create_fn in architectures:
    print(f"\n[3/8] Training {arch_name}...")
    
    model, base_model = create_fn()
    
    # Phase 1: Frozen base
    print(f"Phase 1: Training with frozen base...")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=CombinedLoss(focal_alpha=0.97, focal_gamma=2.7),
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    lr_schedule_1 = keras.callbacks.LearningRateScheduler(
        make_epoch_lr_fn(0.001, 12, warmup_epochs=2),
        verbose=0
    )
    
    history_1 = model.fit(
        train_ds, epochs=12, validation_data=val_ds,
        callbacks=[lr_schedule_1], verbose=1
    )
    
    # Phase 2: Fine-tune
    print(f"Phase 2: Fine-tuning base model...")
    base_model.trainable = True
    
    for layer in base_model.layers[:80]:
        layer.trainable = False
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss=CombinedLoss(focal_alpha=0.97, focal_gamma=2.7),
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    lr_schedule_2 = keras.callbacks.LearningRateScheduler(
        make_epoch_lr_fn(0.0001, 25, warmup_epochs=1),
        verbose=0
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_auc', mode='max', patience=6, restore_best_weights=True
    )
    
    history_2 = model.fit(
        train_ds, epochs=25, validation_data=val_ds,
        callbacks=[lr_schedule_2, early_stopping], verbose=1
    )
    
    models_dict[arch_name] = model

print("\n[4/8] Models trained successfully!")

# =============================================================================
# ENSEMBLE PREDICTION
# =============================================================================

print("\n[5/8] Generating ensemble predictions...")

def get_ensemble_predictions(models_dict, dataset):
    """Ensemble averaging"""
    all_probs = []
    for model_name, model in models_dict.items():
        probs = model.predict(dataset).flatten()
        all_probs.append(probs)
    ensemble_probs = np.mean(all_probs, axis=0)
    return ensemble_probs, all_probs


val_probs_ensemble, _ = get_ensemble_predictions(models_dict, val_ds)
test_probs_ensemble, _ = get_ensemble_predictions(models_dict, test_ds)

val_labels = val_df['risk_label'].values
test_labels = test_df['risk_label'].values

# Find best threshold
fpr, tpr, thresholds = roc_curve(val_labels, val_probs_ensemble)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thresholds[best_idx]

print(f"Best threshold: {best_threshold:.3f}")

# Evaluate
y_pred = (test_probs_ensemble > best_threshold).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred, zero_division=0)
recall = recall_score(test_labels, y_pred, zero_division=0)
f1 = f1_score(test_labels, y_pred, zero_division=0)
roc_auc = auc(fpr, tpr)

print("\n[6/8] ENSEMBLE TEST RESULTS:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, y_pred, 
                          target_names=['Low-risk', 'High-risk']))

# =============================================================================
# SAVE MODELS & RESULTS
# =============================================================================

print("\n[7/8] Saving models and results...")

os.makedirs('ensemble_risk_results', exist_ok=True)

for arch_name, model in models_dict.items():
    model.save(f'ensemble_risk_results/{arch_name.lower()}_model.h5')

results = {
    'architecture': 'Ensemble (MobileNetV2 + EfficientNetB3 + DenseNet201)',
    'best_threshold': float(best_threshold),
    'test_accuracy': float(accuracy),
    'test_precision': float(precision),
    'test_recall': float(recall),
    'test_f1': float(f1),
    'test_roc_auc': float(roc_auc),
    'train_samples': len(train_df),
    'val_samples': len(val_df),
    'test_samples': len(test_df)
}

import json
with open('ensemble_risk_results/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# =============================================================================
# VISUALIZATION
# =============================================================================

print("[8/8] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ROC Curve
ax = axes[0, 0]
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Ensemble ROC Curve')
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

# Confusion Matrix
ax = axes[0, 1]
cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
ax.set_xticklabels(['Low-risk', 'High-risk'])
ax.set_yticklabels(['Low-risk', 'High-risk'])

# Probability distribution
ax = axes[1, 0]
ax.hist(test_probs_ensemble[test_labels == 0], bins=30, alpha=0.5, label='Low-risk', color='blue')
ax.hist(test_probs_ensemble[test_labels == 1], bins=30, alpha=0.5, label='High-risk', color='red')
ax.axvline(best_threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {best_threshold:.3f}')
ax.set_xlabel('Prediction Probability')
ax.set_ylabel('Frequency')
ax.set_title('Ensemble Prediction Distribution')
ax.legend()
ax.grid(alpha=0.3)

# Performance metrics
ax = axes[1, 1]
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
ax.barh(metrics_names, metrics_values, color='lightgreen')
ax.set_xlim([0, 1])
ax.set_xlabel('Score')
ax.set_title('Ensemble Model Performance')
for i, v in enumerate(metrics_values):
    ax.text(v + 0.02, i, f'{v:.3f}', va='center')
ax.grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('ensemble_risk_results/evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation.png")

plt.close()

print("\n" + "="*80)
print("✓ ENSEMBLE MODEL TRAINING COMPLETE")
print("="*80)
print(f"\nResults saved to: ensemble_risk_results/")
print(f"Models: MobileNetV2, EfficientNetB3, DenseNet201")
print(f"Ensemble ROC-AUC: {roc_auc:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
