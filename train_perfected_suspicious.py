"""
PERFECTED MODEL - Advanced Ensemble & Optimization
Multiple Architectures: MobileNetV2, Inception V3, Xception
Advanced Loss Functions & Training Strategies
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PERFECTED ENSEMBLE MODEL - SUSPICIOUS vs NON-SUSPICIOUS")
print("="*80)

# =============================================================================
# FOCAL LOSS - Better handling of class imbalance
# =============================================================================

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.85, gamma=2.5, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce_loss = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
        weight = y_true * self.alpha * tf.pow(1.0 - y_pred, self.gamma) + \
                (1.0 - y_true) * (1.0 - self.alpha) * tf.pow(y_pred, self.gamma)
        return tf.reduce_mean(weight * ce_loss)


# =============================================================================
# LOAD & PREPARE DATA
# =============================================================================

print("\n[Step 1] Loading GDC dataset...")
df = pd.read_csv('data/gdc_oral_cancer_dataset/processed_labels.csv')
df = df[df['suspicious_label'].notna()].reset_index(drop=True)

print(f"Total images: {len(df)}")
print(f"Suspicious: {(df['suspicious_label'] == 1).sum()}")
print(f"Non-suspicious: {(df['suspicious_label'] == 0).sum()}")

# Stratified patient-aware split
patients = df['case_no'].unique()
patient_labels = df.groupby('case_no')['suspicious_label'].first().values

train_patients, temp_patients = train_test_split(
    patients, test_size=0.3, stratify=patient_labels, random_state=42
)
temp_labels = df[df['case_no'].isin(temp_patients)].groupby('case_no')['suspicious_label'].first().values
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, stratify=temp_labels, random_state=42
)

train_df = df[df['case_no'].isin(train_patients)].reset_index(drop=True)
val_df = df[df['case_no'].isin(val_patients)].reset_index(drop=True)
test_df = df[df['case_no'].isin(test_patients)].reset_index(drop=True)

# Oversample minority
from collections import Counter
counts = Counter(train_df['suspicious_label'])
max_count = max(counts.values())
for label in [0, 1]:
    class_df = train_df[train_df['suspicious_label'] == label]
    if len(class_df) < max_count:
        oversample_count = max_count - len(class_df)
        oversample_indices = np.random.choice(class_df.index, oversample_count, replace=True)
        train_df = pd.concat([train_df, df.loc[oversample_indices]], ignore_index=True)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nTrain: {len(train_df)} images")
print(f"Val: {len(val_df)} images")
print(f"Test: {len(test_df)} images")

# =============================================================================
# CREATE DATASETS
# =============================================================================

def create_dataset(df_data, batch_size=32, augment=False):
    images = []
    for img_path in df_data['image_path'].values:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
        except:
            images.append(np.zeros((224, 224, 3), dtype=np.float32))
    
    images = np.array(images, dtype=np.float32)
    labels = df_data['suspicious_label'].values.astype(np.float32)
    
    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if augment:
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            if tf.random.uniform(()) > 0.5:
                k = tf.random.uniform((), 1, 4, dtype=tf.int32)
                img = tf.image.rot90(img, k=k)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            return img, lbl
        
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds

print("\n[Step 2] Creating tf.data pipelines...")
train_ds = create_dataset(train_df, batch_size=32, augment=True)
val_ds = create_dataset(val_df, batch_size=32, augment=False)
test_ds = create_dataset(test_df, batch_size=32, augment=False)

# =============================================================================
# CREATE MODELS
# =============================================================================

def create_mobilenet():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base

def create_inceptionv3():
    base = tf.keras.applications.InceptionV3(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base

def create_xception():
    base = tf.keras.applications.Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid', dtype='float32')
    ])
    return model, base

# =============================================================================
# TRAIN MODELS
# =============================================================================

print("\n[Step 3] Training ensemble models...")

models_dict = {}
architectures = [
    ('MobileNetV2', create_mobilenet),
    ('InceptionV3', create_inceptionv3),
    ('Xception', create_xception)
]

for arch_name, create_fn in architectures:
    print(f"\n  Training {arch_name}...")
    model, base_model = create_fn()
    
    # Phase 1
    model.compile(optimizer=optimizers.Adam(0.001),
                 loss=FocalLoss(0.85, 2.5),
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    
    model.fit(train_ds, epochs=12, validation_data=val_ds, verbose=0,
             callbacks=[callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)])
    
    # Phase 2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(optimizer=optimizers.Adam(0.0001),
                 loss=FocalLoss(0.85, 2.5),
                 metrics=['accuracy', tf.keras.metrics.AUC()])
    
    model.fit(train_ds, epochs=15, validation_data=val_ds, verbose=0,
             callbacks=[callbacks.EarlyStopping(monitor='val_auc', patience=4, restore_best_weights=True)])
    
    models_dict[arch_name] = model
    print(f"  ✓ {arch_name} trained")

# =============================================================================
# ENSEMBLE EVALUATION
# =============================================================================

print("\n[Step 4] Evaluating ensemble...")

val_probs_all = []
for model in models_dict.values():
    probs = model.predict(val_ds, verbose=0).flatten()
    val_probs_all.append(probs)

ensemble_val_probs = np.mean(val_probs_all, axis=0)
val_labels = val_df['suspicious_label'].values

fpr, tpr, thresholds = roc_curve(val_labels, ensemble_val_probs)
youden = tpr - fpr
best_idx = np.argmax(youden)
best_threshold = thresholds[best_idx]

print(f"  Best threshold: {best_threshold:.3f}")

# Test evaluation
test_probs_all = []
for model in models_dict.values():
    probs = model.predict(test_ds, verbose=0).flatten()
    test_probs_all.append(probs)

ensemble_test_probs = np.mean(test_probs_all, axis=0)
test_labels = test_df['suspicious_label'].values
y_pred = (ensemble_test_probs > best_threshold).astype(int)

accuracy = accuracy_score(test_labels, y_pred)
f1 = f1_score(test_labels, y_pred, zero_division=0)
roc_auc = auc(*roc_curve(test_labels, ensemble_test_probs)[:2])

print("\n" + "="*60)
print("ENSEMBLE TEST RESULTS - SUSPICIOUS CLASSIFIER")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(classification_report(test_labels, y_pred, target_names=['Non-suspicious', 'Suspicious']))

# =============================================================================
# SAVE RESULTS
# =============================================================================

print("\n[Step 5] Saving models and results...")

os.makedirs('perfected_models_suspicious', exist_ok=True)

for arch_name, model in models_dict.items():
    model.save(f'perfected_models_suspicious/{arch_name.lower()}.h5')

import json
results = {
    'architecture': 'Ensemble (MobileNetV2, InceptionV3, Xception)',
    'threshold': float(best_threshold),
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'roc_auc': float(roc_auc),
    'test_samples': len(test_df)
}

with open('perfected_models_suspicious/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

fpr, tpr, _ = roc_curve(test_labels, ensemble_test_probs)

axes[0, 0].plot(fpr, tpr, lw=2, label=f'Ensemble (AUC={roc_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0, 0].set_xlabel('FPR')
axes[0, 0].set_ylabel('TPR')
axes[0, 0].set_title('ROC Curve')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

cm = confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1], cbar=False)
axes[0, 1].set_title('Confusion Matrix')

axes[1, 0].hist(ensemble_test_probs[test_labels==0], bins=20, alpha=0.5, label='Non-suspicious')
axes[1, 0].hist(ensemble_test_probs[test_labels==1], bins=20, alpha=0.5, label='Suspicious')
axes[1, 0].axvline(best_threshold, color='red', linestyle='--', label=f'Threshold: {best_threshold:.3f}')
axes[1, 0].set_xlabel('Prediction')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Prediction Distribution')
axes[1, 0].legend()

metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
values = [accuracy, f1, roc_auc]
axes[1, 1].bar(metrics, values, color=['skyblue', 'lightgreen', 'orange'])
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_title('Performance Metrics')
for i, v in enumerate(values):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig('perfected_models_suspicious/results.png', dpi=300)
print("✓ Saved results and models")

print("\n" + "="*60)
print("✓ SUSPICIOUS CLASSIFIER COMPLETE")
print("="*60)
