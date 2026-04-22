"""
🚀 GDC SUSPICIOUS vs NON-SUSPICIOUS BINARY CLASSIFICATION
=========================================================
Android-deployable binary classifier using MobileNetV2
Optimizations:
- tf.data pipeline with GPU-accelerated augmentation
- Mixed precision training for Tensor Cores
- Prefetching and caching for better GPU utilization
- Patient-aware splitting to prevent data leakage
- Two-phase training (freeze → fine-tune)
- Ready for TFLite conversion
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, mixed_precision
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os
import json
import argparse

# Enable mixed precision
mixed_precision.set_global_policy('mixed_float16')

# Enable XLA
tf.config.optimizer.set_jit(True)

np.random.seed(42)
tf.random.set_seed(42)
keras.utils.set_random_seed(42)

print("🚀 GDC SUSPICIOUS vs NON-SUSPICIOUS BINARY CLASSIFICATION")
print("=" * 70)

CONFIG = {
    'backbone': 'efficientnetb0',
    'img_height': 224,
    'img_width': 224,
    'batch_size': 8,
    'initial_lr': 2e-4,
    'fine_tune_lr': 2e-5,
    'initial_epochs': 22,
    'fine_tune_epochs': 20,
    'fine_tune_trainable_layers': 60,
    'focal_loss_alpha': 0.78,
    'focal_loss_gamma': 2.0,
    'warmup_epochs': 2,
    'target_pos_ratio': 0.50,
    'early_stopping_patience': 10,
    'tta_flip': True,
    'monitor_metric': 'val_auc',
    'recall_target': 0.93,
    'hard_negative_mining': True,
    'hnm_threshold': 0.35,
    'hnm_oversample_factor': 2,
}

data_dir = 'data/gdc_oral_cancer_dataset'
images_dir = os.path.join(data_dir, 'images')
csv_path = os.path.join(data_dir, 'processed_labels.csv')
results_dir = 'gdc_suspicious_results_ratio_1_1'
os.makedirs(results_dir, exist_ok=True)


def get_backbone_and_preprocess(backbone_name):
    name = backbone_name.lower()
    if name == 'efficientnetb5':
        return keras.applications.EfficientNetB5, keras.applications.efficientnet.preprocess_input
    if name == 'efficientnetb3':
        return keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input
    if name == 'efficientnetb0':
        return keras.applications.EfficientNetB0, keras.applications.efficientnet.preprocess_input
    if name == 'mobilenetv2':
        return keras.applications.MobileNetV2, keras.applications.mobilenet_v2.preprocess_input
    raise ValueError(f"Unsupported backbone: {backbone_name}")


BACKBONE_FN, PREPROCESS_FN = get_backbone_and_preprocess(CONFIG['backbone'])


class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
        
        y_true = tf.cast(y_true, tf.float32)
        
        # Binary focal loss
        loss_1 = -self.alpha * keras.backend.pow(1 - y_pred, self.gamma) * y_true * keras.backend.log(y_pred)
        loss_0 = -(1 - self.alpha) * keras.backend.pow(y_pred, self.gamma) * (1 - y_true) * keras.backend.log(1 - y_pred)
        
        return loss_1 + loss_0
    
    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}


def augment_image(image, label):
    """GPU-accelerated augmentation in pixel space [0, 255]."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.08)
    image = tf.image.random_contrast(image, 0.85, 1.15)
    image = tf.image.random_saturation(image, 0.85, 1.15)
    image = tf.image.random_hue(image, 0.03)
    
    # Mild random zoom
    shape = tf.shape(image)
    crop_size = tf.random.uniform([], 0.90, 1.0)
    h = tf.cast(tf.cast(shape[0], tf.float32) * crop_size, tf.int32)
    w = tf.cast(tf.cast(shape[1], tf.float32) * crop_size, tf.int32)
    image = tf.image.random_crop(image, [h, w, 3])
    image = tf.image.resize(image, [CONFIG['img_height'], CONFIG['img_width']])

    # Add mild Gaussian noise
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=3.0, dtype=image.dtype)
    image = tf.clip_by_value(image + noise, 0.0, 255.0)
    return image, label


def augment_image_heavy(image, label):
    """Extra-aggressive augmentation for minority (suspicious) class.
    Simulates SMOTE-like diversity: rotations, heavy color shifts, coarse dropout.
    """
    image, label = augment_image(image, label)

    # Random 90/180/270 degree rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)

    # Stronger brightness + contrast swing
    image = tf.image.random_brightness(image, 0.20)
    image = tf.image.random_contrast(image, 0.70, 1.40)

    # Coarse dropout: blank out a random rectangle (simulates occlusion)
    h = CONFIG['img_height']
    w = CONFIG['img_width']
    patch_h = tf.random.uniform([], 16, 56, dtype=tf.int32)
    patch_w = tf.random.uniform([], 16, 56, dtype=tf.int32)
    y0 = tf.random.uniform([], 0, h - 56, dtype=tf.int32)
    x0 = tf.random.uniform([], 0, w - 56, dtype=tf.int32)
    top    = tf.ones([y0, w, 3], dtype=image.dtype)
    mid_l  = tf.ones([patch_h, x0, 3], dtype=image.dtype)
    blank  = tf.zeros([patch_h, patch_w, 3], dtype=image.dtype)
    mid_r  = tf.ones([patch_h, w - x0 - patch_w, 3], dtype=image.dtype)
    bottom = tf.ones([h - y0 - patch_h, w, 3], dtype=image.dtype)
    mid    = tf.concat([mid_l, blank, mid_r], axis=1)
    mask   = tf.concat([top, mid, bottom], axis=0)
    image  = image * mask
    image = tf.clip_by_value(image, 0.0, 255.0)
    return image, label


def create_balanced_dataset(paths, labels, batch_size):
    """Build a balanced training dataset using tf.data interleaving.

    Separates suspicious (minority) and non-suspicious (majority) into two
    parallel streams. Minority gets heavy augmentation, majority gets standard.
    Batches are zipped 1:1 so every batch always contains ~50% suspicious.
    Requires steps_per_epoch in model.fit() because it uses .repeat().
    """
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [CONFIG['img_height'], CONFIG['img_width']])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        return image, label

    def to_input(image, label):
        return PREPROCESS_FN(image), label

    paths = np.array(paths)
    labels = np.array(labels)
    pos_mask = labels == 1
    neg_mask = labels == 0

    pos_paths, pos_labels = paths[pos_mask], labels[pos_mask]
    neg_paths, neg_labels = paths[neg_mask], labels[neg_mask]

    half = batch_size // 2

    # Minority (suspicious): shuffle + repeat + heavy augment
    pos_ds = tf.data.Dataset.from_tensor_slices((pos_paths, pos_labels))
    pos_ds = pos_ds.shuffle(len(pos_paths) * 10, seed=42, reshuffle_each_iteration=True)
    pos_ds = pos_ds.repeat()
    pos_ds = pos_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    pos_ds = pos_ds.map(augment_image_heavy, num_parallel_calls=tf.data.AUTOTUNE)
    pos_ds = pos_ds.map(to_input, num_parallel_calls=tf.data.AUTOTUNE)
    pos_ds = pos_ds.batch(half)

    # Majority (non-suspicious): shuffle + repeat + standard augment
    neg_ds = tf.data.Dataset.from_tensor_slices((neg_paths, neg_labels))
    neg_ds = neg_ds.shuffle(len(neg_paths) * 3, seed=42, reshuffle_each_iteration=True)
    neg_ds = neg_ds.repeat()
    neg_ds = neg_ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    neg_ds = neg_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    neg_ds = neg_ds.map(to_input, num_parallel_calls=tf.data.AUTOTUNE)
    neg_ds = neg_ds.batch(half)

    combined = tf.data.Dataset.zip((pos_ds, neg_ds))
    def merge_batches(pos_batch, neg_batch):
        images = tf.concat([pos_batch[0], neg_batch[0]], axis=0)
        labs   = tf.concat([pos_batch[1], neg_batch[1]], axis=0)
        return images, labs
    combined = combined.map(merge_batches, num_parallel_calls=tf.data.AUTOTUNE)
    combined = combined.prefetch(tf.data.AUTOTUNE)
    return combined


def create_dataset(paths, labels, batch_size, shuffle=False, augment=False, cache=True):
    """Create optimized tf.data pipeline"""
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [CONFIG['img_height'], CONFIG['img_width']])
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        return image, label

    def to_mobilenet_input(image, label):
        image = PREPROCESS_FN(image)
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
    
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if cache:
        dataset = dataset.cache()
    
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.map(to_mobilenet_input, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def create_flipped_dataset(paths, labels, batch_size):
    """TTA dataset with horizontal flip."""
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [CONFIG['img_height'], CONFIG['img_width']])
        image = tf.image.flip_left_right(image)
        image = tf.cast(image, tf.float32)
        image = PREPROCESS_FN(image)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def stratified_patient_split(df, label_col, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    """Patient-aware stratified split to keep label distribution in each split"""
    patient_labels = df.groupby('case_no')[label_col].first()
    patients = patient_labels.index.values
    labels = patient_labels.values

    train_patients, temp_patients, train_labels, temp_labels = train_test_split(
        patients, labels, test_size=val_ratio + test_ratio, stratify=labels, random_state=42
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=1 - val_size, stratify=temp_labels, random_state=42
    )

    train_df = df[df['case_no'].isin(train_patients)]
    val_df = df[df['case_no'].isin(val_patients)]
    test_df = df[df['case_no'].isin(test_patients)]

    return train_df, val_df, test_df


def oversample_minority_by_patient(train_df, label_col, target_pos_ratio=0.5):
    """Oversample minority patients to approach target positive ratio"""
    patient_labels = train_df.groupby('case_no')[label_col].first()
    pos_patients = patient_labels[patient_labels == 1].index.tolist()
    neg_patients = patient_labels[patient_labels == 0].index.tolist()

    if not pos_patients or not neg_patients:
        return train_df

    n_pos = len(pos_patients)
    n_neg = len(neg_patients)
    current_ratio = n_pos / (n_pos + n_neg)

    if current_ratio >= target_pos_ratio:
        return train_df

    # how many positive patients to add (with replacement)
    target_pos = int(target_pos_ratio * (n_pos + n_neg) / (1 - target_pos_ratio))
    needed = max(0, target_pos - n_pos)
    sampled = np.random.choice(pos_patients, size=needed, replace=True)

    extra_chunks = [train_df[train_df['case_no'] == p].copy() for p in sampled]
    if not extra_chunks:
        return train_df
    extra = pd.concat(extra_chunks, axis=0)
    balanced_df = pd.concat([train_df, extra], axis=0, ignore_index=True)
    return balanced_df


def build_exact_ratio_1_to_1(train_df, label_col):
    """Use all positives and sample equal number of negatives (exact 1:1)."""
    pos_df = train_df[train_df[label_col] == 1].copy()
    neg_df = train_df[train_df[label_col] == 0].copy()

    if len(pos_df) == 0 or len(neg_df) == 0:
        return train_df

    n_pos = len(pos_df)
    replace = len(neg_df) < n_pos
    neg_sample = neg_df.sample(n=n_pos, random_state=42, replace=replace)
    out = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return out


def load_and_clean_df(label_col):
    """Load processed labels and remove duplicates / invalid rows."""
    df = pd.read_csv(csv_path)

    # Keep only rows with label and unique image per case
    df = df[df[label_col].notna()].copy()
    before = len(df)
    df = df.drop_duplicates(subset=['case_no', 'image_path']).copy()
    deduped = before - len(df)

    # Full path and existing files only
    df['full_path'] = df['image_path'].apply(lambda x: os.path.join(images_dir, x))
    df = df[df['full_path'].apply(os.path.exists)].copy()

    # Remove contradictory labels within same patient if any
    patient_unique = df.groupby('case_no')[label_col].nunique()
    bad_patients = patient_unique[patient_unique > 1].index
    if len(bad_patients) > 0:
        df = df[~df['case_no'].isin(bad_patients)].copy()
        print(f"⚠️ Removed {len(bad_patients)} patients with mixed labels for {label_col}")

    print(f"🧹 Removed duplicate rows: {deduped}")
    return df


def choose_best_threshold_by_f1(y_true, y_probs):
    """Pick threshold using F2-score (beta=2 weights recall 4x over precision).

    In medical screening, false negatives (missed suspicious) are far worse
    than false positives (unnecessary follow-up). F2 reflects that priority.
    Threshold is capped at 0.45 to ensure active flagging of suspicious cases.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    f2 = f2[:-1]  # drop last point (threshold=1.0)
    best_idx = int(np.argmax(f2))
    best_threshold = float(np.clip(thresholds[best_idx], 0.05, 0.45))
    best_f2 = float(f2[best_idx])
    return best_threshold, best_f2


def choose_threshold_with_recall_constraint(y_true, y_probs, min_recall=0.93):
    """Select threshold that maximizes specificity while meeting recall floor.

    For screening, maintain high recall first, then improve specificity.
    """
    thresholds = np.linspace(0.05, 0.95, 181)
    best = None

    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        f2_num = (1 + 2.0**2) * (tp / (tp + fp + 1e-8)) * recall
        f2_den = (2.0**2) * (tp / (tp + fp + 1e-8)) + recall + 1e-8
        f2 = f2_num / f2_den

        if recall >= min_recall:
            cand = (specificity, f2, -t, t, recall)
            if best is None or cand > best:
                best = cand

    if best is not None:
        return float(best[3]), float(best[4]), True

    # Fallback: no threshold can satisfy recall floor.
    # Choose threshold with highest recall, then best F2.
    fallback = None
    for t in thresholds:
        y_pred = (y_probs > t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f2_num = (1 + 2.0**2) * precision * recall
        f2_den = (2.0**2) * precision + recall + 1e-8
        f2 = f2_num / f2_den
        cand = (recall, f2, -t, t)
        if fallback is None or cand > fallback:
            fallback = cand

    return float(fallback[3]), float(fallback[0]), False


def build_hard_negative_boosted_df(model, train_df):
    """Mine hard negatives (false alarms) and upsample them for phase-2."""
    eval_ds = create_dataset(
        train_df['full_path'].values,
        train_df['suspicious_label'].values,
        CONFIG['batch_size'],
        shuffle=False,
        augment=False,
        cache=False,
    )
    train_probs = model.predict(eval_ds, verbose=0).flatten()
    train_labels = train_df['suspicious_label'].values
    hard_neg_mask = (train_labels == 0) & (train_probs >= CONFIG['hnm_threshold'])
    hard_negs = train_df.loc[hard_neg_mask].copy()

    if len(hard_negs) == 0:
        return train_df, 0

    boosted_parts = [train_df] + [hard_negs.copy() for _ in range(CONFIG['hnm_oversample_factor'])]
    boosted_df = pd.concat(boosted_parts, ignore_index=True)
    return boosted_df, len(hard_negs)


def create_model():
    """Create backbone-based binary classifier (Android-ready)."""
    base_model = BACKBONE_FN(
        input_shape=(CONFIG['img_height'], CONFIG['img_width'], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(CONFIG['img_height'], CONFIG['img_width'], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)  # float32 for stability
    
    model = Model(inputs, outputs)
    return model, base_model


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, steps_per_epoch, total_epochs, warmup_epochs=2, final_lr_ratio=0.1, name=None):
        super().__init__()
        self.base_lr = base_lr
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.final_lr_ratio = final_lr_ratio
        self.total_steps = steps_per_epoch * total_epochs
        self.warmup_steps = steps_per_epoch * warmup_epochs
        self.cosine_steps = max(self.total_steps - self.warmup_steps, 1)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=base_lr,
            decay_steps=self.cosine_steps,
            alpha=final_lr_ratio
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        warmup_lr = self.base_lr * (step + 1.0) / tf.maximum(warmup_steps, 1.0)
        decay_lr = self.cosine_decay(tf.maximum(step - warmup_steps, 0.0))
        return tf.convert_to_tensor(tf.where(step < warmup_steps, warmup_lr, decay_lr), dtype=tf.float32)

    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'steps_per_epoch': self.steps_per_epoch,
            'total_epochs': self.total_epochs,
            'warmup_epochs': self.warmup_epochs,
            'final_lr_ratio': self.final_lr_ratio
        }


def make_epoch_lr_fn(base_lr, total_epochs, warmup_epochs=2, final_lr_ratio=0.1):
    def lr_fn(epoch):
        if epoch < warmup_epochs:
            return float(base_lr * (epoch + 1) / max(warmup_epochs, 1))
        # cosine from warmup to end
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return float((final_lr_ratio + (1 - final_lr_ratio) * cosine_decay) * base_lr)
    return lr_fn


def plot_metrics(history, phase_name):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title(f'{phase_name} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'{phase_name} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{phase_name.lower().replace(" ", "_")}_history.png'), dpi=300)
    plt.close()


def main(eval_only=False):
    # Load data
    df = load_and_clean_df('suspicious_label')
    print(f"\n📊 Dataset Statistics:")
    print(f"Total images: {len(df)}")
    print(f"Total cases: {df['case_no'].nunique()}")
    print(f"Suspicious: {(df['suspicious_label']==1).sum()}")
    print(f"Non-suspicious: {(df['suspicious_label']==0).sum()}")
    
    # Split by patient (stratified)
    # CRITICAL: compute class weights from ORIGINAL split BEFORE oversampling
    # so they reflect the true ~2.2:1 imbalance, not the post-oversample ratio.
    # Boost minority weight by 1.5x: false negatives (missed suspicious) are
    # clinically worse than false positives (unnecessary follow-up).
    train_df, val_df, test_df = stratified_patient_split(df, 'suspicious_label')
    orig_counts = Counter(train_df['suspicious_label'])
    orig_total = sum(orig_counts.values())
    recall_boost = 1.5
    class_weights = {
        0: float(orig_total / (2 * orig_counts[0])),
        1: float(orig_total / (2 * orig_counts[1])) * recall_boost
    }
    train_df = build_exact_ratio_1_to_1(train_df, 'suspicious_label')

    print(f"\n📂 Split:")
    print(f"Train: {len(train_df)} images from {train_df['case_no'].nunique()} patients (exact 1:1 image ratio)")
    print(f"Val: {len(val_df)} images from {val_df['case_no'].nunique()} patients")
    print(f"Test: {len(test_df)} images from {test_df['case_no'].nunique()} patients")

    # Create balanced dataset: minority gets heavy augment every batch
    train_ds = create_balanced_dataset(
        train_df['full_path'].values,
        train_df['suspicious_label'].values,
        CONFIG['batch_size']
    )
    # steps_per_epoch required because balanced dataset uses .repeat()
    minority_count = int((train_df['suspicious_label'] == 1).sum())
    steps_per_epoch = max(50, (minority_count * 2) // CONFIG['batch_size'])
    print(f"⚖️  Balanced batching: {minority_count} minority samples → {steps_per_epoch} steps/epoch")

    val_ds = create_dataset(
        val_df['full_path'].values,
        val_df['suspicious_label'].values,
        CONFIG['batch_size'],
        shuffle=False,
        augment=False
    )

    test_ds = create_dataset(
        test_df['full_path'].values,
        test_df['suspicious_label'].values,
        CONFIG['batch_size'],
        shuffle=False,
        augment=False
    )

    CONFIG['class_weights'] = class_weights
    print(f"\n⚖️ Class weights (from original split + recall boost): {class_weights}")
    
    if not eval_only:
        # Create model
        model, base_model = create_model()
        print("\n🏗️ Model Architecture:")
        model.summary()
        
        # Phase 1: Train with frozen base
        print("\n" + "="*70)
        print(f"PHASE 1: Training with frozen {CONFIG['backbone']} base")
        print("="*70)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
            loss=FocalLoss(alpha=CONFIG['focal_loss_alpha'], gamma=CONFIG['focal_loss_gamma']),
            metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='pr_auc', curve='PR')]
        )

        lr_callback = keras.callbacks.LearningRateScheduler(
            make_epoch_lr_fn(
                base_lr=CONFIG['initial_lr'],
                total_epochs=CONFIG['initial_epochs'],
                warmup_epochs=CONFIG['warmup_epochs'],
                final_lr_ratio=0.1
            ), verbose=0
        )
        
        phase1_weights_path = os.path.join(results_dir, 'best_phase1.weights.h5')
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=CONFIG['initial_epochs'],
            steps_per_epoch=steps_per_epoch,
            class_weight=class_weights,
            callbacks=[
                lr_callback,
                keras.callbacks.ModelCheckpoint(
                    phase1_weights_path,
                    monitor=CONFIG['monitor_metric'],
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    patience=CONFIG['early_stopping_patience'],
                    restore_best_weights=True,
                    monitor=CONFIG['monitor_metric'],
                    mode='max'
                ),
            ]
        )
        
        # Load best weights; save_weights() avoids TF 2.10 EagerTensor JSON crash in model.save()
        if os.path.exists(phase1_weights_path):
            model.load_weights(phase1_weights_path)
        model.save_weights(os.path.join(results_dir, 'best_phase1.h5'))
        print('✅ Phase 1 weights saved.')
        plot_metrics(history1, 'Phase 1')
        model.save_weights(os.path.join(results_dir, 'suspicious_classifier_initial.h5'))
        
        # Phase 2: Fine-tune
        print("\n" + "="*70)
        print("PHASE 2: Fine-tuning top backbone layers")
        print("="*70)

        train_df_phase2 = train_df
        if CONFIG['hard_negative_mining']:
            train_df_phase2, hard_neg_count = build_hard_negative_boosted_df(model, train_df)
            print(f"🎯 Hard-negative mining: found {hard_neg_count} hard negatives")
            if hard_neg_count > 0:
                print(f"🎯 Phase-2 train set boosted to {len(train_df_phase2)} images")

        train_ds_phase2 = create_balanced_dataset(
            train_df_phase2['full_path'].values,
            train_df_phase2['suspicious_label'].values,
            CONFIG['batch_size']
        )
        minority_count_phase2 = int((train_df_phase2['suspicious_label'] == 1).sum())
        steps_per_epoch_phase2 = max(50, (minority_count_phase2 * 2) // CONFIG['batch_size'])
        
        base_model.trainable = True
        fine_tune_at = max(0, len(base_model.layers) - CONFIG['fine_tune_trainable_layers'])
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=CONFIG['fine_tune_lr']),
            loss=FocalLoss(alpha=CONFIG['focal_loss_alpha'], gamma=CONFIG['focal_loss_gamma']),
            metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='pr_auc', curve='PR')]
        )

        lr_callback_ft = keras.callbacks.LearningRateScheduler(
            make_epoch_lr_fn(
                base_lr=CONFIG['fine_tune_lr'],
                total_epochs=CONFIG['fine_tune_epochs'],
                warmup_epochs=max(1, CONFIG['warmup_epochs'] // 2),
                final_lr_ratio=0.1
            ), verbose=0
        )
        
        phase2_weights_path = os.path.join(results_dir, 'best_phase2.weights.h5')
        history2 = model.fit(
            train_ds_phase2,
            validation_data=val_ds,
            epochs=CONFIG['fine_tune_epochs'],
            steps_per_epoch=steps_per_epoch_phase2,
            class_weight=class_weights,
            callbacks=[
                lr_callback_ft,
                keras.callbacks.ModelCheckpoint(
                    phase2_weights_path,
                    monitor=CONFIG['monitor_metric'],
                    mode='max',
                    save_best_only=True,
                    save_weights_only=True,
                    verbose=1,
                ),
                keras.callbacks.EarlyStopping(
                    patience=CONFIG['early_stopping_patience'],
                    restore_best_weights=True,
                    monitor=CONFIG['monitor_metric'],
                    mode='max'
                ),
            ]
        )
        
        # Load best weights; save_weights() avoids TF 2.10 EagerTensor JSON crash in model.save()
        if os.path.exists(phase2_weights_path):
            model.load_weights(phase2_weights_path)
        model.save_weights(os.path.join(results_dir, 'best_phase2.h5'))
        print('✅ Phase 2 weights saved.')
        plot_metrics(history2, 'Phase 2')
        model.save_weights(os.path.join(results_dir, 'suspicious_classifier_final.h5'))
        
        # Combined history plot
        combined_history = {
            'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
            'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
            'loss': history1.history['loss'] + history2.history['loss'],
            'val_loss': history1.history['val_loss'] + history2.history['val_loss']
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(combined_history['accuracy'], label='Train Accuracy')
        axes[0].plot(combined_history['val_accuracy'], label='Val Accuracy')
        axes[0].axvline(x=len(history1.history['accuracy']), color='r', linestyle='--', label='Fine-tune Start')
        axes[0].set_title('Combined Training - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(combined_history['loss'], label='Train Loss')
        axes[1].plot(combined_history['val_loss'], label='Val Loss')
        axes[1].axvline(x=len(history1.history['loss']), color='r', linestyle='--', label='Fine-tune Start')
        axes[1].set_title('Combined Training - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'combined_training_history.png'), dpi=300)
        plt.close()

        # Compare Phase 1 vs Phase 2 best val metric, use the better one
        best_p1_val = max(history1.history.get(CONFIG['monitor_metric'], [0]))
        best_p2_val = max(history2.history.get(CONFIG['monitor_metric'], [0]))
        print(f"\n📊 Phase 1 best {CONFIG['monitor_metric']}: {best_p1_val:.4f}")
        print(f"📊 Phase 2 best {CONFIG['monitor_metric']}: {best_p2_val:.4f}")

        # Reload best weights into current model (weights-only format, no JSON config)
        best_phase2 = os.path.join(results_dir, 'best_phase2.h5')
        best_phase1 = os.path.join(results_dir, 'best_phase1.h5')
        if best_p2_val >= best_p1_val and os.path.exists(best_phase2):
            model.load_weights(best_phase2)
            print("✅ Loaded best fine-tuned checkpoint (Phase 2 is better)")
        elif os.path.exists(best_phase1):
            # Phase 2 degraded — Phase 1 weights saved with backbone fully frozen.
            # Must freeze backbone FIRST before load_weights to match saved weight layout.
            base_model.trainable = False
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=CONFIG['initial_lr']),
                loss=FocalLoss(alpha=CONFIG['focal_loss_alpha'], gamma=CONFIG['focal_loss_gamma']),
                metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.AUC(name='pr_auc', curve='PR')]
            )
            model.load_weights(best_phase1)
            print("✅ Loaded best phase-1 checkpoint (Phase 1 is better)")
    else:
        # eval_only: rebuild architecture then load weights
        model, base_model = create_model()
        # Try Phase 2 weights (saved while backbone was partially unfrozen - must match that state)
        phase2_loaded = False
        for wname in ['best_phase2.weights.h5', 'best_phase2_weights.h5', 'best_phase2.h5']:
            wpath = os.path.join(results_dir, wname)
            if os.path.exists(wpath):
                # Recreate Phase 2 trainability to match saved weight layout
                base_model.trainable = True
                fine_tune_at = max(0, len(base_model.layers) - CONFIG['fine_tune_trainable_layers'])
                for layer in base_model.layers[:fine_tune_at]:
                    layer.trainable = False
                model.load_weights(wpath)
                base_model.trainable = False  # freeze for inference
                print(f"\n\u2705 Loaded fine-tuned weights: {wname}")
                phase2_loaded = True
                break
        if not phase2_loaded:
            for wname in ['best_phase1.weights.h5', 'best_phase1_weights.h5', 'best_phase1.h5']:
                wpath = os.path.join(results_dir, wname)
                if os.path.exists(wpath):
                    model.load_weights(wpath)
                    print(f"\n\u2705 Loaded phase-1 weights: {wname}")
                    break
    
    # Evaluation
    print(f"\n" + "="*70)
    print("FINAL EVALUATION ON TEST SET")
    print("="*70)

    # Find threshold on validation set with recall constraint
    val_probs = model.predict(val_ds).flatten()
    if CONFIG['tta_flip']:
        val_ds_flip = create_flipped_dataset(
            val_df['full_path'].values,
            val_df['suspicious_label'].values,
            CONFIG['batch_size']
        )
        val_probs_flip = model.predict(val_ds_flip).flatten()
        val_probs = (val_probs + val_probs_flip) / 2.0

    val_labels = val_df['suspicious_label'].values
    best_threshold, val_recall_or_fallback, recall_target_met = choose_threshold_with_recall_constraint(
        val_labels, val_probs, min_recall=CONFIG['recall_target']
    )
    if recall_target_met:
        print(
            f"\n🔍 Best threshold from validation (recall>={CONFIG['recall_target']:.2f}): "
            f"{best_threshold:.3f} (val_recall={val_recall_or_fallback:.4f})"
        )
    else:
        print(
            f"\n⚠️ Could not meet recall target {CONFIG['recall_target']:.2f} on validation. "
            f"Using fallback threshold {best_threshold:.3f} (best val_recall={val_recall_or_fallback:.4f})"
        )

    y_pred_probs = model.predict(test_ds).flatten()
    if CONFIG['tta_flip']:
        test_ds_flip = create_flipped_dataset(
            test_df['full_path'].values,
            test_df['suspicious_label'].values,
            CONFIG['batch_size']
        )
        y_pred_probs_flip = model.predict(test_ds_flip).flatten()
        y_pred_probs = (y_pred_probs + y_pred_probs_flip) / 2.0

    y_pred = (y_pred_probs > best_threshold).astype(int)
    y_true = test_df['suspicious_label'].values
    
    # Metrics
    print("\n📈 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-suspicious', 'Suspicious'], digits=4))
    
    # ROC-AUC
    auc = roc_auc_score(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    print(f"\n🎯 ROC-AUC Score: {auc:.4f}")
    print(f"🎯 PR-AUC Score: {pr_auc:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-suspicious', 'Suspicious'],
                yticklabels=['Non-suspicious', 'Suspicious'])
    plt.title('Confusion Matrix - Suspicious vs Non-suspicious')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Suspicious vs Non-suspicious')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300)
    plt.close()
    
    # Save results
    # Make a JSON-serializable copy of CONFIG (convert any tensors)
    config_for_json = {}
    for k, v in CONFIG.items():
        if isinstance(v, (tf.Tensor, np.ndarray)):
            config_for_json[k] = float(v) if v.shape == () else v.tolist()
        elif isinstance(v, dict):
            config_for_json[k] = {kk: (float(vv) if isinstance(vv, (tf.Tensor, np.ndarray)) and vv.shape == () else vv.tolist() if isinstance(vv, (tf.Tensor, np.ndarray)) else vv) for kk, vv in v.items()}
        else:
            config_for_json[k] = v
    
    results = {
        'config': config_for_json,
        'test_accuracy': float(np.mean(y_pred == y_true)),
        'roc_auc': float(auc),
        'pr_auc': float(pr_auc),
        'best_threshold': float(best_threshold),
        'val_recall_at_best_threshold': float(val_recall_or_fallback),
        'recall_target': float(CONFIG['recall_target']),
        'recall_target_met': bool(recall_target_met),
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            'train': dict(Counter(train_df['suspicious_label'])),
            'val': dict(Counter(val_df['suspicious_label'])),
            'test': dict(Counter(test_df['suspicious_label']))
        }
    }
    
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
        f.write("GDC SUSPICIOUS vs NON-SUSPICIOUS CLASSIFICATION\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Test Accuracy: {results['test_accuracy']:.4f}\n")
        f.write(f"ROC-AUC: {auc:.4f}\n\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=['Non-suspicious', 'Suspicious'], digits=4))
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    
    print(f"\n✅ Results saved to {results_dir}/")
    print(f"📊 Model ready for TFLite conversion!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate existing model')
    args = parser.parse_args()
    
    main(eval_only=args.eval_only)
