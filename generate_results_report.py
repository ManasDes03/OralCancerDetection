"""
generate_results_report.py
==========================
Generates a full, presentation-ready results report for the GDC oral cancer
two-stage classifier (Stage 1: Suspicious, Stage 2: Risk).

Outputs  →  results_presentation/
    01_confusion_matrices.png     side-by-side CMs for both stages
    02_roc_curves.png             overlaid ROC curves
    03_pr_curves.png              overlaid Precision-Recall curves
    04_summary_table.png          key metrics table
    05_pipeline_overview.png      mini CMs + bar chart in one figure
    06_suspicious_training_history.png   (copied from gdc_suspicious_results if present)
    07_risk_training_history.png         (copied from gdc_risk_results if present)
    report_summary.txt            plain-text summary

Usage:
    conda run -n efficientnet_env --no-capture-output python generate_results_report.py
"""

import os, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, precision_recall_curve,
    average_precision_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, mixed_precision

# ── Setup ──────────────────────────────────────────────────────────────────────
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)
np.random.seed(42)
tf.random.set_seed(42)

OUTPUT_DIR = 'results_presentation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Consistent presentation style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Paths ───────────────────────────────────────────────────────────────────────
DATA_DIR   = 'data/gdc_oral_cancer_dataset'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
CSV_PATH   = os.path.join(DATA_DIR, 'processed_labels.csv')
SUSP_DIR   = 'gdc_suspicious_results'
RISK_DIR   = 'gdc_risk_results'

IMG_H = IMG_W = 224
BATCH = 8
SUSP_FT_LAYERS = 60   # must match train_gdc_suspicious.py CONFIG
RISK_FT_LAYERS = 30   # must match train_gdc_risk.py CONFIG


# ── Focal Loss (required to rebuild Keras graphs) ───────────────────────────────
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        eps = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, eps, 1.0 - eps)
        y_true = tf.cast(y_true, tf.float32)
        l1 = -self.alpha * keras.backend.pow(1 - y_pred, self.gamma) * y_true * keras.backend.log(y_pred)
        l0 = -(1 - self.alpha) * keras.backend.pow(y_pred, self.gamma) * (1 - y_true) * keras.backend.log(1 - y_pred)
        return l1 + l0

    def get_config(self):
        return {'alpha': self.alpha, 'gamma': self.gamma}


# ── Model builders (architectures must match training scripts exactly) ──────────
def build_suspicious_model():
    """EfficientNetB0 → GAP → BN → D(256) → BN → D(96) → sigmoid"""
    base = keras.applications.EfficientNetB0(
        input_shape=(IMG_H, IMG_W, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(shape=(IMG_H, IMG_W, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inp, out), base


def build_risk_model():
    """EfficientNetB0 → GAP → BN → D(256) → BN → D(128) → sigmoid"""
    base = keras.applications.EfficientNetB0(
        input_shape=(IMG_H, IMG_W, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(shape=(IMG_H, IMG_W, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inp, out), base


# ── Weight loader ───────────────────────────────────────────────────────────────
def load_weights(model, base_model, results_dir, ft_layers):
    """Load best available checkpoint matching the correct trainability state."""
    for wname in ['best_phase2_weights.h5', 'best_phase2.h5']:
        p = os.path.join(results_dir, wname)
        if os.path.exists(p):
            # Recreate Phase 2 trainability layout before loading
            base_model.trainable = True
            ft_at = max(0, len(base_model.layers) - ft_layers)
            for layer in base_model.layers[:ft_at]:
                layer.trainable = False
            model.load_weights(p)
            base_model.trainable = False   # freeze for inference
            print(f'  ✅ Loaded {wname}')
            return True
    for wname in ['best_phase1_weights.h5', 'best_phase1.h5']:
        p = os.path.join(results_dir, wname)
        if os.path.exists(p):
            # Phase 1 weights were saved with backbone fully frozen
            base_model.trainable = False
            model.load_weights(p)
            print(f'  ✅ Loaded {wname}')
            return True
    print(f'  ⚠️  No checkpoint found in {results_dir}')
    return False


# ── Dataset factory ─────────────────────────────────────────────────────────────
def make_ds(paths, labels):
    preprocess_fn = keras.applications.efficientnet.preprocess_input

    def parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_H, IMG_W])
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        return img, tf.cast(label, tf.float32)

    return (tf.data.Dataset.from_tensor_slices((np.array(paths), np.array(labels)))
            .map(parse, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))


def make_ds_flip(paths, labels):
    """Horizontally-flipped dataset for TTA."""
    preprocess_fn = keras.applications.efficientnet.preprocess_input

    def parse(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [IMG_H, IMG_W])
        img = tf.image.flip_left_right(img)
        img = tf.cast(img, tf.float32)
        img = preprocess_fn(img)
        return img, tf.cast(label, tf.float32)

    return (tf.data.Dataset.from_tensor_slices((np.array(paths), np.array(labels)))
            .map(parse, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH)
            .prefetch(tf.data.AUTOTUNE))


# ── Patient-stratified split ────────────────────────────────────────────────────
def get_splits(df, label_col):
    """Reproduce the exact same val/test split used during training (seed=42)."""
    patient_labels = df.groupby('case_no')[label_col].first()
    patients = patient_labels.index.values
    labels   = patient_labels.values

    _, temp_p, _, temp_l = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=42)
    val_p, test_p = train_test_split(
        temp_p, test_size=0.50, stratify=temp_l, random_state=42)

    return df[df['case_no'].isin(val_p)], df[df['case_no'].isin(test_p)]


# ── TTA predict ─────────────────────────────────────────────────────────────────
def predict_tta(model, paths, labels):
    ds      = make_ds(paths, labels)
    ds_flip = make_ds_flip(paths, labels)
    p1 = model.predict(ds,      verbose=0).flatten()
    p2 = model.predict(ds_flip, verbose=0).flatten()
    return (p1 + p2) / 2.0


# ── F2 threshold (recall-biased, capped at 0.45) ───────────────────────────────
def best_threshold_f2(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-8)
    f2 = f2[:-1]
    best_idx = int(np.argmax(f2))
    return float(np.clip(thresholds[best_idx], 0.05, 0.45))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    preprocess_fn = keras.applications.efficientnet.preprocess_input

    # ── Load and clean data ──────────────────────────────────────────────────
    print("📂 Loading dataset...")
    df_raw = pd.read_csv(CSV_PATH)
    df_raw['full_path'] = df_raw['image_path'].apply(
        lambda x: os.path.join(IMAGES_DIR, x))

    # ── Build & load Suspicious model ────────────────────────────────────────
    print("\n⏳ Building Suspicious model...")
    susp_model, susp_base = build_suspicious_model()
    load_weights(susp_model, susp_base, SUSP_DIR, SUSP_FT_LAYERS)

    # ── Build & load Risk model ───────────────────────────────────────────────
    print("\n⏳ Building Risk model...")
    risk_model, risk_base = build_risk_model()
    load_weights(risk_model, risk_base, RISK_DIR, RISK_FT_LAYERS)

    # ── Suspicious evaluation ────────────────────────────────────────────────
    print("\n📊 Evaluating Stage 1 — Suspicious...")
    df_s = (df_raw[df_raw['suspicious_label'].notna()]
            .drop_duplicates(subset=['case_no', 'image_path'])
            .copy())
    df_s = df_s[df_s['full_path'].apply(os.path.exists)].copy()

    val_s, test_s = get_splits(df_s, 'suspicious_label')

    val_s_probs  = predict_tta(susp_model, val_s['full_path'].values,  val_s['suspicious_label'].values)
    thr_s        = best_threshold_f2(val_s['suspicious_label'].values, val_s_probs)
    test_s_probs = predict_tta(susp_model, test_s['full_path'].values, test_s['suspicious_label'].values)
    y_true_s = test_s['suspicious_label'].values.astype(int)
    y_pred_s = (test_s_probs >= thr_s).astype(int)

    roc_auc_s = roc_auc_score(y_true_s, test_s_probs)
    pr_auc_s  = average_precision_score(y_true_s, test_s_probs)
    cm_s      = confusion_matrix(y_true_s, y_pred_s)
    fpr_s, tpr_s, _ = roc_curve(y_true_s, test_s_probs)
    prec_s, rec_s, _ = precision_recall_curve(y_true_s, test_s_probs)

    # ── Risk evaluation ──────────────────────────────────────────────────────
    print("\n📊 Evaluating Stage 2 — Risk...")
    df_r = (df_raw[df_raw['risk_label'].notna()]
            .drop_duplicates(subset=['case_no', 'image_path'])
            .copy())
    df_r = df_r[df_r['full_path'].apply(os.path.exists)].copy()

    val_r, test_r = get_splits(df_r, 'risk_label')

    val_r_probs  = predict_tta(risk_model, val_r['full_path'].values,  val_r['risk_label'].values)
    thr_r        = best_threshold_f2(val_r['risk_label'].values, val_r_probs)
    test_r_probs = predict_tta(risk_model, test_r['full_path'].values, test_r['risk_label'].values)
    y_true_r = test_r['risk_label'].values.astype(int)
    y_pred_r = (test_r_probs >= thr_r).astype(int)

    roc_auc_r = roc_auc_score(y_true_r, test_r_probs)
    pr_auc_r  = average_precision_score(y_true_r, test_r_probs)
    cm_r      = confusion_matrix(y_true_r, y_pred_r)
    fpr_r, tpr_r, _ = roc_curve(y_true_r, test_r_probs)
    prec_r, rec_r, _ = precision_recall_curve(y_true_r, test_r_probs)

    # Derived metrics
    susp_recall = cm_s[1, 1] / max(cm_s[1].sum(), 1)
    susp_prec_v = cm_s[1, 1] / max(cm_s[:, 1].sum(), 1)
    susp_f1     = 2 * susp_prec_v * susp_recall / max(susp_prec_v + susp_recall, 1e-8)
    risk_recall = cm_r[1, 1] / max(cm_r[1].sum(), 1)
    risk_prec_v = cm_r[1, 1] / max(cm_r[:, 1].sum(), 1)
    risk_f1     = 2 * risk_prec_v * risk_recall / max(risk_prec_v + risk_recall, 1e-8)
    acc_s = float(np.mean(y_pred_s == y_true_s))
    acc_r = float(np.mean(y_pred_r == y_true_r))

    # ── Print classification reports ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 — SUSPICIOUS CLASSIFIER  (test set)")
    print("=" * 70)
    print(classification_report(y_true_s, y_pred_s,
          target_names=['Non-suspicious', 'Suspicious'], digits=4))
    print(f"  ROC-AUC : {roc_auc_s:.4f}   PR-AUC : {pr_auc_s:.4f}   Threshold : {thr_s:.3f}")

    print("\n" + "=" * 70)
    print("STAGE 2 — RISK CLASSIFIER  (test set)")
    print("=" * 70)
    print(classification_report(y_true_r, y_pred_r,
          target_names=['Low-risk', 'High-risk'], digits=4))
    print(f"  ROC-AUC : {roc_auc_r:.4f}   PR-AUC : {pr_auc_r:.4f}   Threshold : {thr_r:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 1 — Side-by-side Confusion Matrices
    # ─────────────────────────────────────────────────────────────────────────
    print("\n🎨 Generating charts...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GDC Oral Cancer — Confusion Matrices (Test Set)',
                 fontsize=15, fontweight='bold', y=1.01)

    sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Non-suspicious', 'Suspicious'],
                yticklabels=['Non-suspicious', 'Suspicious'],
                linewidths=0.5, linecolor='#cccccc', cbar=True, annot_kws={'size': 15})
    axes[0].set_title(
        f'Stage 1 — Suspicious Classifier\nROC-AUC={roc_auc_s:.3f}  '
        f'Recall={susp_recall:.3f}  Threshold={thr_s:.3f}', pad=10)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['Low-risk', 'High-risk'],
                yticklabels=['Low-risk', 'High-risk'],
                linewidths=0.5, linecolor='#cccccc', cbar=True, annot_kws={'size': 15})
    axes[1].set_title(
        f'Stage 2 — Risk Classifier\nROC-AUC={roc_auc_r:.3f}  '
        f'Recall={risk_recall:.3f}  Threshold={thr_r:.3f}', pad=10)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    _save(fig, '01_confusion_matrices.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 2 — Overlaid ROC Curves
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr_s, tpr_s, lw=2.5, color='steelblue',
            label=f'Stage 1 – Suspicious  (AUC = {roc_auc_s:.3f})')
    ax.plot(fpr_r, tpr_r, lw=2.5, color='crimson',
            label=f'Stage 2 – Risk  (AUC = {roc_auc_r:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.5, label='Random Classifier')
    ax.fill_between(fpr_s, tpr_s, alpha=0.08, color='steelblue')
    ax.fill_between(fpr_r, tpr_r, alpha=0.08, color='crimson')
    ax.set_xlabel('False Positive Rate (1 – Specificity)')
    ax.set_ylabel('True Positive Rate (Recall / Sensitivity)')
    ax.set_title('ROC Curves — GDC Oral Cancer Two-Stage Classifier')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    _save(fig, '02_roc_curves.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 3 — Overlaid Precision-Recall Curves
    # ─────────────────────────────────────────────────────────────────────────
    pos_rate_s = y_true_s.mean()
    pos_rate_r = y_true_r.mean()

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.step(rec_s, prec_s, lw=2.5, color='steelblue', where='post',
            label=f'Stage 1 – Suspicious  (AP = {pr_auc_s:.3f})')
    ax.step(rec_r, prec_r, lw=2.5, color='crimson', where='post',
            label=f'Stage 2 – Risk  (AP = {pr_auc_r:.3f})')
    ax.axhline(pos_rate_s, color='steelblue', lw=1.0, linestyle=':',
               alpha=0.7, label=f'Susp. baseline ({pos_rate_s:.2f})')
    ax.axhline(pos_rate_r, color='crimson', lw=1.0, linestyle=':',
               alpha=0.7, label=f'Risk baseline ({pos_rate_r:.2f})')
    ax.fill_between(rec_s, prec_s, alpha=0.07, color='steelblue')
    ax.fill_between(rec_r, prec_r, alpha=0.07, color='crimson')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves — GDC Oral Cancer Two-Stage Classifier')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    _save(fig, '03_pr_curves.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 4 — Metrics Summary Table
    # ─────────────────────────────────────────────────────────────────────────
    rows = [
        ['Stage 1 – Suspicious',
         f'{roc_auc_s:.4f}', f'{pr_auc_s:.4f}',
         f'{susp_recall:.4f}', f'{susp_prec_v:.4f}', f'{susp_f1:.4f}',
         f'{acc_s:.4f}', f'{thr_s:.3f}'],
        ['Stage 2 – High-Risk',
         f'{roc_auc_r:.4f}', f'{pr_auc_r:.4f}',
         f'{risk_recall:.4f}', f'{risk_prec_v:.4f}', f'{risk_f1:.4f}',
         f'{acc_r:.4f}', f'{thr_r:.3f}'],
    ]
    cols = ['Model', 'ROC-AUC', 'PR-AUC',
            'Recall\n(pos class)', 'Precision\n(pos class)', 'F1\n(pos class)',
            'Test Acc', 'Threshold\n(F2-based)']

    fig, ax = plt.subplots(figsize=(15, 2.8))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=cols, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10.5)
    tbl.scale(1.0, 2.4)
    for j in range(len(cols)):                          # header row
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(len(cols)):                          # Suspicious row
        tbl[1, j].set_facecolor('#D6EAF8')
    for j in range(len(cols)):                          # Risk row
        tbl[2, j].set_facecolor('#FADBD8')
    ax.set_title('GDC Oral Cancer Two-Stage Classifier — Test Set Performance',
                 fontsize=13, fontweight='bold', pad=20, loc='center')
    plt.tight_layout()
    _save(fig, '04_summary_table.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 5 — Pipeline Overview (mini CMs + bar chart)
    # ─────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('GDC Oral Cancer — Two-Stage Pipeline Overview',
                 fontsize=15, fontweight='bold')

    # GridSpec: 2 small CM columns + 1 wider bar chart
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.35)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    sns.heatmap(cm_s, annot=True, fmt='d', cmap='Blues', ax=ax0,
                xticklabels=['Non-susp', 'Susp'],
                yticklabels=['Non-susp', 'Susp'],
                linewidths=0.5, cbar=False, annot_kws={'size': 13})
    ax0.set_title(f'Stage 1: Suspicious\nAUC={roc_auc_s:.3f}  Recall={susp_recall:.3f}')
    ax0.set_ylabel('True'); ax0.set_xlabel('Predicted')

    sns.heatmap(cm_r, annot=True, fmt='d', cmap='Reds', ax=ax1,
                xticklabels=['Low', 'High'],
                yticklabels=['Low', 'High'],
                linewidths=0.5, cbar=False, annot_kws={'size': 13})
    ax1.set_title(f'Stage 2: Risk\nAUC={roc_auc_r:.3f}  Recall={risk_recall:.3f}')
    ax1.set_ylabel('True'); ax1.set_xlabel('Predicted')

    labels_bar = ['Susp\nRecall', 'Susp\nPrecision', 'Susp\nROC-AUC',
                  'Risk\nRecall', 'Risk\nPrecision', 'Risk\nROC-AUC']
    values_bar = [susp_recall, susp_prec_v, roc_auc_s,
                  risk_recall, risk_prec_v, roc_auc_r]
    colors_bar = ['#2E86AB', '#2E86AB', '#2E86AB', '#E84855', '#E84855', '#E84855']

    bars = ax2.bar(labels_bar, values_bar, color=colors_bar, width=0.55,
                   edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, values_bar):
        ax2.text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.012,
                 f'{val:.3f}', ha='center', va='bottom',
                 fontsize=10.5, fontweight='bold')
    ax2.set_ylim(0, 1.22)
    ax2.set_ylabel('Score')
    ax2.set_title('Key Metrics at a Glance')
    ax2.grid(axis='y', alpha=0.3)
    legend_elems = [
        mpatches.Patch(facecolor='#2E86AB', label='Stage 1 – Suspicious'),
        mpatches.Patch(facecolor='#E84855', label='Stage 2 – Risk'),
    ]
    ax2.legend(handles=legend_elems, loc='upper right', fontsize=10)

    plt.tight_layout()
    _save(fig, '05_pipeline_overview.png')

    # ─────────────────────────────────────────────────────────────────────────
    # CHART 6 — Probability distribution histograms
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Predicted Probability Distributions (Test Set)',
                 fontsize=14, fontweight='bold')

    for ax, probs, true, thr, title, c_pos, c_neg in [
        (axes[0], test_s_probs, y_true_s, thr_s,
         'Stage 1 – Suspicious', '#2980B9', '#BDC3C7'),
        (axes[1], test_r_probs, y_true_r, thr_r,
         'Stage 2 – Risk',       '#C0392B', '#BDC3C7'),
    ]:
        ax.hist(probs[true == 0], bins=30, alpha=0.65, color=c_neg,
                label='Negative class', edgecolor='white', linewidth=0.4)
        ax.hist(probs[true == 1], bins=30, alpha=0.65, color=c_pos,
                label='Positive class', edgecolor='white', linewidth=0.4)
        ax.axvline(thr, color='black', lw=1.8, linestyle='--',
                   label=f'Threshold = {thr:.3f}')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Count')
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    _save(fig, '06_probability_distributions.png')

    # ─────────────────────────────────────────────────────────────────────────
    # Copy existing training history PNGs
    # ─────────────────────────────────────────────────────────────────────────
    for src, dst in [
        (os.path.join(SUSP_DIR, 'combined_training_history.png'),
         '07_suspicious_training_history.png'),
        (os.path.join(RISK_DIR, 'combined_training_history.png'),
         '08_risk_training_history.png'),
    ]:
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(OUTPUT_DIR, dst))
            print(f'  ✅ Copied {dst}')

    # ─────────────────────────────────────────────────────────────────────────
    # Text summary
    # ─────────────────────────────────────────────────────────────────────────
    summary = f"""GDC ORAL CANCER — TWO-STAGE CLASSIFIER RESULTS
{'='*65}
Architecture  : EfficientNetB0, two-phase training (frozen → fine-tune)
Input         : 224×224 RGB  |  EfficientNet preprocessing
TTA           : horizontal flip averaging
Threshold     : F2-score maximisation (recall-biased), capped at 0.45
Training      : Focal loss  |  balanced batch interleaving  |  heavy minority augment

STAGE 1 — SUSPICIOUS vs NON-SUSPICIOUS
{'─'*65}
  Test ROC-AUC         : {roc_auc_s:.4f}
  Test PR-AUC          : {pr_auc_s:.4f}
  Classification threshold : {thr_s:.3f}
  Suspicious  Recall   : {susp_recall:.4f}
  Suspicious  Precision: {susp_prec_v:.4f}
  Suspicious  F1       : {susp_f1:.4f}
  Test Accuracy        : {acc_s:.4f}
  Confusion Matrix:
{cm_s}

STAGE 2 — HIGH-RISK vs LOW-RISK
{'─'*65}
  Test ROC-AUC         : {roc_auc_r:.4f}
  Test PR-AUC          : {pr_auc_r:.4f}
  Classification threshold : {thr_r:.3f}
  High-risk  Recall    : {risk_recall:.4f}
  High-risk  Precision : {risk_prec_v:.4f}
  High-risk  F1        : {risk_f1:.4f}
  Test Accuracy        : {acc_r:.4f}
  Confusion Matrix:
{cm_r}

{'='*65}
Charts saved to: {OUTPUT_DIR}/
"""
    print(summary)
    with open(os.path.join(OUTPUT_DIR, 'report_summary.txt'), 'w') as f:
        f.write(summary)
    print(f'  ✅ Saved report_summary.txt')
    print(f'\n🎉  All results saved to  {OUTPUT_DIR}/')


# ── Helper ──────────────────────────────────────────────────────────────────────
def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ Saved {filename}')


if __name__ == '__main__':
    main()
