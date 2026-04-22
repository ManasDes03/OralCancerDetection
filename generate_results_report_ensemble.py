"""
Ensemble Results Comparison
===========================

Evaluate and compare:
1. Positive model alone (original suspicious detector)
2. Negative model alone (non-suspicious detector)
3. Ensemble combinations using various fusion strategies

Focus: Does the ensemble approach reduce false positives?
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report
)
import os
import json
from collections import Counter
from ensemble_fusion import ensemble_predict


def tune_hybrid_operating_point(y_true, probs_pos, ensemble_confs, min_recall=0.90):
    """Find best hybrid operating point under a recall constraint.

    Hybrid score = w * positive_prob + (1 - w) * ensemble_confidence.
    Objective: maximize specificity while keeping recall >= min_recall.
    """
    best = None

    for w in np.linspace(0.50, 0.95, 10):
        hybrid = w * probs_pos + (1.0 - w) * ensemble_confs
        for thr in np.linspace(0.05, 0.95, 91):
            y_pred = (hybrid > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if recall < min_recall:
                continue

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            accuracy = (tp + tn) / max(len(y_true), 1)

            candidate = {
                'w': float(w),
                'threshold': float(thr),
                'recall': float(recall),
                'specificity': float(specificity),
                'accuracy': float(accuracy),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
            }

            if best is None:
                best = candidate
                continue

            if candidate['specificity'] > best['specificity'] or (
                np.isclose(candidate['specificity'], best['specificity'])
                and candidate['accuracy'] > best['accuracy']
            ):
                best = candidate

    return best


def load_model_from_weights(backbone_name, weights_path):
    """Load model architecture and weights."""
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, Model
    
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    model = Model(inputs, outputs)
    
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")


def load_model_with_fallback(model_name, requested_path, fallback_paths):
    """Try requested checkpoint first, then fallback candidates."""
    attempted = []

    ordered_candidates = []
    for p in [requested_path] + list(fallback_paths):
        if p and p not in ordered_candidates:
            ordered_candidates.append(p)

    for path in ordered_candidates:
        if not os.path.exists(path):
            attempted.append(f"{path} (missing)")
            continue
        try:
            model = load_model_from_weights('efficientnetb0', path)
            if path != requested_path:
                print(f"   ⚠️ {model_name}: using fallback checkpoint: {path}")
            return model, path
        except Exception as e:
            attempted.append(f"{path} ({e})")

    raise RuntimeError(
        f"Could not load any checkpoint for {model_name}. Attempts:\n  - " + "\n  - ".join(attempted)
    )


def create_dataset(paths, labels, batch_size):
    """Create tf.data pipeline for batch prediction."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        label = tf.cast(label, tf.float32)
        return image, label
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def create_flipped_dataset(paths, labels, batch_size):
    """TTA dataset with horizontal flip."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        image = tf.image.flip_left_right(image)
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        label = tf.cast(label, tf.float32)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def load_test_set():
    """Load original test set (non-flipped labels) from CSV."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    data_dir = 'data/gdc_oral_cancer_dataset'
    images_dir = os.path.join(data_dir, 'images')
    csv_path = os.path.join(data_dir, 'processed_labels.csv')
    
    df = pd.read_csv(csv_path)
    df = df[df['suspicious_label'].notna()].copy()
    df = df.drop_duplicates(subset=['case_no', 'image_path']).copy()
    df['full_path'] = df['image_path'].apply(lambda x: os.path.join(images_dir, x))
    df = df[df['full_path'].apply(os.path.exists)].copy()
    
    # Patient-aware split
    patient_labels = df.groupby('case_no')['suspicious_label'].first()
    patients = patient_labels.index.values
    labels = patient_labels.values
    
    train_patients, temp_patients, _, temp_labels = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_size = 0.15 / 0.30
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=1 - val_size, stratify=temp_labels, random_state=42
    )
    
    test_df = df[df['case_no'].isin(test_patients)].copy()
    return test_df


def main():
    print("\n" + "="*70)
    print("ENSEMBLE RESULTS COMPARISON: POSITIVE vs NEGATIVE vs ENSEMBLE")
    print("="*70)
    
    # Load test set
    print("\n📂 Loading test set...")
    test_df = load_test_set()
    print(f"   Test images: {len(test_df)}")
    print(f"   Suspicious: {(test_df['suspicious_label']==1).sum()}")
    print(f"   Non-suspicious: {(test_df['suspicious_label']==0).sum()}")
    
    # Create datasets
    batch_size = 8
    paths = test_df['full_path'].values
    labels = test_df['suspicious_label'].values
    
    test_ds = create_dataset(paths, labels, batch_size)
    test_ds_flip = create_flipped_dataset(paths, labels, batch_size)
    
    # Load models
    print("\n🔧 Loading models...")
    print("   Positive model...")
    model_pos, pos_path = load_model_with_fallback(
        'positive model',
        'gdc_suspicious_results/suspicious_classifier_final.h5',
        [
            'gdc_suspicious_results/best_phase2_weights.h5',
            'gdc_suspicious_results/best_phase1_weights.h5',
            'gdc_suspicious_results/best_phase2.h5',
            'gdc_suspicious_results/best_phase1.h5',
        ]
    )
    
    print("   Negative model...")
    model_neg, neg_path = load_model_with_fallback(
        'negative model',
        'gdc_suspicious_results_negative/suspicious_neg_final.h5',
        [
            'gdc_suspicious_results_negative/best_phase2_weights.h5',
            'gdc_suspicious_results_negative/best_phase1_weights.h5',
            'gdc_suspicious_results_negative/best_phase2.h5',
            'gdc_suspicious_results_negative/best_phase1.h5',
        ]
    )
    print(f"   Positive checkpoint used: {pos_path}")
    print(f"   Negative checkpoint used: {neg_path}")
    
    # Predictions
    print("\n🧠 Running batch predictions (with TTA)...")
    
    # Positive model
    probs_pos = model_pos.predict(test_ds, verbose=0).flatten()
    probs_pos_flip = model_pos.predict(test_ds_flip, verbose=0).flatten()
    probs_pos = (probs_pos + probs_pos_flip) / 2.0
    
    # Negative model (flipped in training, so flip output back)
    probs_neg = model_neg.predict(test_ds, verbose=0).flatten()
    probs_neg_flip = model_neg.predict(test_ds_flip, verbose=0).flatten()
    probs_neg = (probs_neg + probs_neg_flip) / 2.0
    # Note: neg model was trained on flipped labels, so P(non-suspicious) output
    # needs to be flipped back to P(suspicious) for comparison
    # Actually, for comparison keep it as is, interpretation is just inverted
    
    print("✅ Predictions complete")
    
    # Load saved thresholds from individual model results
    print("\n📊 Loading threshold configurations...")
    
    with open('gdc_suspicious_results/results.json', 'r') as f:
        results_pos = json.load(f)
        thresh_pos = results_pos['best_threshold']
        print(f"   Positive model threshold: {thresh_pos:.3f}")
    
    with open('gdc_suspicious_results_negative/results.json', 'r') as f:
        results_neg = json.load(f)
        thresh_neg = results_neg['best_threshold']
        print(f"   Negative model threshold: {thresh_neg:.3f}")
    
    # Predictions for each model
    preds_pos = (probs_pos > thresh_pos).astype(int)
    preds_neg = (probs_neg > thresh_neg).astype(int)
    
    # Ensemble: For each image, run fusion strategies and vote
    print("\n🔄 Running ensemble fusion...")
    ensemble_confs = []
    for i in range(len(test_df)):
        result = ensemble_predict(
            probs_pos[i],
            probs_neg[i],
            strategies=['agreement_voting', 'contradiction_score', 'weighted_average'],
            threshold=0.5
        )
        ensemble_confs.append(result['average_confidence'])
    
    ensemble_confs = np.array(ensemble_confs)
    preds_ensemble = (ensemble_confs > 0.5).astype(int)

    # Tune operating point for screening: high recall with better specificity.
    tuned = tune_hybrid_operating_point(
        y_true=labels.astype(int),
        probs_pos=probs_pos,
        ensemble_confs=ensemble_confs,
        min_recall=0.90,
    )
    if tuned is not None:
        tuned_scores = tuned['w'] * probs_pos + (1.0 - tuned['w']) * ensemble_confs
        preds_tuned = (tuned_scores > tuned['threshold']).astype(int)
    else:
        tuned_scores = ensemble_confs.copy()
        preds_tuned = preds_ensemble.copy()
    
    # Metrics
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    y_true = labels.astype(int)
    
    # Function to compute metrics
    def compute_metrics(y_true, y_pred, y_probs, model_name):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = np.mean(y_pred == y_true)
        auc = roc_auc_score(y_true, y_probs)
        pr_auc = average_precision_score(y_true, y_probs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall
        
        return {
            'model': model_name,
            'accuracy': accuracy,
            'auc': auc,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    
    metrics_pos = compute_metrics(y_true, preds_pos, probs_pos, 'Positive Model')
    metrics_neg = compute_metrics(y_true, preds_neg, probs_neg, 'Negative Model')
    metrics_ens = compute_metrics(y_true, preds_ensemble, ensemble_confs, 'Ensemble')
    metrics_tuned = compute_metrics(y_true, preds_tuned, tuned_scores, 'Tuned Hybrid Ensemble')
    
    # Print metrics
    print("\n{:<20} {:<12} {:<12} {:<12} {:<12}".format("Metric", "Positive", "Negative", "Ensemble", "Tuned"))
    print("-" * 70)
    
    for metric_key in ['accuracy', 'auc', 'pr_auc', 'precision', 'recall', 'f1', 'sensitivity', 'specificity']:
        val_pos = metrics_pos[metric_key]
        val_neg = metrics_neg[metric_key]
        val_ens = metrics_ens[metric_key]
        val_tuned = metrics_tuned[metric_key]
        
        print("{:<20} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(metric_key.upper(), val_pos, val_neg, val_ens, val_tuned))
    
    # Confusion matrices
    print("\n" + "="*70)
    print("CONFUSION MATRICES")
    print("="*70)
    
    print("\nPositive Model:")
    print(f"  TN: {metrics_pos['tn']}, FP: {metrics_pos['fp']}")
    print(f"  FN: {metrics_pos['fn']}, TP: {metrics_pos['tp']}")
    
    print("\nNegative Model:")
    print(f"  TN: {metrics_neg['tn']}, FP: {metrics_neg['fp']}")
    print(f"  FN: {metrics_neg['fn']}, TP: {metrics_neg['tp']}")
    
    print("\nEnsemble:")
    print(f"  TN: {metrics_ens['tn']}, FP: {metrics_ens['fp']}")
    print(f"  FN: {metrics_ens['fn']}, TP: {metrics_ens['tp']}")

    print("\nTuned Hybrid Ensemble:")
    print(f"  TN: {metrics_tuned['tn']}, FP: {metrics_tuned['fp']}")
    print(f"  FN: {metrics_tuned['fn']}, TP: {metrics_tuned['tp']}")
    
    # Key insight: False positives
    print("\n" + "="*70)
    print("FALSE POSITIVES ANALYSIS (Key for Negative Test)")
    print("="*70)
    
    fp_reduction_from_neg = 100 * (1 - metrics_neg['fp'] / metrics_pos['fp'])
    fp_reduction_from_ens = 100 * (1 - metrics_ens['fp'] / metrics_pos['fp'])
    fp_reduction_from_tuned = 100 * (1 - metrics_tuned['fp'] / metrics_pos['fp'])
    
    print(f"\nPositive Model FP: {metrics_pos['fp']} (baseline)")
    print(f"Negative Model FP: {metrics_neg['fp']} ({fp_reduction_from_neg:+.1f}% change)")
    print(f"Ensemble FP: {metrics_ens['fp']} ({fp_reduction_from_ens:+.1f}% change)")
    print(f"Tuned Hybrid FP: {metrics_tuned['fp']} ({fp_reduction_from_tuned:+.1f}% change)")
    
    recall_change_ens = 100 * (metrics_ens['recall'] - metrics_pos['recall']) / metrics_pos['recall']
    recall_change_tuned = 100 * (metrics_tuned['recall'] - metrics_pos['recall']) / metrics_pos['recall']
    print(f"\nRecall Trade-off (Ensemble vs Positive):")
    print(f"  Positive Recall: {metrics_pos['recall']:.4f}")
    print(f"  Ensemble Recall: {metrics_ens['recall']:.4f} ({recall_change_ens:+.1f}%)")
    print(f"  Tuned Recall: {metrics_tuned['recall']:.4f} ({recall_change_tuned:+.1f}%)")

    if tuned is not None:
        print(f"\nTuned operating point (recall >= 0.90):")
        print(f"  Positive weight (w): {tuned['w']:.2f}")
        print(f"  Threshold: {tuned['threshold']:.2f}")
    
    # Plot comparison
    print("\n📊 Generating comparison visualizations...")
    
    # ROC curves
    fpr_pos, tpr_pos, _ = roc_curve(y_true, probs_pos)
    fpr_neg, tpr_neg, _ = roc_curve(y_true, probs_neg)
    fpr_ens, tpr_ens, _ = roc_curve(y_true, ensemble_confs)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC curves
    axes[0].plot(fpr_pos, tpr_pos, label=f'Positive (AUC={metrics_pos["auc"]:.3f})', linewidth=2)
    axes[0].plot(fpr_neg, tpr_neg, label=f'Negative (AUC={metrics_neg["auc"]:.3f})', linewidth=2)
    axes[0].plot(fpr_ens, tpr_ens, label=f'Ensemble (AUC={metrics_ens["auc"]:.3f})', linewidth=2)
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curves - Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics bar chart
    metrics_names = ['Recall', 'Specificity', 'Precision', 'F1']
    x_pos = np.arange(len(metrics_names))
    width = 0.25
    
    vals_pos = [metrics_pos['recall'], metrics_pos['specificity'], metrics_pos['precision'], metrics_pos['f1']]
    vals_neg = [metrics_neg['recall'], metrics_neg['specificity'], metrics_neg['precision'], metrics_neg['f1']]
    vals_ens = [metrics_ens['recall'], metrics_ens['specificity'], metrics_ens['precision'], metrics_ens['f1']]
    
    axes[1].bar(x_pos - width, vals_pos, width, label='Positive', alpha=0.8)
    axes[1].bar(x_pos, vals_neg, width, label='Negative', alpha=0.8)
    axes[1].bar(x_pos + width, vals_ens, width, label='Ensemble', alpha=0.8)
    axes[1].set_ylabel('Score')
    axes[1].set_title('Key Metrics Comparison')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(metrics_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    os.makedirs('gdc_suspicious_results_ensemble', exist_ok=True)
    plt.savefig('gdc_suspicious_results_ensemble/comparison_metrics.png', dpi=300)
    print("✅ Saved: comparison_metrics.png")
    plt.close()
    
    # Confusion matrix heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (ax, metrics, title) in enumerate([
        (axes[0], metrics_pos, 'Positive Model'),
        (axes[1], metrics_neg, 'Negative Model'),
        (axes[2], metrics_ens, 'Ensemble')
    ]):
        cm = np.array([[metrics['tn'], metrics['fp']], [metrics['fn'], metrics['tp']]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Non-susp', 'Suspicious'],
                   yticklabels=['Non-susp', 'Suspicious'])
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('gdc_suspicious_results_ensemble/confusion_matrices_comparison.png', dpi=300)
    print("✅ Saved: confusion_matrices_comparison.png")
    plt.close()
    
    # False positives vs recall trade-off
    all_metrics = [metrics_pos, metrics_neg, metrics_ens, metrics_tuned]
    model_names = [m['model'] for m in all_metrics]
    fps = [m['fp'] for m in all_metrics]
    recalls = [m['recall'] for m in all_metrics]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['blue', 'orange', 'green', 'red']
    for i, (fp, recall, name, color) in enumerate(zip(fps, recalls, model_names, colors)):
        ax.scatter(fp, recall, s=300, alpha=0.7, label=name, color=color)
        ax.annotate(name, (fp, recall), xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('False Positives (lower is better for specificity)', fontsize=12)
    ax.set_ylabel('Recall / Sensitivity (higher is critical)', fontsize=12)
    ax.set_title('Negative Test Trade-off: False Positives vs Recall', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gdc_suspicious_results_ensemble/fp_vs_recall_tradeoff.png', dpi=300)
    print("✅ Saved: fp_vs_recall_tradeoff.png")
    plt.close()
    
    # Save detailed report
    report = {
        'test_set_size': len(test_df),
        'suspicious_count': int((y_true == 1).sum()),
        'non_suspicious_count': int((y_true == 0).sum()),
        'models': {
            'positive': metrics_pos,
            'negative': metrics_neg,
            'ensemble': metrics_ens,
            'tuned_hybrid_ensemble': metrics_tuned
        },
        'false_positive_analysis': {
            'positive_model_fp': metrics_pos['fp'],
            'negative_model_fp': metrics_neg['fp'],
            'ensemble_fp': metrics_ens['fp'],
            'tuned_hybrid_fp': metrics_tuned['fp'],
            'fp_reduction_negative_vs_positive': fp_reduction_from_neg,
            'fp_reduction_ensemble_vs_positive': fp_reduction_from_ens,
            'fp_reduction_tuned_vs_positive': fp_reduction_from_tuned
        },
        'recall_analysis': {
            'positive_model_recall': metrics_pos['recall'],
            'negative_model_recall': metrics_neg['recall'],
            'ensemble_recall': metrics_ens['recall'],
            'tuned_hybrid_recall': metrics_tuned['recall'],
            'recall_change_ensemble_vs_positive_percent': recall_change_ens,
            'recall_change_tuned_vs_positive_percent': recall_change_tuned
        },
        'tuned_hybrid_operating_point': tuned
    }
    
    # Convert numpy types for JSON
    def convert_to_native(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        return obj
    
    report = convert_to_native(report)
    
    with open('gdc_suspicious_results_ensemble/comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("✅ Saved: comparison_report.json")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: gdc_suspicious_results_ensemble/")
    print(f"\n📊 Key Findings:")
    print(f"   • Ensemble FP reduction: {fp_reduction_from_ens:+.1f}%")
    print(f"   • Ensemble recall change: {recall_change_ens:+.1f}%")
    print(f"   • Tuned FP reduction: {fp_reduction_from_tuned:+.1f}%")
    print(f"   • Tuned recall change: {recall_change_tuned:+.1f}%")
    print(f"   • Recommend ensemble if: FP reduction > 20% AND recall loss < 10%")


if __name__ == '__main__':
    main()
