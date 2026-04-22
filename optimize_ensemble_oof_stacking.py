"""
OOF Stacking Optimizer (Patient-Level CV)
=========================================

Goal:
- Improve ensemble robustness by training meta-learner on out-of-fold predictions.
- Keep test split untouched.
- Enforce high-recall operating point for screening.
- Provide statistical significance vs positive baseline.

Outputs:
- gdc_suspicious_results_ensemble/oof_stacking_report.json
- gdc_suspicious_results_ensemble/oof_stacking_summary.txt
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_model_from_weights(weights_path: str):
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, Model

    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = Model(inputs, outputs)
    model.load_weights(weights_path)
    return model


def load_model_with_fallback(model_name: str, requested_path: str, fallback_paths):
    attempts = []
    candidates = []
    for p in [requested_path] + list(fallback_paths):
        if p and p not in candidates:
            candidates.append(p)

    for path in candidates:
        if not os.path.exists(path):
            attempts.append(f"{path} (missing)")
            continue
        try:
            model = load_model_from_weights(path)
            if path != requested_path:
                print(f"  {model_name}: fallback -> {path}")
            else:
                print(f"  {model_name}: {path}")
            return model, path
        except Exception as e:
            attempts.append(f"{path} ({e})")

    raise RuntimeError(
        f"Could not load any checkpoint for {model_name}. Attempts:\n  - " + "\n  - ".join(attempts)
    )


def create_dataset(paths, labels, batch_size=8, flip=False):
    from tensorflow.keras.applications.efficientnet import preprocess_input

    def parse_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [224, 224])
        if flip:
            image = tf.image.flip_left_right(image)
        image = tf.cast(image, tf.float32)
        image = preprocess_input(image)
        label = tf.cast(label, tf.float32)
        return image, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def get_probs(model, paths, labels):
    ds = create_dataset(paths, labels, flip=False)
    ds_flip = create_dataset(paths, labels, flip=True)
    p = model.predict(ds, verbose=0).flatten()
    p_flip = model.predict(ds_flip, verbose=0).flatten()
    return (p + p_flip) / 2.0


def load_df():
    data_dir = "data/gdc_oral_cancer_dataset"
    images_dir = os.path.join(data_dir, "images")
    csv_path = os.path.join(data_dir, "processed_labels.csv")

    df = pd.read_csv(csv_path)
    df = df[df["suspicious_label"].notna()].copy()
    df = df.drop_duplicates(subset=["case_no", "image_path"]).copy()
    df["full_path"] = df["image_path"].apply(lambda x: os.path.join(images_dir, x))
    df = df[df["full_path"].apply(os.path.exists)].copy()
    return df


def patient_split(df):
    patient_labels = df.groupby("case_no")["suspicious_label"].first()
    patients = patient_labels.index.values
    labels = patient_labels.values

    train_patients, temp_patients, train_labels, temp_labels = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_size = 0.15 / 0.30
    val_patients, test_patients, _, _ = train_test_split(
        temp_patients, temp_labels, test_size=1 - val_size, stratify=temp_labels, random_state=SEED
    )

    train_df = df[df["case_no"].isin(train_patients)].copy()
    val_df = df[df["case_no"].isin(val_patients)].copy()
    test_df = df[df["case_no"].isin(test_patients)].copy()
    return train_df, val_df, test_df


@dataclass
class EvalResult:
    name: str
    threshold: float
    auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int


def evaluate(y_true, y_score, threshold, name):
    y_pred = (y_score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return EvalResult(
        name=name,
        threshold=float(threshold),
        auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def find_threshold_recall_constrained(y_true, y_score, min_recall=0.93):
    best = None
    for thr in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        accuracy = (tp + tn) / len(y_true)

        if recall < min_recall:
            continue

        candidate = {
            "threshold": float(thr),
            "recall": float(recall),
            "specificity": float(specificity),
            "accuracy": float(accuracy),
            "fp": int(fp),
        }

        if best is None:
            best = candidate
        elif candidate["specificity"] > best["specificity"] or (
            np.isclose(candidate["specificity"], best["specificity"]) and candidate["accuracy"] > best["accuracy"]
        ):
            best = candidate

    return best


def bootstrap_ci(metric_values, alpha=0.05):
    lo = np.percentile(metric_values, 100 * (alpha / 2))
    hi = np.percentile(metric_values, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def bootstrap_diff_ci(y_true, score_a, thr_a, score_b, thr_b, n_boot=2000):
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    diff_recall = []
    diff_fp = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]

        pa = (score_a[idx] > thr_a).astype(int)
        pb = (score_b[idx] > thr_b).astype(int)

        tn_a, fp_a, fn_a, tp_a = confusion_matrix(yt, pa).ravel()
        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yt, pb).ravel()

        rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) else 0.0
        rec_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) else 0.0

        diff_recall.append(rec_b - rec_a)
        diff_fp.append(fp_b - fp_a)

    return bootstrap_ci(np.array(diff_recall)), bootstrap_ci(np.array(diff_fp))


def mcnemar_pvalue(y_true, pred_a, pred_b):
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))
    if b + c == 0:
        return 1.0, b, c

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    p = math.exp(-chi2 / 2.0)
    return float(p), b, c


def as_dict(res: EvalResult):
    return {
        "name": res.name,
        "threshold": res.threshold,
        "auc": res.auc,
        "pr_auc": res.pr_auc,
        "accuracy": res.accuracy,
        "precision": res.precision,
        "recall": res.recall,
        "specificity": res.specificity,
        "f1": res.f1,
        "tn": res.tn,
        "fp": res.fp,
        "fn": res.fn,
        "tp": res.tp,
    }


def main():
    print("\nOOF Stacking Optimization")
    print("=" * 70)

    target_recall = 0.93

    print("\n1) Loading data + patient split...")
    df = load_df()
    train_df, val_df, test_df = patient_split(df)
    print(f"   Train={len(train_df)} Val={len(val_df)} Test={len(test_df)}")

    print("\n2) Loading base models...")
    model_pos, pos_path = load_model_with_fallback(
        "positive model",
        "gdc_suspicious_results/suspicious_classifier_final.h5",
        [
            "gdc_suspicious_results/best_phase2_weights.h5",
            "gdc_suspicious_results/best_phase1_weights.h5",
            "gdc_suspicious_results/best_phase2.h5",
            "gdc_suspicious_results/best_phase1.h5",
        ],
    )
    model_neg, neg_path = load_model_with_fallback(
        "negative model",
        "gdc_suspicious_results_negative/suspicious_neg_final.h5",
        [
            "gdc_suspicious_results_negative/best_phase2_weights.h5",
            "gdc_suspicious_results_negative/best_phase1_weights.h5",
            "gdc_suspicious_results_negative/best_phase2.h5",
            "gdc_suspicious_results_negative/best_phase1.h5",
        ],
    )

    print("\n3) Getting train/val/test probabilities (TTA)...")
    y_train = train_df["suspicious_label"].values.astype(int)
    y_val = val_df["suspicious_label"].values.astype(int)
    y_test = test_df["suspicious_label"].values.astype(int)

    ppos_train = get_probs(model_pos, train_df["full_path"].values, y_train)
    pneg_train = get_probs(model_neg, train_df["full_path"].values, y_train)
    ppos_val = get_probs(model_pos, val_df["full_path"].values, y_val)
    pneg_val = get_probs(model_neg, val_df["full_path"].values, y_val)
    ppos_test = get_probs(model_pos, test_df["full_path"].values, y_test)
    pneg_test = get_probs(model_neg, test_df["full_path"].values, y_test)

    pneg_susp_train = 1.0 - pneg_train
    pneg_susp_val = 1.0 - pneg_val
    pneg_susp_test = 1.0 - pneg_test

    print("\n4) Calibrating base probabilities on train+val reference...")
    # Use train+val for calibrators, preserve test untouched.
    y_cal = np.concatenate([y_train, y_val])
    ppos_cal = np.concatenate([ppos_train, ppos_val])
    pneg_cal = np.concatenate([pneg_susp_train, pneg_susp_val])

    iso_pos = IsotonicRegression(out_of_bounds="clip")
    iso_pos.fit(ppos_cal, y_cal)
    iso_neg = IsotonicRegression(out_of_bounds="clip")
    iso_neg.fit(pneg_cal, y_cal)

    ppos_train_iso = iso_pos.transform(ppos_train)
    pneg_train_iso = iso_neg.transform(pneg_susp_train)
    ppos_val_iso = iso_pos.transform(ppos_val)
    pneg_val_iso = iso_neg.transform(pneg_susp_val)
    ppos_test_iso = iso_pos.transform(ppos_test)
    pneg_test_iso = iso_neg.transform(pneg_susp_test)

    print("\n5) Patient-level OOF stacking on train split...")
    patient_train = train_df.groupby("case_no")["suspicious_label"].first().reset_index()
    patient_ids = patient_train["case_no"].values
    patient_y = patient_train["suspicious_label"].values.astype(int)

    # Map image rows to patient fold through patient id.
    train_patient_index = {pid: i for i, pid in enumerate(patient_ids)}
    row_patient_idx = np.array([train_patient_index[pid] for pid in train_df["case_no"].values])

    X_train_all = np.column_stack([
        ppos_train_iso,
        pneg_train_iso,
        np.abs(ppos_train_iso - pneg_train_iso),
        ppos_train_iso * pneg_train_iso,
    ])

    oof_score = np.zeros(len(train_df), dtype=float)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr_p_idx, va_p_idx) in enumerate(skf.split(patient_ids, patient_y), start=1):
        tr_patients = set(patient_ids[tr_p_idx])
        va_patients = set(patient_ids[va_p_idx])

        tr_mask = np.array([pid in tr_patients for pid in train_df["case_no"].values])
        va_mask = np.array([pid in va_patients for pid in train_df["case_no"].values])

        X_tr = X_train_all[tr_mask]
        y_tr = y_train[tr_mask]
        X_va = X_train_all[va_mask]

        meta = LogisticRegression(random_state=SEED + fold, max_iter=1000, class_weight="balanced")
        meta.fit(X_tr, y_tr)
        oof_score[va_mask] = meta.predict_proba(X_va)[:, 1]

    # Fit final stacker on full train for val/test inference.
    meta_final = LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced")
    meta_final.fit(X_train_all, y_train)

    X_val = np.column_stack([
        ppos_val_iso,
        pneg_val_iso,
        np.abs(ppos_val_iso - pneg_val_iso),
        ppos_val_iso * pneg_val_iso,
    ])
    X_test = np.column_stack([
        ppos_test_iso,
        pneg_test_iso,
        np.abs(ppos_test_iso - pneg_test_iso),
        ppos_test_iso * pneg_test_iso,
    ])

    score_val = meta_final.predict_proba(X_val)[:, 1]
    score_test = meta_final.predict_proba(X_test)[:, 1]

    print("\n6) Build baseline and choose thresholds on validation...")
    base_thr_info = find_threshold_recall_constrained(y_val, ppos_val_iso, min_recall=target_recall)
    if base_thr_info is None:
        raise RuntimeError("Could not find baseline threshold satisfying recall constraint.")

    stack_thr_info = find_threshold_recall_constrained(y_val, score_val, min_recall=target_recall)
    if stack_thr_info is None:
        raise RuntimeError("Could not find stack threshold satisfying recall constraint.")

    print(f"   Baseline thr={base_thr_info['threshold']:.3f}")
    print(f"   OOF stack thr={stack_thr_info['threshold']:.3f}")

    print("\n7) Final test evaluation...")
    baseline = evaluate(y_test, ppos_test_iso, base_thr_info["threshold"], "positive_isotonic_baseline")
    optimized = evaluate(y_test, score_test, stack_thr_info["threshold"], "oof_stacked_ensemble")

    print(f"   Baseline: recall={baseline.recall:.4f}, specificity={baseline.specificity:.4f}, fp={baseline.fp}, f1={baseline.f1:.4f}")
    print(f"   OOF stack: recall={optimized.recall:.4f}, specificity={optimized.specificity:.4f}, fp={optimized.fp}, f1={optimized.f1:.4f}")

    print("\n8) Significance tests...")
    pred_base = (ppos_test_iso > base_thr_info["threshold"]).astype(int)
    pred_opt = (score_test > stack_thr_info["threshold"]).astype(int)

    p_mcnemar, b, c = mcnemar_pvalue(y_test, pred_base, pred_opt)
    recall_ci, fp_ci = bootstrap_diff_ci(
        y_test,
        ppos_test_iso,
        base_thr_info["threshold"],
        score_test,
        stack_thr_info["threshold"],
        n_boot=2000,
    )

    fp_reduction_pct = 100.0 * (1.0 - optimized.fp / max(baseline.fp, 1))
    recall_change_pct = 100.0 * ((optimized.recall - baseline.recall) / max(baseline.recall, 1e-8))

    out_dir = "gdc_suspicious_results_ensemble"
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "checkpoints": {"positive_used": pos_path, "negative_used": neg_path},
        "target_recall_constraint": target_recall,
        "validation_thresholds": {
            "baseline": base_thr_info,
            "oof_stacked": stack_thr_info,
        },
        "test_metrics": {
            "baseline": as_dict(baseline),
            "oof_stacked": as_dict(optimized),
        },
        "improvement": {
            "fp_reduction_percent": float(fp_reduction_pct),
            "recall_change_percent": float(recall_change_pct),
            "specificity_delta": float(optimized.specificity - baseline.specificity),
            "f1_delta": float(optimized.f1 - baseline.f1),
        },
        "significance": {
            "mcnemar_pvalue": float(p_mcnemar),
            "mcnemar_b": int(b),
            "mcnemar_c": int(c),
            "recall_diff_95ci": [float(recall_ci[0]), float(recall_ci[1])],
            "fp_diff_95ci": [float(fp_ci[0]), float(fp_ci[1])],
        },
    }

    report_path = os.path.join(out_dir, "oof_stacking_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_path = os.path.join(out_dir, "oof_stacking_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("OOF Stacking Optimization Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Target recall: {target_recall:.2f}\n\n")
        f.write("Baseline (positive isotonic):\n")
        f.write(f"  recall={baseline.recall:.4f}, specificity={baseline.specificity:.4f}, fp={baseline.fp}, f1={baseline.f1:.4f}\n")
        f.write("OOF stacked ensemble:\n")
        f.write(f"  recall={optimized.recall:.4f}, specificity={optimized.specificity:.4f}, fp={optimized.fp}, f1={optimized.f1:.4f}\n\n")
        f.write("Delta:\n")
        f.write(f"  fp_reduction_percent={fp_reduction_pct:+.2f}%\n")
        f.write(f"  recall_change_percent={recall_change_pct:+.2f}%\n")
        f.write(f"  specificity_delta={optimized.specificity - baseline.specificity:+.4f}\n")
        f.write(f"  f1_delta={optimized.f1 - baseline.f1:+.4f}\n\n")
        f.write("Significance:\n")
        f.write(f"  McNemar p-value={p_mcnemar:.6f} (b={b}, c={c})\n")
        f.write(f"  Recall diff 95% CI={recall_ci}\n")
        f.write(f"  FP diff 95% CI={fp_ci}\n")

    print("\nDone.")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
