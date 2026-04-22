"""
Advanced Ensemble Optimizer with Statistical Significance
=========================================================

What this script does:
1. Uses patient-aware 70/15/15 split (same random state) and keeps test untouched.
2. Tunes ensemble on validation only (no test leakage).
3. Compares multiple scoring schemes including a stacked meta-learner.
4. Selects model under clinical recall constraints.
5. Reports statistical significance with bootstrap confidence intervals and McNemar test.

Outputs:
- gdc_suspicious_results_ensemble/advanced_optimization_report.json
- gdc_suspicious_results_ensemble/advanced_optimization_summary.txt
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
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


def load_patient_split_df() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = "data/gdc_oral_cancer_dataset"
    images_dir = os.path.join(data_dir, "images")
    csv_path = os.path.join(data_dir, "processed_labels.csv")

    df = pd.read_csv(csv_path)
    df = df[df["suspicious_label"].notna()].copy()
    df = df.drop_duplicates(subset=["case_no", "image_path"]).copy()
    df["full_path"] = df["image_path"].apply(lambda x: os.path.join(images_dir, x))
    df = df[df["full_path"].apply(os.path.exists)].copy()

    patient_labels = df.groupby("case_no")["suspicious_label"].first()
    patients = patient_labels.index.values
    labels = patient_labels.values

    _, temp_patients, _, temp_labels = train_test_split(
        patients, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_size = 0.15 / 0.30
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=1 - val_size, stratify=temp_labels, random_state=SEED
    )

    val_df = df[df["case_no"].isin(val_patients)].copy()
    test_df = df[df["case_no"].isin(test_patients)].copy()
    return val_df, test_df


def get_probs(model, paths, labels):
    ds = create_dataset(paths, labels, flip=False)
    ds_flip = create_dataset(paths, labels, flip=True)
    p = model.predict(ds, verbose=0).flatten()
    p_flip = model.predict(ds_flip, verbose=0).flatten()
    return (p + p_flip) / 2.0


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

    rec_ci = bootstrap_ci(np.array(diff_recall))
    fp_ci = bootstrap_ci(np.array(diff_fp))
    return rec_ci, fp_ci


def mcnemar_pvalue(y_true, pred_a, pred_b):
    # b: A wrong, B correct ; c: A correct, B wrong
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true

    b = int(np.sum((~correct_a) & correct_b))
    c = int(np.sum(correct_a & (~correct_b)))

    # continuity-corrected chi-square for df=1
    if b + c == 0:
        return 1.0, b, c

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    # For df=1, survival function is exp(-x/2)
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
    print("\nAdvanced Ensemble Optimization + Significance")
    print("=" * 70)

    print("\n1) Loading patient-aware val/test split...")
    val_df, test_df = load_patient_split_df()
    print(f"   Val images: {len(val_df)} | Test images: {len(test_df)}")

    print("\n2) Loading models with fallback checkpoints...")
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

    print("\n3) Predicting probabilities (TTA) on val/test...")
    y_val = val_df["suspicious_label"].values.astype(int)
    y_test = test_df["suspicious_label"].values.astype(int)

    ppos_val = get_probs(model_pos, val_df["full_path"].values, y_val)
    pneg_val = get_probs(model_neg, val_df["full_path"].values, y_val)
    ppos_test = get_probs(model_pos, test_df["full_path"].values, y_test)
    pneg_test = get_probs(model_neg, test_df["full_path"].values, y_test)

    # Re-express negative output as suspicious probability.
    pneg_susp_val = 1.0 - pneg_val
    pneg_susp_test = 1.0 - pneg_test

    print("\n4) Building candidate ensemble scores...")
    # Candidate 1: simple average
    score_avg_val = 0.5 * (ppos_val + pneg_susp_val)
    score_avg_test = 0.5 * (ppos_test + pneg_susp_test)

    # Candidate 2: contradiction-aware penalty
    disagreement_val = np.abs(ppos_val - pneg_susp_val)
    disagreement_test = np.abs(ppos_test - pneg_susp_test)
    score_contra_val = (0.5 * (ppos_val + pneg_susp_val)) * (1.0 - 0.5 * disagreement_val)
    score_contra_test = (0.5 * (ppos_test + pneg_susp_test)) * (1.0 - 0.5 * disagreement_test)

    # Candidate 3: stacked meta-learner (trained only on val)
    X_val = np.column_stack([
        ppos_val,
        pneg_susp_val,
        np.abs(ppos_val - pneg_susp_val),
        ppos_val * pneg_susp_val,
    ])
    X_test = np.column_stack([
        ppos_test,
        pneg_susp_test,
        np.abs(ppos_test - pneg_susp_test),
        ppos_test * pneg_susp_test,
    ])

    stacker = LogisticRegression(random_state=SEED, max_iter=1000, class_weight="balanced")
    stacker.fit(X_val, y_val)
    score_stack_val = stacker.predict_proba(X_val)[:, 1]
    score_stack_test = stacker.predict_proba(X_test)[:, 1]

    print("\n5) Validation-only threshold tuning under recall constraints...")
    target_recall = 0.93

    # Baseline threshold tuned on val
    best_pos = find_threshold_recall_constrained(y_val, ppos_val, min_recall=target_recall)
    if best_pos is None:
        raise RuntimeError("Could not find positive-model threshold satisfying recall constraint.")

    candidates = {
        "avg_ensemble": (score_avg_val, score_avg_test),
        "contra_ensemble": (score_contra_val, score_contra_test),
        "stacked_ensemble": (score_stack_val, score_stack_test),
    }

    tuned_thresholds = {}
    for name, (sv, _) in candidates.items():
        best = find_threshold_recall_constrained(y_val, sv, min_recall=target_recall)
        if best is not None:
            tuned_thresholds[name] = best

    if not tuned_thresholds:
        raise RuntimeError("No ensemble candidate met recall constraint on validation.")

    # Select winner by highest specificity on validation, then accuracy.
    winner_name = sorted(
        tuned_thresholds.keys(),
        key=lambda n: (tuned_thresholds[n]["specificity"], tuned_thresholds[n]["accuracy"]),
        reverse=True,
    )[0]

    print(f"   Winner on validation: {winner_name}")
    print(f"   Val threshold: {tuned_thresholds[winner_name]['threshold']:.3f}")

    print("\n6) Final evaluation on untouched test split...")
    baseline = evaluate(y_test, ppos_test, best_pos["threshold"], "positive_baseline")

    winner_val_score, winner_test_score = candidates[winner_name]
    winner_thr = tuned_thresholds[winner_name]["threshold"]
    optimized = evaluate(y_test, winner_test_score, winner_thr, winner_name)

    print("\nBaseline vs Optimized (test):")
    print(f"  Baseline recall={baseline.recall:.4f}, specificity={baseline.specificity:.4f}, fp={baseline.fp}")
    print(f"  Optimized recall={optimized.recall:.4f}, specificity={optimized.specificity:.4f}, fp={optimized.fp}")

    print("\n7) Statistical significance analysis...")
    pred_base = (ppos_test > best_pos["threshold"]).astype(int)
    pred_opt = (winner_test_score > winner_thr).astype(int)

    p_mcnemar, b, c = mcnemar_pvalue(y_test, pred_base, pred_opt)
    recall_diff_ci, fp_diff_ci = bootstrap_diff_ci(
        y_test,
        ppos_test,
        best_pos["threshold"],
        winner_test_score,
        winner_thr,
        n_boot=2000,
    )

    fp_reduction_pct = 100.0 * (1.0 - (optimized.fp / max(baseline.fp, 1)))
    recall_change_pct = 100.0 * ((optimized.recall - baseline.recall) / max(baseline.recall, 1e-8))

    out_dir = "gdc_suspicious_results_ensemble"
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "checkpoints": {
            "positive_used": pos_path,
            "negative_used": neg_path,
        },
        "split": {
            "val_size": int(len(val_df)),
            "test_size": int(len(test_df)),
            "val_suspicious": int(np.sum(y_val == 1)),
            "test_suspicious": int(np.sum(y_test == 1)),
        },
        "target_recall_constraint": target_recall,
        "baseline_threshold_on_val": best_pos,
        "ensemble_candidates_thresholds_on_val": tuned_thresholds,
        "winner": winner_name,
        "winner_threshold": float(winner_thr),
        "test_metrics": {
            "baseline": as_dict(baseline),
            "optimized": as_dict(optimized),
        },
        "improvement": {
            "fp_reduction_percent": float(fp_reduction_pct),
            "recall_change_percent": float(recall_change_pct),
            "f1_delta": float(optimized.f1 - baseline.f1),
            "specificity_delta": float(optimized.specificity - baseline.specificity),
        },
        "significance": {
            "mcnemar_pvalue": float(p_mcnemar),
            "mcnemar_b": int(b),
            "mcnemar_c": int(c),
            "recall_diff_95ci": [float(recall_diff_ci[0]), float(recall_diff_ci[1])],
            "fp_diff_95ci": [float(fp_diff_ci[0]), float(fp_diff_ci[1])],
        },
    }

    report_path = os.path.join(out_dir, "advanced_optimization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary_path = os.path.join(out_dir, "advanced_optimization_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Advanced Ensemble Optimization Summary\n")
        f.write("=" * 48 + "\n\n")
        f.write(f"Winner: {winner_name}\n")
        f.write(f"Threshold: {winner_thr:.3f}\n")
        f.write(f"Target recall constraint: {target_recall:.2f}\n\n")
        f.write("Baseline (test):\n")
        f.write(f"  recall={baseline.recall:.4f}, specificity={baseline.specificity:.4f}, fp={baseline.fp}, f1={baseline.f1:.4f}\n")
        f.write("Optimized (test):\n")
        f.write(f"  recall={optimized.recall:.4f}, specificity={optimized.specificity:.4f}, fp={optimized.fp}, f1={optimized.f1:.4f}\n\n")
        f.write("Delta:\n")
        f.write(f"  fp_reduction_percent={fp_reduction_pct:+.2f}%\n")
        f.write(f"  recall_change_percent={recall_change_pct:+.2f}%\n")
        f.write(f"  specificity_delta={optimized.specificity - baseline.specificity:+.4f}\n")
        f.write(f"  f1_delta={optimized.f1 - baseline.f1:+.4f}\n\n")
        f.write("Significance:\n")
        f.write(f"  McNemar p-value={p_mcnemar:.6f} (b={b}, c={c})\n")
        f.write(f"  Recall diff 95% CI={recall_diff_ci}\n")
        f.write(f"  FP diff 95% CI={fp_diff_ci}\n")

    print("\nDone.")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
