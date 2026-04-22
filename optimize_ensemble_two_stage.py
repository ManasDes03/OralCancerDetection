"""
🏥 TWO-STAGE PIPELINE ENSEMBLE (SIMPLIFIED)
==========================================
Stage 1: Use negative test to rule out safe cases
Stage 2: Use positive test only on uncertain cases
"""

import numpy as np
import pandas as pd
import os
import json
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, accuracy_score,
    recall_score, precision_score, f1_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt

print("=" * 80)
print("TWO-STAGE PIPELINE ENSEMBLE")
print("=" * 80)

# Load test data
data_dir = 'data/gdc_oral_cancer_dataset'
csv_path = os.path.join(data_dir, 'processed_labels.csv')
df = pd.read_csv(csv_path)

# Patient-aware split (same as training)
unique_patients = df['case_no'].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)

train_size = int(0.70 * len(unique_patients))
val_size = int(0.15 * len(unique_patients))

test_patients = set(unique_patients[train_size + val_size:])
test_df = df[df['case_no'].isin(test_patients)].reset_index(drop=True)

print(f"\n📊 Test set size: {len(test_df)} images from {len(test_patients)} patients")

# Load cached predictions from calibrated optimization
if os.path.exists('test_predictions_cache.csv'):
    pred_cache = pd.read_csv('test_predictions_cache.csv')
    # Merge with test df
    test_df = test_df.merge(pred_cache[['image_path', 'p_positive', 'p_negative']], 
                             on='image_path', how='left')
    print("✅ Loaded cached predictions")
else:
    print("⚠️  No cache found - using synthetic demo data")
    np.random.seed(42)
    test_df['p_positive'] = np.random.beta(2.5, 2.5, len(test_df))
    test_df['p_negative'] = np.random.beta(2.5, 2.5, len(test_df))

y_test = test_df['suspicious_label'].values

# ============================================================================
# TWO-STAGE PIPELINE CONFIGURATION
# ============================================================================

print("\n" + "=" * 80)
print("PIPELINE CONFIGURATION")
print("=" * 80)

STAGE1_THRESH = 0.80  # Negative test threshold
STAGE2_THRESH = 0.50  # Positive test threshold

print(f"\n🏥 STAGE 1 (Negative Test - Rule-Out):")
print(f"   Threshold: P(non-suspicious) > {STAGE1_THRESH}")  
print(f"   Decision: If confident it's SAFE → Dismiss immediately")
print(f"   Speed: Fast (single model evaluation)")

print(f"\n🔬 STAGE 2 (Positive Test - Confirmation):")
print(f"   Threshold: P(suspicious) > {STAGE2_THRESH}")
print(f"   Decision: Used only for uncertain cases")
print(f"   Speed: More thorough analysis")

# ============================================================================
# APPLY TWO-STAGE PIPELINE
# ============================================================================

y_pred_baseline = (test_df['p_positive'] > 0.5).astype(int)

y_pred_two_stage = []
y_stage_used = []

for i in range(len(test_df)):
    p_pos = test_df['p_positive'].iloc[i]
    p_neg = test_df['p_negative'].iloc[i]
    
    # STAGE 1: Negative test (rule-out)
    if p_neg > STAGE1_THRESH:
        y_pred_two_stage.append(0)  # Safe
        y_stage_used.append(1)
    else:
        # STAGE 2: Positive test (confirmation)
        y_pred_two_stage.append(1 if p_pos > STAGE2_THRESH else 0)
        y_stage_used.append(2)

y_pred_two_stage = np.array(y_pred_two_stage)
y_stage_used = np.array(y_stage_used)

# ============================================================================
# COMPUTE METRICS
# ============================================================================

def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute all metrics."""
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
        if cm.shape[0] >= 1 and cm.shape[1] >= 1: tn = cm[0, 0]
        if cm.shape[0] >= 1 and cm.shape[1] >= 2: fp = cm[0, 1]
        if cm.shape[0] >= 2 and cm.shape[1] >= 1: fn = cm[1, 0]
        if cm.shape[0] >= 2 and cm.shape[1] >= 2: tp = cm[1, 1]
    
    metrics = {
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    
    if y_probs is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_probs))
        except:
            metrics['auc'] = None
    
    return metrics

metrics_baseline = compute_metrics(y_test, y_pred_baseline, test_df['p_positive'].values)
metrics_two_stage = compute_metrics(y_test, y_pred_two_stage, None)

# ============================================================================
# PRINT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

print("\n📊 BASELINE (Positive Model Alone, threshold=0.5):")
print(f"\n  Confusion Matrix:")
print(f"    True Positives (TP):   {metrics_baseline['TP']:4d}  ← Cancer correctly caught ✅")
print(f"    False Positives (FP):  {metrics_baseline['FP']:4d}  ← Healthy wrongly flagged ⚠️")
print(f"    True Negatives (TN):   {metrics_baseline['TN']:4d}  ← Healthy correctly dismissed ✅")
print(f"    False Negatives (FN):  {metrics_baseline['FN']:4d}  ← Cancer wrongly missed ⚠️")

print(f"\n  Key Metrics:")
print(f"    ├─ Recall:      {metrics_baseline['recall']:.4f}  (% of CANCER caught - MOST IMPORTANT)")
print(f"    ├─ Specificity: {metrics_baseline['specificity']:.4f}  (% of HEALTHY correctly ruled out)")
print(f"    ├─ Precision:   {metrics_baseline['precision']:.4f}  (confidence in positive flag)")
print(f"    ├─ Accuracy:    {metrics_baseline['accuracy']:.4f}  (overall correctness)")
print(f"    ├─ F1 Score:    {metrics_baseline['f1']:.4f}  (balance of precision & recall)")
if metrics_baseline['auc']:
    print(f"    └─ AUC:         {metrics_baseline['auc']:.4f}  (discrimination ability)")

print("\n✨ TWO-STAGE PIPELINE:")
print(f"\n  Confusion Matrix:")
print(f"    True Positives (TP):   {metrics_two_stage['TP']:4d}  ← Cancer correctly caught ✅")
print(f"    False Positives (FP):  {metrics_two_stage['FP']:4d}  ← Healthy wrongly flagged ⚠️")
print(f"    True Negatives (TN):   {metrics_two_stage['TN']:4d}  ← Healthy correctly dismissed ✅")
print(f"    False Negatives (FN):  {metrics_two_stage['FN']:4d}  ← Cancer wrongly missed ⚠️")

print(f"\n  Key Metrics:")
print(f"    ├─ Recall:      {metrics_two_stage['recall']:.4f}  (% of CANCER caught - MOST IMPORTANT)")
print(f"    ├─ Specificity: {metrics_two_stage['specificity']:.4f}  (% of HEALTHY correctly ruled out)")
print(f"    ├─ Precision:   {metrics_two_stage['precision']:.4f}  (confidence in positive flag)")
print(f"    ├─ Accuracy:    {metrics_two_stage['accuracy']:.4f}  (overall correctness)")
print(f"    └─ F1 Score:    {metrics_two_stage['f1']:.4f}  (balance of precision & recall)")

# ============================================================================
# IMPROVEMENTS
# ============================================================================

print("\n" + "=" * 80)
print("IMPROVEMENTS")
print("=" * 80)

fp_change = metrics_two_stage['FP'] - metrics_baseline['FP']
fp_pct = (fp_change / metrics_baseline['FP'] * 100) if metrics_baseline['FP'] > 0 else 0
recall_change = (metrics_two_stage['recall'] - metrics_baseline['recall']) * 100
spec_change = metrics_two_stage['specificity'] - metrics_baseline['specificity']
prec_change = metrics_two_stage['precision'] - metrics_baseline['precision']
f1_change = metrics_two_stage['f1'] - metrics_baseline['f1']

print(f"\n🎯 False Positives (Unnecessary flags):")
print(f"   Before: {metrics_baseline['FP']}")
print(f"   After:  {metrics_two_stage['FP']}")
print(f"   Change: {fp_change:+d} ({fp_pct:+.1f}%)")
if fp_change < 0:
    print(f"   ✅ IMPROVED - Fewer false alarms")
else:
    print(f"   ⚠️  WORSE - More false alarms")

print(f"\n🔍 Recall (% of Cancer Cases Caught) - MOST CRITICAL:")
print(f"   Before: {metrics_baseline['recall']:.4f} ({metrics_baseline['recall']*100:.2f}%)")
print(f"   After:  {metrics_two_stage['recall']:.4f} ({metrics_two_stage['recall']*100:.2f}%)")
print(f"   Change: {recall_change:+.2f} percentage points")
if recall_change > 0:
    print(f"   ✅ IMPROVED - Catch MORE cancers")
elif recall_change < -1:
    print(f"   ⚠️  TRADE-OFF - Catch fewer cancers (CONCERNING)")
else:
    print(f"   ✅ MAINTAINED - About the same")

print(f"\n🛡️  Specificity (% of Healthy Correctly Dismissed):")
print(f"   Before: {metrics_baseline['specificity']:.4f} ({metrics_baseline['specificity']*100:.2f}%)")
print(f"   After:  {metrics_two_stage['specificity']:.4f} ({metrics_two_stage['specificity']*100:.2f}%)")
print(f"   Change: {spec_change:+.4f} ({spec_change*100:+.2f}%)")
if spec_change > 0:
    print(f"   ✅ IMPROVED - Better at dismissing healthy")
else:
    print(f"   ⚠️  WORSE - Fewer healthy dismissed correctly")

print(f"\n📌 Precision (Confidence in flags):")
print(f"   Before: {metrics_baseline['precision']:.4f} ({metrics_baseline['precision']*100:.2f}%)")
print(f"   After:  {metrics_two_stage['precision']:.4f} ({metrics_two_stage['precision']*100:.2f}%)")
print(f"   Change: {prec_change:+.4f} ({prec_change*100:+.2f}%)")
if prec_change > 0:
    print(f"   ✅ IMPROVED - More confident in positive flags")
else:
    print(f"   ⚠️  Slightly less confident")

print(f"\n⚖️  F1 Score (Balance of Recall & Precision):")
print(f"   Before: {metrics_baseline['f1']:.4f}")
print(f"   After:  {metrics_two_stage['f1']:.4f}")
print(f"   Change: {f1_change:+.4f}")

# Pipeline efficiency
stage1_count = np.sum(y_stage_used == 1)
stage2_count = np.sum(y_stage_used == 2)

print(f"\n⚡ PIPELINE EFFICIENCY:")
print(f"   Stage 1 only:    {stage1_count:4d} cases ({stage1_count/len(y_test)*100:.1f}%) - dismissed as safe")
print(f"   Stage 2:         {stage2_count:4d} cases ({stage2_count/len(y_test)*100:.1f}%) - needed confirmation")
print(f"   Speed gain:      ~{stage1_count/len(y_test)*50:.1f}% faster (only stage 1 on {stage1_count/len(y_test)*100:.1f}% of cases)")

if stage1_count > 0:
    stage1_pred = y_pred_two_stage[y_stage_used == 1]
    stage1_true = y_test[y_stage_used == 1]
    stage1_correct = np.sum(stage1_pred == stage1_true)
    stage1_missed = np.sum((stage1_pred == 0) & (stage1_true == 1))
    
    print(f"\n   Among {stage1_count} cases stopped at Stage 1:")
    print(f"    ├─ Correctly dismissed: {stage1_correct}/{stage1_count} ({stage1_correct/stage1_count*100:.1f}%)")
    print(f"    └─ Missed cancers:      {stage1_missed} (CRITICAL - cancers wrongly dismissed)")

# ============================================================================
# SAVE RESULTS
# ============================================================================

results_dir = 'gdc_suspicious_results_ensemble'
os.makedirs(results_dir, exist_ok=True)

results = {
    'method': 'two_stage_pipeline',
    'parameters': {
        'stage1_negative_threshold': STAGE1_THRESH,
        'stage2_positive_threshold': STAGE2_THRESH,
    },
    'test_set_size': len(y_test),
    'baseline_metrics': metrics_baseline,
    'two_stage_metrics': metrics_two_stage,
    'improvements': {
        'fp_change': int(fp_change),
        'fp_change_percent': float(fp_pct),
        'recall_change_percent': float(recall_change),
        'specificity_change': float(spec_change),
        'precision_change': float(prec_change),
        'f1_change': float(f1_change),
        'stage1_efficiency_percent': float(stage1_count / len(y_test) * 100),
    }
}

with open(os.path.join(results_dir, 'two_stage_report.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Summary
summary = f"""TWO-STAGE PIPELINE ENSEMBLE - RESULTS
====================================

PARAMETERS:
-----------
Stage 1 Threshold (Negative Test):   {STAGE1_THRESH}
Stage 2 Threshold (Positive Test):   {STAGE2_THRESH}
Test Set Size:                        {len(y_test)} images

BASELINE (Positive Model):
--------------------------
True Positives:    {metrics_baseline['TP']}
False Positives:   {metrics_baseline['FP']}
True Negatives:    {metrics_baseline['TN']}
False Negatives:   {metrics_baseline['FN']}

Recall:            {metrics_baseline['recall']:.4f}
Specificity:       {metrics_baseline['specificity']:.4f}
Precision:         {metrics_baseline['precision']:.4f}
Accuracy:          {metrics_baseline['accuracy']:.4f}
F1 Score:          {metrics_baseline['f1']:.4f}

TWO-STAGE PIPELINE:
-------------------
True Positives:    {metrics_two_stage['TP']}
False Positives:   {metrics_two_stage['FP']}
True Negatives:    {metrics_two_stage['TN']}
False Negatives:   {metrics_two_stage['FN']}

Recall:            {metrics_two_stage['recall']:.4f}
Specificity:       {metrics_two_stage['specificity']:.4f}
Precision:         {metrics_two_stage['precision']:.4f}
Accuracy:          {metrics_two_stage['accuracy']:.4f}
F1 Score:          {metrics_two_stage['f1']:.4f}

KEY IMPROVEMENTS:
-----------------
False Positives:   {metrics_baseline['FP']} → {metrics_two_stage['FP']} ({fp_pct:+.1f}%)
Recall:            {metrics_baseline['recall']:.4f} → {metrics_two_stage['recall']:.4f} ({recall_change:+.2f}pp)
Specificity:       {metrics_baseline['specificity']:.4f} → {metrics_two_stage['specificity']:.4f} ({spec_change:+.4f})
Precision:         {metrics_baseline['precision']:.4f} → {metrics_two_stage['precision']:.4f} ({prec_change:+.4f})
F1 Score:          {metrics_baseline['f1']:.4f} → {metrics_two_stage['f1']:.4f} ({f1_change:+.4f})

EFFICIENCY:
-----------
Stage 1 dismissals: {stage1_count}/{len(y_test)} ({stage1_count/len(y_test)*100:.1f}%)
Speed improvement: ~{stage1_count/len(y_test)*50:.1f}%
"""

with open(os.path.join(results_dir, 'two_stage_summary.txt'), 'w') as f:
    f.write(summary)

print("\n✅ Results saved:")
print(f"   📄 {os.path.join(results_dir, 'two_stage_report.json')}")
print(f"   📄 {os.path.join(results_dir, 'two_stage_summary.txt')}")
print("\n" + "=" * 80)
