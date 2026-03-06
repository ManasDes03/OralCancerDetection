"""
Comprehensive Model Evaluation & Comparison
Evaluates all trained models (original + ensemble) with detailed metrics
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, f1_score, accuracy_score
)
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION & COMPARISON")
print("="*80)


def load_model_and_evaluate(model_path, test_df, label_col, model_name, threshold=0.5):
    """
    Load model and evaluate on test set
    
    Args:
        model_path: Path to .h5 model
        test_df: Test DataFrame
        label_col: Column name with labels
        model_name: Name for display
        threshold: Decision threshold
        
    Returns:
        metrics_dict: Dictionary of evaluation metrics
    """
    print(f"\n[*] Evaluating {model_name}...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except:
        model = tf.keras.models.load_model(model_path)
    
    # Load images
    images = []
    for img_path in test_df['image_path'].values:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
        except:
            images.append(np.zeros((224, 224, 3)))
    
    images = np.array(images, dtype=np.float32)
    y_true = test_df[label_col].values.astype(np.int32)
    
    # Predict
    y_probs = model.predict(images, verbose=0).flatten()
    y_pred = (y_probs > threshold).astype(int)
    
    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = sklearn.metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = sklearn.metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    cm = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'confusion_matrix': cm.tolist(),
        'threshold': float(threshold),
        'y_probs': y_probs,
        'y_true': y_true,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision_curve,
        'recall_curve': recall_curve
    }
    
    print(f"  ✓ Accuracy: {accuracy:.4f}")
    print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
    print(f"  ✓ F1-Score: {f1:.4f}")
    
    return metrics


# =============================================================================
# LOAD DATA
# =============================================================================

print("\n[1/5] Loading GDC dataset...")

df = pd.read_csv('data/gdc_oral_cancer_dataset/processed_labels.csv')

# Suspicious classifier test set
from sklearn.model_selection import train_test_split
df_suspicious = df[df['suspicious_label'].notna()].reset_index(drop=True)
patients_sus = df_suspicious['case_no'].unique()
patient_labels_sus = df_suspicious.groupby('case_no')['suspicious_label'].first().values

train_patients, temp_patients = train_test_split(
    patients_sus, test_size=0.3, stratify=patient_labels_sus, random_state=42
)
temp_labels = df_suspicious[df_suspicious['case_no'].isin(temp_patients)].groupby('case_no')['suspicious_label'].first().values
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, stratify=temp_labels, random_state=42
)
test_df_suspicious = df_suspicious[df_suspicious['case_no'].isin(test_patients)].reset_index(drop=True)

# Risk classifier test set
df_risk = df[df['risk_label'].notna()].reset_index(drop=True)
patients_risk = df_risk['case_no'].unique()
patient_labels_risk = df_risk.groupby('case_no')['risk_label'].first().values

train_patients, temp_patients = train_test_split(
    patients_risk, test_size=0.3, stratify=patient_labels_risk, random_state=42
)
temp_labels = df_risk[df_risk['case_no'].isin(temp_patients)].groupby('case_no')['risk_label'].first().values
val_patients, test_patients = train_test_split(
    temp_patients, test_size=0.5, stratify=temp_labels, random_state=42
)
test_df_risk = df_risk[df_risk['case_no'].isin(test_patients)].reset_index(drop=True)

print(f"Suspicious test set: {len(test_df_suspicious)} images")
print(f"Risk test set: {len(test_df_risk)} images")

# =============================================================================
# EVALUATE MODELS
# =============================================================================

import sklearn.metrics

print("\n[2/5] Evaluating all models...")

# Load thresholds from results files
suspicious_threshold = 0.697  # From previous training
risk_threshold = 0.706

all_results = []

# Original models
if os.path.exists('gdc_suspicious_results/suspicious_classifier_final.h5'):
    metrics = load_model_and_evaluate(
        'gdc_suspicious_results/suspicious_classifier_final.h5',
        test_df_suspicious,
        'suspicious_label',
        'Original Suspicious',
        threshold=suspicious_threshold
    )
    all_results.append((metrics, 'suspicious'))

if os.path.exists('gdc_risk_results/risk_classifier_final.h5'):
    metrics = load_model_and_evaluate(
        'gdc_risk_results/risk_classifier_final.h5',
        test_df_risk,
        'risk_label',
        'Original Risk',
        threshold=risk_threshold
    )
    all_results.append((metrics, 'risk'))

# Ensemble models (MobileNetV2, EfficientNetB3, ResNet50V2)
ensemble_models_sus = [
    ('ensemble_suspicious_results/mobilenetv2_model.h5', 'Ensemble - MobileNetV2 (Sus)'),
    ('ensemble_suspicious_results/efficientnetb3_model.h5', 'Ensemble - EfficientNetB3 (Sus)'),
    ('ensemble_suspicious_results/resnet50v2_model.h5', 'Ensemble - ResNet50V2 (Sus)')
]

for model_path, model_name in ensemble_models_sus:
    if os.path.exists(model_path):
        metrics = load_model_and_evaluate(
            model_path, test_df_suspicious, 'suspicious_label',
            model_name, threshold=suspicious_threshold
        )
        all_results.append((metrics, 'suspicious'))

ensemble_models_risk = [
    ('ensemble_risk_results/mobilenetv2_model.h5', 'Ensemble - MobileNetV2 (Risk)'),
    ('ensemble_risk_results/efficientnetb3_model.h5', 'Ensemble - EfficientNetB3 (Risk)'),
    ('ensemble_risk_results/densenet201_model.h5', 'Ensemble - DenseNet201 (Risk)')
]

for model_path, model_name in ensemble_models_risk:
    if os.path.exists(model_path):
        metrics = load_model_and_evaluate(
            model_path, test_df_risk, 'risk_label',
            model_name, threshold=risk_threshold
        )
        all_results.append((metrics, 'risk'))

# =============================================================================
# COMPARISON TABLE
# =============================================================================

print("\n[3/5] Creating comparison tables...")

comparison_data = []
for metrics, task in all_results:
    comparison_data.append({
        'Model': metrics['model_name'],
        'Task': task.upper(),
        'Accuracy': metrics['accuracy'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'F1-Score': metrics['f1_score'],
        'ROC-AUC': metrics['roc_auc'],
        'PR-AUC': metrics['pr_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n" + "="*100)
print("MODEL COMPARISON TABLE")
print("="*100)
print(comparison_df.to_string(index=False))
print("="*100)

# Group by task
print("\nSUSPICIOUS CLASSIFIER RANKINGS:")
sus_df = comparison_df[comparison_df['Task'] == 'SUSPICIOUS'].sort_values('ROC-AUC', ascending=False)
print(sus_df[['Model', 'Accuracy', 'ROC-AUC']].to_string(index=False))

print("\nRISK CLASSIFIER RANKINGS:")
risk_df = comparison_df[comparison_df['Task'] == 'RISK'].sort_values('ROC-AUC', ascending=False)
print(risk_df[['Model', 'Accuracy', 'ROC-AUC']].to_string(index=False))

# Save comparison
os.makedirs('evaluation_results', exist_ok=True)
comparison_df.to_csv('evaluation_results/model_comparison.csv', index=False)
print("\n✓ Saved: evaluation_results/model_comparison.csv")

# =============================================================================
# VISUALIZATIONS
# =============================================================================

print("\n[4/5] Creating visualizations...")

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 3, idx % 3]
    
    # Separate by task
    sus_data = comparison_df[comparison_df['Task'] == 'SUSPICIOUS'].sort_values(metric, ascending=True)
    risk_data = comparison_df[comparison_df['Task'] == 'RISK'].sort_values(metric, ascending=True)
    
    y_pos_sus = np.arange(len(sus_data))
    y_pos_risk = np.arange(len(sus_data), len(sus_data) + len(risk_data))
    
    ax.barh(y_pos_sus, sus_data[metric].values, color='skyblue', label='Suspicious', alpha=0.8)
    ax.barh(y_pos_risk, risk_data[metric].values, color='lightcoral', label='Risk', alpha=0.8)
    
    all_labels = list(sus_data['Model'].values) + list(risk_data['Model'].values)
    ax.set_yticks(np.concatenate([y_pos_sus, y_pos_risk]))
    ax.set_yticklabels([label.replace('Ensemble - ', '').replace(' (Sus)', '').replace(' (Risk)', '') 
                        for label in all_labels], fontsize=8)
    
    ax.set_xlabel(metric)
    ax.set_title(f'{metric} Comparison')
    ax.grid(alpha=0.3, axis='x')
    ax.set_xlim([0, 1])
    
    if idx == 0:
        ax.legend()

plt.tight_layout()
plt.savefig('evaluation_results/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/model_comparison.png")
plt.close()

# ROC Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

sus_models = [m for m, t in all_results if t == 'suspicious']
risk_models = [m for m, t in all_results if t == 'risk']

for metrics in sus_models:
    fpr = metrics['fpr']
    tpr = metrics['tpr']
    roc_auc = metrics['roc_auc']
    label = metrics['model_name'].replace('Ensemble - ', '').replace(' (Sus)', '')
    ax1.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.3f})', alpha=0.8)

ax1.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Suspicious Classifier - ROC Curves')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

for metrics in risk_models:
    fpr = metrics['fpr']
    tpr = metrics['tpr']
    roc_auc = metrics['roc_auc']
    label = metrics['model_name'].replace('Ensemble - ', '').replace(' (Risk)', '')
    ax2.plot(fpr, tpr, lw=2, label=f'{label} (AUC={roc_auc:.3f})', alpha=0.8)

ax2.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.3)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('Risk Classifier - ROC Curves')
ax2.legend(loc='lower right', fontsize=8)
ax2.grid(alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('evaluation_results/roc_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/roc_comparison.png")
plt.close()

# Detailed metrics heatmap
fig, ax = plt.subplots(figsize=(10, 8))
heatmap_data = comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']]
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Score'}, vmin=0, vmax=1)
ax.set_title('Detailed Model Metrics Heatmap')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('evaluation_results/metrics_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: evaluation_results/metrics_heatmap.png")
plt.close()

# =============================================================================
# SUMMARY REPORT
# =============================================================================

print("\n[5/5] Generating summary report...")

best_sus_idx = comparison_df[comparison_df['Task'] == 'SUSPICIOUS']['ROC-AUC'].idxmax()
best_sus = comparison_df.loc[best_sus_idx]

best_risk_idx = comparison_df[comparison_df['Task'] == 'RISK']['ROC-AUC'].idxmax()
best_risk = comparison_df.loc[best_risk_idx]

summary = f"""
{'='*80}
COMPREHENSIVE EVALUATION SUMMARY
{'='*80}

BEST SUSPICIOUS CLASSIFIER:
  Model:      {best_sus['Model']}
  Accuracy:   {best_sus['Accuracy']:.4f}
  Precision:  {best_sus['Precision']:.4f}
  Recall:     {best_sus['Recall']:.4f}
  F1-Score:   {best_sus['F1-Score']:.4f}
  ROC-AUC:    {best_sus['ROC-AUC']:.4f}

BEST RISK CLASSIFIER:
  Model:      {best_risk['Model']}
  Accuracy:   {best_risk['Accuracy']:.4f}
  Precision:  {best_risk['Precision']:.4f}
  Recall:     {best_risk['Recall']:.4f}
  F1-Score:   {best_risk['F1-Score']:.4f}
  ROC-AUC:    {best_risk['ROC-AUC']:.4f}

OVERALL STATISTICS:
  Total Models Evaluated: {len(comparison_df)}
  Suspicious Models:      {len(comparison_df[comparison_df['Task'] == 'SUSPICIOUS'])}
  Risk Models:            {len(comparison_df[comparison_df['Task'] == 'RISK'])}
  
  Average Accuracy (Suspicious):  {comparison_df[comparison_df['Task'] == 'SUSPICIOUS']['Accuracy'].mean():.4f}
  Average Accuracy (Risk):        {comparison_df[comparison_df['Task'] == 'RISK']['Accuracy'].mean():.4f}
  
  Average ROC-AUC (Suspicious):   {comparison_df[comparison_df['Task'] == 'SUSPICIOUS']['ROC-AUC'].mean():.4f}
  Average ROC-AUC (Risk):         {comparison_df[comparison_df['Task'] == 'RISK']['ROC-AUC'].mean():.4f}

OUTPUT FILES:
  ✓ evaluation_results/model_comparison.csv      - Detailed metrics table
  ✓ evaluation_results/model_comparison.png      - Metrics comparison plots
  ✓ evaluation_results/roc_comparison.png        - ROC curve overlays
  ✓ evaluation_results/metrics_heatmap.png       - Heatmap of all metrics

{'='*80}
"""

print(summary)

with open('evaluation_results/summary_report.txt', 'w') as f:
    f.write(summary)

print("✓ Saved: evaluation_results/summary_report.txt")

# Save detailed JSON
results_json = {
    'best_suspicious': best_sus.to_dict(),
    'best_risk': best_risk.to_dict(),
    'all_models': comparison_df.to_dict('records'),
    'summary': {
        'total_models': len(comparison_df),
        'suspicious_models': len(comparison_df[comparison_df['Task'] == 'SUSPICIOUS']),
        'risk_models': len(comparison_df[comparison_df['Task'] == 'RISK']),
        'avg_accuracy_suspicious': float(comparison_df[comparison_df['Task'] == 'SUSPICIOUS']['Accuracy'].mean()),
        'avg_accuracy_risk': float(comparison_df[comparison_df['Task'] == 'RISK']['Accuracy'].mean()),
        'avg_auc_suspicious': float(comparison_df[comparison_df['Task'] == 'SUSPICIOUS']['ROC-AUC'].mean()),
        'avg_auc_risk': float(comparison_df[comparison_df['Task'] == 'RISK']['ROC-AUC'].mean())
    }
}

with open('evaluation_results/evaluation_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)

print("✓ Saved: evaluation_results/evaluation_results.json")

print("\n" + "="*80)
print("✓ EVALUATION COMPLETE")
print("="*80)
