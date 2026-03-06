"""
5-Fold Cross-Validation Training
Provides more robust performance estimates and better generalization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks, models, layers, mixed_precision
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print("="*80)
print("5-FOLD CROSS-VALIDATION TRAINING")
print("="*80)


class CrossValidationTrainer:
    """5-Fold Cross-Validation training for robust model evaluation"""
    
    def __init__(self, df, label_col, task_name, n_splits=5):
        """
        Args:
            df: DataFrame with image paths and labels
            label_col: Column name with binary labels
            task_name: 'suspicious' or 'risk'
            n_splits: Number of folds
        """
        self.df = df
        self.label_col = label_col
        self.task_name = task_name
        self.n_splits = n_splits
        self.fold_results = []
        self.models = []
    
    def create_model(self):
        """Create fresh EfficientNetB3 model"""
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(224, 224, 3),
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
    
    def create_dataset(self, image_paths, labels, augment=True, batch_size=32):
        """Create tf.data pipeline"""
        images = []
        for img_path in image_paths:
            try:
                img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.utils.img_to_array(img) / 255.0
                images.append(img_array)
            except:
                images.append(np.zeros((224, 224, 3)))
        
        images = np.array(images, dtype=np.float32)
        
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        
        if augment:
            ds = ds.map(self._augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        
        ds = ds.batch(batch_size)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    def _augment_fn(self, image, label):
        """Augmentation function"""
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
        
        if tf.random.uniform(()) > 0.5:
            k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
            image = tf.image.rot90(image, k=k)
        
        zoom_range = tf.random.uniform((), minval=0.75, maxval=1.0)
        h, w = tf.shape(image)[0], tf.shape(image)[1]
        crop_h = tf.cast(tf.cast(h, tf.float32) * zoom_range, tf.int32)
        crop_w = tf.cast(tf.cast(w, tf.float32) * zoom_range, tf.int32)
        
        image = tf.image.random_crop(image, [crop_h, crop_w, 3])
        image = tf.image.resize(image, [224, 224])
        
        image = tf.image.random_brightness(image, 0.25)
        image = tf.image.random_contrast(image, 0.75, 1.25)
        image = tf.image.random_saturation(image, 0.75, 1.25)
        
        noise = tf.random.normal(shape=tf.shape(image), stddev=0.02)
        image = image + noise
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def train_fold(self, fold_idx, train_indices, val_indices):
        """Train single fold"""
        print(f"\n[Fold {fold_idx + 1}/{self.n_splits}] Training...")
        
        # Split data
        train_df = self.df.iloc[train_indices].reset_index(drop=True)
        val_df = self.df.iloc[val_indices].reset_index(drop=True)
        
        print(f"  Train: {len(train_df)} images, Val: {len(val_df)} images")
        
        # Create datasets
        train_ds = self.create_dataset(
            train_df['image_path'].values,
            train_df[self.label_col].values.astype(np.float32),
            augment=True
        )
        
        val_ds = self.create_dataset(
            val_df['image_path'].values,
            val_df[self.label_col].values.astype(np.float32),
            augment=False
        )
        
        # Create model
        model, base_model = self.create_model()
        
        # Phase 1: Frozen base
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        history_1 = model.fit(
            train_ds, epochs=10, validation_data=val_ds,
            verbose=1, callbacks=[
                callbacks.EarlyStopping(monitor='val_auc', patience=3, restore_best_weights=True)
            ]
        )
        
        # Phase 2: Fine-tune
        base_model.trainable = True
        for layer in base_model.layers[:100]:
            layer.trainable = False
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        history_2 = model.fit(
            train_ds, epochs=15, validation_data=val_ds,
            verbose=1, callbacks=[
                callbacks.EarlyStopping(monitor='val_auc', patience=4, restore_best_weights=True)
            ]
        )
        
        # Evaluate
        val_probs = model.predict(val_ds, verbose=0).flatten()
        val_labels = val_df[self.label_col].values
        
        fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_threshold = thresholds[best_idx]
        
        y_pred = (val_probs > best_threshold).astype(int)
        
        accuracy = accuracy_score(val_labels, y_pred)
        f1 = f1_score(val_labels, y_pred, zero_division=0)
        roc_auc = auc(fpr, tpr)
        
        fold_result = {
            'fold': fold_idx + 1,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'threshold': best_threshold,
            'val_probs': val_probs,
            'val_labels': val_labels
        }
        
        self.fold_results.append(fold_result)
        self.models.append(model)
        
        print(f"  ✓ Fold {fold_idx + 1} Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
        
        return fold_result
    
    def run_cross_validation(self):
        """Run 5-fold cross validation"""
        print(f"\nStarting {self.n_splits}-Fold Cross-Validation for {self.task_name}...")
        
        # Get labels for stratification
        labels = self.df[self.label_col].values
        
        # Create stratified k-fold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.df, labels)):
            self.train_fold(fold_idx, train_idx, val_idx)
        
        # Summarize results
        self.print_summary()
        return self.fold_results
    
    def print_summary(self):
        """Print cross-validation summary"""
        accuracies = [r['accuracy'] for r in self.fold_results]
        f1_scores = [r['f1_score'] for r in self.fold_results]
        aucs = [r['roc_auc'] for r in self.fold_results]
        
        print("\n" + "="*60)
        print(f"5-FOLD CROSS-VALIDATION SUMMARY - {self.task_name.upper()}")
        print("="*60)
        print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"F1-Score:  {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
        print(f"ROC-AUC:   {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print("="*60)
        
        print("\nPer-Fold Results:")
        for fold_result in self.fold_results:
            print(f"  Fold {fold_result['fold']}: "
                  f"Acc={fold_result['accuracy']:.4f}, "
                  f"F1={fold_result['f1_score']:.4f}, "
                  f"AUC={fold_result['roc_auc']:.4f}")
    
    def save_results(self, output_dir):
        """Save cross-validation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model.save(f'{output_dir}/fold_{i+1}_model.h5')
        
        # Save summary
        results_summary = {
            'task': self.task_name,
            'n_splits': self.n_splits,
            'fold_results': []
        }
        
        for fold_result in self.fold_results:
            results_summary['fold_results'].append({
                'fold': fold_result['fold'],
                'accuracy': float(fold_result['accuracy']),
                'f1_score': float(fold_result['f1_score']),
                'roc_auc': float(fold_result['roc_auc']),
                'threshold': float(fold_result['threshold'])
            })
        
        # Add summary statistics
        accuracies = [r['accuracy'] for r in self.fold_results]
        f1_scores = [r['f1_score'] for r in self.fold_results]
        aucs = [r['roc_auc'] for r in self.fold_results]
        
        results_summary['summary'] = {
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies)),
            'mean_f1': float(np.mean(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs))
        }
        
        import json
        with open(f'{output_dir}/cv_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        folds = [r['fold'] for r in self.fold_results]
        
        axes[0].plot(folds, accuracies, marker='o', linewidth=2, markersize=8)
        axes[0].axhline(np.mean(accuracies), color='r', linestyle='--', label=f'Mean: {np.mean(accuracies):.4f}')
        axes[0].fill_between(folds, np.mean(accuracies) - np.std(accuracies), 
                             np.mean(accuracies) + np.std(accuracies), alpha=0.2)
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Accuracy per Fold')
        axes[0].grid(alpha=0.3)
        axes[0].legend()
        
        axes[1].plot(folds, f1_scores, marker='s', linewidth=2, markersize=8, color='green')
        axes[1].axhline(np.mean(f1_scores), color='r', linestyle='--', label=f'Mean: {np.mean(f1_scores):.4f}')
        axes[1].fill_between(folds, np.mean(f1_scores) - np.std(f1_scores),
                             np.mean(f1_scores) + np.std(f1_scores), alpha=0.2)
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_title('F1-Score per Fold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()
        
        axes[2].plot(folds, aucs, marker='^', linewidth=2, markersize=8, color='purple')
        axes[2].axhline(np.mean(aucs), color='r', linestyle='--', label=f'Mean: {np.mean(aucs):.4f}')
        axes[2].fill_between(folds, np.mean(aucs) - np.std(aucs),
                             np.mean(aucs) + np.std(aucs), alpha=0.2)
        axes[2].set_xlabel('Fold')
        axes[2].set_ylabel('ROC-AUC')
        axes[2].set_title('ROC-AUC per Fold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/cv_results.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/cv_results.png")
        plt.close()


# =============================================================================
# RUN CROSS-VALIDATION
# =============================================================================

print("\n[1/2] Loading dataset...")

df = pd.read_csv('data/gdc_oral_cancer_dataset/processed_labels.csv')

# Suspicious classifier CV
print("\n[2/2] Running 5-Fold CV for Suspicious Classifier...")

df_sus = df[df['suspicious_label'].notna()].reset_index(drop=True)

trainer_sus = CrossValidationTrainer(df_sus, 'suspicious_label', 'suspicious', n_splits=5)
trainer_sus.run_cross_validation()
trainer_sus.save_results('cv_suspicious_results')

print("\n" + "="*80)
print("Running 5-Fold CV for Risk Classifier...")

df_risk = df[df['risk_label'].notna()].reset_index(drop=True)

trainer_risk = CrossValidationTrainer(df_risk, 'risk_label', 'risk', n_splits=5)
trainer_risk.run_cross_validation()
trainer_risk.save_results('cv_risk_results')

print("\n" + "="*80)
print("✓ CROSS-VALIDATION COMPLETE")
print("="*80)
print("\nResults saved to:")
print("  - cv_suspicious_results/")
print("  - cv_risk_results/")
