

"""
main.py
--------
Oral Cancer Detection: Publication-Ready 10-Fold Cross-Validation Pipeline

- Robust multi-class image classification
- Advanced metrics, class imbalance analysis
- Model checkpointing, experiment logging, misclassification visualization
- Configurable via config.yaml
"""

import yaml
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
import tensorflow as tf
from dataloaders.dataloaders import get_classification_data_loader
from model.model import OralCancerModel
import numpy as np
import pandas as pd
import os

def load_config(config_path="config.yaml"):
    """Load YAML configuration file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    
    Args:
        alpha: Weighting factor for rare class (cancer). Higher = more focus on cancer.
        gamma: Focusing parameter. Higher = more focus on hard examples.
    
    Returns:
        Focal loss function
    """
    def focal_loss_fn(y_true, y_pred):
        # Ensure consistent data types
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1.0 - 1e-8)
        
        # Calculate cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        # Calculate focal weight
        p_t = tf.where(tf.equal(y_true, 1.0), y_pred, 1 - y_pred)
        alpha_t = tf.where(tf.equal(y_true, 1.0), alpha, 1 - alpha)
        
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Calculate focal loss
        focal_loss = focal_weight * cross_entropy
        
        return tf.reduce_mean(focal_loss)
    
    return focal_loss_fn

def compile_model(model, config):
    """Compile the Keras model with optimizer, loss, and metrics from config."""
    # Use Focal Loss for severe class imbalance (cancer detection)
    # Alpha=0.75 gives more weight to cancer class, gamma=2.0 focuses on hard examples
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss(alpha=0.75, gamma=2.0),  # Focal loss for imbalanced data
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

def get_full_dataset(config):
    """Load all file paths and labels for cross-validation, with label encoding."""
    # Loads all file paths and labels for cross-validation
    image_size = tuple(config['dataset']['image_size'])
    dataset_name = config['dataset']['name']
    batch_size = config['dataset']['batch_size']
    if dataset_name == "Sri_Lankan":
        images_dir = config['paths'][dataset_name]['images_dir']
        annotations_file = config['paths'][dataset_name]['annotations_file']
        df = pd.read_csv(annotations_file)
        valid_file_paths = []
        valid_labels = []
        for fname, label in zip(df['Image Name'], df['Category']):
            img_path = os.path.join(images_dir, str(fname) + ".jpg" if not str(fname).endswith('.jpg') else str(fname))
            if os.path.isfile(img_path):
                valid_file_paths.append(img_path)
                # Map to binary: OCA->1 (cancer), others->0 (non-cancer)
                if label == 'OCA':
                    valid_labels.append(1)
                else:
                    valid_labels.append(0)
        labels = np.array(valid_labels)
        label_mapping = {0: 'non-cancer', 1: 'cancer'}
        file_paths = valid_file_paths
    else:
        raise NotImplementedError("10-fold CV only implemented for Sri_Lankan dataset in this script.")
    return np.array(file_paths), np.array(labels), label_mapping

def process_image(file_path, label, image_size):
    """Read, decode, resize, and normalize an image for the pipeline."""
    # Log the filename before decoding for robust error tracing
    tf.print("[DECODE] Attempting to decode:", file_path)
    image = tf.io.read_file(file_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3)
    except Exception as e:
        tf.print("[ERROR] Could not decode image:", file_path, ". Skipping. Error:", e)
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, image_size)
    image = image / 255.0
    return image, label

def make_tf_dataset(file_paths, labels, image_size, batch_size, augment=False):
    """Create a tf.data.Dataset from file paths and labels."""
    def _process(file_path, label):
        return process_image(file_path, label, image_size)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(_process)
    ds = ds.batch(batch_size, drop_remainder=False)
    return ds


def main():
    """Main entry point: runs stratified k-fold cross-validation, training, evaluation, and reporting."""
    import matplotlib.pyplot as plt

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore')

    config = load_config()
    file_paths, labels, label_mapping = get_full_dataset(config)
    num_classes = config['dataset']['num_classes']
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    epochs = config.get('training', {}).get('epochs', 20)

    # Single train/val split: 80% train, 20% val
    from sklearn.model_selection import train_test_split
    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )

    def make_balanced_loader(files, labels, augment=False):
        """Create a balanced dataset by oversampling cancer cases with aggressive augmentation."""
        import dataloaders.dataloaders
        
        # Separate cancer and non-cancer samples
        files_array = np.array(files)
        labels_array = np.array(labels)
        
        cancer_files = files_array[labels_array == 1]
        non_cancer_files = files_array[labels_array == 0]
        
        # For cancer samples: replicate multiple times with aggressive augmentation
        cancer_multiplier = 8 if augment else 1  # Oversample cancer 8x during training
        
        final_files = list(non_cancer_files)
        final_labels = [0] * len(non_cancer_files)
        
        # Add multiple copies of cancer samples
        for _ in range(cancer_multiplier):
            final_files.extend(cancer_files)
            final_labels.extend([1] * len(cancer_files))
        
        if augment:
            # More aggressive augmentation for training
            ds = tf.data.Dataset.from_tensor_slices((final_files, final_labels))
            ds = ds.map(lambda f, l: dataloaders.dataloaders.process_image_aug(f, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            # No augmentation for validation
            ds = tf.data.Dataset.from_tensor_slices((final_files, final_labels))
            ds = ds.map(lambda f, l: dataloaders.dataloaders.process_image(f, l, image_size), num_parallel_calls=tf.data.AUTOTUNE)
            
        # Shuffle more aggressively for training
        if augment:
            ds = ds.shuffle(buffer_size=len(final_files), seed=42)
        ds = ds.batch(batch_size, drop_remainder=False)
        return ds

    train_ds = make_balanced_loader(train_files, train_labels, augment=True)
    val_ds = make_balanced_loader(val_files, val_labels, augment=False)

    # Calculate class weights to handle severe imbalance
    from sklearn.utils.class_weight import compute_class_weight
    unique_labels = np.unique(train_labels)
    class_weights_array = compute_class_weight('balanced', classes=unique_labels, y=train_labels)
    class_weights = {int(unique_labels[i]): class_weights_array[i] for i in range(len(unique_labels))}
    
    print(f"Class distribution in training set:")
    print(f"Non-cancer (0): {np.sum(train_labels == 0)} samples")
    print(f"Cancer (1): {np.sum(train_labels == 1)} samples")
    print(f"Class weights: {class_weights}")

    model = OralCancerModel(config)
    compile_model(model, config)

    # Custom callback to monitor cancer recall specifically
    class CancerRecallCallback(tf.keras.callbacks.Callback):
        def __init__(self, val_ds, val_labels):
            self.val_ds = val_ds
            self.val_labels = val_labels
            self.best_cancer_recall = 0.0
            
        def on_epoch_end(self, epoch, logs=None):
            # Get predictions on validation set
            y_pred_probs = self.model.predict(self.val_ds, verbose=0)
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            
            # Calculate cancer-specific metrics
            cancer_mask = self.val_labels == 1
            if np.sum(cancer_mask) > 0:
                cancer_recall = np.sum((y_pred == 1) & cancer_mask) / np.sum(cancer_mask)
                cancer_precision = np.sum((y_pred == 1) & cancer_mask) / max(np.sum(y_pred == 1), 1)
                
                print(f"\nEpoch {epoch + 1} - Cancer Recall: {cancer_recall:.4f}, Cancer Precision: {cancer_precision:.4f}")
                
                if cancer_recall > self.best_cancer_recall:
                    self.best_cancer_recall = cancer_recall
                    print(f"New best cancer recall: {cancer_recall:.4f}")
    
    # Multiple callbacks for comprehensive monitoring
    cancer_callback = CancerRecallCallback(val_ds, val_labels)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=7,  # Increase patience for imbalanced learning
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    
    # Reduce learning rate on plateau
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=2,
        class_weight=class_weights,  # Apply class weights to penalize missed cancer cases
        callbacks=[cancer_callback, early_stopping, lr_scheduler]
    )

    # Save training history
    os.makedirs('fold_results', exist_ok=True)
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('fold_results/training_history.csv', index=False)

    # Evaluate on validation set
    val_pred = []
    val_true = []
    val_prob = []
    for images, labels_batch in val_ds:
        preds = model.predict(images)
        val_prob.extend(preds.flatten())
        val_pred.extend((preds.flatten() > 0.5).astype(int))
        val_true.extend(labels_batch.numpy())

    acc = accuracy_score(val_true, val_pred)
    cm = confusion_matrix(val_true, val_pred)
    report = classification_report(val_true, val_pred, digits=4)
    auc = roc_auc_score(val_true, val_prob)
    fpr, tpr, _ = roc_curve(val_true, val_prob)

    # Save metrics and plots
    with open('fold_results/results.txt', 'w') as f:
        f.write(f'Validation Accuracy: {acc:.4f}\n')
        f.write(f'AUC: {auc:.4f}\n')
        f.write('Confusion Matrix:\n')
        f.write(np.array2string(cm))
        f.write('\nClassification Report:\n')
        f.write(report)

    # Plot confusion matrix
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(label_mapping.values()), yticklabels=list(label_mapping.values()))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('fold_results/confusion_matrix.png')
    plt.close()

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig('fold_results/roc_curve.png')
    plt.close()

    # Plot accuracy/loss curves
    acc_hist = history.history.get('accuracy', [])
    val_acc_hist = history.history.get('val_accuracy', [])
    loss_hist = history.history.get('loss', [])
    val_loss_hist = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc_hist) + 1)
    plt.figure()
    plt.plot(epochs_range, acc_hist, label='Train Accuracy')
    plt.plot(epochs_range, val_acc_hist, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig('fold_results/accuracy.png')
    plt.close()
    plt.figure()
    plt.plot(epochs_range, loss_hist, label='Train Loss')
    plt.plot(epochs_range, val_loss_hist, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('fold_results/loss.png')
    plt.close()

    print("[INFO] Training complete. Results and plots saved in fold_results/.")


if __name__ == "__main__":
    main()
