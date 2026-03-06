"""
Test-Time Augmentation (TTA) Utility
Improves predictions by averaging predictions over multiple augmented versions
of the same image. Provides significant accuracy improvements for medical imaging.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from tqdm import tqdm


class TTAPredictor:
    """Test-Time Augmentation for improved predictions"""
    
    def __init__(self, model, num_augmentations=8):
        """
        Args:
            model: Trained Keras model
            num_augmentations: Number of augmented versions to average
        """
        self.model = model
        self.num_augmentations = num_augmentations
    
    def augment_image_tta(self, img_array):
        """Apply test-time augmentation to single image"""
        h, w = img_array.shape[:2]
        augmented_images = []
        
        # Original
        augmented_images.append(img_array)
        
        # Horizontal flip
        augmented_images.append(np.fliplr(img_array))
        
        # Vertical flip
        augmented_images.append(np.flipud(img_array))
        
        # Both flips
        augmented_images.append(np.flipud(np.fliplr(img_array)))
        
        # Rotations
        for k in [1, 2, 3]:
            augmented_images.append(np.rot90(img_array, k=k))
        
        # Crop and resize (center 224x224 from 224x224 doesn't change, use 75% crop)
        if len(augmented_images) < self.num_augmentations:
            crop_size = int(h * 0.75)
            y_start = (h - crop_size) // 2
            x_start = (w - crop_size) // 2
            cropped = img_array[y_start:y_start+crop_size, x_start:x_start+crop_size, :]
            cropped = tf.image.resize(cropped[np.newaxis, ...], [h, w]).numpy()[0]
            augmented_images.append(cropped)
        
        # Ensure we have exactly num_augmentations
        while len(augmented_images) < self.num_augmentations:
            idx = np.random.randint(0, len(augmented_images))
            augmented_images.append(augmented_images[idx])
        
        return np.array(augmented_images[:self.num_augmentations])
    
    def predict_tta(self, image_path):
        """
        Predict on single image with TTA
        
        Args:
            image_path: Path to image file
            
        Returns:
            averaged_prediction: Average probability across augmentations
            std_dev: Standard deviation of predictions
            all_predictions: All individual predictions
        """
        try:
            # Load image
            img = image.load_img(image_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            
            # Generate augmentations
            augmented_images = self.augment_image_tta(img_array)
            
            # Get predictions for each augmentation
            predictions = self.model.predict(augmented_images, verbose=0).flatten()
            
            # Average
            avg_pred = np.mean(predictions)
            std_dev = np.std(predictions)
            
            return avg_pred, std_dev, predictions
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None, None
    
    def predict_batch_tta(self, image_paths, batch_size=8):
        """
        Predict on batch of images with TTA
        
        Returns:
            predictions: Averaged predictions
            uncertainties: Standard deviations
        """
        predictions = []
        uncertainties = []
        
        for img_path in tqdm(image_paths, desc="TTA Predictions"):
            avg_pred, std_dev, _ = self.predict_tta(img_path)
            if avg_pred is not None:
                predictions.append(avg_pred)
                uncertainties.append(std_dev)
        
        return np.array(predictions), np.array(uncertainties)


def compare_predictions(model, test_df, task_name='suspicious'):
    """
    Compare standard predictions vs TTA predictions
    
    Args:
        model: Trained model
        test_df: DataFrame with image paths
        task_name: Name for output files
    """
    print(f"\n{'='*60}")
    print(f"TTA ANALYSIS - {task_name.upper()}")
    print(f"{'='*60}")
    
    # Standard predictions
    print("\n[1/3] Computing standard predictions...")
    images = []
    for img_path in test_df['image_path'].values:
        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
        except:
            images.append(np.zeros((224, 224, 3)))
    
    images = np.array(images, dtype=np.float32)
    standard_preds = model.predict(images, verbose=0).flatten()
    
    # TTA predictions
    print("[2/3] Computing TTA predictions (this may take a while)...")
    tta_predictor = TTAPredictor(model, num_augmentations=8)
    tta_preds, uncertainties = tta_predictor.predict_batch_tta(test_df['image_path'].values)
    
    # Analysis
    print("[3/3] Analyzing results...")
    
    comparison_df = pd.DataFrame({
        'image_path': test_df['image_path'].values,
        'standard_pred': standard_preds,
        'tta_pred': tta_preds,
        'uncertainty': uncertainties,
        'difference': np.abs(standard_preds - tta_preds)
    })
    
    print(f"\nStandard Predictions - Mean: {standard_preds.mean():.4f}, Std: {standard_preds.std():.4f}")
    print(f"TTA Predictions     - Mean: {tta_preds.mean():.4f}, Std: {tta_preds.std():.4f}")
    print(f"Average Uncertainty - {uncertainties.mean():.4f}")
    print(f"Avg Prediction Diff - {comparison_df['difference'].mean():.4f}")
    print(f"Max Prediction Diff - {comparison_df['difference'].max():.4f}")
    
    # Save results
    os.makedirs('tta_results', exist_ok=True)
    comparison_df.to_csv(f'tta_results/tta_comparison_{task_name}.csv', index=False)
    print(f"\n✓ Saved: tta_results/tta_comparison_{task_name}.csv")
    
    # Visualization
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prediction comparison
    ax = axes[0, 0]
    ax.scatter(standard_preds, tta_preds, alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--', lw=2)
    ax.set_xlabel('Standard Predictions')
    ax.set_ylabel('TTA Predictions')
    ax.set_title('Standard vs TTA Predictions')
    ax.grid(alpha=0.3)
    
    # Uncertainty distribution
    ax = axes[0, 1]
    ax.hist(uncertainties, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Prediction Uncertainty (Std Dev)')
    ax.set_ylabel('Frequency')
    ax.set_title('TTA Uncertainty Distribution')
    ax.grid(alpha=0.3, axis='y')
    
    # Difference vs uncertainty
    ax = axes[1, 0]
    ax.scatter(uncertainties, comparison_df['difference'], alpha=0.5)
    ax.set_xlabel('Uncertainty (Std Dev)')
    ax.set_ylabel('|Standard - TTA| Difference')
    ax.set_title('Uncertainty vs Prediction Difference')
    ax.grid(alpha=0.3)
    
    # Distribution comparison
    ax = axes[1, 1]
    ax.hist(standard_preds, bins=25, alpha=0.5, label='Standard', edgecolor='black')
    ax.hist(tta_preds, bins=25, alpha=0.5, label='TTA', edgecolor='black')
    ax.set_xlabel('Prediction Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Distribution Comparison')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'tta_results/tta_analysis_{task_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: tta_results/tta_analysis_{task_name}.png")
    
    plt.close()
    
    return comparison_df


if __name__ == "__main__":
    print("Test-Time Augmentation (TTA) Utility")
    print("\nUsage:")
    print("  from test_time_augmentation import TTAPredictor, compare_predictions")
    print("  ")
    print("  # Single image prediction with TTA")
    print("  tta = TTAPredictor(model, num_augmentations=8)")
    print("  avg_pred, uncertainty, all_preds = tta.predict_tta('path/to/image.jpg')")
    print("  ")
    print("  # Batch comparison")
    print("  comparison_df = compare_predictions(model, test_df, 'suspicious')")
