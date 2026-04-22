"""
Ensemble Inference: Combine Positive & Negative Model Predictions
==================================================================

Two-model approach for Stage 1 (Suspicious Detection):
- Model A (Positive): "Is this SUSPICIOUS?"
- Model B (Negative): "Is this NON-SUSPICIOUS?"

Uses multiple fusion strategies to reduce false positives through agreement voting.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import argparse
import json
from ensemble_fusion import ensemble_predict


def load_model_from_weights(backbone_name, weights_path):
    """Load model architecture and weights (weights-only format)."""
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras import layers, Model, mixed_precision
    
    # Reconstruct architecture
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
    
    # Load weights
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"✅ Loaded weights from {weights_path}")
        return model
    else:
        raise FileNotFoundError(f"Weights not found: {weights_path}")


def load_model_with_fallback(model_name, requested_path, fallback_paths):
    """Try requested checkpoint first, then fallback candidates."""
    attempted = []

    # Keep order but deduplicate paths.
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
                print(f"⚠️  {model_name}: requested checkpoint failed/unavailable, using fallback: {path}")
            return model, path
        except Exception as e:
            attempted.append(f"{path} ({e})")

    raise RuntimeError(
        f"Could not load any checkpoint for {model_name}. Attempts:\n  - " + "\n  - ".join(attempted)
    )


def preprocess_image(image_path):
    """Load and preprocess a single image."""
    from tensorflow.keras.applications.efficientnet import preprocess_input
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    
    return image


def predict_with_tta(model, image, use_tta=True):
    """
    Predict with optional Test-Time Augmentation (TTA).
    
    TTA: average predictions from original + horizontal flip.
    """
    # Original
    image = tf.expand_dims(image, 0)
    pred_orig = model.predict(image, verbose=0)[0, 0]
    
    if use_tta:
        # Flipped
        image_flipped = tf.image.flip_left_right(tf.squeeze(image, 0))
        image_flipped = tf.expand_dims(image_flipped, 0)
        pred_flip = model.predict(image_flipped, verbose=0)[0, 0]
        
        # Average
        pred = (pred_orig + pred_flip) / 2.0
    else:
        pred = pred_orig
    
    return float(pred)


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference for suspicious detection using negative test approach')
    parser.add_argument('--image', type=str, required=True, help='Path to image to predict')
    parser.add_argument('--positive-weights', type=str, default='gdc_suspicious_results/suspicious_classifier_final.h5',
                       help='Path to positive model weights')
    parser.add_argument('--negative-weights', type=str, default='gdc_suspicious_results_negative/suspicious_neg_final.h5',
                       help='Path to negative model weights')
    parser.add_argument('--threshold', type=float, default=0.5, help='Decision threshold')
    parser.add_argument('--use-tta', action='store_true', default=True, help='Use test-time augmentation')
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=['agreement_voting', 'contradiction_score', 'weighted_average'],
                       help='Fusion strategies to use')
    
    args = parser.parse_args()
    
    # Check image exists
    if not os.path.exists(args.image):
        print(f"❌ Image not found: {args.image}")
        return
    
    print("\n" + "="*70)
    print("ENSEMBLE INFERENCE: SUSPICIOUS DETECTION WITH NEGATIVE TEST")
    print("="*70)
    print(f"\n📷 Image: {args.image}")
    
    # Load models
    print("\n🔧 Loading models...")
    print("  Positive model (\"Is this suspicious?\")...")
    try:
        positive_fallbacks = [
            'gdc_suspicious_results/best_phase2_weights.h5',
            'gdc_suspicious_results/best_phase1_weights.h5',
            'gdc_suspicious_results/best_phase2.h5',
            'gdc_suspicious_results/best_phase1.h5',
            'gdc_suspicious_results/suspicious_classifier_final.h5',
        ]
        model_positive, positive_loaded_path = load_model_with_fallback(
            'positive model', args.positive_weights, positive_fallbacks
        )
    except Exception as e:
        print(f"❌ Error loading positive model: {e}")
        return
    
    print("  Negative model (\"Is this non-suspicious?\")...")
    try:
        negative_fallbacks = [
            'gdc_suspicious_results_negative/best_phase2_weights.h5',
            'gdc_suspicious_results_negative/best_phase1_weights.h5',
            'gdc_suspicious_results_negative/best_phase2.h5',
            'gdc_suspicious_results_negative/best_phase1.h5',
            'gdc_suspicious_results_negative/suspicious_neg_final.h5',
        ]
        model_negative, negative_loaded_path = load_model_with_fallback(
            'negative model', args.negative_weights, negative_fallbacks
        )
    except Exception as e:
        print(f"❌ Error loading negative model: {e}")
        return

    print(f"  Positive checkpoint used: {positive_loaded_path}")
    print(f"  Negative checkpoint used: {negative_loaded_path}")
    
    # Preprocess image
    print("\n📸 Preprocessing image...")
    image = preprocess_image(args.image)
    print(f"  Image shape: {image.shape}")
    
    # Get predictions from both models
    print("\n🧠 Running predictions...")
    p_suspicious = predict_with_tta(model_positive, image, use_tta=args.use_tta)
    p_not_suspicious = predict_with_tta(model_negative, image, use_tta=args.use_tta)
    
    print(f"  Positive model (P(suspicious)): {p_suspicious:.4f}")
    print(f"  Negative model (P(non-suspicious)): {p_not_suspicious:.4f}")
    print(f"            ↓ converted ↓")
    print(f"  Negative model (P(suspicious) = 1-{p_not_suspicious:.4f}): {1.0-p_not_suspicious:.4f}")
    
    # Run ensemble fusion
    print("\n⚙️ Ensemble Fusion Strategies:")
    print("-" * 70)
    
    ensemble_result = ensemble_predict(
        p_suspicious,
        p_not_suspicious,
        strategies=args.strategies,
        threshold=args.threshold
    )
    
    # Print individual strategy results
    for strat_name, result in ensemble_result['individual_strategies'].items():
        print(f"\n  Strategy: {strat_name.upper()}")
        print(f"    Confidence: {result['combined_confidence']:.4f}")
        print(f"    Reasoning: {result.get('reasoning', result.get('summary', 'N/A'))}")
    
    # Print ensemble summary
    print("\n" + "="*70)
    print("ENSEMBLE SUMMARY")
    print("="*70)
    print(f"\nAverage Confidence (across all strategies): {ensemble_result['average_confidence']:.4f}")
    print(f"Votes for Suspicious: {ensemble_result['votes_suspicious']}/{ensemble_result['total_strategies']}")
    print(f"Consensus Strength: {ensemble_result['consensus_strength']*100:.1f}%")
    print(f"\n🎯 FINAL DECISION: {ensemble_result['final_decision']}")
    print(f"   {ensemble_result['summary']}")
    
    # Clinical recommendation
    print("\n" + "="*70)
    print("CLINICAL RECOMMENDATION")
    print("="*70)
    
    avg_conf = ensemble_result['average_confidence']
    consensus = ensemble_result['consensus_strength']
    
    if ensemble_result['final_decision'] == 'SUSPICIOUS':
        if consensus >= 0.9:
            print(f"🚨 HIGH CONFIDENCE: {consensus*100:.0f}% of strategies agree (avg={avg_conf:.3f})")
            print("   → REFER TO SPECIALIST for detailed examination")
        elif consensus >= 0.5:
            print(f"⚠️  MODERATE CONFIDENCE: {consensus*100:.0f}% of strategies agree (avg={avg_conf:.3f})")
            print("   → FLAG FOR REVIEW - models show some disagreement")
            print("   → Consider specialist consultation, may warrant follow-up")
        else:
            print(f"❓ LOW CONFIDENCE: {consensus*100:.0f}% of strategies agree (avg={avg_conf:.3f})")
            print("   → BORDERLINE CASE - models disagree significantly")
            print("   → Recommend human expert review before specialist referral")
    else:  # NON-SUSPICIOUS
        if consensus >= 0.9:
            print(f"✅ HIGH CONFIDENCE: {consensus*100:.0f}% of strategies agree (avg={avg_conf:.3f})")
            print("   → LIKELY SAFE - routine follow-up only")
        else:
            print(f"⚠️  MODERATE CONFIDENCE: {consensus*100:.0f}% of strategies agree (avg={avg_conf:.3f})")
            print("   → BORDERLINE - models show some concern")
            print("   → Consider conservative approach, follow-up imaging may be warranted")
    
    # Save results
    output_json = {
        'image': args.image,
        'positive_checkpoint_used': positive_loaded_path,
        'negative_checkpoint_used': negative_loaded_path,
        'positive_model_output': float(p_suspicious),
        'negative_model_output': float(p_not_suspicious),
        'ensemble_average_confidence': float(ensemble_result['average_confidence']),
        'consensus_strength': float(ensemble_result['consensus_strength']),
        'votes_suspicious': ensemble_result['votes_suspicious'],
        'total_strategies': ensemble_result['total_strategies'],
        'final_decision': ensemble_result['final_decision'],
        'threshold': args.threshold,
        'strategies_used': args.strategies
    }
    
    # Save per-strategy results
    for strat_name, result in ensemble_result['individual_strategies'].items():
        output_json[f'strategy_{strat_name}'] = {
            'confidence': float(result['combined_confidence']),
            'reasoning': result.get('reasoning', result.get('summary', 'N/A'))
        }
    
    output_file = os.path.join('gdc_suspicious_results_ensemble', 'last_prediction.json')
    os.makedirs('gdc_suspicious_results_ensemble', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n📊 Results saved to {output_file}")


if __name__ == '__main__':
    main()
