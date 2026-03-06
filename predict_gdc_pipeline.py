"""
Two-stage GDC inference pipeline.

Stage 1: Suspicious vs Non-suspicious
Stage 2: (if suspicious) High-risk vs Low-risk

Final classes:
- Non-suspicious
- Suspicious-Low-risk
- Suspicious-High-risk
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

SUSPICIOUS_DIR = "gdc_suspicious_results"
RISK_DIR = "gdc_risk_results"


def _build_suspicious_model():
    """Rebuild suspicious classifier architecture (EfficientNetB0 head)."""
    from tensorflow.keras import layers, Model
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=False, weights=None)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(96, activation='relu')(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inputs, outputs)


def _build_risk_model():
    """Rebuild risk classifier architecture (EfficientNetB0 head)."""
    from tensorflow.keras import layers, Model
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3), include_top=False, weights=None)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = backbone(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return Model(inputs, outputs)


_FINE_TUNE_LAYERS = 60  # must match CONFIG['fine_tune_trainable_layers'] in training scripts


def _load_best_model(results_dir: str, model_type: str):
    """Build model architecture and load saved weights (weights-only format)."""
    # Use clear_session so layer names are consistent with training (e.g., 'efficientnetb0')
    keras.backend.clear_session()

    if model_type == "suspicious":
        model = _build_suspicious_model()
    else:
        model = _build_risk_model()

    backbone = model.layers[1]  # EfficientNetB0 is always layer index 1

    # Phase 2 weights were saved while backbone was partially unfrozen.
    # Must recreate that trainability state before loading, then re-freeze for inference.
    for weights_name in ["best_phase2_weights.h5", "best_phase2.h5"]:
        wpath = os.path.join(results_dir, weights_name)
        if os.path.exists(wpath):
            backbone.trainable = True
            fine_tune_at = max(0, len(backbone.layers) - _FINE_TUNE_LAYERS)
            for layer in backbone.layers[:fine_tune_at]:
                layer.trainable = False
            model.load_weights(wpath)
            backbone.trainable = False  # freeze for inference
            print(f"  [{model_type}] Loaded fine-tuned weights: {os.path.basename(wpath)}")
            return model

    # Fall back to Phase 1 weights (backbone was fully frozen when saved - no state change needed)
    for weights_name in ["best_phase1_weights.h5", "best_phase1.h5"]:
        wpath = os.path.join(results_dir, weights_name)
        if os.path.exists(wpath):
            model.load_weights(wpath)
            print(f"  [{model_type}] Loaded phase-1 weights: {os.path.basename(wpath)}")
            return model

    raise FileNotFoundError(
        f"No weights found in {results_dir}. Run training first.")


def _load_threshold(results_dir: str, default_threshold: float = 0.5):
    results_json = os.path.join(results_dir, "results.json")
    if not os.path.exists(results_json):
        return default_threshold
    with open(results_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload.get("best_threshold", default_threshold))


def _detect_preprocess_and_size(model):
    model_name = model.name.lower()
    if 'efficientnet' in model_name:
        preprocess_fn = keras.applications.efficientnet.preprocess_input
    elif 'mobilenet' in model_name:
        preprocess_fn = keras.applications.mobilenet_v2.preprocess_input
    else:
        # fallback: search in layers
        layer_names = ' '.join([layer.name.lower() for layer in model.layers])
        if 'efficientnet' in layer_names:
            preprocess_fn = keras.applications.efficientnet.preprocess_input
        else:
            preprocess_fn = keras.applications.mobilenet_v2.preprocess_input

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    height = int(input_shape[1])
    width = int(input_shape[2])
    return preprocess_fn, (height, width)


def _preprocess_image(image_path: str, img_size, preprocess_fn):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = tf.keras.utils.load_img(image_path, target_size=img_size)
    arr = tf.keras.utils.img_to_array(image)
    arr = preprocess_fn(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def classify(image_path: str):
    suspicious_model = _load_best_model(SUSPICIOUS_DIR, "suspicious")
    risk_model = _load_best_model(RISK_DIR, "risk")

    suspicious_threshold = _load_threshold(SUSPICIOUS_DIR, 0.5)
    risk_threshold = _load_threshold(RISK_DIR, 0.5)

    suspicious_preprocess, suspicious_size = _detect_preprocess_and_size(suspicious_model)
    risk_preprocess, risk_size = _detect_preprocess_and_size(risk_model)

    x_suspicious = _preprocess_image(image_path, suspicious_size, suspicious_preprocess)
    x_risk = _preprocess_image(image_path, risk_size, risk_preprocess)

    suspicious_prob = float(suspicious_model.predict(x_suspicious, verbose=0).reshape(-1)[0])
    is_suspicious = suspicious_prob >= suspicious_threshold

    risk_prob = float(risk_model.predict(x_risk, verbose=0).reshape(-1)[0])
    is_high_risk = risk_prob >= risk_threshold

    if not is_suspicious:
        final_class = "Non-suspicious"
    elif is_high_risk:
        final_class = "Suspicious-High-risk"
    else:
        final_class = "Suspicious-Low-risk"

    return {
        "image": image_path,
        "final_class": final_class,
        "suspicious_probability": suspicious_prob,
        "suspicious_threshold": suspicious_threshold,
        "risk_probability": risk_prob,
        "risk_threshold": risk_threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/gdc.jpg", help="Path to image for classification")
    args = parser.parse_args()

    result = classify(args.image)

    print("=" * 72)
    print("GDC TWO-STAGE CLASSIFICATION")
    print("=" * 72)
    print(f"Image: {result['image']}")
    print(f"Final class: {result['final_class']}")
    print(f"Suspicious probability: {result['suspicious_probability']:.4f} (thr={result['suspicious_threshold']:.4f})")
    print(f"Risk probability: {result['risk_probability']:.4f} (thr={result['risk_threshold']:.4f})")


if __name__ == "__main__":
    main()
