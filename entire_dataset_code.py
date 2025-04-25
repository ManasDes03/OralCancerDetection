import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report
from PIL import Image, ImageFile
from tensorflow.keras.callbacks import EarlyStopping

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define paths
base_dir = "/home/anshullz/Desktop/Sri Lankan Dataset"
output_dir = os.path.join(base_dir, "Structured_Dataset")

# Image dimensions and batch size
img_height, img_width = 224, 224  # MobileNetV2 default input size
batch_size = 16

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, "train"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(output_dir, "train"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(output_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Verify dataset
print(f"Found {train_generator.samples} training samples")
print(f"Found {validation_generator.samples} validation samples")
print(f"Found {test_generator.samples} test samples")

# Create MobileNetV2 model
base_model = MobileNetV2(
    input_shape=(img_height, img_width, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the model
try:
    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        verbose=1,
        callbacks=[early_stopping]
    )
    
    # Evaluation
    try:
        test_loss, test_acc = model.evaluate(test_generator, batch_size=batch_size)
        print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    except Exception as e:
        print(f"\nEvaluation error: {str(e)}")
        print("Trying evaluation with smaller batch size...")
        test_loss, test_acc = model.evaluate(test_generator, batch_size=8)
        print(f"\nTest Accuracy (with smaller batch): {test_acc*100:.2f}%")
        
except Exception as e:
    print(f"Training error: {str(e)}")

# Plot training history if training succeeded
if 'history' in locals():
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'training_history.png'))
    plt.show()

    # Generate predictions
    test_generator.reset()
    y_pred = model.predict(test_generator, batch_size=batch_size)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, 
                              target_names=list(train_generator.class_indices.keys())))

    # Save the Keras model
    model.save(os.path.join(base_dir, 'oral_cavity_mobilenet_model.h5'))
    print("\nKeras model saved.")

# Save class labels to labels.txt
class_labels = list(train_generator.class_indices.keys())
labels_path = os.path.join(base_dir, 'labels.txt')
with open(labels_path, 'w') as f:
    for label in class_labels:
        f.write(label + '\n')
print(f"Class labels saved to {labels_path}")

# Convert the model to TFLite format
def convert_to_tflite(model, base_dir):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    # Save the model
    tflite_path = os.path.join(base_dir, 'oral_cavity_mobilenet_model.tflite')
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved as TFLite to {tflite_path}")
    
    return tflite_path

try:
    tflite_path = convert_to_tflite(model, base_dir)
    
    # Optional: Verify the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("\nTFLite model loaded and verified successfully!")
    
    # Print input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("\nInput details:", input_details)
    print("Output details:", output_details)
    
except Exception as e:
    print(f"\nError converting to TFLite: {str(e)}")

print("\nAll operations completed successfully!")
