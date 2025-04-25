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
import shutil
import random

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define paths
base_dir = "/home/anshullz/Desktop/Sri Lankan Dataset"
output_dir = os.path.join(base_dir, "Structured_Dataset")
reduced_dir = os.path.join(base_dir, "Reduced_Dataset_25percent")

# Create reduced dataset directories
def create_reduced_dataset(source_dir, target_dir, sampling_ratio=0.25):
    """
    Creates a reduced dataset by sampling a percentage of images from each class
    
    Args:
        source_dir: Original dataset directory
        target_dir: Output directory for reduced dataset
        sampling_ratio: Percentage of images to keep (0.25 = 25%)
    """
    # Clear existing reduced directory if it exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # Create main split directories
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "test"), exist_ok=True)
    
    # Process training data
    train_source = os.path.join(source_dir, "train")
    train_target = os.path.join(target_dir, "train")
    
    for class_name in os.listdir(train_source):
        class_dir = os.path.join(train_source, class_name)
        if os.path.isdir(class_dir):
            # Create class directory in the target
            os.makedirs(os.path.join(train_target, class_name), exist_ok=True)
            
            # Get all image files
            all_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Calculate how many to keep
            num_to_keep = int(len(all_images) * sampling_ratio)
            # Sample randomly
            sampled_images = random.sample(all_images, num_to_keep)
            
            # Copy the sampled images
            for img in sampled_images:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(train_target, class_name, img)
                shutil.copy2(src_path, dst_path)
            
            print(f"Train - {class_name}: Reduced from {len(all_images)} to {num_to_keep} images")
    
    # Process test data
    test_source = os.path.join(source_dir, "test")
    test_target = os.path.join(target_dir, "test")
    
    for class_name in os.listdir(test_source):
        class_dir = os.path.join(test_source, class_name)
        if os.path.isdir(class_dir):
            # Create class directory in the target
            os.makedirs(os.path.join(test_target, class_name), exist_ok=True)
            
            # Get all image files
            all_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Calculate how many to keep
            num_to_keep = int(len(all_images) * sampling_ratio)
            # Sample randomly
            sampled_images = random.sample(all_images, num_to_keep)
            
            # Copy the sampled images
            for img in sampled_images:
                src_path = os.path.join(class_dir, img)
                dst_path = os.path.join(test_target, class_name, img)
                shutil.copy2(src_path, dst_path)
            
            print(f"Test - {class_name}: Reduced from {len(all_images)} to {num_to_keep} images")
    
    return target_dir

# Create the reduced dataset
print("Creating reduced dataset (25% of original)...")
reduced_dataset_dir = create_reduced_dataset(output_dir, reduced_dir, 0.25)
print(f"Reduced dataset created at: {reduced_dataset_dir}")

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

# Use the REDUCED dataset for training
train_generator = train_datagen.flow_from_directory(
    os.path.join(reduced_dir, "train"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    os.path.join(reduced_dir, "train"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(reduced_dir, "test"),
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
    patience=5,  # Increased patience for smaller dataset
    restore_best_weights=True
)

# Train the model
try:
    history = model.fit(
        train_generator,
        epochs=30,  # Increased epochs for smaller dataset
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
    plt.title('Model Accuracy (25% Dataset)')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (25% Dataset)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'training_history_25percent.png'))
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
    model.save(os.path.join(base_dir, 'oral_cavity_mobilenet_model_25percent.h5'))
    print("\nKeras model saved.")

# Save class labels to labels.txt
class_labels = list(train_generator.class_indices.keys())
labels_path = os.path.join(base_dir, 'labels_25percent.txt')
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
    tflite_path = os.path.join(base_dir, 'oral_cavity_mobilenet_model_25percent.tflite')
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

# ----- VISUALIZE MODEL PREDICTIONS ON SAMPLE IMAGES -----

def visualize_predictions(model, test_dir, class_indices, num_images=8):
    """
    Visualizes model predictions on sample images from the test set
    
    Args:
        model: Trained Keras model
        test_dir: Directory containing test images
        class_indices: Dictionary mapping class indices to class names
        num_images: Number of images to visualize
    """
    # Reverse the class indices dictionary
    idx_to_class = {v: k for k, v in class_indices.items()}
    
    # Create a figure for visualization
    plt.figure(figsize=(15, 12))
    
    # Track how many images we've plotted
    plot_count = 0
    
    # Get a few images from each class
    classes = list(class_indices.keys())
    images_per_class = max(1, num_images // len(classes))
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Get image files from this class
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Take a sample of images
        sample_size = min(images_per_class, len(image_files))
        if sample_size == 0:
            continue
            
        sampled_images = random.sample(image_files, sample_size)
        
        for img_file in sampled_images:
            img_path = os.path.join(class_dir, img_file)
            
            # Load and preprocess the image
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(img_height, img_width))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = img_array / 255.0  # Rescale
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            
            # Make prediction
            predictions = model.predict(img_array)
            pred_class_idx = np.argmax(predictions[0])
            pred_class = idx_to_class[pred_class_idx]
            confidence = predictions[0][pred_class_idx] * 100
            
            # Determine if prediction is correct
            is_correct = (pred_class == class_name)
            title_color = 'green' if is_correct else 'red'
            
            # Plot the image
            plt.subplot(4, 2, plot_count + 1)
            plt.imshow(img)
            plt.title(f"True: {class_name}\nPred: {pred_class} ({confidence:.1f}%)", 
                    color=title_color)
            plt.axis('off')
            
            plot_count += 1
            if plot_count >= num_images:
                break
                
        if plot_count >= num_images:
            break
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'sample_predictions_25percent.png'))
    plt.show()

# Visualize sample predictions
print("\nVisualizing sample predictions...")
test_dir = os.path.join(reduced_dir, "test")
try:
    visualize_predictions(model, test_dir, train_generator.class_indices)
    print("Prediction visualization completed and saved!")
except Exception as e:
    print(f"Error visualizing predictions: {str(e)}")

print("\nAll operations completed successfully!")