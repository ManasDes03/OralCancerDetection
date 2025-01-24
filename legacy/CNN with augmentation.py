import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image
import sys
import io
import contextlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import random

# Define paths
base_dir = r"/home/anshullz/Desktop/SOP/Oral cancer Dataset 2.0"
cancer_dir = os.path.join(base_dir, "CANCER")
non_cancer_dir = os.path.join(base_dir, "NON CANCER")

# Create directories for train, validation, and test sets
def create_split_directories(base_dir):
    # Create main split directories
    split_dirs = ['train', 'validation', 'test']
    classes = ['CANCER', 'NON CANCER']
    
    for split in split_dirs:
        split_path = os.path.join(base_dir, split)
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        os.makedirs(split_path)
        
        # Create class subdirectories
        for cls in classes:
            os.makedirs(os.path.join(split_path, cls))

# Function to check and remove corrupted images
def remove_corrupted_images(directory):
    corrupted = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f'Corrupted image found: {file_path}')
            corrupted.append(file_path)
    
    for file_path in corrupted:
        os.remove(file_path)
        print(f'Removed corrupted image: {file_path}')

# Split data into train, validation, and test sets
def split_data(source_dir, train_dir, val_dir, test_dir):
    files = os.listdir(source_dir)
    random.shuffle(files)
    
    total = len(files)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    
    train_files = files[:train_size]
    val_files = files[train_size:train_size + val_size]
    test_files = files[train_size + val_size:]
    
    # Copy files to respective directories
    for f in train_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(test_dir, f))
    
    return len(train_files), len(val_files), len(test_files)

# Set random seed for reproducibility
random.seed(42)

# Create split directories
create_split_directories(base_dir)

# Remove corrupted images
remove_corrupted_images(cancer_dir)
remove_corrupted_images(non_cancer_dir)

# Split and copy data
print("Splitting data...")
# Split cancer data
cancer_train = os.path.join(base_dir, 'train', 'CANCER')
cancer_val = os.path.join(base_dir, 'validation', 'CANCER')
cancer_test = os.path.join(base_dir, 'test', 'CANCER')
c_train, c_val, c_test = split_data(cancer_dir, cancer_train, cancer_val, cancer_test)

# Split non-cancer data
non_cancer_train = os.path.join(base_dir, 'train', 'NON CANCER')
non_cancer_val = os.path.join(base_dir, 'validation', 'NON CANCER')
non_cancer_test = os.path.join(base_dir, 'test', 'NON CANCER')
nc_train, nc_val, nc_test = split_data(non_cancer_dir, non_cancer_train, non_cancer_val, non_cancer_test)

# Print split statistics
print("\nData Split Statistics:")
print(f"Training samples: {c_train + nc_train}")
print(f"Validation samples: {c_val + nc_val}")
print(f"Test samples: {c_test + nc_test}")

# Set up data generators
img_height, img_width = 224, 224
batch_size = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

def adjust_contrast_brightness(image):
    # Adjust the contrast and brightness of the image
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)  # Contrast adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)       # Brightness adjustment
    return image

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    channel_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.2, 1.5], 
    preprocessing_function=adjust_contrast_brightness  
)


# Create data generators without augmentation for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Set up generators
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    subset='training'  # Specify the subset for training
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(base_dir, 'validation'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Define the model using functional API
inputs = Input(shape=(img_height, img_width, 3))
x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

try:
    # Train the model
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator
    )
    
    # Evaluate on all sets
    training_accuracy = history.history['accuracy'][-1]
    validation_accuracy = history.history['val_accuracy'][-1]
    
    # Evaluate on test set
    test_results = model.evaluate(test_generator)
    test_loss, test_accuracy = test_results
    
    # Generate predictions for confusion matrix
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Print comprehensive results
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {training_accuracy * 100:.2f}%")
    print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Performance gaps
    train_val_diff = abs(training_accuracy - validation_accuracy) * 100
    train_test_diff = abs(training_accuracy - test_accuracy) * 100
    print(f"\n=== Performance Gaps ===")
    print(f"Train-Validation Gap: {train_val_diff:.2f}%")
    print(f"Train-Test Gap: {train_test_diff:.2f}%")
    
    # Detailed metrics
    print("\n=== Detailed Test Set Metrics ===")
    cm = confusion_matrix(y_true, y_pred_classes)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=['Non-Cancer', 'Cancer']))
    
    # Model evaluation
    if training_accuracy > validation_accuracy * 1.1 or training_accuracy > test_accuracy * 1.1:
        print("\nWarning: The model might be overfitting. Consider using regularization techniques.")
    elif validation_accuracy > training_accuracy or test_accuracy > training_accuracy:
        print("\nNote: Validation/Test accuracy is higher than training accuracy. This is unusual but can happen.")
    else:
        print("\nThe model seems to be generalizing well.")

    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.axhline(y=test_accuracy, color='r', linestyle='--', label='Test Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=test_loss, color='r', linestyle='--', label='Test Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model
    model.save('oral_cancer_detection_model.h5')
    print("\nModel training completed and saved.")

except Exception as e:
    print(f"An error occurred during training: {str(e)}")

# Clean up split directories after training (optional)
# shutil.rmtree(os.path.join(base_dir, 'train'))
# shutil.rmtree(os.path.join(base_dir, 'validation'))
# shutil.rmtree(os.path.join(base_dir, 'test'))
