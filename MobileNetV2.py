import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Define paths
base_dir = r"/home/anshullz/Desktop/SOP/Oral cancer Dataset 2.0"
cancer_dir = os.path.join(base_dir, "CANCER")
non_cancer_dir = os.path.join(base_dir, "NON CANCER")

# Function to check and remove corrupted images
def remove_corrupted_images(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = Image.open(file_path)
            img.verify()
        except (IOError, SyntaxError) as e:
            print(f'Removing corrupted image: {file_path}')
            os.remove(file_path)

# Remove corrupted images
remove_corrupted_images(cancer_dir)
remove_corrupted_images(non_cancer_dir)

# Set up data generators
img_height, img_width = 224, 224
batch_size = 32

# Create three separate data generators for train, validation, and test
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    channel_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.3  # This will be split into validation and test sets
)

test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for test set

# Generate training data (70% of total)
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

# Generate validation data (first half of the remaining 30%)
validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Create a separate directory structure for test set
import shutil

def create_test_set(base_dir, test_dir, split=0.5):
    """
    Creates a test set from the validation split
    split: proportion of validation data to use as test (0.5 means equal split between val and test)
    """
    test_base = os.path.join(test_dir, "test")
    os.makedirs(test_base, exist_ok=True)
    
    for class_name in ["CANCER", "NON CANCER"]:
        # Create directories
        test_class_dir = os.path.join(test_base, class_name)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Get list of files in validation split
        source_dir = os.path.join(base_dir, class_name)
        files = os.listdir(source_dir)
        total_files = len(files)
        
        # Calculate number of files for validation split (30% of total)
        val_test_size = int(total_files * 0.3)
        # Calculate number of files for test (half of val_test_size)
        test_size = int(val_test_size * split)
        
        # Move files to test directory
        for file_name in files[-test_size:]:
            src = os.path.join(source_dir, file_name)
            dst = os.path.join(test_class_dir, file_name)
            shutil.copy2(src, dst)

# Create test directory and move files
test_dir = os.path.join(os.path.dirname(base_dir), "test_set")
create_test_set(base_dir, test_dir)

# Create test generator
test_generator = test_datagen.flow_from_directory(
    os.path.join(test_dir, "test"),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")
print(f"Number of test samples: {test_generator.samples}")

# Create MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Evaluate on all three sets
train_loss, train_acc = model.evaluate(train_generator)
val_loss, val_acc = model.evaluate(validation_generator)
test_loss, test_acc = model.evaluate(test_generator)

print("\nFinal Results:")
print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Validation Accuracy: {val_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Generate predictions on the test set
test_generator.reset()
y_pred = model.predict(test_generator)
y_pred_classes = (y_pred > 0.5).astype(int).reshape(-1)
y_true = test_generator.classes[:len(y_pred_classes)]

# Compute and display confusion matrix for test set
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Cancer', 'Cancer'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Test Set)')
plt.show()

# Print classification report for test set
print("\nTest Set Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=['Non-Cancer', 'Cancer']))

# Save the model
model.save('oral_cancer_detection_mobilenetv2_model.h5')
print("Model training completed and saved.")