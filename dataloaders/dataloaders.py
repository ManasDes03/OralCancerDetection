import tensorflow as tf
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2


# --- Module-level image processing functions for use in main.py ---
def process_image(file_path, label, image_size=(224,224)):
    image = tf.io.read_file(file_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    except Exception as e:
        tf.print("[ERROR] Could not decode image:", file_path, ". Skipping. Error:", e)
        image = tf.zeros([*image_size, 3], dtype=tf.float32)
        label = tf.constant(-1, dtype=tf.int32)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    tf.debugging.assert_type(image, tf.float32, message="Image is not float32")
    tf.debugging.assert_type(label, tf.int32, message="Label is not int32")
    tf.print("[DEBUG][PROCESS_IMAGE] image dtype:", image.dtype, ", label:", label)
    return image, label

def process_image_aug(file_path, label, image_size=(224,224)):
    image = tf.io.read_file(file_path)
    try:
        image = tf.image.decode_jpeg(image, channels=3, try_recover_truncated=True)
    except Exception as e:
        tf.print("[ERROR] Could not decode image:", file_path, ". Skipping. Error:", e)
        image = tf.zeros([*image_size, 3], dtype=tf.float32)
        label = tf.constant(-1, dtype=tf.int32)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    tf.debugging.assert_type(image, tf.float32, message="Image is not float32 (aug)")
    tf.debugging.assert_type(label, tf.int32, message="Label is not int32 (aug)")
    tf.print("[DEBUG][PROCESS_IMAGE_AUG] image dtype:", image.dtype, ", label:", label)
    # ...existing code...
    # --- Data Augmentation ---
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    # Random zoom
    import numpy as np
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [y1, x1, y2, x2]
    def random_crop(img):
        crop_size = tf.shape(img)[:2]
        crops = tf.image.crop_and_resize(
            tf.expand_dims(img, 0),
            boxes=boxes,
            box_indices=np.zeros(len(scales)),
            crop_size=crop_size
        )
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: random_crop(image), lambda: image)
    # Random translation
    def random_translate(img):
        tx = tf.random.uniform([], -0.1, 0.1) * tf.cast(tf.shape(img)[0], tf.float32)
        ty = tf.random.uniform([], -0.1, 0.1) * tf.cast(tf.shape(img)[1], tf.float32)
        import tensorflow_addons as tfa
        return tfa.image.translate(img, [tx, ty])
    try:
        import tensorflow_addons as tfa
        image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: random_translate(image), lambda: image)
    except ImportError:
        pass
    # Color jitter (hue, saturation)
    image = tf.image.random_hue(image, max_delta=0.08)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Gaussian noise
    def add_noise(img):
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.05, dtype=tf.float32)
        return tf.clip_by_value(img + noise, 0.0, 1.0)
    image = tf.cond(tf.random.uniform([], 0, 1) > 0.7, lambda: add_noise(image), lambda: image)
    # --- End Data Augmentation ---
    image = image / 255.0
    tf.print("[DEBUG][PROCESS_IMAGE_AUG] image dtype:", image.dtype, ", label:", label)
    return image, tf.cast(label, tf.int32)

def get_classification_data_loader(config):
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    dataset_name = config['dataset']['name']

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
        labels = valid_labels
        label_mapping = {0: 'non-cancer', 1: 'cancer'}
        file_paths = valid_file_paths
    else:
        cancer_dir = config['paths'][dataset_name]['cancer_dir']
        non_cancer_dir = config['paths'][dataset_name]['non_cancer_dir']
        cancer_images = [os.path.join(cancer_dir, f) for f in os.listdir(cancer_dir) if f.endswith(".jpg")]
        non_cancer_images = [os.path.join(non_cancer_dir, f) for f in os.listdir(non_cancer_dir) if f.endswith(".jpg")]
        file_paths = cancer_images + non_cancer_images
        labels = [1] * len(cancer_images) + [0] * len(non_cancer_images)

    dataset_size = len(file_paths)
    if dataset_size == 0:
        raise ValueError("No valid image files found for dataset. Please check your image directory and annotation file.")
    indices = list(range(dataset_size))
    random.shuffle(indices)
    train_end = int(0.8 * dataset_size)
    val_end = train_end + int(0.1 * dataset_size)

    # Ensure types are correct
    file_paths = [str(f) for f in file_paths]
    labels = [int(l) for l in labels]


    train_files = [file_paths[i] for i in indices[:train_end]]
    train_labels = [labels[i] for i in indices[:train_end]]
    val_files = [file_paths[i] for i in indices[train_end:val_end]]
    val_labels = [labels[i] for i in indices[train_end:val_end]]
    test_files = [file_paths[i] for i in indices[val_end:]]
    test_labels = [labels[i] for i in indices[val_end:]]

    # Trim val/test splits to a multiple of batch size
    def trim_to_batch(files, labels, batch_size):
        n = (len(files) // batch_size) * batch_size
        return files[:n], labels[:n]

    val_files, val_labels = trim_to_batch(val_files, val_labels, batch_size)
    test_files, test_labels = trim_to_batch(test_files, test_labels, batch_size)

    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = (
        train_ds
        .shuffle(buffer_size=len(train_files))
        .map(process_image_aug, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda img, lbl: lbl > -1)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda img, lbl: lbl > -1)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((test_files, test_labels))
        .map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
        .filter(lambda img, lbl: lbl > -1)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Debug: Print batch shapes as they are yielded
    def print_batch_shape(images, labels, split_name):
        tf.print("[DEBUG] Batch from", split_name, ": images shape =", tf.shape(images), ", labels shape =", tf.shape(labels))
        return images, labels

    train_ds = train_ds.map(lambda x, y: print_batch_shape(x, y, "train"))
    val_ds = val_ds.map(lambda x, y: print_batch_shape(x, y, "val"))
    test_ds = test_ds.map(lambda x, y: print_batch_shape(x, y, "test"))

    return train_ds, val_ds, test_ds, test_files, test_labels, label_mapping

def get_mscoco_data_loader(config):
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    images_dir = config['paths']['Sri_Lankan']['images_dir']
    annotations_file = config['paths']['Sri_Lankan']['annotations_file']
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    file_paths = []
    annotations_map = {}
    
    for annotation in annotations['images']:
        image_path = os.path.join(images_dir, annotation['file_name'])
        file_paths.append(image_path)
        image_id = annotation['id']
        annotations_map[image_path] = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
    
    dataset_size = len(file_paths)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_end = int(0.8 * dataset_size)
    val_end = train_end + int(0.1 * dataset_size)
    
    train_files = [file_paths[i] for i in indices[:train_end]]
    val_files = [file_paths[i] for i in indices[train_end:val_end]]
    test_files = [file_paths[i] for i in indices[val_end:]]
    
    return train_files, val_files, test_files, annotations_map

if __name__ == "__main__":
    config = {
        "dataset": {
            "name": "Sri_Lankan",
            "image_size": [224, 224],
            "batch_size": 32
        },
        "paths": {
            "Sri_Lankan": {
                "images_dir": "./data/Sri_Lankan_Dataset/Images",
                "annotations_file": "./data/Sri_Lankan_Dataset/Annotation.json"
            }
        }
    }
    
    train_files, val_files, test_files, annotations_map = get_mscoco_data_loader(config)
    print("Training set size:", len(train_files))
    print("Validation set size:", len(val_files))
    print("Testing set size:", len(test_files))
    
    # Show a random test image with segmentation
    random_index = random.randint(0, len(test_files) - 1)
    image_path = test_files[random_index]
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = image.numpy()
    
    plt.imshow(image / 255.0)
    plt.axis("off")
    plt.title("Lesion and Oral Cavity Segmentation")
    
    # Plot segmentation masks
    if image_path in annotations_map:
        for annotation in annotations_map[image_path]:
            segmentation = annotation['segmentation']
            for segment in segmentation:
                polygon = np.array(segment).reshape((-1, 2))
                polygon[:, 0] *= image.shape[1] / annotation['width']
                polygon[:, 1] *= image.shape[0] / annotation['height']
                polygon = polygon.astype(np.int32)
                cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        plt.imshow(image / 255.0)
    
    plt.show()
