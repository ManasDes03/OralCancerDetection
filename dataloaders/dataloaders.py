import tensorflow as tf
import pandas as pd
import os
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

def get_classification_data_loader(config):
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']
    dataset_name = config['dataset']['name']
    
    def process_image(file_path, label):
        image = tf.io.read_file(file_path)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            image = tf.image.decode_jpeg(image, channels=1)
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, image_size)
        image = image / 255.0
        return image, label

    def process_image_aug(file_path, label):
        image = tf.io.read_file(file_path)
        try:
            image = tf.image.decode_jpeg(image, channels=3)
        except:
            image = tf.image.decode_jpeg(image, channels=1)
            image = tf.image.grayscale_to_rgb(image)
        image = tf.image.resize(image, image_size)
        # --- Data Augmentation ---
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        # --- End Data Augmentation ---
        image = image / 255.0
        return image, label

    if dataset_name == "Sri_Lankan":
        images_dir = config['paths'][dataset_name]['images_dir']
        annotations_file = config['paths'][dataset_name]['annotations_file']
        df = pd.read_csv(annotations_file)
        # Only keep rows where the image file exists
        valid_file_paths = []
        valid_labels = []
        for fname, label in zip(df['image_name'], df['Category']):
            img_path = os.path.join(images_dir, str(fname) + ".jpg" if not str(fname).endswith('.jpg') else str(fname))
            if os.path.isfile(img_path):
                valid_file_paths.append(img_path)
                valid_labels.append(label)
        # Use LabelEncoder for multi-class string labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels = le.fit_transform(valid_labels)
        label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
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
    train_ds = train_ds.shuffle(buffer_size=len(train_files)).map(process_image_aug).apply(tf.data.experimental.ignore_errors()).batch(batch_size, drop_remainder=True)
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels)).map(process_image).apply(tf.data.experimental.ignore_errors()).batch(batch_size, drop_remainder=True)
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels)).map(process_image).apply(tf.data.experimental.ignore_errors()).batch(batch_size, drop_remainder=True)

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
