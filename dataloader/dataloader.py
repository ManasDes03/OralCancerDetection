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
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = image / 255.0  # Normalize
        return image, label
    
    cancer_dir = config['paths'][dataset_name]['cancer_dir']
    non_cancer_dir = config['paths'][dataset_name]['non_cancer_dir']
    
    cancer_images = [os.path.join(cancer_dir, f) for f in os.listdir(cancer_dir) if f.endswith(".jpg")]
    non_cancer_images = [os.path.join(non_cancer_dir, f) for f in os.listdir(non_cancer_dir) if f.endswith(".jpg")]
    
    file_paths = cancer_images + non_cancer_images
    labels = [1] * len(cancer_images) + [0] * len(non_cancer_images)
    
    dataset_size = len(file_paths)
    indices = list(range(dataset_size))
    random.shuffle(indices)
    
    train_end = int(0.8 * dataset_size)
    val_end = train_end + int(0.1 * dataset_size)
    
    train_files = [file_paths[i] for i in indices[:train_end]]
    train_labels = [labels[i] for i in indices[:train_end]]
    val_files = [file_paths[i] for i in indices[train_end:val_end]]
    val_labels = [labels[i] for i in indices[train_end:val_end]]
    test_files = [file_paths[i] for i in indices[val_end:]]
    test_labels = [labels[i] for i in indices[val_end:]]
    
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels)).map(process_image).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((val_files, val_labels)).map(process_image).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels)).map(process_image).batch(batch_size)
    
    return train_ds, val_ds, test_ds, test_files, test_labels

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
