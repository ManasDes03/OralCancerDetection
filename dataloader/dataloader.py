import tensorflow as tf
import pandas as pd
import os

def get_data_loaders(config):
    dataset_name = config['dataset']['name']
    image_size = tuple(config['dataset']['image_size'])
    batch_size = config['dataset']['batch_size']

    if dataset_name in ["Oral_Cancer", "Oral_Cancer_2.0"]:
        # For Oral Cancer and Oral Cancer 2.0 datasets
        train_dir = config['paths'][dataset_name]['train_dir']
        val_dir = config['paths'][dataset_name]['val_dir']

        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )

        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            val_dir,
            image_size=image_size,
            batch_size=batch_size,
            label_mode='categorical'
        )

    elif dataset_name == "Sri_Lankan":
        # For Sri Lankan Dataset
        images_dir = config['paths'][dataset_name]['images_dir']
        annotations_file = config['paths'][dataset_name]['annotations_file']

        # Load CSV annotations
        df = pd.read_csv(annotations_file)

        # Function to process images and labels
        def process_image(file_path, label):
            image = tf.io.read_file(file_path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, image_size)
            image = image / 255.0  # Normalize the image
            return image, tf.one_hot(label, depth=config['dataset']['num_classes'])

        # Map CSV entries to full image paths and labels
        file_paths = df['image_name'].apply(lambda x: os.path.join(images_dir, x)).values
        labels = df['label'].values  # Assuming label column contains 0 (non-cancer) or 1 (cancer)

        # Create TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = dataset.map(process_image)

        # Split into training and validation (80-20 split)
        dataset_size = len(df)
        train_size = int(0.8 * dataset_size)
        train_ds = dataset.take(train_size).batch(batch_size)
        val_ds = dataset.skip(train_size).batch(batch_size)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_ds, val_ds
