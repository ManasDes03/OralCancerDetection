import yaml
import tensorflow as tf
from dataloaders.dataloaders import get_classification_data_loader
from model.model import OralCancerModel
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Compile the model with specified optimizer, loss, and metrics
def compile_model(model, config):
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    metrics = []
    for m in config['training']['metrics']:
        if m == "sparse_categorical_accuracy":
            metrics.append(tf.keras.metrics.SparseCategoricalAccuracy())
        elif m == "BinaryPrecision":
            metrics.append(tf.keras.metrics.Precision(thresholds=0.5))
        elif m == "BinaryRecall":
            metrics.append(tf.keras.metrics.Recall(thresholds=0.5))
        else:
            metrics.append(m)

    # Dynamic optimizer selection
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        # "sgd": tf.keras.optimizers.SGD,
        # "rmsprop": tf.keras.optimizers.RMSprop
    }

    optimizer = optimizers.get(optimizer_name, tf.keras.optimizers.Adam)(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss_function,
        metrics=metrics
    )

# Train the model
def train_model(model, train_ds, val_ds, config):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['training']['epochs']
    )
    return history

# Main function
def main():
    # Load configuration
    config = load_config()


    # Load datasets
    train_ds, val_ds, test_ds, test_files, test_labels, label_mapping = get_classification_data_loader(config)

    # Build the model
    model = OralCancerModel(config)

    # Compile the model
    compile_model(model, config)

    # Train the model
    history = train_model(model, train_ds, val_ds, config)

    # Plot training/validation accuracy and loss
    acc = history.history.get('sparse_categorical_accuracy', [])
    val_acc = history.history.get('val_sparse_categorical_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Train Acc')
    plt.plot(epochs, val_acc, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


    # Ensure model is compiled before evaluation (important if loaded from file)
    if not hasattr(model.model, 'optimizer') or model.model.optimizer is None:
        print("[INFO] Compiling model before evaluation...")
        model.model.compile(
            optimizer=config['training']['optimizer'],
            loss=config['training']['loss'],
            metrics=[config['training']['metrics']]
        )
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


    # Predict on test set for confusion matrix and classification report
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.model.predict(images)
        preds = np.argmax(preds, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

    # Map integer labels back to class names for reporting
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [inv_label_mapping[i] for i in sorted(inv_label_mapping.keys())]
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", class_report)

    # Save confusion matrix as image
    import seaborn as sns
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Save test results to file for report generator
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(class_report)

    # Save the trained model
    model_save_path = f"saved_models/{config['model']['name']}_oral_cancer_model.h5"
    model.model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
