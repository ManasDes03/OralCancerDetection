import yaml
import tensorflow as tf
from dataloaders.dataloaders import get_data_loaders
from models.models import OralCancerModel

# Load configuration from config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Compile the model with specified optimizer, loss, and metrics
def compile_model(model, config):
    optimizer_name = config['training']['optimizer']
    learning_rate = config['training']['learning_rate']
    loss_function = config['training']['loss_function']
    metrics = config['training']['metrics']

    # Dynamic optimizer selection
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop
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
    train_ds, val_ds = get_data_loaders(config)

    # Build the model
    model = OralCancerModel(config)

    # Compile the model
    compile_model(model, config)

    # Train the model
    history = train_model(model, train_ds, val_ds, config)

    # Save the trained model
    model_save_path = f"saved_models/{config['model']['name']}_oral_cancer_model.h5"
    model.model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
