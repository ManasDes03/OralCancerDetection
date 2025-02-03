import yaml
from dataloaders.dataloaders import get_data_loaders
from models.models import OralCancerModel
import tensorflow as tf

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Prepare data
train_data, val_data = get_data_loaders(config)

# Initialize model
model = OralCancerModel(config)

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_data,
          validation_data=val_data,
          epochs=config['training']['epochs'])
