import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    EfficientNetB0,
    MobileNetV2,
    ResNet50,
    InceptionV3
)

# Mapping model names to their corresponding Keras applications
PRETRAINED_MODELS = {
    "EfficientNetB0": EfficientNetB0,
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3
}

class OralCancerModel(tf.keras.Model):
    def __init__(self, config):
        super(OralCancerModel, self).__init__()
        self.config = config
        self.num_classes = config['dataset']['num_classes']
        self.image_size = tuple(config['dataset']['image_size'])
        self.base_model = self._get_base_model()

        # Adding custom classification head
        self.model = models.Sequential([
            self.base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')
        ])

    def _get_base_model(self):
        model_name = self.config['model']['name']
        weights = self.config['model'].get('weights', 'imagenet')  # Default to ImageNet weights

        if model_name in PRETRAINED_MODELS:
            base_model = PRETRAINED_MODELS[model_name](
                include_top=False,
                weights=weights,
                input_shape=(*self.image_size, 3)
            )
            base_model.trainable = self.config['model'].get('trainable', False)  # Fine-tuning option
            return base_model
        else:
            raise ValueError(f"Unsupported model: {model_name}. Available models: {list(PRETRAINED_MODELS.keys())}")

    def call(self, inputs):
        return self.model(inputs)
