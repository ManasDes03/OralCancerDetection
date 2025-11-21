
import tensorflow as tf

def OralCancerModel(config):
    """
    MobileNetV2 transfer learning model for binary oral cancer classification.
    """
    num_classes = config['dataset']['num_classes']
    image_size = tuple(config['dataset']['image_size'])
    if len(image_size) == 2:
        input_shape = (*image_size, 3)
    else:
        input_shape = (*image_size[:2], 3)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False  # Freeze base

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model
