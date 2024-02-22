# inspired by
# https://www.tensorflow.org/tutorials/images/classification


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from consts import INPUT_IMAGE_SHAPE, CLASS_NAMES

normalization_layer = layers.Rescaling(1./255)
num_classes = len(CLASS_NAMES)
img_height, img_width = INPUT_IMAGE_SHAPE
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("vertical",
                        input_shape=(img_height,
                                    img_width,
                                    1)),
        layers.RandomRotation(0.05),
        keras.layers.GaussianNoise(0.03),
        layers.RandomContrast(0.02),
        layers.RandomTranslation(0.05,0.05),
        layers.RandomZoom(0.05),
  ]
)

markers_model = Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(4, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.05),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax', name="outputs")
])
