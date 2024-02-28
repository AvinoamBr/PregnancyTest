import os
import pathlib
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
STICK_CATEGORY_ID = 0
tf.config.run_functions_eagerly(True)

# inspiration: https://www.tensorflow.org/tutorials/load_data/images

# data_loader
class ObjectDetectionDataLoader(object):
    def __init__(self):
        data_dir = pathlib.Path('/home/avinoam/workspace/Salignostics/data/roboflow_20240219/')
        train_dir = pathlib.Path(data_dir,'train')
        dataset_dir = train_dir
        image_count = len(list(train_dir.glob('*/*.jpg')))
        list_ds = tf.data.Dataset.list_files(str(dataset_dir/'images/*'), shuffle=False)
        list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

        val_size = int(image_count * 0.2)
        train_ds = list_ds.skip(val_size)
        val_ds = list_ds.take(val_size)

        AUTOTUNE = tf.data.AUTOTUNE
        # data = list(map(self.process_path, train_ds))
        self.train_ds = train_ds.map(self.process_path)
        self.val_ds = val_ds.map(self.process_path)
        pass

    def get_label(self, file_path):
      .2d49f6789d6d2c3d963b6bb9f731e36a.jpg'
        fn = tf.strings.as_string(file_path)
        fn = tf.strings.regex_replace(fn, "images", "labels")
        fn = tf.strings.regex_replace(fn, "jpg$", "txt")
        labels = tf.io.read_file(fn)
        labels = tf.strings.split(tf.strings.split(labels, os.linesep), " ")
        labels = tf.reshape(labels, (-1,5))
        labels = tf.strings.to_number(labels)
        labels = labels[0][1:]

        return labels

    def decode_img(self, img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(self, file_path):
        label = self.get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return [img, label]


def stick_model():
    '''
    model_display_name = 'SSD MobileNet V2 FPNLite 320x320' # @param ['CenterNet HourGlass104 512x512','CenterNet HourGlass104 Keypoints 512x512','CenterNet HourGlass104 1024x1024','CenterNet HourGlass104 Keypoints 1024x1024','CenterNet Resnet50 V1 FPN 512x512','CenterNet Resnet50 V1 FPN Keypoints 512x512','CenterNet Resnet101 V1 FPN 512x512','CenterNet Resnet50 V2 512x512','CenterNet Resnet50 V2 Keypoints 512x512','EfficientDet D0 512x512','EfficientDet D1 640x640','EfficientDet D2 768x768','Effici… R-CNN ResNet50 V1 640x640','Faster R-CNN ResNet50 V1 1024x1024','Faster R-CNN ResNet50 V1 800x1333','Faster R-CNN ResNet101 V1 640x640','Faster R-CNN ResNet101 V1 1024x1024','Faster R-CNN ResNet101 V1 800x1333','Faster R-CNN ResNet152 V1 640x640','Faster R-CNN ResNet152 V1 1024x1024','Faster R-CNN ResNet152 V1 800x1333','Faster R-CNN Inception ResNet V2 640x640','Faster R-CNN Inception ResNet V2 1024x1024','Mask R-CNN Inception ResNet V2 1024x1024']
    model_handle = ALL_MODELS[model_display_name]


    print('loading model...')
    print(model_handle)
    hub_model = hub.load(model_handle)
    print('model loaded!')
    '''

    mobilenet_model = keras.applications.MobileNet(weights='imagenet')
    stick_model = keras.Sequential(mobilenet_model.layers[:-2])
    for layer in stick_model.layers:
        layer.trainable = False
    stick_model.add(layers.Dense(4, activation="relu"))


(img_height, img_width) = (320,320)

batch_size = 4
seed = 999
validation_split = 0.2
def train():
    data_path = '/home/avinoam/workspace/Salignostics/data/roboflow_20240219/data.yaml'
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        color_mode='grayscale',
        subset="training",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        color_mode='grayscale',
        subset="validation",
        seed=seed,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = model()
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
    #                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    #                      metrics=['accuracy'])

if __name__ == "__main__":
    # model = stick_model()
    data_loader = ObjectDetectionDataLoader()
    pass
    # train()