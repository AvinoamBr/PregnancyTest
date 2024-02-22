import glob
from argparse import ArgumentParser
import tensorflow as tf
from matplotlib import pyplot as plt

from consts import EPOCHS, CHECKPOINT_PATH
from markers_classifier.markers_model import markers_model, INPUT_IMAGE_SHAPE




DEBUG_VISUALIZE_TRAIN_DATA = False
batch_size = 32

img_height, img_width = INPUT_IMAGE_SHAPE

seed = 999
validation_split = 0.2

class TrainModel(object):
    def __init__(self):
        self._history = None # train logging history
        self._train_ds = None
        self._val_ds = None
        self._model = markers_model

    def load_data_sets(self, data_dir):
        data_dir  = data_dir
        self._train_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                color_mode= 'grayscale',
                subset="training",
                seed=seed,
                image_size=(img_height, img_width),
                batch_size=batch_size)

        self._val_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                validation_split=validation_split,
                color_mode= 'grayscale',
                subset="validation",
                seed=seed,
                image_size=(img_height, img_width),
                batch_size=batch_size)

        image_count = len(list(glob.glob(f"{data_dir}/*/*.jpg")))
        print(f"number of images in data dir: {image_count}")
        self._class_names = self._train_ds.class_names

        AUTOTUNE = tf.data.AUTOTUNE
        self._train_ds = self._train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self._val_ds = self._val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def visualize_train_data(self):
        class_names = self._class_names
        plt.figure(figsize=(10, 10))
        for images, labels in self._train_ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[labels[i].numpy()])
                plt.axis("off")
            plt.show()

    def train(self):
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                         loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                         metrics=['accuracy'])

        self._model.summary()
        epochs=EPOCHS
        history = self._model.fit(
            self._train_ds,
            validation_data=self._val_ds,
            epochs=epochs
        )
        self._history = history

    def dump_weights(self, check_point_path):
        self._model.save_weights(check_point_path)

    def plot_history(self):
        history = self._history
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t","--data_path")
    parser.add_argument("-c", "--cp",help="load from check point", default=None)
    parser.add_argument("--save_to", help="model checkpoint", default=CHECKPOINT_PATH)
    args = parser.parse_args()

    train_model = TrainModel()
    train_model.load_data_sets(args.data_path)
    # train_model.visualize_train_data()
    train_model.train()
    train_model.dump_weights(args.save_to)
    train_model.plot_history()