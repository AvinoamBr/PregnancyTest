import os

import cv2
import glob
from argparse import ArgumentParser
from consts import CHECKPOINT_PATH, INPUT_IMAGE_SHAPE
import tensorflow as tf
from tensorflow import keras
from markers_classifier.markers_model import markers_model
import numpy as np
from matplotlib import pyplot as plt
img_height, img_width = INPUT_IMAGE_SHAPE


class TestMarkersModel(object):
    def __init__(self, args):
        self._args = args
        self._model = markers_model
        self._model.load_weights(CHECKPOINT_PATH)
        self._data_set = self.data_set()
        keras.utils.set_random_seed(0)

    def data_set(self):
        batch_size = 4
        data_set = tf.keras.utils.image_dataset_from_directory(
            self._args.data,
            shuffle=False,
            image_size=(img_height, img_width),
            color_mode='grayscale',
            batch_size=batch_size)
        self._file_paths = np.array(data_set.file_paths)

        window_files = []
        for f in self._file_paths:
            matchs = [fn for fn in glob.glob(f'{"/".join(f.split(os.sep)[:-3])}/*/*/*') if os.path.split(f)[1] in fn]
            match_window = [m for m in matchs if "window" in m][0]
            window_files.append(match_window)
        self._window_files = np.array(window_files)

        return data_set

    def predict(self):
        ds = self._data_set
        self._preds = preds = self._model.predict(ds)

        y_train = list(map(lambda x: x[1].numpy(), ds))
        Y = []
        [Y.extend(y) for y in y_train]

        self._Y = Y = np.array(Y)
        self._pred = pred = np.argmax(preds, axis=1)

        pd = positive_detected = pred == 1
        nd = negative_detected = np.logical_not(pd)

        pr = positive_required = Y == 1
        nr = negative_required = np.logical_not(pr)

        self._TruePositive = np.where(np.logical_and(pd,pr))[0]
        self._TrueNegative = TN = np.where(np.logical_and(nd,nr))[0]
        self._FalsePositive = FP = np.where(np.logical_and(pd,nr))[0]
        self._FalseNegative = FN = np.where(np.logical_and(nd,pr))[0]

        TP = len(self._TruePositive)
        TN = len(self._TrueNegative)
        FP = len(self._FalsePositive)
        FN = len(self._FalseNegative)

        accuracy = (TP+TN)/ len(Y)
        precision = TP / (TP+FP)
        recall =  TP /(TP+FN)
        print (f"num samples: {len(Y)}, accuracy: {accuracy:.2f}, precision:{precision:.2f}, recall:{recall:.2f}")

        self.display_sample(self._TruePositive,"true positive")
        self.display_sample(self._TrueNegative,"true negative")

        self.display_sample(self._FalsePositive,"false positive")
        self.display_sample(self._FalseNegative,"false negative")

    def display_sample(self, sample_ids, title, permutate=True):
        plt.figure(figsize=(10,25))
        if permutate:
            sample_ids = np.random.permutation(sample_ids)
        for i in range (min((len(sample_ids), 10))):
            idx = sample_ids[i]
            plt.subplot(2,5,i+1)
            f = self._window_files[idx]
            im1 = plt.imread(f)
            # plt.imshow(im)
            f = self._file_paths[idx]
            im2 = plt.imread(f)
            im2 = cv2.cvtColor(im2, cv2.COLOR_GRAY2BGR).astype(np.uint8)
            im = cv2.vconcat( (im1,im2))
            im[im1.shape[0]-10:im1.shape[0]+10][:]= [200,100,100]
            plt.imshow(im)
            conf = self._preds[idx][self._pred[idx]]
            conf = self._preds[idx][self._pred[idx]]
            plt.text(10,im1.shape[0]-3, f"{conf:.2f}")
            plt.title(os.path.split(f)[1].replace(".jpg",""))
        plt.suptitle(title)
        plt.show()



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data', help="input data path")
    args = parser.parse_args()

    test_model = TestMarkersModel(args)
    predictions = test_model.predict()
    pass