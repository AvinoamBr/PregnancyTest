import argparse
import logging
import os
import sys

from matplotlib import pyplot as plt
from tqdm import tqdm
from ultralytics import YOLO
import glob
import cv2
import numpy as np

from consts import text_color
from consts import YOLO_CP
from consts import OBJECT_DETECTION_CLASS_NAMES as CLASS_NAMES
from markers_detector.markers_detector import MarkersDetector

class WindowDetector(object):
    '''
    Class to handle the task of detecting the markers (test/control) window out of image,
    and producing new image that contain aligned (horizontal) image of the window.
    '''
    def __init__(self, detection_model, args):
        self._detection_model = detection_model
        self._args = args
        self._names_dict = {}
        self._PLOT = False

    def __call__(self,fn):
        self.load(fn)
        self.detect_boxes()
        self.get_image_rotation_angle(use_stick=True)
        self.rotate_window(PLOT=False)

    def load(self, fn):
        self._fn = fn
        self._source_image = cv2.imread(fn)
        self._rotation_angle = None
        self._rotattion_center = (None, None)  # (cx,cy)
        self._original_window_bb = None

    def detect_boxes(self):
        '''
        use DL model to detect stick, window, inner_window
        set detections as class attributes
        '''
        res = self._detection_model.predict([self._source_image], verbose=False)
        box = res[0].boxes
        self._names_dict = res[0].names

        # read YOLO results to class attributes
        for att in CLASS_NAMES:
            setattr(self, f"_{att}_id", list(self._names_dict.values()).index(att))

        for att in CLASS_NAMES:
            id = getattr(self,f"_{att}_id")
            try:
                idx = np.where(box.cls.numpy() == id)[0][0]
                setattr(self, f"_box_{att}", res[0].boxes[idx])
            except:
                # print(f"failed to find box for {att}")
                setattr(self, f"_box_{att}", None)

    def get_image_rotation_angle(self,use_stick=False, use_window=False):
        '''
        calculate a recommended rotation angle that will allign stick to horizonal line
        '''
        if use_stick:
            x, y, x_b, y_b = self._box_Sticks.xyxy.numpy()[0].astype(int)
            stick = np.array(self._source_image[y:y_b, x:x_b])

            self._blurred_image = np.array(stick, copy=True)
            self._blurred_image = cv2.GaussianBlur(self._blurred_image,(55,55) , sigmaX=55,sigmaY=55)
            pixel_vals = self._blurred_image.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 2
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria,
                                                 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            # segmented_data = centers[labels.flatten()]
            self._segmented_image = labels.flatten().reshape((self._blurred_image.shape[:2]))
            bg_index = np.argmin(centers.mean(axis=1)) # bg is expected to be  darker than stick
            stick[self._segmented_image==bg_index] = [0,0,0]

            img_before = stick
            img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 40, 100, apertureSize=3)
            self._lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 100, minLineLength=300, maxLineGap=50)

            if self._lines.shape[0]>20:
                self._lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 200, minLineLength=300, maxLineGap=50)

        if use_window or self._lines.shape[0]>20:
            x, y, x_b, y_b = self._box_window.xyxy.numpy()[0].astype(int)
            window = self._source_image[y:y_b, x:x_b]
            img_before = window
            img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 40, 100, apertureSize=3)
            self._lines = np.array([])
            i = 0
            while self._lines.shape[0]<5:
                self._lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 30-i, minLineLength=30-i, maxLineGap=10)
                if i==10:
                    raise Exception("failed to find lines for image allignment")
        # else:
        #     raise Exception("stick and window were not detected by model")

        angles = []
        weights = []
        edge_image = np.array(img_before)
        for [[x1, y1, x2, y2]] in self._lines:
            # print([x1, y1, x2, y2])
            edge_image = cv2.line(edge_image, (x1, y1), (x2, y2), (255, np.random.randint(255), 100), 7)
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.rad2deg(angle)
            angles.append(angle_deg)
            weights.append(np.linalg.norm([y2 - y1, x2 - x1]))

        sort_ids = np.argsort(angles)
        angles = np.array(angles)[sort_ids]
        weights = np.array(weights)[sort_ids]
        half_sum = weights.sum()/2
        sum_of_weights = np.add.accumulate(weights)
        weighted_median_idx = np.where(sum_of_weights>half_sum)[0][0]


        # median_angle = np.median(angles)
        median_angle = angles[weighted_median_idx]
        self._rotation_angle = median_angle
        return median_angle

    def plot_rotation_model(self):
        plt.subplot(2,1,1)
        plt.imshow(self._blurred_image)
        for l in self._lines:
            x1, y1, x2, y2 = l[0]
            plt.plot([x1,x2],[y1,y2])
        plt.subplot(2,1,2)
        plt.imshow(self._segmented_image)
        plt.show()

    def rotate_window(self, PLOT=False):
        '''
        sets the class variable self._rotated_window to alligned according to rotation angle
        '''
        self._PLOT = PLOT
        box = x, y, x_b, y_b = self._box_window.xyxy.numpy()[0].astype(int)
        self._rotated_window = self._rotate_patch(box)
        self._PLOT = False

    @property
    def rotated_window(self):
        return self._rotated_window

    def _rotate_patch(self, box):
        '''
        :param box: a bounding box to crop from self._source_image
        :return: cropped and rotated box.
        '''
        x, y, x_b, y_b = box
        img = self._source_image[y:y_b, x:x_b]
        h,w = img.shape[:2]
        if h>w:
            pad_im = np.zeros((h,h,3)).astype(np.uint8)
            left = int((h-w)/2)
            pad_im[:,left:left+w]=img
            img = pad_im
        elif w>h:
            pad_im = np.zeros((w,w,3)).astype(np.uint8)
            top = int((w-h)/2)
            pad_im[top:top+h,:]=img
            img = pad_im

        rotation_angle = self._rotation_angle
        cx,cy = (np.array(img.shape[:2])/2).astype(np.uint16)[::-1]
        M = cv2.getRotationMatrix2D((cx, cy), rotation_angle, 1.0)
        img_rotated = cv2.warpAffine(img, M, (cx*2,cy*2))

        return img_rotated

    def plot(self):
        plt.subplot(1,2,1)
        plt.imshow(self._source_image)
        plt.subplot(1,2,2)
        plt.imshow(self.rotated_window)
        plt.suptitle(os.path.split(self._fn)[1])
        plt.show()


