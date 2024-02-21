import argparse
import os

from matplotlib import pyplot as plt
from ultralytics import YOLO
import glob
import cv2
import numpy as np

from bounding_box_detection.utils.Gaussian import gaussuian_filter

STICKS = 'Sticks'
INNER_WINDOW = "inner_window"
WINDOW = 'window'
CLASS_NAMES = [STICKS, INNER_WINDOW, WINDOW]

cp = "/home/avinoam/Desktop/autobrains/DL_Engineer/assignment_files/runs/detect/train14/weights/best.pt"
model = YOLO(cp)

class PatternMatch(object):
    '''
    A class that perform classical CV methods to enhance image, and return image in standard shape
    that present a better map for pregnant test markers.
    '''
    MATCH_SHAPE = (80,80)
    def __init__(self):
        self._window = None
        self._standard_window = None
        self.vertical_edges = None
        self.horizontal_edges = None
        self.marker_match = None
        self.marker_candidates = None

    def plot(self):
        plt.subplots(2,3)
        for i,mat in enumerate((self._standard_window, self.vertical_edges, self.horizontal_edges,
                                self._vertical_fixed,  self.marker_match, self.marker_candidates)):
            plt.subplot(2,3,i+1)
            plt.imshow(mat)
        plt.show()


    def load(self,window):
        self._window = window
        self._standard_window = cv2.resize(window, self.MATCH_SHAPE)

    def __call__(self,window):
        self.load(window)
        self.find_edges()
        self.find_markers_candidates()

    def find_edges(self):
        reds = self._standard_window[:,:,2]
        image_mask = self._standard_window.sum(axis=-1) !=0
        self.vertical_edges = cv2.filter2D(src=reds, ddepth=-1, kernel=self.horizontal_kernel).astype(np.int16)
        self.vertical_edges = image_mask * self.vertical_edges
        self.horizontal_edges = cv2.filter2D(src=reds, ddepth=-1, kernel=self.vertical_kernel).astype(np.int16)
        self.horizontal_edges = self.horizontal_edges * image_mask
        self._vertical_fixed = self.vertical_edges - self.horizontal_edges

    def find_markers_candidates(self):
        marker_match = cv2.filter2D(src=self._vertical_fixed, ddepth=-1, kernel=self.marker_kernel).astype(np.int16)
        distance_filter =  gaussuian_filter(marker_match.shape)
        marker_match = marker_match * distance_filter
        marker_match[marker_match < 30] = 0
        self.marker_match = marker_match

        marker_candidates = np.zeros_like(marker_match)
        output = cv2.connectedComponentsWithStats(marker_match.astype(np.uint8), 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        vals = [marker_match[labels == i].sum() for i in range(numLabels)]
        for i in range(numLabels):
            marker_candidates[labels == i] = vals[i]
        marker_candidates = (marker_candidates / marker_candidates.max() * 255).astype(np.uint8)
        self.marker_candidates = marker_candidates

    @property
    def horizontal_kernel(self):
        '''
        :return: a horizontal derivation kernel (to detect vertical lines)
        '''
        if hasattr(self, "_horizontal_kernel"):
            return self._horizontal_kernel
        horizontal_kernel = np.ones((7,9))
        mean = (horizontal_kernel.shape[1])/2
        xs = np.linspace(0,2*mean, horizontal_kernel.shape[1])
        # pdf = np.exp(-0.5 * (  (xs-mean)/std)* ((xs-mean)/std))
        pdf = mean-np.abs(xs-mean)
        horizontal_kernel[:,]=pdf
        horizontal_kernel = (horizontal_kernel - horizontal_kernel.mean()) * -2
        self._horizonal_kernel = horizontal_kernel
        return self._horizonal_kernel

    @property
    def vertical_kernel(self):
        '''
        :return: a vertical derivation kernel (to detect horizonal lines)
        '''
        if hasattr(self, "_vertical_kernel"):
            return self._vertical_kernel
        vertical_kernel = np.ones((9,3))
        mean = (self.horizontal_kernel.shape[0])/2
        ys = np.linspace(0,2*mean, vertical_kernel.shape[0])
        pdf = mean-np.abs(ys-mean)
        vertical_kernel.T[:,] = pdf
        vertical_kernel = (vertical_kernel - vertical_kernel.mean()) * -2
        self._vertical_kernel = vertical_kernel
        return self._vertical_kernel
    @property
    def marker_kernel(self):
        marker_kernel = np.ones((15, 8))
        marker_kernel_l = [-1, 0, 1, 2, 2, 1, 0, -1]
        marker_kernel[:, ] = marker_kernel_l
        marker_kernel = (marker_kernel - marker_kernel.mean())
        marker_kernel = marker_kernel / marker_kernel.size
        return marker_kernel

class WindowDetector(object):
    def __init__(self, detection_model, args):
        self._detection_model = detection_model
        self._args = args
        self._pattern_match = PatternMatch()
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
        res = self._detection_model.predict([self._source_image])
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
                print(f"failed to find box for {att}")
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
            # filter stick from background using kmeans
            if self._lines.shape[0]>20:
                print ("more than 20 lines were found, probably bakground.. detecting angle by window")
                plt.imshow(stick)
                for l in self._lines:
                    x,y,xb,yb = l[0]
                    plt.plot([x,xb],[y,yb])
                plt.show()
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

    def match_pattern(self):
        window = window_detector.rotated_window
        self._pattern_match(window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--file", required=False, help="single file to run")
    parser.add_argument('-o','--output_path',help='path to save outputs', default=None)

    args = parser.parse_args()
    if args.file:
        images = [f"{args.data_path}/{args.file}.jpg"]
    else:
        images = glob.glob(f"{args.data_path}/*")
        images = np.random.permutation(images)

    window_detector = WindowDetector(model, args)

    for im in images:
        try:
            window_detector(im)
            window_detector.plot_rotation_model()
            window_detector.plot()

            window_detector.match_pattern()

        except Exception as e:
            print(e)
            continue
        continue
