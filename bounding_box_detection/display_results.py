import imutils
from matplotlib import pyplot as plt
from ultralytics import YOLO
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import ndimage

from bounding_box_detection.utils.Gaussian import gaussuian_filter

STICKS = 'Sticks'
INNER_WINDOW = "inner_window"
WINDOW = 'window'
CLASS_NAMES = [STICKS, INNER_WINDOW, WINDOW]

cp = "/home/avinoam/Desktop/autobrains/DL_Engineer/assignment_files/runs/detect/train14/weights/best.pt"
images_path = "/home/avinoam/workspace/Salignostics/data/roboflow/train/images/"
model = YOLO(cp)

def match_pattern(window):
    standard_window = cv2.resize(window, (80,80))
    reds = standard_window[:,:,2]

    horizonal_kernel = np.ones((7,9))
    # kernel[:,3:8]= -1
    mean = (horizonal_kernel.shape[1])/2
    xs = np.linspace(0,2*mean, horizonal_kernel.shape[1])
    # pdf = np.exp(-0.5 * (  (xs-mean)/std)* ((xs-mean)/std))
    pdf = mean-np.abs(xs-mean)
    horizonal_kernel[:,]=pdf
    horizonal_kernel = (horizonal_kernel - horizonal_kernel.mean()) * -2
    # kernel = kernel/kernel.sum()
    red_match = cv2.filter2D(src=reds, ddepth=-1, kernel=horizonal_kernel).astype(np.int16)

    vertical_kernel = np.ones((9,3))
    mean = (horizonal_kernel.shape[0])/2
    ys = np.linspace(0,2*mean, vertical_kernel.shape[0])
    pdf = mean-np.abs(ys-mean)
    vertical_kernel.T[:,] = pdf
    vertical_kernel = (vertical_kernel - vertical_kernel.mean()) * -2
    vertical_match = cv2.filter2D(src=reds, ddepth=-1, kernel=vertical_kernel).astype(np.int16)
    vertical_match[vertical_match<255] = 0
    # red_match = red_match - vertical_match

    img_gray = cv2.cvtColor(standard_window, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(standard_window, 40, 100, apertureSize=3)
    red_fixed = np.clip(red_match-vertical_match,0,255)
    marker_kernel = np.ones((15,8))
    marker_kernel_l = [-1,0,1,2,2,1,0,-1]
    marker_kernel[:,] = marker_kernel_l
    marker_kernel = (marker_kernel - marker_kernel.mean())
    marker_kernel = marker_kernel  / marker_kernel.size
    marker_match = cv2.filter2D(src=red_fixed, ddepth=-1, kernel=marker_kernel).astype(np.int16)

    distance_filter =  gaussuian_filter(marker_match.shape)
    marker_match = marker_match * distance_filter

    marker_match[marker_match<30]=0

    output = cv2.connectedComponentsWithStats(marker_match.astype(np.uint8), 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    vals = [marker_match[labels==i].sum() for i in range(numLabels)]
    for i in range(numLabels):
        marker_match[labels==i]=vals[i]
    marker_match = (marker_match / marker_match.max() * 255).astype(np.uint8)

    if True:
        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(standard_window)
        y_max = np.where((marker_match == marker_match.max()))[0].mean()
        x_max = np.where((marker_match == marker_match.max()))[1].mean()
        ax1.scatter(x_max, y_max)

        plt.subplot(1, 2, 2)
        plt.imshow(standard_window)
        plt.imshow(marker_match, alpha=0.5)
        plt.show()

    return

    ax1 = plt.subplot(3, 2, 1)
    plt.imshow(standard_window)

    plt.subplot(3,2,2)
    plt.imshow(red_match.astype(np.uint8))

    plt.subplot(3,2,3)
    plt.imshow(vertical_match.astype(np.uint8))

    plt.subplot(3,2,4)
    plt.imshow(red_fixed)

    plt.subplot(3,2,5)
    plt.imshow(marker_match)

    plt.subplot(3,2,6)

    # marker_match[marker_match==0] = np.nan
    plt.imshow(standard_window)
    plt.imshow(marker_match, alpha=0.2)
    y_max = np.where((marker_match==marker_match.max()))[0].mean()
    x_max = np.where((marker_match==marker_match.max()))[1].mean()
    # y_max , x_max = np.unravel_index(np.argmax(marker_match), marker_match.shape)
    ax1.scatter(x_max,y_max)
    plt.show()
    pass

class WindowDetector(object):
    def __init__(self, image, detection_model):
        self._detection_model = detection_model
        self._source_image = image
        self._rotation_angle = None
        self._rotattion_center = (None,None) # (cx,cy)
        self._original_window_bb = None
        self._names_dict = {}
        self._PLOT = False

    def detect_boxes(self):
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
        :param image_before:

        :return: the median angle for dominant lines in image
        '''
        if use_stick:
            x, y, x_b, y_b = self._box_Sticks.xyxy.numpy()[0].astype(int)
            stick = np.array(self._source_image[y:y_b, x:x_b])

            image = np.array(stick, copy=True)
            image = cv2.GaussianBlur(image,(55,55) , sigmaX=25,sigmaY=25)
            pixel_vals = image.reshape((-1, 3))
            pixel_vals = np.float32(pixel_vals)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            k = 2
            retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria,
                                                 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = np.uint8(centers)
            # segmented_data = centers[labels.flatten()]
            segmented_image = labels.flatten().reshape((image.shape[:2]))
            bg_index = np.argmin(centers.mean(axis=1)) # bg is expected to be  darker than stick
            stick[segmented_image==bg_index] = [0,0,0]

            img_before = stick
            # filter stick from background using kmeans

        elif use_window:
            x, y, x_b, y_b = self._window_stick.xyxy.numpy()[0].astype(int)
            window = self._source_image[y:y_b, x:x_b]
            img_before = window
        else:
            img_before = self._source_image



        # img_before = cv2.GaussianBlur(img_before, (5,5),0)

        img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        img_edges = cv2.Canny(img_gray, 40, 100, apertureSize=3)
        lines = cv2.HoughLinesP(img_edges, 1, np.pi / 180.0, 100, minLineLength=300, maxLineGap=50)

        angles = []

        edge_image = np.array(img_before)
        for [[x1, y1, x2, y2]] in lines:
            # print([x1, y1, x2, y2])
            edge_image = cv2.line(edge_image, (x1, y1), (x2, y2), (255, np.random.randint(255), 100), 7)
            angle = np.arctan2(y2 - y1, x2 - x1)
            angle_deg = np.rad2deg(angle)
            angles.append(angle_deg)

        median_angle = np.median(angles)
        self._rotation_angle = median_angle
        return median_angle

    def rotate_window(self, PLOT=False):
        self._PLOT = PLOT
        box = x, y, x_b, y_b = self._box_window.xyxy.numpy()[0].astype(int)
        self._rotated_window = self._rotate_image(box)
        self._PLOT = False

    @property
    def rotated_window(self):
        return self._rotated_window
    def _rotate_image(self, box):
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

        median_angle = self._rotation_angle
        cx,cy = (np.array(img.shape[:2])/2).astype(np.uint16)[::-1]
        M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
        # print(cx,cy)
        top_left = M.dot(np.array((0,0,1)))
        # bottom_right = M.dot(np.array((x_b,y_b,1)))
        #w = 2*np.absolute((cx-top_left[0])).astype(np.uint16)
        #h = 2*np.absolute((cy-top_left[1])).astype(np.uint16)
        img_rotated = cv2.warpAffine(img, M, (cx*2,cy*2))
        # cv2.imshow("Rotated by 45 Degrees", rotated)
        #
        # img_rotated = ndimage.rotate(img_before, median_angle)



        if self._PLOT:
            plt.subplot(3, 1, 1)
            plt.imshow(self._source_image)


            plt.subplot(3,1,2)
            plt.scatter(cx,cy)
            plt.scatter(*top_left)
            plt.imshow(img)


            print(f"Angle is {median_angle:.04f}")
            plt.subplot(3,1,3)
            plt.scatter(cx,cy)
            plt.scatter(*top_left)

            plt.imshow(img_rotated)
            plt.show()

        return img_rotated



images = glob.glob(f"{images_path}/*")
for im in images:
    try:
        im = cv2.imread(im)
        window_detector = WindowDetector(im, model)
        window_detector.detect_boxes()
        window_detector.get_image_rotation_angle(use_stick=True)
        window_detector.rotate_window(PLOT=False)
        window = window_detector.rotated_window
        match_pattern(window)
    except Exception as e:
        raise(e)
    continue
