import cv2
import numpy as np
from markers_detector.utils.Gaussian import gaussuian_filter
from matplotlib import pyplot as plt

class MarkersDetector(object):
    '''
    A class that perform classical CV methods to enhance image, and return image in standard shape
    that present a better map for pregnant test markers.
    '''
    MATCH_SHAPE = (80,80)
    def __init__(self, args):
        self._args = args
        self._window = None
        self._standard_window = None
        self.vertical_edges = None
        self.horizontal_edges = None
        self.marker_match = None
        self.marker_candidates = None

    def plot(self):
        plt.subplots(2,3)
        for i,mat in enumerate((self._standard_window, self.vertical_edges, self.horizontal_edges,
                                self.vertical_fixed, self.marker_match, self.marker_candidates)):
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
        self.vertical_fixed = self.vertical_edges - self.horizontal_edges

    def find_markers_candidates(self):
        marker_match = cv2.filter2D(src=self.vertical_fixed, ddepth=-1, kernel=self.marker_kernel).astype(np.int16)
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
        if hasattr(self, "_horizontal_kernel") and self._horizontal_kernel is not None:
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
