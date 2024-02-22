import argparse
import logging
import os

import cv2
import numpy as np
from ultralytics import YOLO

from consts import text_color, YOLO_CP, CHECKPOINT_PATH, CLASS_CONF_THRESH

from window_detector.window_detector import WindowDetector
from markers_detector.markers_detector import MarkersDetector
from markers_classifier.markers_model import markers_model

class PregnantTest(object):
    def __init__(self, args):
        self._args = args
        model = YOLO(YOLO_CP)
        self.window_detector = WindowDetector(model,args)
        self.pattern_detector = MarkersDetector(args)
        self.markers_classifier = markers_model
        self.markers_classifier.load_weights(CHECKPOINT_PATH)

        self._fn = None
        self.res = None
        self.pred_class = None
        self.conf = None
        self.best_idx = None

    def load(self, fn):
        self._fn = fn
    def __call__(self, fn):
        self.load(fn)
        self.detect_window()
        self.match_pattern()
        self.classify_markers()

    def classify_markers(self):
        markers = self.pattern_detector.marker_candidates
        self.res = self.markers_classifier.predict(np.array([markers]))[0]
        self.best_idx = np.argmax(self.res)
        self.pred_class = ['negative','positive'][self.best_idx]
        self.conf = self.res[self.best_idx]
        if self.conf<CLASS_CONF_THRESH:
            self.pred_class = 'unknown'
        pass

    def detect_window(self):
        self.window_detector(self._fn)

    def match_pattern(self):
        window = self.window_detector.rotated_window
        self.pattern_detector(window)

    def save_patterns(self):
        try:
            markers = self.pattern_detector.marker_candidates
            output_path = self._args.output_path
            assert output_path != None, "please add --output_path to cml"
            pattern_out_path = f"{output_path}/markers/"
            os.makedirs(pattern_out_path, exist_ok=True)
            fn = pattern_out_path + os.path.split(self._fn)[1]
            cv2.imwrite(fn, markers)

            window_out_path = f"{output_path}/window/"
            os.makedirs(window_out_path, exist_ok=True)
            fn = window_out_path + os.path.split(self._fn)[1]
            cv2.imwrite(fn, self.pattern_detector._standard_window)
            logging.info(f"saved patterns into {output_path}")
        except Exception as e:
            raise Exception(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=False, help="single file to run")
    args = parser.parse_args()

    try:
        fn = args.file
        assert os.path.isfile(fn) , f"file {fn} not exist!"
        pregnant_test = PregnantTest(args)
        pregnant_test(fn)
        print(f"{text_color.OKCYAN}\nim: {fn}, predicted: {pregnant_test.pred_class}, "
              f"{pregnant_test.conf:.2f}{text_color.ENDC}")
    except Exception as e:
        print(f"{text_color.FAIL}{e}{text_color.ENDC}")

