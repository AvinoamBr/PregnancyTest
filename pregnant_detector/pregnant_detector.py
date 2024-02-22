import argparse
import glob
import logging
import os
import sys

import cv2
from tqdm import tqdm
import numpy as np
from ultralytics import YOLO

from consts import text_color, YOLO_CP, CHECKPOINT_PATH, CLASS_NAMES, CLASS_CONF_THRESH

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
    parser.add_argument("--data_path")
    parser.add_argument("--file", required=False, help="single file to run")
    parser.add_argument('-o','--output_path',help='path to save outputs', default=None)
    parser.add_argument("--skip_exist", type=bool, default=False)
    args = parser.parse_args()

    #  --- save command line ---
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)
    howto_fn = f"{args.output_path}/howto.txt"
    with open(howto_fn, 'w') as f:
        f.write(" ".join(sys.argv) + os.linesep)
    # ---------------------------

    # --------- load images list --------
    if args.file:
        images = [f"{args.data_path}/{args.file}.jpg"]
    else:
        images = glob.glob(f"{args.data_path}/*")
        images = np.random.permutation(images)
    # ---------------------------

    # init
    pregnant_test = PregnantTest(args)
    success = []
    fail = []

    # -- main loop --
    progress_bar = tqdm(images, total=len(images))
    for im in images:
        progress_bar.set_description(f"Processing {im}")
        progress_bar.update()
        pattern_out_path = f"{args.output_path}/markers/"
        fn = pattern_out_path + os.path.split(im)[1]
        if os.path.exists(fn) and args.skip_exist:
            print(f"{fn} exist...")
            continue

        try:
            pregnant_test(im)
            # pregnant_test.detect_window()
            # pregnant_test.match_pattern()
            # pregnant_test.save_patterns()
            # pregnant_test.pattern_detector.plot()
            print(f"{text_color.OKCYAN}\nim: {im}, predicted: {pregnant_test.pred_class}, "
                  f"{pregnant_test.conf:.2f}{text_color.ENDC}")
            success.append(im)
        except Exception as e:
            print(f"{text_color.FAIL}e{text_color.ENDC}")
            fail.append(f"{im} {e}")
        continue
    progress_bar.close()
    # -- /main loop ---

    # ---- Save summary ----
    success_fn = f"{args.output_path}/success.txt"
    with open(success_fn,'w') as f:
        f.write("\n".join(success)+ os.linesep)
    print(f"{len(success)}, saved into {success_fn}")

    fail_fn = f"{args.output_path}/fail.txt"
    with open(fail_fn,'w') as f:
        f.write("\n".join(fail)+ os.linesep)
    print(f"{len(fail)}, saved into {fail_fn}")
    # /---- Save summary ----
