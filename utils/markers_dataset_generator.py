import argparse
import glob
import os
import sys

import numpy as np
from tqdm import tqdm

from consts import text_color
from pregnant_detector.pregnant_detector import PregnantTest

# from pregnant_detector import PregnantTest


'''
Iterates over path of images containing sticks
extract window and markers by PregnantTest class 
save markers to output location

Use this script to prepare a train/test set for training markers-classifier,
    or as a pre-process for fast performance evaluation. 
'''
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
        progress_bar.set_description(f"Processing {im} {len(success)}-succedd. {len(fail)}-fail ")
        progress_bar.update()
        pattern_out_path = f"{args.output_path}/markers/"
        fn = pattern_out_path + os.path.split(im)[1]
        if os.path.exists(fn) and args.skip_exist:
            print(f"{fn} exist...")
            continue

        try:
            pregnant_test.load(im)
            pregnant_test.detect_window()
            pregnant_test.match_pattern()
            pregnant_test.save_patterns()
            # pregnant_test.pattern_detector.plot()
            success.append(im)
        except Exception as e:
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
