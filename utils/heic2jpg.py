import os
from argparse import ArgumentParser
import glob

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm
register_heif_opener()

def convert_images(heic_path, jpg_path):
    src_files = glob.glob(heic_path+"/*.heic")
    for src_fn in tqdm(src_files):
        fn = os.path.split(src_fn)[1].replace(".heic",".jpg")
        dest_fn = f"{jpg_path}/{fn}"

        image = Image.open(src_fn)
        image = np.array(image)
        # Convert RGB to BGR
        open_cv_image = image[:, :, ::-1].copy()
        # w ,h = 1080,720
        # dest_size = [(w,h), (h,w)][open_cv_image.shape[0]>open_cv_image.shape[1]]
        # image = cv2.resize(image,dest_size)
        cv2.imwrite(dest_fn, image)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--heic_path", help = "source heic path to be converted")
    parser.add_argument("--jpg_path", help="destination jpg path")
    args = parser.parse_args()
    convert_images(args.heic_path, args.jpg_path)