import os
import sys
import glob
import argparse
import tqdm
from tqdm import tqdm
import numpy as np

from train import TEST_DATA_PATH, TRAIN_DATA_PATH

all_data = "/home/avinoam/workspace/Salignostics/data/patterns/"


labels = ['positive','negative']
split = 0.8

if __name__ == "__main__":
    for label in labels:
        test_data_path_l = f"{TEST_DATA_PATH}/{label}"
        train_data_path_l =  f"{TRAIN_DATA_PATH}/{label}"
        for p in [TEST_DATA_PATH, TRAIN_DATA_PATH, test_data_path_l, train_data_path_l]:
            if not os.path.exists(p):
                os.mkdir(p)

    for label in labels:
        files = glob.glob(f"{all_data}/{label}/markers/*.jpg")
        files = np.random.permutation(files)
        for (i,f) in tqdm(enumerate(files)):
            if i/len(files)<split:
                path = f"{TRAIN_DATA_PATH}/{label}/{os.path.split(f)[1]}"
            else:
                path = f"{TEST_DATA_PATH}/{label}/{os.path.split(f)[1]}"
            os.link(f,path)
