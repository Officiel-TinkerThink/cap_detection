import argparse
import glob
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


class Relabeler:
    def __init__(self, idx2label, labels_value):
        self.idx2label = idx2label
        # self.label2idx = {value:key for key, value in self.idx2label.items()}
        self.labels_value = labels_value

    def classify(self, array):
        hsv = cv2.cvtColor(array, cv2.COLOR_BGR2HSV)
        mean_h = hsv[:, :, 0].mean()
        mean_s = hsv[:, :, 1].mean()
        mean_v = hsv[:, :, 2].mean()
        if 75 <= mean_h <= 130 and mean_s > 40:
            if mean_v > 100:
                label = 0  # light blue
            else:
                label = 1  # dark blue
        else:
            label = 2
        # labeling based on the treshold
        return label, (mean_h, mean_s, mean_v)

    def _process(self, image, bboxes):
        height, width = image.shape[:2]
        data = []

        # for each bboxes, crop the image
        for i, bbox in enumerate(bboxes):
            # crop the image based (get the numpy array) on the bbox given
            x_min = int((bbox[1] - bbox[3] / 2) * width)
            x_max = int((bbox[1] + bbox[3] / 2) * width)
            y_min = int((bbox[2] - bbox[4] / 2) * height)
            y_max = int((bbox[2] + bbox[4] / 2) * height)

            log.debug(f"box: {bbox}")
            log.debug(f"array: {i} --> {(x_min, y_min), (x_max, y_max)}")

            box_array = image[y_min:y_max, x_min:x_max, :]
            label, (h, s, v) = self.classify(box_array)

            log.debug(f"{i} ---> {label}")
            log.info(f"{label == bbox[0]} --> pred: {label}, default: {bbox[0]}")
            log.info(f"h: {h}, s: {s}, v: {v}")
            data.append(
                {"default_label": bbox[0], "pred": label, "h": h, "s": s, "v": v}
            )
        return data

    def process(self, image_dir):
        overall_data = []
        for im_path in sorted(glob.glob(f"{image_dir}/*.jpg")):
            label_path = im_path.replace(".jpg", ".txt")
            labels = np.loadtxt(label_path, delimiter=" ")
            image = cv2.imread(im_path)
            datum = labeler._process(image, labels)
            overall_data.append(datum)

        df = pd.DataFrame(overall_data)
        df.to_csv("hsv_analysis.csv", sep=",", index=False, header=df.columns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(args.debug)
    if args.debug:
        logging.basicConfig(
            level="DEBUG",
            format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format)
        )
    else:
        logging.basicConfig(
            level="INFO",
            format="%(asctime)s - %(levelname)s - %(message)s",  # Define the log message format)
        )
    log = logging.getLogger()
    labeler = Relabeler("k", 1)
    labeler.process("sample")
