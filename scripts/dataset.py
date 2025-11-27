import glob
import os
import random
import re
import shutil
from pathlib import Path
from typing import Optional


class DataSplitter:
    """
    Split the dataset into train and val following the yolo structure
    target/
      images/
        train/
        val/
      labels/
        train/
        val/
    """

    def __init__(self, source: str, target: str, train_ratio: float = 0.7):
        """
        Initialize the DataSplitter class
        Args:
          source (str): the path to the source directory
          target (str): the path to the target directory
          train_ratio (float): the ratio of the training set
        """
        self.source = source
        self.target = target
        self.train_ratio = train_ratio
        output = self.restart_target()
        self.im_train = output["im_train"]
        self.im_val = output["im_val"]
        self.lab_train = output["lab_train"]
        self.lab_val = output["lab_val"]
        self.bucket = {}

    def restart_target(self) -> dict[str, Path]:
        """
        Restart the target directory
        """
        if os.path.exists(self.target):
            shutil.rmtree(self.target)
        os.makedirs(os.path.join(self.target, "images", "train"))
        os.makedirs(os.path.join(self.target, "images", "val"))
        os.makedirs(os.path.join(self.target, "labels", "train"))
        os.makedirs(os.path.join(self.target, "labels", "val"))

        return {
            "im_train": os.path.join(self.target, "images", "train"),
            "im_val": os.path.join(self.target, "images", "val"),
            "lab_train": os.path.join(self.target, "labels", "train"),
            "lab_val": os.path.join(self.target, "labels", "val"),
        }

    def mapping(self) -> None:
        """
        Mapping the label files to the bucket based on code
        """
        for im_path in glob.glob(f"{self.source}/*.jpg"):
            label_file = Path(im_path.replace(".jpg", ".txt"))
            if label_file.exists():
                match = re.search(r"_(b\d)_", label_file.name)
                if not match:
                    print("Find the image but label file does not match the pattern")
                    continue
                code = match.group(1)
                self.bucket.setdefault(code, []).append(label_file.stem)

    def split(self, seed: Optional[int] = None) -> None:
        """
        Split the dataset into train and val

        Args:
          seed (Optional[int]): the seed for random
        """
        self.mapping()

        if seed:
            random.seed(seed)
        trains = []
        vals = []
        for _, filenames in self.bucket.items():
            if len(filenames) > 1:
                train = random.sample(filenames, int(len(filenames) * self.train_ratio))
                trains.extend(train)
                val = [filename for filename in filenames if filename not in train]
                vals.extend(val)
            else:
                trains.extend(filenames)

        # copy and paste the file from source into target
        for file_name in trains:
            im_path = Path(self.source, f"{file_name}.jpg")
            lab_path = Path(self.source, f"{file_name}.txt")
            shutil.copy(im_path, self.im_train)
            shutil.copy(lab_path, self.lab_train)

        for file_name in vals:
            im_path = Path(self.source, f"{file_name}.jpg")
            lab_path = Path(self.source, f"{file_name}.txt")
            shutil.copy(im_path, self.im_val)
            shutil.copy(lab_path, self.lab_val)

        print(
            f"Splitting {len(trains)} files into train and {len(vals)} files into val"
        )


if __name__ == "__main__":
    splitter = DataSplitter("sample", "data_split")
    splitter.split(seed=42)
