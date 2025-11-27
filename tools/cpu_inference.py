import os
import time

import cv2
import ncnn
import numpy as np

# YOLO settings
INPUT_SIZE = 320  # depends on your model (check Ultralytics export log)
CONF_THRES = 0.25
NMS_THRES = 0.45


def load_ncnn_model(param_path: str, bin_path: str) -> ncnn.Net:
    """Load an NCNN model from .param and .bin files.

    Args:
        param_path (str): Path to the NCNN `.param` file.
        bin_path (str): Path to the NCNN `.bin` file.

    Returns:
        ncnn.Net: Loaded NCNN network object.
    """
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    return net


def preprocess(image: np.ndarray) -> np.ndarray:
    """Resize and normalize an image for NCNN YOLO input.

    Args:
        image (np.ndarray): Original input image (HWC BGR format).

    Returns:
        np.ndarray: Preprocessed CHW float32 image normalized to [0, 1].
    """
    img = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return img


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert YOLO-format bounding boxes (cx, cy, w, h) to (x1, y1, x2, y2).

    Args:
        x (np.ndarray): Bounding boxes in [cx, cy, w, h] format.

    Returns:
        np.ndarray: Bounding boxes converted to [x1, y1, x2, y2].
    """
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def run_inference(
    model_dir: str, image_path: str, num_runs: int = 1000, num_warmups: int = 5
):
    """Run NCNN YOLO inference multiple times and measure speed statistics.

    This function loads an NCNN-exported YOLO model (`model.ncnn.param` and
    `model.ncnn.bin`), preprocesses an input image, performs several warm-up
    iterations, and then measures inference latency over multiple runs.

    Args:
        model_dir (str): Directory containing `model.ncnn.param` and `model.ncnn.bin`.
        image_path (str): Path to the input image for inference.
        num_runs (int, optional): Number of timed inference runs. Defaults to 1000.
        num_warmups (int, optional): Number of warm-up runs to stabilize performance. Defaults to 5.

    Prints:
        Max, Min, Avg, and Std (standard deviation) inference times in milliseconds.
    """
    param_path = os.path.join(model_dir, "model.ncnn.param")
    bin_path = os.path.join(model_dir, "model.ncnn.bin")

    img = cv2.imread(str(image_path))
    img = preprocess(img)

    net = load_ncnn_model(param_path, bin_path)

    in_name = net.input_names()[0]
    out_name = net.output_names()[0]

    time_records = []

    # Warm-up runs
    for _ in range(num_warmups):
        ex = net.create_extractor()
        ex.input(in_name, ncnn.Mat(img))
        ex.extract(out_name)

    # Timed runs
    for _ in range(num_runs):
        ex = net.create_extractor()
        ex.input(in_name, ncnn.Mat(img))
        start_time = time.perf_counter()
        ex.extract(out_name)
        end_time = time.perf_counter()
        time_records.append((end_time - start_time) * 1000)

    max_t = np.max(time_records)
    min_t = np.min(time_records)
    avg_t = np.mean(time_records)
    std_t = np.std(time_records)

    print(f"Max: {max_t:.4f} ms")
    print(f"Min: {min_t:.4f} ms")
    print(f"Avg: {avg_t:.4f} ± {std_t:.4f} ms")


if __name__ == "__main__":
    model_dir = "runs/detect/train/weights/best_ncnn_model"
    img_path = "data/images/val/raw-250110_dc_s001_b2_15.jpg"
    run_inference(model_dir, img_path)
