"""Throughput evaluation script for different conv2d implementations on various datasets.
This script loads datasets from torchvision, applies different conv2d functions to each image,
and measures the throughput in images per second.
"""

import numpy as np
import constants
from evl_lib import main_eval


if __name__ == '__main__':
    # define a simple 3x3 averaging kernel for RGB images
    # shape: (out_channels, kernel_height, kernel_width, in_channels)
    kernel_rgb = np.ones((3, 3, 3, 3), dtype=np.float32) / 27
    bias = np.zeros(3, dtype=np.float32)
    main_eval(kernel_rgb, bias, constants.THROUGHPUT_EVAL_DATASET, "throughput_results.csv")