"""This module defines constants for different versions of convolution functions"""

# --- version mapping ---
VERSION_FUNCTIONS = {
    "version1_0": "conv2d_v1_wrapped",
    "version2_0": "conv2d_v2_wrapped",
    "version3_0": "conv2d",
    "version4_0_winograd": "conv_winograd",
    "version4_0_im2col": "conv2d_im2col",
    "version4_0_im2col_lib_baseline": "im2col_lib_baseline",
    "version4_0_fft": "conv2d_fft"
}


# define datasets for throughput evaluation
# CIFAR10, 32x32x3
# CIFAR100, 32x32x3
# SVHN, 32x32x3
# STL10, 96x96x3
# PCAM, 96x96x3
THROUGHPUT_EVAL_DATASET = ['CIFAR10', 'CIFAR100', 'SVHN', 'STL10']

# define datasets for image scalability evaluation
# CIFAR100, 32x32x3
# STL10, 96x96x3
# CelebA, 178x218x3
# ImageNet1000Mini, 224x224x3 （you need to download ImageNet1000Mini dataset separately) link:https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000
# Places365, 256x256x3
# DTD, 224x224x3
# Food101, at most 512x512x3
# Celeba_hq 1024x1024x3 (you need to download CelebA-HQ dataset separately) link:https://www.kaggle.com/datasets/lamsimon/celebahq
IMG_SCALABILITY_EVAL_DATASET = ['CIFAR100', 'STL10', 'ImageNet1000Mini', 'DTD']

# define tiling size for version 4.0
TILING_SIZE = 64