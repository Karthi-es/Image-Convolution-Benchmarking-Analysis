'''
3.0 Navie convolution function
1.accept multiple channel input(RGB image)
2.support multiple channel output
3.support bias and activate function
4.support parallel calculation
'''
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def pad_image(image, padding):
    i_h, i_w, i_c = image.shape
    padded = np.zeros((i_h + 2 * padding, i_w + 2 * padding, i_c), dtype=np.float32)
    for h in range(i_h):
        for w in range(i_w):
            for c in range(i_c):
                padded[h + padding, w + padding, c] = image[h, w, c]
    return padded

@njit(parallel=True)
def conv2d(image, kernels, bias=None, stride=1, padding=0, activation=None):
    i_h, i_w, i_c = image.shape
    num_filters, k_h, k_w, k_c = kernels.shape
    assert i_c == k_c, "channel number must match"

    if padding > 0:
        image = pad_image(image, padding)

    o_h = (i_h + 2 * padding - k_h) // stride + 1
    o_w = (i_w + 2 * padding - k_w) // stride + 1
    output = np.zeros((o_h, o_w, num_filters), dtype=np.float32)

    for f in prange(num_filters):
        kernel = kernels[f]
        for i in range(o_h):
            for j in range(o_w):
                start_i = i * stride
                start_j = j * stride
                val = 0.0
                for ki in range(k_h):
                    for kj in range(k_w):
                        for c in range(i_c):
                            val += image[start_i + ki, start_j + kj, c] * kernel[ki, kj, c]
                if bias is not None:
                    val += bias[f]
                output[i, j, f] = val

    if activation is not None:
        if activation == "relu":
            for i in prange(o_h):
                for j in range(o_w):
                    for f in range(num_filters):
                        output[i, j, f] = max(0, output[i, j, f])
        elif activation == "sigmoid":
            for i in prange(o_h):
                for j in range(o_w):
                    for f in range(num_filters):
                        output[i, j, f] = 1 / (1 + np.exp(-output[i, j, f]))
        elif activation == "tanh":
            for i in prange(o_h):
                for j in range(o_w):
                    for f in range(num_filters):
                        output[i, j, f] = np.tanh(output[i, j, f])
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    return output

image_rgb = np.random.randint(0, 256, (5, 5, 3)).astype(np.float32)
kernel_rgb = np.ones((3, 3, 3, 3), dtype=np.float32) / 27  # blur kernel
bias = np.zeros(3, dtype=np.float32)

result = conv2d(image_rgb, kernel_rgb, bias=bias, stride=1, padding=1, activation="relu")
