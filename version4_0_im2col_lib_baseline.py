'''
4.0
implement convolution using Im2Col + GEMM with parallel computation
'''
import numpy as np
from numba import njit
from version3_0 import pad_image


# --- auxiliary function: Im2Col ---
@njit(parallel=False)  # Im2Col does not benefit much from parallelization, so we keep it serial
def im2col(image, k_h, k_w, stride):
    """
    convert input image to column matrix M_col for convolution.
    """
    i_h, i_w, i_c = image.shape

    # calculate output dimensions
    o_h = (i_h - k_h) // stride + 1
    o_w = (i_w - k_w) // stride + 1

    # 1. dimensions of M_col
    col_rows = k_h * k_w * i_c
    # col_cols = O_h * O_w (output height * output width)
    col_cols = o_h * o_w

    M_col = np.zeros((col_rows, col_cols), dtype=np.float32)

    # 2. visit each sliding window position
    col_idx = 0
    for i in range(o_h):  # output height
        for j in range(o_w):  # output width

            # determine the starting point of the current window
            start_i = i * stride
            start_j = j * stride

            # 3. visit each element in the window
            window_idx = 0
            for ki in range(k_h):
                for kj in range(k_w):
                    for c in range(i_c):
                        # get the value from the image
                        val = image[start_i + ki, start_j + kj, c]
                        # pad into M_col
                        M_col[window_idx, col_idx] = val
                        window_idx += 1

            col_idx += 1

    return M_col


@njit(parallel=False)
def im2col_lib_baseline(image, kernels, bias=None, stride=1, padding=0, activation=None):
    # 1. data preparation: padding
    if padding > 0:
        image = pad_image(image, padding)
        i_h, i_w, i_c = image.shape
    else:
        i_h, i_w, i_c = image.shape

    num_filters, k_h, k_w, k_c = kernels.shape
    assert i_c == k_c, "channel number must match"

    # 2.build M_kernel
    # size: (F, K_h * K_w * C)
    M_kernel = kernels.reshape(num_filters, k_h * k_w * k_c)

    # 3. Im2Col operation
    # M_col size: (K_h * K_w * C, O_h * O_w)
    M_col = im2col(image, k_h, k_w, stride)

    # 4. GEMM operation
    # M_output size: (F, O_h * O_w)
    M_output = M_kernel @ M_col

    # 5. Bias addition
    if bias is not None:
        # add bias to each filter's output
        M_output += bias.reshape(num_filters, 1)

    # 6. Activation function
    if activation is not None:
        # apply activation function element-wise
        if activation == "relu":
            # Numba parallel loop for ReLU
            M_output = np.maximum(np.float32(0.0), M_output)
        if activation == "sigmoid":
            M_output = 1 / (1 + np.exp(-M_output))
        if activation == "tanh":
            M_output = np.tanh(M_output)

    # 7. Reshape output to (O_h, O_w, F)
    # alternatively, calculate O_h and O_w directly
    o_h = (i_h - k_h) // stride + 1
    o_w = (i_w - k_w) // stride + 1

    # size: (O_h, O_w, F)
    final_output = M_output.T.reshape(o_h, o_w, num_filters)

    return final_output