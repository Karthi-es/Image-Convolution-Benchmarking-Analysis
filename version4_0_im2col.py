'''
4.0
implement convolution using Im2Col + GEMM with parallel computation
'''
import numpy as np
from numba import njit, prange
from version3_0 import pad_image


# --- auxiliary function: Im2Col ---
@njit(parallel=True)
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


# --- auxiliary function: parallel GEMM ---
@njit(parallel=True)
def parallel_gemm(A, B):
    """
    perform matrix multiplication C = A * B in parallel
    A: shape (M, K)
    B: shape (K, N)
    C: shape (M, N)
    """
    M, K = A.shape
    K_prime, N = B.shape
    assert K == K_prime

    C = np.zeros((M, N), dtype=np.float32)

    # core parallel loop
    for i in prange(M):
        # Numba automatically parallelizes this outer loop
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    return C


@njit(parallel=False)
def conv2d_im2col(image, kernels, bias=None, stride=1, padding=0, activation=None):
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
    M_output = parallel_gemm(M_kernel, M_col)

    # 5. Bias addition
    if bias is not None:
        # add bias to each filter's output
        M_output += bias.reshape(num_filters, 1)

    # 6. Activation function
    if activation is not None:
        # apply activation function element-wise
        if activation == "relu":
            # Numba parallel loop for ReLU
            for idx in prange(M_output.size):
                i = idx // M_output.shape[1]
                j = idx % M_output.shape[1]
                M_output[i, j] = max(0.0, M_output[i, j])
        if activation == "sigmoid":
            for idx in prange(M_output.size):
                i = idx // M_output.shape[1]
                j = idx % M_output.shape[1]
                M_output[i, j] = 1 / (1 + np.exp(-M_output[i, j]))
        if activation == "tanh":
            for idx in prange(M_output.size):
                i = idx // M_output.shape[1]
                j = idx % M_output.shape[1]
                M_output[i, j] = np.tanh(M_output[i, j])

    # 7. Reshape output to (O_h, O_w, F)
    # alternatively, calculate O_h and O_w directly
    o_h = (i_h - k_h) // stride + 1
    o_w = (i_w - k_w) // stride + 1

    # size: (O_h, O_w, F)
    final_output = M_output.T.reshape(o_h, o_w, num_filters)

    return final_output