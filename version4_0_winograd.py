'''
4.0
implement convolution using winograd F(2x2, 3x3) algorithm with parallel computation
'''
import numpy as np
from numba import njit, prange
import cProfile
import pstats


# --- auxiliary function: Pad Image for Winograd ---
@njit
def pad_image_winograd(image, block_size=4):
    """
    Pad image to make height and width divisible by block_size (4 for F(2x2, 3x3)).
    We assume stride=1 and padding=0 is handled externally if needed for standard conv.
    Here we pad to ensure tiling works correctly.
    """
    i_h, i_w, i_c = image.shape

    # Winograd F(2x2, 3x3) uses 4x4 input tiles.
    # The image must be padded such that (H - 2) and (W - 2) are divisible by 2.
    # A simpler requirement is padding H and W to be divisible by the tile step,
    # which is often handled by a surrounding buffer. For simplicity, we pad to make
    # the output map size divisible by 2.

    target_h = ((i_h + block_size - 1) // 2) * 2 + 2
    target_w = ((i_w + block_size - 1) // 2) * 2 + 2

    # The actual padding required
    pad_h = target_h - i_h
    pad_w = target_w - i_w

    # Simplification: pad only at the bottom/right for now (zero padding)
    new_h = i_h + pad_h
    new_w = i_w + pad_w

    if new_h == i_h and new_w == i_w:
        return image

    padded_image = np.zeros((new_h, new_w, i_c), dtype=np.float32)
    padded_image[:i_h, :i_w, :] = image
    return padded_image


# --- core Winograd Matrices ---
def get_winograd_matrices():
    """
    Define the constant transformation matrices for F(2x2, 3x3)
    """
    # G (Kernel Transform) is 4x3
    G = np.array([
        [1.0, 0.0, 0.0],
        [-2.0 / 3.0, -2.0 / 3.0, -2.0 / 3.0],
        [-2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    ], dtype=np.float32)

    # BT (Input Transform) is 4x4
    BT = np.array([
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 1.0]
    ], dtype=np.float32)

    # AT (Output Transform) is 2x4
    AT = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, -1.0]
    ], dtype=np.float32)

    return G, BT, AT


# --- step 1: pre-transform kernels ---
@njit
def transform_kernel_U(kernels, G):
    """
    Pre-calculate U = G * W * G.T for all kernels W.
    Input: kernels (F, 3, 3, C_in)
    Output: U_transformed (F, 4, 4, C_in)
    """
    num_filters, k_h, k_w, i_c = kernels.shape
    assert k_h == 3 and k_w == 3

    GT = G.T
    U_transformed = np.zeros((num_filters, 4, 4, i_c), dtype=np.float32)

    for f in prange(num_filters):
        for c in range(i_c):
            W = np.ascontiguousarray(kernels[f, :, :, c])
            U_temp = G @ W
            U_transformed[f, :, :, c] = U_temp @ GT

    return U_transformed


# --- step 2: Winograd Convolution ---
@njit(parallel=True)
def conv2d_winograd_F2x2_K3x3(image, U_transformed, G, BT, AT, bias=None, activation=None):
    """
    Perform 2D convolution using Winograd F(2x2, 3x3).
    G, BT, AT are passed in (NumPy arrays compiled into Numba types).
    """
    # 1. Get dimensions
    num_filters, _, _, i_c = U_transformed.shape
    i_h, i_w, _ = image.shape

    # Output map size (O_h, O_w)
    o_h = (i_h - 2) // 2
    o_w = (i_w - 2) // 2

    final_output = np.zeros((o_h, o_w, num_filters), dtype=np.float32)

    # Use passed matrices; compute transposes
    B = BT.T  # 4x4
    A = AT.T  # 4x2

    # Number of tiles
    num_tiles_h = o_h // 2
    num_tiles_w = o_w // 2

    B = BT.T  # 4x4
    A = AT.T  # 4x2

    # Pre-allocate temporaries
    V_temp = np.zeros((4, 4), dtype=np.float32)
    V = np.zeros((4, 4), dtype=np.float32)
    X = np.zeros((4, 4), dtype=np.float32)

    # Parallelize over tiles
    for tile_idx in prange(num_tiles_h * num_tiles_w):
        tile_i = tile_idx // num_tiles_w
        tile_j = tile_idx % num_tiles_w
        start_i = tile_i * 2
        start_j = tile_j * 2

        # 1.Initialize Tile_Z_accumulated for this tile
        # size: (F, 4, 4)
        Tile_Z_accumulated = np.zeros((num_filters, 4, 4), dtype=np.float32)

        # 2.loop over input channels
        for c in range(i_c):
            # A. abstract input tile X_c
            for r in range(4):
                for col in range(4):
                    X[r, col] = image[start_i + r, start_j + col, c]

            # B. calculate V_c = BT * X_c * B
            matmul_4x4_4x4(BT, X, V_temp)  # V_temp = BT @ X
            matmul_4x4_4x4(V_temp, B, V)  # V = V_temp @ B

            # C. abstract U_c
            U_c = U_transformed[:, :, :, c]  # Size (F, 4, 4)

            # D. use Element-wise multiplication and accumulate
            # loop F (Filter dimension)
            for f in range(num_filters):
                for r in range(4):
                    for col in range(4):
                        Tile_Z_accumulated[f, r, col] += U_c[f, r, col] * V[r, col]

        Y_temp = np.zeros((2, 4), dtype=np.float32)  # Move Y_temp here
        Y = np.zeros((2, 2), dtype=np.float32)  # Move Y here

        # 3. Output transformation for each filter
        for f in range(num_filters):
            Z_f = Tile_Z_accumulated[f]  # Size (4, 4)

            # E. Y_temp = AT @ Z_f
            matmul_2x4_4x4(AT, Z_f, Y_temp)

            # F. Y = Y_temp @ A
            matmul_2x4_4x2(Y_temp, A, Y)

            # G. Write back and apply bias/activation
            final_output[start_i:start_i + 2, start_j:start_j + 2, f] = Y

            if bias is not None:
                final_output[start_i:start_i + 2, start_j:start_j + 2, f] += bias[f]

            # Activation (Relu)
            if activation == "relu":
                # Manual ReLU for Numba efficiency
                for r in range(2):
                    for col in range(2):
                        val = final_output[start_i + r, start_j + col, f]
                        final_output[start_i + r, start_j + col, f] = max(0.0, val)
            elif activation == "sigmoid":
                for r in range(2):
                    for c in range(2):
                        val = final_output[start_i + r, start_j + c, f]
                        final_output[start_i + r, start_j + c, f] = 1.0 / (1.0 + np.exp(-val))

            elif activation == "tanh":
                for r in range(2):
                    for c in range(2):
                        val = final_output[start_i + r, start_j + c, f]
                        final_output[start_i + r, start_j + c, f] = np.tanh(val)

    return final_output


@njit(fastmath=True)
def matmul_2x4_4x2(M1, M2, result):
    """
    2x4 * 4x2 matrix multiplication expansion
    M1: 2x4
    M2: 4x2
    result: 2x2
    """
    for i in range(2):
        for j in range(2):
            # C[i, j] = sum(M1[i, k] * M2[k, j]) for k=0 to 3
            res = 0.0
            for k in range(4):
                res += M1[i, k] * M2[k, j]
            result[i, j] = res


@njit(fastmath=True)
def matmul_2x4_4x4(M1, M2, result):
    """
    2x4 * 4x4 matrix multiplication expansion
    M1: 2x4
    M2: 4x4
    result: 2x4
    """
    for i in range(2):
        for j in range(4):
            # C[i, j] = sum(M1[i, k] * M2[k, j]) for k=0 to 3
            res = 0.0
            for k in range(4):
                res += M1[i, k] * M2[k, j]
            result[i, j] = res


@njit(fastmath=True)
def matmul_4x4_4x4(M1, M2, result):
    """
    4x4 * 4x4 matrix multiplication expansion
    M1: 4x4
    M2: 4x4
    result: 4x4
    """
    for i in range(4):
        for j in range(4):
            # C[i, j] = sum(M1[i, k] * M2[k, j]) for k=0 to 3
            res = 0.0
            for k in range(4):
                res += M1[i, k] * M2[k, j]
            result[i, j] = res

# --- Example Usage (Non-Numba code for testing) ---
def main():
    # Setup dummy data (Simulating a single layer)
    H, W, C_in = 10000, 10000, 3
    F = 16
    K = 3

    image = np.random.rand(H, W, C_in).astype(np.float32)
    kernels = np.random.rand(F, K, K, C_in).astype(np.float32)
    bias = np.random.rand(F).astype(np.float32)

    # Get matrices (non-njit)
    G, BT, AT = get_winograd_matrices()

    # Force C-contiguous and correct dtype before calling njit functions
    image = np.array(image, dtype=np.float32, order='C')
    kernels = np.array(kernels, dtype=np.float32, order='C')
    G = np.array(G, dtype=np.float32, order='C')
    BT = np.array(BT, dtype=np.float32, order='C')
    AT = np.array(AT, dtype=np.float32, order='C')
    bias = np.array(bias, dtype=np.float32, order='C')

    # Pre-transform kernels (will compile transform_kernel_U)
    U_transformed = transform_kernel_U(kernels, G)

    # Execute the Winograd pipeline (this will call the njit conv function)
    output = conv2d_winograd_F2x2_K3x3(image, U_transformed, G, BT, AT, bias, "relu")

    print("\n--- Winograd Execution Summary ---")
    print(f"Original Image Size: {image.shape}")
    print(f"Output Feature Map Size: {output.shape}")
    print(f"Total time complexity reduction for a single 3x3 convolution: 9 Multiplications -> 4 Multiplications.")


def conv_winograd(image, kernels, bias=None, stride=1, padding=0, activation=None):
    # Get matrices
    G, BT, AT = get_winograd_matrices()

    # Force C-contiguous and correct dtype before calling njit functions
    image = np.array(image, dtype=np.float32, order='C')
    kernels = np.array(kernels, dtype=np.float32, order='C')
    G = np.array(G, dtype=np.float32, order='C')
    BT = np.array(BT, dtype=np.float32, order='C')
    AT = np.array(AT, dtype=np.float32, order='C')
    bias = np.array(bias, dtype=np.float32, order='C')

    # Pre-transform kernels (will compile transform_kernel_U)
    U_transformed = transform_kernel_U(kernels, G)

    # Execute the Winograd pipeline (this will call the njit conv function)
    return conv2d_winograd_F2x2_K3x3(image, U_transformed, G, BT, AT, bias, activation)


def main_profiler():
    # 禁用 print 输出以获得清晰的性能数据
    print("Starting profiler...")

    # 核心：使用 cProfile 运行你的代码
    # 将结果写入文件 'profile_output.prof'
    main()

    cProfile.runctx('main()',
    globals(),
    locals(),
    'profile_output.prof')

    print("Profiling finished. Analyzing results...")

    # 打印分析结果 (按总时间排序)
    p = pstats.Stats('profile_output.prof')
    p.strip_dirs().sort_stats('tottime').print_stats(20) # 打印前 20 个最耗时的函数

if __name__ == '__main__':
    main_profiler()
