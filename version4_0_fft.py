'''
4.0
implement convolution using fft with batch processing and optimizations
'''
import cProfile
import pstats

import numpy as np
from scipy.fft import next_fast_len


def pad_image_with_zeros(image, padding):
    """
    Auxiliary function: use np.pad to pad the image with zeros.

    Args:
        image (np.ndarray): The input image.
        padding (int): The amount of zero-padding to apply to height and width.

    Returns:
        np.ndarray: The zero-padded image.
    """
    if padding <= 0:
        return image

    # Padding size for height and width dimensions: (padding, padding) for H and W
    pad_hw = ((padding, padding),) * 2

    # No padding for the channel dimension
    pad_c = (0, 0)

    # Combine to form the full padding tuple: ((ph_start, ph_end), (pw_start, pw_end), (pc_start, pc_end))
    pads = pad_hw + (pad_c,)

    # Use np.pad to pad the image
    padded_image = np.pad(image, pads, mode='constant', constant_values=0)
    return padded_image


def conv2d_fft(image, kernels, bias=None, stride=1, padding=0, activation=None):
    """
    Performs 2D convolution on multi-channel images using the Fast Fourier Transform (FFT).

    Args:
        image (np.ndarray): Input image of shape (i_h, i_w, i_c).
        kernels (np.ndarray): Convolution kernels of shape (num_filters, k_h, k_w, k_c).
        bias (np.ndarray): Bias vector of shape (num_filters,).
        stride (int): Stride S.
        padding (int): Padding P (zero-padding before FFT).
        activation (str): Activation function ("relu", "sigmoid", "tanh").

    Returns:
        np.ndarray: Output feature map of shape (o_h, o_w, num_filters).
    """
    i_h, i_w, i_c = image.shape
    num_filters, k_h, k_w, k_c = kernels.shape

    # --- Step 0: Padding ---
    # FFT convolution requires zero-padding the input image before the FFT.
    if padding > 0:
        image_padded = pad_image_with_zeros(image, padding)
    else:
        image_padded = image

    p_h, p_w, _ = image_padded.shape  # Padded dimensions

    # --- Step 1: Determine Output and FFT Dimensions ---

    # a) Calculate the expected output spatial dimensions (o_h, o_w) *after* striding
    # This uses the standard convolution output size formula (Padded - Kernel) / Stride + 1
    o_h = (p_h - k_h) // stride + 1
    o_w = (p_w - k_w) // stride + 1

    # --- Step 2: Core FFT Operation ---

    # 1. Determine optimal FFT size for zero-padding (to avoid circular wrap-around)
    # Size should be at least (I + K - 1). next_fast_len optimizes for speed.
    fft_h = next_fast_len(p_h + k_h - 1)
    fft_w = next_fast_len(p_w + k_w - 1)

    # A) Optimized Kernel FFT (Batch Processing)
    # -----------------------------------------
    # Reshape/Transpose kernels from (Nf, Kh, Kw, Ic) to a view that allows batching: (Nf * Ic, Kh, Kw)

    # 1. Transpose: (Nf, Kh, Kw, Ic) -> (Nf, Ic, Kh, Kw)
    kernels_transposed = np.transpose(kernels, (0, 3, 1, 2))

    # 2. Reshape to allow batch FFT: (Nf, Ic, Kh, Kw) -> (Nf * Ic, Kh, Kw)
    kernels_reshaped = kernels_transposed.reshape(-1, k_h, k_w)

    # 3. Vectorized Shift (Roll) Operation: Required for correlation/convolution via FFT
    # Shift along the spatial axes (-2, -1)
    kernels_shifted = np.fft.fftshift(kernels_reshaped, axes=(-2, -1))

    # 4. Batch FFT
    # s=(fft_h, fft_w) ensures zero-padding to the full FFT size.
    # Axes (1, 2) correspond to k_h and k_w.
    kernels_fft_batched = np.fft.fftn(kernels_shifted,
                                      s=(fft_h, fft_w),
                                      axes=(1, 2))

    # 5. Restore shape for element-wise multiplication: (Nf * Ic, fft_h, fft_w) -> (Nf, Ic, fft_h, fft_w)
    kernels_fft = kernels_fft_batched.reshape(num_filters, i_c, fft_h, fft_w)

    # B) Optimized Image FFT
    # ----------------------
    # image_padded shape: (p_h, p_w, i_c)
    # Perform 2D FFT on spatial axes (0, 1). Channel axis (2) remains.
    image_padded_fft = np.fft.fftn(image_padded,
                                   s=(fft_h, fft_w),
                                   axes=(0, 1))
    # image_padded_fft shape: (fft_h, fft_w, i_c)

    # C) Core Multiplication and Summation (using Batch Operations)
    # ------------------------------------------------------------
    # kernels_fft shape: (Nf, Ic, fft_h, fft_w)
    # image_padded_fft shape: (fft_h, fft_w, Ic)

    # Adjust Kernel axes: move Ic to the last dimension for einsum
    kernels_fft_adjusted = np.transpose(kernels_fft, (0, 2, 3, 1))  # (Nf, fft_h, fft_w, Ic)

    # Einsum: For each filter f, sum over all input channels c: Sum_c (Kernel[f, h, w, c] * Image[h, w, c])
    # Result shape: (Nf, fft_h, fft_w)
    conv_result_f = np.einsum('fhwc, hwc -> fhw', kernels_fft_adjusted, image_padded_fft)

    # D) Batch Inverse FFT (IFFT)
    # ---------------------------
    # Perform IFFT on spatial axes (1, 2)
    # conv_result_f shape: (Nf, fft_h, fft_w)
    result_spatial_batched = np.fft.ifftn(conv_result_f, axes=(1, 2))

    # The full convolution result (spatial domain)
    # 1. Take the real part (due to potential numerical noise from IFFT)
    # 2. Transpose axes: (Nf, H, W) -> (H, W, Nf)
    full_output = np.real(np.transpose(result_spatial_batched, (1, 2, 0)))

    # Add Bias (Vectorized)
    if bias is not None:
        # Use broadcasting (np.newaxis for H and W dimensions)
        full_output += bias[np.newaxis, np.newaxis, :]

    # --- Step 3: Cropping and Striding ---

    # The offset to start cropping is (k_h // 2, k_w // 2) because of the kernel shift.
    # However, since the output is defined by the valid part of the convolution result
    # (which is the first p_h - k_h + 1 rows/cols), we crop to that size.
    # The FFT result naturally aligns such that the "valid" region starts at (0, 0)
    # of the full_output array after the IFFT and shift.

    # Cropping to the 'valid' convolution region size: (p_h - k_h + 1, p_w - k_w + 1)
    cropped_valid = full_output[:p_h - k_h + 1, :p_w - k_w + 1, :]

    # Apply stride sampling
    final_output = cropped_valid[::stride, ::stride, :]

    # Sanity check: ensure the dimensions match the expected o_h, o_w
    assert final_output.shape[0] == o_h and final_output.shape[1] == o_w, \
        f"Output shape mismatch! Expected ({o_h}, {o_w}), Actual {final_output.shape[:2]}"

    # --- Step 4: Activation Function ---
    if activation is not None:
        if activation == "relu":
            np.maximum(0, final_output, out=final_output)  # Use inplace operation
        elif activation == "sigmoid":
            # Must create a new array as sigmoid is not an inplace operation
            final_output = 1 / (1 + np.exp(-final_output))
        elif activation == "tanh":
            # Must create a new array
            final_output = np.tanh(final_output)
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    return final_output


def main_profiler():
    # Disable print output for clear performance data
    print("Starting profiler...")
    # Large image for meaningful profiling
    image_rgb = np.random.randint(0, 256, (5000, 5000, 3)).astype(np.float32)
    # Kernel shape: (num_filters, k_h, k_w, k_c) = (3, 3, 3, 3)
    kernel_rgb = np.ones((3, 3, 3, 3), dtype=np.float32) / 27
    bias = np.zeros(3, dtype=np.float32)


    # Core: Run the code using cProfile and write results to a file
    cProfile.runctx('conv2d_fft(image_rgb, kernel_rgb, bias=bias, activation="relu")',
    globals(),
    locals(),
    'profile_output.prof')

    print("Profiling finished. Analyzing results...")

    # Print analysis results (sorted by total time spent in the function itself)
    p = pstats.Stats('profile_output.prof')
    p.strip_dirs().sort_stats('tottime').print_stats(20) # Print top 20 time-consuming functions

if __name__ == '__main__':
    main_profiler()