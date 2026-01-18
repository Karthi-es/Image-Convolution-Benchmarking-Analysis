"""Wrapper functions to adapt early convolution versions to the common API.

Common API:
    (image, kernels, bias=None, stride=1, padding=0, activation=None) -> (Oh, Ow, F)
    - image: (H, W, C)
    - kernels: (F, Kh, Kw, C)
    - bias: (F,) or None
    - stride: int >= 1
    - padding: int >= 0 (zero-padding on H/W)
    - activation: None | 'relu' | 'sigmoid' | 'tanh'
"""
from __future__ import annotations

import numpy as np

import version1_0 as v1
import version2_0 as v2


def _apply_activation(x: np.ndarray, activation: str | None) -> np.ndarray:
    if activation is None:
        return x
    if activation == "relu":
        return np.maximum(0.0, x)
    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown activation: {activation}")


def _pad_image(image: np.ndarray, padding: int) -> np.ndarray:
    if padding <= 0:
        return image
    # pad spatial dims only, zero pad
    return np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="constant")


def conv2d_v1_wrapped(image: np.ndarray,
                       kernels: np.ndarray,
                       bias: np.ndarray | None = None,
                       stride: int = 1,
                       padding: int = 0,
                       activation: str | None = None) -> np.ndarray:
    """Use version1_0.conv2d_basic (single-channel) across channels and filters.

    Implements multi-channel summation and multi-filter looping here, plus padding/stride/bias/activation.
    """
    assert image.ndim == 3, "image must be (H, W, C)"
    H, W, C = image.shape
    F, Kh, Kw, Ck = kernels.shape
    assert C == Ck, "channel number must match"
    assert stride >= 1 and padding >= 0

    img = _pad_image(image.astype(np.float32, copy=False), padding)
    Hi, Wi, _ = img.shape

    # output spatial dims before stride sampling
    Oh_full = Hi - Kh + 1
    Ow_full = Wi - Kw + 1
    if Oh_full <= 0 or Ow_full <= 0:
        raise ValueError("Kernel larger than (padded) input")

    # allocate full conv result per filter
    # We'll downsample by stride afterwards
    outputs = np.zeros((Oh_full, Ow_full, F), dtype=np.float32)

    for f in range(F):
        acc = np.zeros((Oh_full, Ow_full), dtype=np.float32)
        for c in range(C):
            k2d = kernels[f, :, :, c].astype(np.float32, copy=False)
            acc += v1.conv2d_basic(img[:, :, c], k2d)
        if bias is not None:
            acc += float(bias[f])
        outputs[:, :, f] = acc

    # stride sampling
    outputs = outputs[::stride, ::stride, :]

    # activation
    outputs = _apply_activation(outputs, activation)
    return outputs


def conv2d_v2_wrapped(image: np.ndarray,
                       kernels: np.ndarray,
                       bias: np.ndarray | None = None,
                       stride: int = 1,
                       padding: int = 0,
                       activation: str | None = None) -> np.ndarray:
    """Use version2_0.conv2d_multichannel for each filter, add bias and activation.
    """
    assert image.ndim == 3, "image must be (H, W, C)"
    H, W, C = image.shape
    F, Kh, Kw, Ck = kernels.shape
    assert C == Ck, "channel number must match"
    assert stride >= 1 and padding >= 0

    # v2 handles padding and stride internally
    outputs = []
    for f in range(F):
        k3d = kernels[f].astype(np.float32, copy=False)
        out2d = v2.conv2d_multichannel(image.astype(np.float32, copy=False), k3d, stride=stride, padding=padding)
        if bias is not None:
            out2d = out2d + float(bias[f])
        outputs.append(out2d.astype(np.float32, copy=False))

    y = np.stack(outputs, axis=-1)
    y = _apply_activation(y, activation)
    return y
