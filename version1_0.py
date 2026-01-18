'''
1.0
function include:
1.simple implement convolution
'''
import numpy as np


def conv2d_basic(image, kernel):
    i_h, i_w = image.shape
    k_h, k_w = kernel.shape
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1
    output = np.zeros((o_h, o_w))

    for i in range(o_h):
        for j in range(o_w):
            region = image[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)

    return output

# example
image = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [0, 1, 2, 3]
])

kernel = np.array([
    [1, 0],
    [0, -1]
])

result = conv2d_basic(image, kernel)
print(result)
