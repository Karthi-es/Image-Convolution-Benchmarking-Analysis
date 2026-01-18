'''
2.0
function include:
1.accept multiple channel input(RGB image)
'''
import numpy as np

def conv2d_multichannel(image, kernel, stride=1, padding=0):
    # image: shape (H, W, C)
    # kernel: shape (kH, kW, C)
    i_h, i_w, i_c = image.shape
    k_h, k_w, k_c = kernel.shape
    assert i_c == k_c, "channel number must be match"

    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant', constant_values=0)

    o_h = (i_h + 2 * padding - k_h) // stride + 1
    o_w = (i_w + 2 * padding - k_w) // stride + 1
    output = np.zeros((o_h, o_w))

    for i in range(o_h):
        for j in range(o_w):
            start_i = i * stride
            start_j = j * stride
            region = image[start_i:start_i + k_h, start_j:start_j + k_w, :]
            output[i, j] = np.sum(region * kernel)

    return output

image_rgb = np.random.randint(0, 256, (5, 5, 3))  # randomly generate a 5×5 RGB image
kernel_rgb = np.ones((3, 3, 3)) / 27  # simple blur kernel

result = conv2d_multichannel(image_rgb, kernel_rgb, stride=1, padding=1)
print(result)
