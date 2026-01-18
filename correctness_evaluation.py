'''
4.0
1.compare result with pytorch function torch.nn.Conv2d
'''
import numpy as np
import torch
import torch.nn as nn
from version4_0_im2col import conv2d_im2col


# verify correctness with PyTorch
def verify_correctness(i_h, i_w, i_c, k_s, num_filters, stride, padding):
    # Generate random input image and kernels
    np_image = np.random.rand(i_h, i_w, i_c).astype(np.float32)
    np_kernels = np.random.rand(num_filters, k_s, k_s, i_c).astype(np.float32)
    np_bias = np.random.rand(num_filters).astype(np.float32)

    torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)

    torch_kernels = torch.from_numpy(np_kernels).permute(0, 3, 1, 2)

    torch_bias = torch.from_numpy(np_bias)

    # Create PyTorch Conv2d layer
    conv_layer = nn.Conv2d(in_channels=i_c,
                           out_channels=num_filters,
                           kernel_size=k_s,
                           stride=stride,
                           padding=padding,
                           bias=True)

    with torch.no_grad():
        conv_layer.weight.data = torch_kernels
        conv_layer.bias.data = torch_bias

        torch_output = conv_layer(torch_image)

        torch_output = nn.ReLU()(torch_output)

        torch_result_np = torch_output.squeeze(0).permute(1, 2, 0).numpy()

    # Get result from our conv2d implementation
    my_result_np = conv2d_im2col(np_image, np_kernels, bias=np_bias,
                          stride=stride, padding=padding, activation="relu")

    # Compare results
    is_correct = np.allclose(my_result_np, torch_result_np, rtol=1e-5, atol=1e-8)

    if not is_correct:
        max_diff = np.max(np.abs(my_result_np - torch_result_np))
        print(f"--- FAILED ---")
        print(f"Max Absolute Difference: {max_diff}")
        print(f"My Output Shape: {my_result_np.shape}, PyTorch Output Shape: {torch_result_np.shape}")
    else:
        print(f"--- PASSED ---")
if __name__ == '__main__':
    # run verification
    print("Validation with PyTorch (Stride 1, Padding 1):")
    verify_correctness(i_h=20, i_w=20, i_c=3, k_s=3, num_filters=16, stride=1, padding=1)

    print("\nValidation with PyTorch (Stride 2, Padding 0):")
    verify_correctness(i_h=21, i_w=21, i_c=3, k_s=3, num_filters=8, stride=2, padding=0)