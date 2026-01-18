"""Throughput evaluation script for different conv2d implementations on various datasets.
This script loads datasets from torchvision, applies different conv2d functions to each image,
and measures the throughput in images per second.
"""

import numpy as np
import time

import constants
from perf_monitor import PerfMonitor


# load dataset function
def load_dataset(dataset_name, root='./data'):
    """based on dataset_name, load the corresponding dataset from torchvision.datasets"""
    import torchvision
    import torchvision.transforms as transforms
    # common transform to convert PIL images to tensors
    transform = transforms.Compose([transforms.ToTensor()])

    # based on dataset_name, load the corresponding dataset
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root=root, train=True,
                                               download=True, transform=transform)
    elif dataset_name == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100(root=root, train=True,
                                                download=True, transform=transform)
    elif dataset_name == 'STL10':
        dataset = torchvision.datasets.STL10(root=root, split='train',
                                             download=True, transform=transform)
    elif dataset_name == 'SVHN':
        dataset = torchvision.datasets.SVHN(root=root, split='train',
                                            download=True, transform=transform)
    elif dataset_name == 'Places365':
        dataset = torchvision.datasets.Places365(root=root, split='train-standard',
                                                 download=True, transform=transform)
    elif dataset_name == 'CelebA':
        dataset = torchvision.datasets.CelebA(root=root, split='train',
                                              download=True, transform=transform)
    elif dataset_name == 'PCAM':
        dataset = torchvision.datasets.PCAM(root=root, split='train',
                                           download=True, transform=transform)
    elif dataset_name == 'Food101':
        dataset = torchvision.datasets.Food101(root=root, split='train',
                                               download=True, transform=transform)
    elif dataset_name == 'DTD':
        dataset = torchvision.datasets.DTD(root=root, split='train',
                                           download=True, transform=transform)
    elif dataset_name == 'ImageNet1000Mini':
        dataset_dir = f"{root}/imagenet-mini/train"
        dataset = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)
        print(f"Loaded ImageNet1000Mini from {dataset_dir} with {len(dataset)} images.")
    elif dataset_name == 'Celeba_hq':
        dataset_dir = f"{root}/celeba_hq"
        dataset = torchvision.datasets.ImageFolder(root=dataset_dir, transform=transform)
        print(f"Loaded Celeba-hq from {dataset_dir} with {len(dataset)} images.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return dataset


# iterate over versions
def iterate_by_version(dataset_name, image_h_w_c_warmup, train_set, total_images, kernel_rgb, bias, all_results):
    for version_name, function_name in constants.VERSION_FUNCTIONS.items():

        print(f"\n--- Testing Version: {version_name} on {dataset_name} ---")

        # 1. warm-up Numba compilation
        print("preparing JIT compilation...")
        function_to_call = return_method_by_name(function_name)
        t0_warm = time.perf_counter()
        function_to_call(image_h_w_c_warmup, kernel_rgb, bias=bias, stride=1, padding=1, activation="relu")
        warmup_time = time.perf_counter() - t0_warm
        print(f"compilation done. Warm-up/JIT time: {warmup_time:.4f}s")

        # 2. start performance test
        print(f"starting throughput evaluation on {total_images} images...")
        monitor = PerfMonitor(interval_s=0.2).start()
        start_time = time.perf_counter()
        per_image_ms = []

        # 3.process each image in the dataset
        for idx in range(total_images):
            # get idx-th image from dataset
            image_tensor, _ = train_set[idx]

            # convert to HWC format
            image_numpy_chw = image_tensor.numpy()
            image_h_w_c = np.transpose(image_numpy_chw, (1, 2, 0))

            # convolution operation
            t0 = time.perf_counter()
            _ = function_to_call(image_h_w_c, kernel_rgb, bias=bias, stride=1, padding=1, activation="relu")
            per_image_ms.append((time.perf_counter() - t0) * 1000.0)

        end_time = time.perf_counter()
        summary = monitor.stop_and_summarize()
        total_duration = end_time - start_time
        throughput = total_images / total_duration  # calculate throughput

        # latency stats (ms)
        lat_p50 = float(np.percentile(per_image_ms, 50)) if per_image_ms else None
        lat_p90 = float(np.percentile(per_image_ms, 90)) if per_image_ms else None
        lat_p99 = float(np.percentile(per_image_ms, 99)) if per_image_ms else None
        lat_mean = float(np.mean(per_image_ms)) if per_image_ms else None
        lat_std = float(np.std(per_image_ms)) if per_image_ms else None

        # 4. print results
        print("\n---------------------- RESULTS -----------------------")
        print(f"Dataset: {dataset_name}")
        print(f"Total images processed: {total_images}")
        print(f"Image size (HxW): {image_h_w_c.shape[0]}x{image_h_w_c.shape[1]}")
        print(f"Total time taken: {total_duration:.4f} seconds")
        print(f"Throughput: {throughput:.2f} images/second")
        if per_image_ms:
            print(f"Latency ms (p50/p90/p99/mean±std): {lat_p50:.2f}/{lat_p90:.2f}/{lat_p99:.2f}/{lat_mean:.2f}±{lat_std:.2f}")
        if summary.cpu_pct_avg is not None:
            print(f"CPU% avg/max: {summary.cpu_pct_avg:.1f}%/{summary.cpu_pct_max:.1f}% | "
                  f"Proc CPU% avg/max: {summary.cpu_proc_pct_avg:.1f}%/{summary.cpu_proc_pct_max:.1f}% | "
                  f"RAM% avg/max: {summary.ram_pct_avg:.1f}%/{summary.ram_pct_max:.1f}% | "
                  f"RSS peak: {summary.rss_peak_bytes or 0} bytes | "
                  f"Avg parallelism: {(summary.avg_parallelism or 0):.2f}x")
        else:
            print("Monitoring disabled (psutil not installed).")
        print("------------------------------------------------------")

        # 5. NEW: Record results to the list
        results_entry = {
            "Dataset": dataset_name,
            "Version": version_name,
            "Total Images Processed": total_images,
            "Image Height (H)": image_h_w_c.shape[0],
            "Image Width (W)": image_h_w_c.shape[1],
            "Total Time (s)": total_duration,
            "Throughput (images/s)": throughput,
            "Warmup Time (s)": warmup_time,
            "Latency p50 (ms)": lat_p50,
            "Latency p90 (ms)": lat_p90,
            "Latency p99 (ms)": lat_p99,
            "Latency mean (ms)": lat_mean,
            "Latency std (ms)": lat_std,
            # Monitoring metrics (may be None if psutil missing)
            "CPU% avg": summary.cpu_pct_avg,
            "CPU% max": summary.cpu_pct_max,
            "Proc CPU% avg": summary.cpu_proc_pct_avg,
            "Proc CPU% max": summary.cpu_proc_pct_max,
            "RAM% avg": summary.ram_pct_avg,
            "RAM% max": summary.ram_pct_max,
            "RSS peak (bytes)": summary.rss_peak_bytes,
            "CPU time user (s)": summary.cpu_time_user_s,
            "CPU time system (s)": summary.cpu_time_system_s,
            "CPU time total (s)": summary.cpu_time_total_s,
            "Avg parallelism (x)": summary.avg_parallelism,
            "Monitor samples": summary.samples,
            "Monitor interval (s)": summary.interval_s,
        }
        all_results.append(results_entry)


def main_eval(kernel_rgb=None, bias=None, dataset_list=None, filename='throughput_results.csv'):
    # define a simple 3x3 averaging kernel for RGB images
    # shape: (out_channels, kernel_height, kernel_width, in_channels)
    if kernel_rgb is None:
        kernel_rgb = np.ones((3, 3, 3, 3), dtype=np.float32) / 27
        print("Using default averaging kernel.")
    if bias is None:
        bias = np.zeros(3, dtype=np.float32)
        print("Using default zero bias.")
    if dataset_list is None:
        dataset_list = constants.THROUGHPUT_EVAL_DATASET
        print("Using default dataset list.")

    # NEW: Initialize the list to store all results
    all_results = []

    # iterate over datasets
    for dataset_name in dataset_list:
        print(f"\n==================== Running on {dataset_name} ====================")

        # 1. load dataset
        train_set = load_dataset(dataset_name)
        total_images = len(train_set)

        # get one sample image for JIT warm-up
        image_tensor, _ = train_set[0]
        # convert to HWC format for our conv2d functions
        image_h_w_c_warmup = np.transpose(image_tensor.numpy(), (1, 2, 0))

        # iterate over versions (PASS the results list)
        iterate_by_version(dataset_name, image_h_w_c_warmup, train_set, total_images, kernel_rgb, bias, all_results)

    # NEW: Save results to CSV after all evaluations are complete
    print("\n\n=============== SAVING RESULTS TO CSV ===============")
    if all_results:
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(filename, index=False)
        print(f"Results successfully saved to {filename}")
        print(f"File saved to: {filename}") # Mention the file name for the user
        # Optional: Print the first few rows of the generated DataFrame
        print("\nFinal DataFrame head:")
        print(df.head())
    else:
        print("No results were generated to save.")
    print("=====================================================")

def return_method_by_name(version_name):
    if version_name == "conv2d":
        import version3_0 as v_module
        return v_module.conv2d
    elif version_name == "conv2d_v1_wrapped":
        import wrappers as v_module
        return v_module.conv2d_v1_wrapped
    elif version_name == "conv2d_v2_wrapped":
        import wrappers as v_module
        return v_module.conv2d_v2_wrapped
    elif version_name == "conv2d_im2col":
        import version4_0_im2col as v_module
        return v_module.conv2d_im2col
    elif version_name == "im2col_lib_baseline":
        import version4_0_im2col_lib_baseline as v_module
        return v_module.im2col_lib_baseline
    elif version_name == "conv_winograd":
        import version4_0_winograd as v_module
        return v_module.conv_winograd
    elif version_name == "conv2d_fft":
        import version4_0_fft as v_module
        return v_module.conv2d_fft
    else:
        raise ValueError(f"Unknown version: {version_name}")