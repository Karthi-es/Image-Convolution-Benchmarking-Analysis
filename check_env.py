import platform
import psutil
import cpuinfo
import os
import sys

# Attempt to import optional libraries
try:
    import GPUtil as GPU
except ImportError:
    GPU = None

try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


def get_size(bytes, suffix="B"):
    """
    Format bytes to a human-readable format (e.g., 1024 -> 1.00 KiB)
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f} {unit}{suffix}"
        bytes /= factor
    return f"{bytes:.2f} P{suffix}"


def print_machine_learning_environment():
    print("=" * 40, "SYSTEM INFORMATION", "=" * 40)
    # Operating System Information
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"OS Version: {platform.version()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version.split()[0]} ({platform.python_compiler()})")

    # --- CPU Information ---
    print("\n" + "=" * 40, "CPU INFORMATION", "=" * 40)
    cpu_info = cpuinfo.get_cpu_info()
    print(f"Model: {cpu_info.get('brand_raw', 'N/A')}")
    print(f"Physical Cores: {psutil.cpu_count(logical=False)}")
    print(f"Logical Cores (Threads): {psutil.cpu_count(logical=True)}")
    if psutil.cpu_freq():
        print(f"Max Frequency: {psutil.cpu_freq().max:.2f} MHz")

    # --- Memory Information ---
    print("\n" + "=" * 40, "MEMORY INFORMATION", "=" * 40)
    virtual_memory = psutil.virtual_memory()
    print(f"Total RAM: {get_size(virtual_memory.total)}")
    print(f"Available RAM: {get_size(virtual_memory.available)}")

    # --- GPU (NVIDIA) Information ---
    print("\n" + "=" * 40, "GPU INFORMATION", "=" * 40)
    if GPU:
        try:
            gpus = GPU.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    print(f"--- GPU {i} ---")
                    print(f"  Name: {gpu.name}")
                    print(f"  Total Memory: {gpu.memoryTotal} MiB")
                    print(f"  Used Memory: {gpu.memoryUsed} MiB ({gpu.memoryUtil * 100:.2f}%)")
                    print(f"  Temperature: {gpu.temperature} °C")
            else:
                print("No NVIDIA GPUs detected (or drivers not properly installed).")
        except Exception as e:
            print(f"Error retrieving GPU information: {e}")
            print("Please ensure NVIDIA drivers and GPUtil are installed (`pip install gputil`).")
    else:
        print("GPUtil library not installed (`pip install gputil`), skipping GPU detection.")

    # --- Machine Learning Frameworks Information ---
    print("\n" + "=" * 40, "ML FRAMEWORKS", "=" * 40)

    # TensorFlow Information
    if tf:
        print("\n--- TensorFlow ---")
        print(f"Version: {tf.__version__}")
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            print(f"Detected GPU Devices: {len(gpu_devices)}")
            for i, device in enumerate(gpu_devices):
                print(f"  GPU {i}: {device.name}")
        else:
            print("No TensorFlow-visible GPU devices detected.")
    else:
        print("TensorFlow is not installed (or could not be imported).")

    # PyTorch Information
    if torch:
        print("\n--- PyTorch ---")
        print(f"Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            # Note: cuDNN version is often available via torch.backends.cudnn.version()
            if torch.backends.cudnn.is_available():
                print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            else:
                print("cuDNN is not available.")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i} Name: {torch.cuda.get_device_name(i)}")
        else:
            print("PyTorch is not configured with CUDA support.")
    else:
        print("PyTorch is not installed (or could not be imported).")

    print("\n" + "=" * 87)


if __name__ == "__main__":
    print_machine_learning_environment()