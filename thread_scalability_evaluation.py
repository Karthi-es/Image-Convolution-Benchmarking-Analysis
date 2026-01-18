"""
as the thread num could only be set via environment variables before Numba is imported,
you should invoke thread_scalability_evaluation.sh script to run this test.
"""

import sys

import numpy as np
import time
import os
import csv
import constants
import evl_lib
from perf_monitor import PerfMonitor

# --- configuration ---
IMAGE_SIZE = 2048  # use a large image for performance testing
NUM_FILTERS = 64
KERNEL_SIZE = 3
NUM_RUNS = 5  # run multiple times and take average


def run_performance_test(function_to_call):
    """
    run performance test for a given conv2d function
    """
    # ... initialize random image and kernels
    np.random.seed(42)
    image = np.random.rand(IMAGE_SIZE, IMAGE_SIZE, 3).astype(np.float32)
    kernels = np.random.rand(NUM_FILTERS, KERNEL_SIZE, KERNEL_SIZE, 3).astype(np.float32)
    bias = np.zeros(NUM_FILTERS, dtype=np.float32)

    # 2. just-in-time compilation warm-up
    function_to_call(image, kernels, bias=bias, stride=1, padding=1, activation="relu")

    # 3. performance measurement
    monitor = PerfMonitor(interval_s=0.2).start()
    total_time = 0.0
    for _ in range(NUM_RUNS):
        start_time = time.perf_counter()
        function_to_call(image, kernels, bias=bias, stride=1, padding=1, activation="relu")
        end_time = time.perf_counter()
        total_time += (end_time - start_time)

    summary = monitor.stop_and_summarize()
    return total_time / NUM_RUNS, summary


if __name__ == '__main__':
    # 1.check command line arguments
    if len(sys.argv) < 2:
        print("Error: Missing version argument (e.g., v3_0 or v4_0)", file=sys.stderr)
        sys.exit(1)

    version_name = sys.argv[1] # get version from command line
    if version_name not in constants.VERSION_FUNCTIONS:
        print(f"Error: Unknown version {version_name}", file=sys.stderr)
        sys.exit(1)

    function_name = constants.VERSION_FUNCTIONS[version_name]
    function_to_call = evl_lib.return_method_by_name(function_name)
    current_threads = os.environ.get('NUMBA_NUM_THREADS', '1')

    # run performance test
    avg_runtime, summary = run_performance_test(function_to_call)

    # print results
    print(f"{version_name}, {current_threads}, {avg_runtime:.6f}")

    # Write extended metrics to a side CSV (append)
    metrics_file = os.path.join(os.getcwd(), "thread_scalability_metrics.csv")
    header = [
        "Version", "Threads", "Avg Runtime (s)",
        "CPU% avg", "CPU% max", "Proc CPU% avg", "Proc CPU% max",
        "RAM% avg", "RAM% max", "RSS peak (bytes)",
        "CPU time user (s)", "CPU time system (s)", "CPU time total (s)",
        "Avg parallelism (x)", "Monitor samples", "Monitor interval (s)"
    ]
    row = [
        version_name, current_threads, avg_runtime,
        summary.cpu_pct_avg, summary.cpu_pct_max, summary.cpu_proc_pct_avg, summary.cpu_proc_pct_max,
        summary.ram_pct_avg, summary.ram_pct_max, summary.rss_peak_bytes,
        summary.cpu_time_user_s, summary.cpu_time_system_s, summary.cpu_time_total_s,
        summary.avg_parallelism, summary.samples, summary.interval_s
    ]
    try:
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow(row)
    except Exception:
        pass