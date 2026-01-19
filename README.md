## Project Overview

This repository is a CPU-first laboratory for experimenting with classic convolution implementations and benchmarking their performance on real image datasets. The code evolves from a single-channel sliding window kernel to advanced algorithms such as FFT, Im2Col+GEMM, and Winograd F(2x2, 3x3). A common wrapper interface keeps every version interchangeable so that the evaluation scripts can measure throughput, thread scalability, and correctness under identical conditions.

### Implemented convolution paths
- **version1_0** – naive 2D convolution for grayscale images.
- **version2_0** – multi-channel convolution with padding and stride.
- **version3_0** – fully featured multi-filter conv2d accelerated with Numba parallel loops.
- **version4_0_fft** – frequency-domain convolution using SciPy FFTs.
- **version4_0_im2col / im2col_lib_baseline** – Im2Col + GEMM pipeline (Numba-parallel vs. NumPy baseline).
- **version4_0_winograd** – Winograd F(2x2,3x3) tiling with handcrafted transforms.
- **wrappers.py** exposes a unified API for legacy versions so the evaluators can call them like modern layers.

### Key supporting modules
- **evl_lib.py** – shared evaluation harness (dataset loading, warm-up, timing, CSV export).
- **perf_monitor.py** – lightweight psutil-based sampler for CPU, RAM, and process metrics.
- **constants.py** – dataset lists, version lookup table, and shared hyperparameters.
- **report/** – documentation of experimental results (see folder for details).

## Requirements
- Python 3.9+
- NumPy, Numba, SciPy, Torch, TorchVision, pandas, psutil (install via `pip install -r requirements.txt` if provided, or install packages manually).
- Access to the datasets listed in `constants.py` (some, like ImageNet Mini or CelebA-HQ, require manual download into `data/`).

## Quick Start
```bash
# create and activate a virtual environment (optional but recommended)
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install numpy numba scipy torch torchvision pandas psutil
```

## Running the evaluations

### 1. Throughput benchmarking
Measures images/second for every convolution version on each dataset defined in `constants.THROUGHPUT_EVAL_DATASET`.
```bash
python3 throughput_evaluation.py
# results: throughput_results.csv
```

### 2. Image scalability study
Explores how performance changes as image resolution grows.
```bash
python3 img_scalability_evaluation.py
# results: img_scalability_results.csv
```

### 3. Thread scalability benchmark
Evaluates Numba-threaded versions by sweeping `NUMBA_NUM_THREADS`. Use the provided shell script so the environment variable is set before Python imports Numba.
```bash
bash thread_scalability_evaluation.sh
# raw runtimes: thread_scalability_evaluation_result.csv
# extended metrics: thread_scalability_metrics.csv

# optional: compute speedup/efficiency columns
python3 analyze_thread_scaling.py \
	-i thread_scalability_evaluation_result.csv \
	-o thread_scalability_analysis.csv
```

### 4. Correctness checks
Compares the Im2Col implementation against `torch.nn.Conv2d` outputs for several configurations.
```bash
python3 correctness_evaluation.py
```

### 5. Environment sanity check
```bash
python3 check_env.py
```