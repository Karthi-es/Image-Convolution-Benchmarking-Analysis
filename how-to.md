# How to run this project

This document explains how to set up the environment and run the main scripts in this repository.

## Running from the terminal
Run scripts with `python3`. Typical commands:

- Run throughput evaluation:
  - `python3 throughput_evaluation.py`
  - Output CSV: `throughput_results.csv`

- Run thread scalability evaluation （using bash script）:
    - `bash thread_scalability_evaluation.sh`
  - Output CSV: `thread_scalability_evaluation_result_ece.csv`

- Run image scalability evaluation:
  - `python3 img_scalability_evaluation.py`
  - Output CSV: `img_scalability_results.csv`

- Run correctness checks:
  - `python3 correctness_evaluation.py`


## Running in PyCharm
1. Open PyCharm and open the project at `assignment-1`.
2. Configure the project interpreter to use the created virtual environment: `venv/bin/python`.
3. To run a script, right-click the file (for example `throughput_evaluation.py`) and choose *Run*.
4. If a script requires command-line arguments, add them in *Run > Edit Configurations*.


## Useful files
- `throughput_evaluation.py` — throughput measurement
- `thread_scalability_evaluation.py` — multi-thread scaling
- `img_scalability_evaluation.py` — image size scaling
- `correctness_evaluation.py` — correctness tests
- `data/` — datasets used by evaluations(some need manual download)
- `constants.py` — configuration constants for evaluations
- `check_env.py` — print environment information


## Tips
- some datasets need to be manually downloaded. See comments in `constants.py` for details.
- configure parameters in `constants.py` as needed. Configurable items include methods to test, datasets to use, number of threads, etc.
- some datasets are large; ensure you have sufficient disk space. If you face issues, consider removing unused datasets in `constants.py`.
- Results and introduction are in the `report/` folder.
- Have fun exploring the project!
