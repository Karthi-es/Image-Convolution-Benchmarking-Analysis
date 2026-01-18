#!/bin/bash

# as the thread num could only be set via environment variable for numba,
# we need to run the python script multiple times with different env settings.

# --- configuration ---
PYTHON_SCRIPT="thread_scalability_evaluation.py"
OUTPUT_FILE="thread_scalability_evaluation_result.csv"
VERSIONS="version3_0 version4_0_im2col version4_0_im2col_lib_baseline version4_0_winograd version4_0_fft" # versions to test
THREADS="1 2 4 8 16 32 64"

# get max logical cores, for information (optional)
MAX_CORES=$(nproc)
echo "Max logical cores detected: $MAX_CORES"
echo "Testing versions: $VERSIONS"
echo "Testing threads: $THREADS"
echo "-------------------------------------"


# --- init csv head---
echo "Version, Threads, Runtime_s" > "$OUTPUT_FILE"

# --- test with diff thread num and version ---
for T in $THREADS # outer loop: traverse thread numbers
do
    # 1. set environment variable for number of threads
    export NUMBA_NUM_THREADS=$T
    export MKL_NUM_THREADS=$T
    export OPENBLAS_NUM_THREADS=$T
    export OMP_NUM_THREADS=$T

    for V in $VERSIONS # inner loop: traverse versions
    do
        echo "Running test for $V with NUMBA_NUM_THREADS=$T ..."

        # 2. run the python script and capture output
        # set the version as argument
        RESULT=$(python3 "$PYTHON_SCRIPT" "$V")

        # 3. check if the python script ran successfully
        if [ $? -eq 0 ]; then
            CSV_LINE="$V, $T, $RESULT"

            echo "Finished $V ($T threads). Result: $RESULT"
            # 4. write result to csv
            echo "$CSV_LINE" >> "$OUTPUT_FILE"
        else
            echo "Error running Python script for $V ($T threads)."
            # log error in csv
            echo "$V, $T, ERROR" >> "$OUTPUT_FILE"
        fi
    done

done

echo "-------------------------------------"
echo "Scaling test complete. Results saved to $OUTPUT_FILE"
echo "The new CSV format is: Version, Threads, Runtime_s"
echo "You can now plot the data to show Strong Scaling efficiency for both versions."