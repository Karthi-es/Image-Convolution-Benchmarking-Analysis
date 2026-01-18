"""Post-process thread scalability results to compute speedup and efficiency.

Reads thread_scalability_evaluation_result.csv (format: Version, Threads, Runtime_s)
and writes thread_scalability_analysis.csv with added columns:
- Baseline Runtime (s) for Threads==1 per Version
- Speedup = Baseline / Runtime
- Efficiency = Speedup / Threads

Usage:
    python analyze_thread_scaling.py -i thread_scalability_evaluation_result.csv \
                                     -o thread_scalability_analysis.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="thread_scalability_evaluation_result.csv")
    ap.add_argument("-o", "--output", default="thread_scalability_analysis.csv")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input CSV not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    # Normalize column names if needed
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)

    # Drop rows with ERROR
    def _parse_runtime(x):
        try:
            return float(x)
        except Exception:
            return None

    df["Runtime_s"] = df["Runtime_s"].apply(_parse_runtime)
    df = df.dropna(subset=["Runtime_s"]).copy()

    # Compute baseline per version (Threads==1)
    baselines = (df[df["Threads"] == 1]
                 .groupby("Version")["Runtime_s"]
                 .min()  # in case of duplicates, take best baseline
                 .to_dict())

    speedups = []
    efficiencies = []
    base_runtimes = []
    for _, row in df.iterrows():
        v = row["Version"]
        t = row["Threads"]
        rt = row["Runtime_s"]
        base = baselines.get(v)
        base_runtimes.append(base)
        if base is None or rt is None or rt <= 0 or t is None or t <= 0:
            speedups.append(None)
            efficiencies.append(None)
        else:
            s = base / rt
            e = s / t
            speedups.append(s)
            efficiencies.append(e)

    df["Baseline Runtime (s)"] = base_runtimes
    df["Speedup"] = speedups
    df["Efficiency"] = efficiencies

    df.to_csv(args.output, index=False)
    print(f"Analysis written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
