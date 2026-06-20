#!/usr/bin/env python3
"""Aggregate 1j leave-one-source-out (LOSO) CV results.

Scans results_mygardenbird_1_<platform>/ for *loso_k<K>_fold<N>_* directories,
parses FP32/INT8 test accuracy from each fold's training_report.txt, and prints
the per-fold table plus mean +/- sd.  Compared against the fixed-split baseline
so the domain-shift delta is explicit (Reviewer-2 concern #7).
"""
import argparse
import glob
import os
import platform
import re
import statistics as st
import sys

# Fixed-split 1j x n_mels=64 baseline (Linux/CUDA authoritative).
BASELINE_FP32 = 94.58
BASELINE_INT8 = 94.91


def parse_fold(report_path):
    txt = open(report_path).read()
    m = re.search(r'FP32:\s*([\d.]+)%\s*\|\s*INT8:\s*([\d.]+)%', txt)
    if not m:
        return None
    ts = re.search(r'Test:\s+(\d+)', txt)
    return float(m.group(1)), float(m.group(2)), int(ts.group(1)) if ts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root',
                    default=f"results_mygardenbird_1_{platform.system().lower()}")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.results_root, '*loso_k*_fold*')))
    rows = []
    for d in dirs:
        fold = re.search(r'fold(\d+)', d).group(1)
        rep = os.path.join(d, 'training_report.txt')
        if not os.path.isfile(rep):
            print(f"  (skip incomplete: {os.path.basename(d)})", file=sys.stderr)
            continue
        res = parse_fold(rep)
        if res:
            rows.append((fold, *res))

    if not rows:
        print("No completed LOSO folds found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'fold':<6}{'FP32%':>9}{'INT8%':>9}{'test_n':>9}")
    for f, a, b, n in rows:
        print(f"{f:<6}{a:>9.2f}{b:>9.2f}{(n if n else '?'):>9}")
    print('-' * 33)

    fp = [r[1] for r in rows]
    q8 = [r[2] for r in rows]
    n = len(rows)
    sd_fp = st.stdev(fp) if n > 1 else 0.0
    sd_q8 = st.stdev(q8) if n > 1 else 0.0
    print(f"{'mean':<6}{st.mean(fp):>9.2f}{st.mean(q8):>9.2f}")
    print(f"{'sd':<6}{sd_fp:>9.2f}{sd_q8:>9.2f}   (sample sd, n={n})")
    print()
    print(f"Fixed split 80:10:10 (1j x64):  {BASELINE_FP32:.2f} FP32 / {BASELINE_INT8:.2f} INT8")
    print(f"LOSO mean:                      {st.mean(fp):.2f} FP32 / {st.mean(q8):.2f} INT8")
    print(f"Delta (LOSO - fixed):           {st.mean(fp)-BASELINE_FP32:+.2f} FP32 / "
          f"{st.mean(q8)-BASELINE_INT8:+.2f} INT8")


if __name__ == '__main__':
    main()
