#!/usr/bin/env python3
"""
Generate LaTeX tables for conference paper from verified_results.csv.
Uses only canonical configs (config_ok=True or best canonical within family).
"""
import csv, sys

csv_path = "/home/muneim/Dropbox/Conda/mynanet/verified_results.csv"

# Read CSV
rows = list(csv.DictReader(open(csv_path)))

# Define which rows to include, with LaTeX formatting
# Format: (model_from_csv, display_name, is_canonical_requirement, context)
TABLE_MAIN = [
    ("0_1d_raw", "1D-raw (raw waveform)", "canonical", "Baseline"),
    ("22a_tcn", "22a (baseline TCN)", "canonical", "In-budget 1D"),
    ("22g_tcn", "22g (best TCN)", "canonical", "Best in-budget 1D"),
    ("1a_baseline", "1a (2D CNN baseline)", "canonical", "2D reference"),
    ("4c_shufflenetv2", "4c (ShuffleNetV2-Compact)", "peer", "In-budget peer"),
    ("1j_mbv3", "1j (deployed MynaNet)", "canonical", "Best deployable"),
]

TABLE_TCN_ABLATION = [
    ("22g_tcn", "22g", "no-residual, k=3"),
    ("22d_tcn", "22d", "deeper (10L)"),
    ("22e_tcn", "22e", "small kernel (k=2)"),
    ("22f_tcn", "22f", "large kernel (k=5)"),
    ("22c_tcn", "22c", "wide (128ch)"),
    ("22a_tcn", "22a", "baseline (8L, 64ch)"),
    ("22h_tcn", "22h", "narrow (32ch, 6L)"),
    ("22b_tcn", "22b", "shallow (4L)"),
]

def format_row(model_key, display_name, context=""):
    """Find model in CSV and format for LaTeX table."""
    for row in rows:
        if row["model"] == model_key:
            int8_mean = row.get("int8_mean", "").strip()
            int8_sd = row.get("int8_sd", "").strip()
            kb = row.get("int8_kb", "").strip()
            mcu = row.get("mcu_fit", "").strip()

            if not int8_mean: return None

            int8_str = f"{int8_mean} & {int8_sd}" if int8_sd else int8_mean
            mcu_sym = "\\textbf{yes}" if mcu == "True" else "no"

            return (display_name, int8_str, kb, mcu_sym)
    return None

# Generate TABLE: Main Comparison
print("% TABLE: Main Comparison")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Best Model Per Family: Accuracy (INT8), Size, and Deployability}")
print("\\begin{tabular}{lrrrc}")
print("\\toprule")
print("\\textbf{Model (family)} & \\textbf{INT8 \\%} & \\textbf{$\\pm$sd} & \\textbf{INT8 KB} & \\textbf{MCU?} \\\\")
print("\\midrule")

for model_key, display_name, category, context in TABLE_MAIN:
    fmt = format_row(model_key, display_name)
    if fmt:
        name, int8_str, kb, mcu = fmt
        if category == "canonical":
            print(f"{name:40s} & {int8_str:15s} & {kb:10s} & {mcu} \\\\")
        elif category == "peer":
            print(f"{name:40s} & {int8_str:15s} & {kb:10s} & {mcu} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:main}")
print("\\end{table}")
print()

# Generate TABLE: TCN Ablation
print("% TABLE: TCN Ablation (canonical 22-series only)")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{TCN Ablation: 8 Variants (INT8, 3 Seeds, Canonical Protocol). Sorted by Accuracy.}")
print("\\begin{tabular}{llrrc}")
print("\\toprule")
print("\\textbf{Variant} & \\textbf{Configuration} & \\textbf{INT8 \\% $\\pm$sd} & \\textbf{KB} & \\textbf{MCU?} \\\\")
print("\\midrule")

# Sort by accuracy descending
tcn_data = []
for model_key, variant_label, config_desc in TABLE_TCN_ABLATION:
    for row in rows:
        if row["model"] == model_key:
            try:
                acc = float(row.get("int8_mean", 0))
                tcn_data.append((variant_label, config_desc, acc, row))
            except:
                pass

tcn_data.sort(key=lambda x: -x[2])

for variant_label, config_desc, acc, row in tcn_data:
    int8_str = f"{row['int8_mean']} $\\pm$ {row['int8_sd']}"
    kb = row.get("int8_kb", "")
    mcu = "\\textbf{yes}" if row.get("mcu_fit") == "True" else "no"
    print(f"{variant_label:5s} & {config_desc:25s} & {int8_str:18s} & {kb:8s} & {mcu} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\label{tab:tcn-ablation}")
print("\\end{table}")
