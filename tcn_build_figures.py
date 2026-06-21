#!/usr/bin/env python3
"""
Generate figures for conference paper from verified_results.csv and result files.
- fig_acc_vs_size.pdf: Pareto front (accuracy vs INT8 size)
- fig_perclass_f1.pdf: Per-class F1 for representative models
"""
import csv, os, re, matplotlib.pyplot as plt, numpy as np

csv_path = "/home/muneim/Dropbox/Conda/mynanet/verified_results.csv"

# Read CSV
rows = list(csv.DictReader(open(csv_path)))
rows_dict = {r["model"]: r for r in rows}

# ====================================================================
# FIGURE 1: Accuracy vs Size (Pareto front)
# ====================================================================

def build_accsize_fig():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Models to plot (canonical only)
    models_to_plot = [
        # 1D models
        ("0_1d_raw", "1D-raw", "o", 100, "gray"),
        ("22a_tcn", "22a TCN", "s", 100, "blue"),
        ("22b_tcn", "22b TCN", "s", 80, "blue"),
        ("22d_tcn", "22d TCN", "s", 100, "blue"),
        ("22e_tcn", "22e TCN", "s", 100, "blue"),
        ("22f_tcn", "22f TCN", "s", 100, "blue"),
        ("22g_tcn", "22g TCN (best)", "s", 150, "darkblue"),
        ("22h_tcn", "22h TCN", "s", 100, "blue"),
        # 2D models
        ("1a_baseline", "1a 2D CNN", "^", 120, "red"),
        # Peers
        ("4c_shufflenetv2", "4c ShuffleNet", "D", 120, "orange"),
        ("4a_squeezenet", "4a SqueezeNet", "D", 100, "orange"),
        # Best
        ("1j_mbv3", "1j MynaNet", "*", 300, "green"),
    ]

    for model_key, label, marker, markersize, color in models_to_plot:
        if model_key not in rows_dict:
            continue
        row = rows_dict[model_key]
        try:
            acc = float(row["int8_mean"])
            kb = float(row["int8_kb"])
            mcu = row.get("mcu_fit") == "True"

            # Color by MCU fit
            if mcu:
                color = color  # use specified color
            else:
                color = "lightgray"  # over-budget = lighter

            ax.scatter(kb, acc, marker=marker, s=markersize, color=color,
                      edgecolors="black", linewidth=1.5, zorder=3, label=label)
            ax.annotate(label, (kb, acc), fontsize=8, ha="center", va="bottom",
                       xytext=(0, 3), textcoords="offset points")
        except:
            pass

    # Draw 512 KB budget line
    ax.axvline(x=512, color="red", linestyle="--", linewidth=2, label="512 KB budget", zorder=1)

    # Shade in-budget region
    ax.axvspan(0, 512, alpha=0.1, color="green", zorder=0)

    ax.set_xlabel("INT8 Model Size (KB, log scale)", fontsize=11)
    ax.set_ylabel("INT8 Accuracy (%)", fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(50, 100)
    ax.set_xlim(50, 6000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, ncol=2)

    plt.tight_layout()
    plt.savefig("/home/muneim/Dropbox/Conda/tcn/fig_acc_vs_size.pdf", dpi=150, bbox_inches="tight")
    print("✓ Saved fig_acc_vs_size.pdf")
    plt.close()

# ====================================================================
# FIGURE 2: Per-class F1
# ====================================================================

def extract_perclass_f1(result_dir, seed=42):
    """Extract per-class F1 from classification_report_int8.txt."""
    report_path = os.path.join(result_dir, "classification_report_int8.txt")
    if not os.path.exists(report_path):
        return None

    f1_scores = {}
    species_order = []
    with open(report_path) as f:
        for line in f:
            # Lines like "  Common Iora              1.00      1.00      1.00        60"
            m = re.match(r"\s+([A-Za-z\s]+?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)", line)
            if m:
                species = m.group(1).strip()
                precision = float(m.group(2))
                recall = float(m.group(3))
                f1 = float(m.group(4))
                f1_scores[species] = f1 * 100
                species_order.append(species)

    return f1_scores, species_order

def build_perclass_fig():
    fig, ax = plt.subplots(figsize=(12, 5))

    # Representative models (seed 42)
    models_perclass = [
        ("22b_tcn", "TCN-22b (in-budget)", "blue"),
        ("22g_tcn", "TCN-22g (best)", "darkblue"),
        ("1a_baseline", "2D-CNN-1a", "red"),
    ]

    # Find actual result dirs for each model
    base_dirs = {
        "22b_tcn": "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_2_linux",
        "22g_tcn": "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_2_linux",
        "1a_baseline": "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_1_linux",
    }

    species_labels = None
    x_pos = None
    width = 0.25

    for idx, (model_key, label, color) in enumerate(models_perclass):
        base_dir = base_dirs.get(model_key)
        if not base_dir:
            continue

        # Find the canonical dir for seed 42
        import glob
        pattern = os.path.join(base_dir, f"{model_key}*rand42*split80*10*10*")
        dirs = glob.glob(pattern)
        if not dirs:
            continue

        result_dir = dirs[0]
        data = extract_perclass_f1(result_dir)
        if not data:
            continue

        f1_scores, species_order = data
        if species_labels is None:
            species_labels = species_order
            x_pos = np.arange(len(species_labels))

        f1_values = [f1_scores.get(sp, 0) for sp in species_labels]
        ax.bar(x_pos + idx * width, f1_values, width, label=label, color=color, alpha=0.8)

    if species_labels:
        ax.set_xlabel("Species", fontsize=11)
        ax.set_ylabel("INT8 F1 Score (%)", fontsize=11)
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(species_labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig("/home/muneim/Dropbox/Conda/tcn/fig_perclass_f1.pdf", dpi=150, bbox_inches="tight")
        print("✓ Saved fig_perclass_f1.pdf")
        plt.close()
    else:
        print("⚠ No per-class data found for figures")

if __name__ == "__main__":
    build_accsize_fig()
    build_perclass_fig()
