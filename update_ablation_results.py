#!/usr/bin/env python3
"""
Parse a training_report.txt and upsert the result row into CLAUDE.md.

Usage:
    python3 update_ablation_results.py <output_dir>
"""

import re
import sys
import os
from pathlib import Path
from datetime import datetime

CLAUDE_MD = Path(__file__).parent / "CLAUDE.md"

# Column widths for the markdown table
# Model | Seed | FP32 | INT8 | Drop | TFLite Micro Compatible
HEADER = (
    "| Model | Seed | FP32 % | INT8 % | Drop | MCU |\n"
    "|-------|------|--------|--------|------|-----|\n"
)

MODEL_ORDER = ["1c", "1d", "1d_wide", "1e"]
SEEDS = [42, 100, 786]

MCU_COMPAT = {
    "1c": "✓",   # no attention — only standard ops
    "1d": "✗",   # has BATCH_MATMUL from MHSA
    "1d_wide": "✗",
    "1e": "✗",
}


def parse_report(report_path):
    """Extract FP32 and INT8 accuracy from training_report.txt."""
    text = Path(report_path).read_text()

    fp32 = None
    int8 = None

    # Look for the summary line: "FP32: 94.67% | INT8: 94.67% | Drop: +0.00%"
    m = re.search(r"FP32:\s*([\d.]+)%\s*\|\s*INT8:\s*([\d.]+)%", text)
    if m:
        fp32 = float(m.group(1))
        int8 = float(m.group(2))

    return fp32, int8


def infer_model_key(output_dir):
    """Infer model key (1c / 1d / 1d_wide / 1e) from directory name."""
    name = Path(output_dir).name
    if name.startswith("1c_"):
        return "1c"
    if name.startswith("1d_dscnn_se_res_att_wide"):
        return "1d_wide"
    if name.startswith("1d_"):
        return "1d"
    if name.startswith("1e_"):
        return "1e"
    return "unknown"


def infer_seed(output_dir):
    """Extract random seed from directory name."""
    m = re.search(r"rand(\d+)", Path(output_dir).name)
    return int(m.group(1)) if m else None


def load_existing_results():
    """
    Load existing results from CLAUDE.md.
    Returns dict: {(model_key, seed): (fp32, int8)}
    """
    results = {}
    if not CLAUDE_MD.exists():
        return results

    text = CLAUDE_MD.read_text()
    for line in text.splitlines():
        # Match table data rows: | 1c | 42 | 94.67 | 94.67 | +0.00 | ✓ |
        m = re.match(
            r"\|\s*([\w_]+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|",
            line
        )
        if m:
            model_key = m.group(1)
            seed = int(m.group(2))
            fp32 = float(m.group(3))
            int8 = float(m.group(4))
            results[(model_key, seed)] = (fp32, int8)
    return results


def build_table(results):
    """Build the full markdown results table from the results dict."""
    lines = [HEADER.rstrip("\n")]

    model_descriptions = {
        "1c":      "DS-CNN + SE + Res",
        "1d":      "DS-CNN + SE + Res + Att",
        "1d_wide": "DS-CNN + SE + Res + Att + Wide",
        "1e":      "DS-CNN + SE + Res + Att + Wider",
    }

    for model_key in MODEL_ORDER:
        desc = model_descriptions.get(model_key, model_key)
        mcu = MCU_COMPAT.get(model_key, "?")
        for seed in SEEDS:
            key = (model_key, seed)
            if key in results:
                fp32, int8 = results[key]
                drop = int8 - fp32
                drop_str = f"{drop:+.2f}"
                lines.append(
                    f"| {model_key} | {seed} | {fp32:.2f} | {int8:.2f} | {drop_str} | {mcu} |"
                )
            else:
                lines.append(
                    f"| {model_key} | {seed} | — | — | — | {mcu} |"
                )

    # Append per-model mean rows
    lines.append("|-------|------|--------|--------|------|-----|")
    for model_key in MODEL_ORDER:
        mcu = MCU_COMPAT.get(model_key, "?")
        seed_results = [results[(model_key, s)] for s in SEEDS if (model_key, s) in results]
        if seed_results:
            mean_fp32 = sum(r[0] for r in seed_results) / len(seed_results)
            mean_int8 = sum(r[1] for r in seed_results) / len(seed_results)
            drop = mean_int8 - mean_fp32
            n = len(seed_results)
            lines.append(
                f"| **{model_key} mean** | n={n} | **{mean_fp32:.2f}** | **{mean_int8:.2f}** | {drop:+.2f} | {mcu} |"
            )
        else:
            lines.append(f"| **{model_key} mean** | n=0 | — | — | — | {mcu} |")

    return "\n".join(lines)


def update_claude_md(output_dir):
    """Main entry point: parse result and upsert into CLAUDE.md."""
    report_path = Path(output_dir) / "training_report.txt"
    if not report_path.exists():
        print(f"ERROR: {report_path} not found")
        sys.exit(1)

    fp32, int8 = parse_report(report_path)
    if fp32 is None:
        print(f"ERROR: could not parse FP32/INT8 accuracy from {report_path}")
        sys.exit(1)

    model_key = infer_model_key(output_dir)
    seed = infer_seed(output_dir)
    if model_key == "unknown" or seed is None:
        print(f"ERROR: could not determine model/seed from directory name: {output_dir}")
        sys.exit(1)

    print(f"  → {model_key} seed={seed}: FP32={fp32:.2f}%  INT8={int8:.2f}%")

    # Load existing results and upsert new one
    results = load_existing_results()
    results[(model_key, seed)] = (fp32, int8)

    table = build_table(results)
    updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build full CLAUDE.md content
    content = f"""# MynaNet — Project Context for Claude

## Project
Bird sound classifier targeting deployment on **Arduino Portenta H7** (Cortex-M7).
The deployed model must use only ops supported by TFLite Micro on Portenta H7.

## Key Constraint
`BATCH_MATMUL` (used by Keras `MultiHeadAttention`) is **not** in the TFLite Micro op set
on Portenta H7. Only 1c (no attention) produces a compatible `.tflite`.

## Dataset
- **mygardenbird16khz** — 10-class garden bird dataset, 16 kHz
- Fixed 80:10:10 train/val/test split (CSV-based, no leakage)
- Flat dir: `/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz`
- Splits CSV: `/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv`

## Model Ablation (mels=64, drop=0.05, warmup=70, mixup=0.2, split 80:10:10)

{table}

> MCU column: ✓ = TFLite Micro compatible (Portenta H7), ✗ = uses unsupported ops
> Last updated: {updated_at}

## Architecture Summary
| Model | Key difference |
|-------|----------------|
| 1c | DS-CNN + SE + Residual — no attention |
| 1d | + Multi-Head Self-Attention (2 heads, 32 key_dim) |
| 1e | + wider channels [128,192,156,384] |

## Scripts
- `1c_dscnn_se_res.py` — trains 1c
- `1d_dscnn_se_res_att.py` — trains 1d
- `1e_dscnn_se_res_att_wide.py` — trains 1e
- `run_1c_1d_1e_ablation.sh` — runs all 9 experiments (3 models × 3 seeds)
- `update_ablation_results.py` — parses results and updates this file
"""

    CLAUDE_MD.write_text(content)
    print(f"  → CLAUDE.md updated")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 update_ablation_results.py <output_dir>")
        sys.exit(1)
    update_claude_md(sys.argv[1])
