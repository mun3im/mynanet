#!/usr/bin/env python3
"""
Phase 0: Canonical parser across all result directories.

Scans all results_mygardenbird_*/ dirs, extracts config + accuracy + size,
validates against canonical protocol (mels64, drop05, warm70, mixup, split80:10:10, 3 seeds).
Outputs verified_results.csv with per-model: fp32_mean, fp32_sd, int8_mean, int8_sd,
params, int8_kb, mcu_fit, config_ok, config_note.

Canonical run = mels64, drop05, warm70, mixup0.2, split80:10:10, seeds {42,100,786}.
Use sample std (ddof=1).
"""
import glob, os, re, statistics as st, csv, sys, math
from collections import defaultdict

def sample_std(vals):
    """Compute sample std (ddof=1)."""
    if len(vals) < 2: return 0
    m = st.mean(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))

# All result roots across projects
ROOTS = [
    "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_1_linux",
    "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_2_linux",
    "/home/muneim/Dropbox/Conda/mynanet/results_mygardenbird_3_linux",
    "/home/muneim/Dropbox/Conda/tcn/results_mygardenbird_0_linux",
    "/home/muneim/Dropbox/Conda/tcn/results_mygardenbird_2_linux",
    "/home/muneim/Dropbox/Conda/mobilenet-inspired/results_mygardenbird_4_linux",
    "/home/muneim/Dropbox/Conda/mynanet/benchmarks/results_mygardenbird_4_linux",
]

def acc(path):
    """Extract accuracy from classification_report_*.txt (normalized 0-100)."""
    if not os.path.exists(path): return None
    for line in open(path):
        m = re.search(r"^\s*accuracy\s+([\d.]+)\s+\d+", line)
        if m: return float(m.group(1)) * 100
    return None

def tr_field(path, pat):
    """Extract field from training_report.txt via regex."""
    if not os.path.exists(path): return None
    m = re.search(pat, open(path).read())
    return m.group(1) if m else None

def parse_dir(dirpath):
    """
    Parse a single result dir. Return dict with keys:
      model, seed, config_str, fp32, int8, params, kb, config_ok, config_note
    or None if dir is incomplete/unreadable.
    """
    basename = os.path.basename(dirpath)
    tr = os.path.join(dirpath, "training_report.txt")
    if not os.path.exists(tr): return None

    # Extract from dir name: model prefix, seed, config hints
    # Patterns: 1j_mbv3_se_mels64_drop05_rand42_warm70_mixup0.2_split80:10:10_linux
    #          22g_tcn_64ch_8l_k3_mels64_drop05_rand42_warm70_mixup0.2_split80:10:10_linux
    #          4c_shufflenetv2_compact_drop20_rand100_warm50
    match = re.search(r"^(\d+[a-z]_[^_]+)_.*rand(\d+)_", basename)
    if not match: return None
    model, seed = match.group(1), int(match.group(2))

    # Read training report
    cfg_mels = tr_field(tr, r"Mel Bins.*?:\s*(\d+)")
    cfg_drop = tr_field(tr, r"Dropout Rate.*?:\s*([\d.]+)")
    cfg_warm = tr_field(tr, r"Warmup Epochs.*?:\s*(\d+)")
    cfg_split = tr_field(tr, r"Splits CSV.*?split[s]?_mip_(\d+_\d+_\d+)" )
    if not cfg_split:
        cfg_split = tr_field(tr, r"split[s]?_(\d+:\d+:\d+)" )
    params = tr_field(tr, r"Total Parameters:\s+([\d,]+)")
    kb = tr_field(tr, r"INT8 \(\.tflite\)\s*:\s*([\d.]+)\s*KB")

    # Check if canonical (case-insensitive, flexible on split notation)
    is_canonical = (
        cfg_mels == "64" and
        cfg_drop == "0.05" and
        cfg_warm == "70" and
        (cfg_split in ["80_10_10", "80:10:10", "80_10_10"] or (cfg_split and "80" in cfg_split and "10" in cfg_split))
    )
    config_note = ""
    if not is_canonical:
        config_note = f"mels={cfg_mels} drop={cfg_drop} warm={cfg_warm} split={cfg_split}"

    # Read accuracies
    fp32 = acc(os.path.join(dirpath, "classification_report_fp32.txt"))
    int8 = acc(os.path.join(dirpath, "classification_report_int8.txt"))

    if fp32 is None or int8 is None: return None

    return {
        "model": model,
        "seed": seed,
        "fp32": fp32,
        "int8": int8,
        "params": params,
        "kb": kb,
        "config_ok": is_canonical,
        "config_note": config_note,
    }

# Parse all directories
all_runs = []
for root in ROOTS:
    if not os.path.isdir(root):
        print(f"# Skipping missing root: {root}", file=sys.stderr)
        continue
    for dirpath in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(dirpath): continue
        parsed = parse_dir(dirpath)
        if parsed: all_runs.append(parsed)

# Group by model
by_model = defaultdict(list)
for run in all_runs:
    by_model[run["model"]].append(run)

# Generate CSV output
csv_path = "/home/muneim/Dropbox/Conda/mynanet/verified_results.csv"
fieldnames = [
    "model", "n_seeds", "fp32_mean", "fp32_sd", "int8_mean", "int8_sd",
    "params", "int8_kb", "mcu_fit", "config_ok", "config_note",
]

with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()

    for model in sorted(by_model.keys()):
        runs = by_model[model]
        seeds_found = sorted(set(r["seed"] for r in runs))
        canonical_runs = [r for r in runs if r["config_ok"]]

        if len(canonical_runs) == 0:
            # No canonical run; use what we have (flag in config_ok)
            runs_to_use = runs
            config_ok_str = "False"
            config_note = "; ".join(set(r["config_note"] for r in runs if r["config_note"]))
        else:
            runs_to_use = canonical_runs
            config_ok_str = "True" if len(canonical_runs) == len(seeds_found) else f"Partial ({len(canonical_runs)}/{len(seeds_found)} seeds)"
            config_note = ""

        fp32s = [r["fp32"] for r in runs_to_use]
        int8s = [r["int8"] for r in runs_to_use]

        fp32_mean = st.mean(fp32s) if fp32s else None
        fp32_sd = sample_std(fp32s)
        int8_mean = st.mean(int8s) if int8s else None
        int8_sd = sample_std(int8s)

        params = runs_to_use[0]["params"] if runs_to_use[0]["params"] else ""
        kb = runs_to_use[0]["kb"] if runs_to_use[0]["kb"] else ""
        mcu_fit = "True" if (kb and float(kb) <= 512) else "False" if kb else ""

        w.writerow({
            "model": model,
            "n_seeds": len(runs_to_use),
            "fp32_mean": f"{fp32_mean:.2f}" if fp32_mean else "",
            "fp32_sd": f"{fp32_sd:.2f}" if fp32_sd else "",
            "int8_mean": f"{int8_mean:.2f}" if int8_mean else "",
            "int8_sd": f"{int8_sd:.2f}" if int8_sd else "",
            "params": params,
            "int8_kb": kb,
            "mcu_fit": mcu_fit,
            "config_ok": config_ok_str,
            "config_note": config_note,
        })

print(f"✓ Wrote {csv_path}")
print(f"  {len(all_runs)} runs parsed, {len(by_model)} unique models")
print(f"\n# Models with non-canonical config:")
for model in sorted(by_model.keys()):
    runs = by_model[model]
    non_canonical = [r for r in runs if not r["config_ok"]]
    if non_canonical:
        seeds_nc = sorted(set(r["seed"] for r in non_canonical))
        notes = set(r["config_note"] for r in non_canonical)
        print(f"  {model}: seeds {seeds_nc} — {'; '.join(notes)}")
