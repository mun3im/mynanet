# Linux Authoritative Training Guide

**Date:** February 8, 2026
**Purpose:** Run authoritative multiseed training on Linux (not macOS prototype)
**Duration:** ~18 hours (6 models × 3 hours each)

---

## Why Linux is Authoritative

**macOS (Current):**
- ✓ Fast prototyping with M4 GPU
- ✗ Non-standard hardware (Apple Silicon)
- ✗ Results may vary from deployment environment
- → Use for **rapid iteration only**

**Linux (Tomorrow):**
- ✓ Standard CUDA/cuDNN environment
- ✓ Reproducible results
- ✓ Matches deployment testing environment
- → Use for **final results and publication**

---

## Pre-Run Checklist

### 1. Update File Paths

Edit `run_linux_authoritative.sh` and update these lines:

```bash
SPLITS_CSV="/path/to/seabird_splits_80_10_10_seed42.csv"  # Line 27
FLAT_DIR="/path/to/seabird16khz_flat"                      # Line 28
```

**Typical Linux paths:**
```bash
SPLITS_CSV="/home/username/data/seabird_splits_80_10_10_seed42.csv"
FLAT_DIR="/mnt/data/seabird16khz_flat"
```

### 2. Verify Python Environment

```bash
# Activate your TensorFlow environment
conda activate tf215_gpu  # or your env name

# Verify TensorFlow + GPU
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# Should show:
# 2.15.0 (or similar)
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 3. Verify Files Exist

```bash
# Check training scripts
ls -lh mynanet_v1.py 1e_dscnn_se_res_att_wide.py

# Check data files
ls -lh /path/to/seabird_splits_80_10_10_seed42.csv
ls /path/to/seabird16khz_flat/ | head -10

# Check results directory
mkdir -p results_linux
```

---

## Running the Training

### Launch Training Session

```bash
# Make script executable
chmod +x run_linux_authoritative.sh

# Run in background (survives logout)
nohup ./run_linux_authoritative.sh > authoritative_master.log 2>&1 &

# Get process ID
echo $! > training_pid.txt

# Verify it's running
tail -f authoritative_master.log
```

### Monitor Progress

```bash
# Watch master log
tail -f authoritative_master.log

# Watch individual model logs
tail -f v1_seed42_linux.log
tail -f 1e_seed42_linux.log

# Check GPU usage
nvidia-smi -l 5

# Check which model is training
ps aux | grep python | grep -E "mynanet_v1|1e_dscnn"
```

### Estimate Time Remaining

```bash
# Count completed models
ls results_linux/*seed* -d | wc -l

# Expected: 6 total (3 v1 + 3 1e)
# Each takes ~3 hours
# If 2 completed: ~12 hours remaining
```

---

## Expected Training Sequence

```
Hour 0-3:   v1 seed 42   → results_linux/v1_*_seed42_*
Hour 3-6:   v1 seed 100  → results_linux/v1_*_seed100_*
Hour 6-9:   v1 seed 786  → results_linux/v1_*_seed786_*
Hour 9-12:  1e seed 42   → results_linux/1e_*_seed42_*
Hour 12-15: 1e seed 100  → results_linux/1e_*_seed100_*
Hour 15-18: 1e seed 786  → results_linux/1e_*_seed786_*
```

---

## What Gets Generated

### Log Files
```
authoritative_master.log           # Master orchestration log
v1_seed42_linux.log               # MynaNet v1 seed 42
v1_seed100_linux.log              # MynaNet v1 seed 100
v1_seed786_linux.log              # MynaNet v1 seed 786
1e_seed42_linux.log               # Model 1e seed 42
1e_seed100_linux.log              # Model 1e seed 100
1e_seed786_linux.log              # Model 1e seed 786
linux_authoritative_results_*.txt # Final summary
```

### Results Directories
```
results_linux/
  v1_dscnn_se_res_att_wide_mels80_drop05_rand42_warm70_mixup0.2_seed42_linux/
  v1_dscnn_se_res_att_wide_mels80_drop05_rand100_warm70_mixup0.2_seed100_linux/
  v1_dscnn_se_res_att_wide_mels80_drop05_rand786_warm70_mixup0.2_seed786_linux/
  1e_dscnn_se_res_att_wide_mels80_drop05_rand42_warm70_mixup0.2_seed42_linux/
  1e_dscnn_se_res_att_wide_mels80_drop05_rand100_warm70_mixup0.2_seed100_linux/
  1e_dscnn_se_res_att_wide_mels80_drop05_rand786_warm70_mixup0.2_seed786_linux/
```

Each directory contains:
```
model_int8.tflite              # Quantized model
model_fp32.keras               # Full precision model
training_report.txt            # Detailed metrics
classification_report_*.txt    # Per-class performance
confusion_matrix_*.png         # Visualization
training_history.png           # Loss/accuracy curves
```

---

## Post-Training Analysis

### Quick Results Check

```bash
# Check all INT8 accuracies
grep "INT8 Accuracy:" results_linux/*/training_report.txt | grep -E "v1|1e"

# Check all model sizes
ls -lh results_linux/*/model_int8.tflite | awk '{print $9, $5}'
```

### Extract Summary

The script automatically generates a summary file. View it:

```bash
# Find the summary file
ls -lt linux_authoritative_results_*.txt | head -1

# Display summary
cat linux_authoritative_results_*.txt
```

### Calculate Statistics

```bash
# MynaNet v1 mean accuracy
grep -h "INT8 Accuracy:" results_linux/v1_*seed*/training_report.txt | \
  awk '{sum+=$3; n++} END {print "Mean:", sum/n "%", "N:", n}'

# Model 1e mean accuracy
grep -h "INT8 Accuracy:" results_linux/1e_*seed*/training_report.txt | \
  awk '{sum+=$3; n++} END {print "Mean:", sum/n "%", "N:", n}'
```

---

## Key Questions to Answer

After training completes:

### 1. Reproducibility
- **Q:** Is accuracy consistent across seeds?
- **Check:** Std deviation <0.3% = reproducible
- **Action:** If high variance, model is unstable

### 2. Size Verification
- **Q:** Does 1e truly exceed 512KB on Linux?
- **Check:** `ls -lh results_linux/1e_*/model_int8.tflite`
- **Action:** If >512KB, cannot deploy (confirm v1 as production)

### 3. Performance Gap
- **Q:** How much better is 1e than v1?
- **Check:** Mean accuracy difference
- **Action:** If <1%, v1 wins (size advantage). If >1%, consider tradeoff

### 4. Production Decision
- **Q:** Which model for deployment?
- **Criteria:**
  - v1: <512KB ✓, good accuracy, deployable
  - 1e: May be >512KB, better accuracy, risky
- **Action:** Update LaTeX and README with final decision

---

## Troubleshooting

### Training Stops Early

```bash
# Check if process is still running
cat training_pid.txt
ps -p $(cat training_pid.txt)

# Check for errors in logs
tail -100 authoritative_master.log | grep -i error

# Restart from current position (modify script to skip completed seeds)
```

### GPU Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Kill other processes using GPU
nvidia-smi | grep python
kill -9 <PID>

# Reduce batch size if needed (edit training scripts)
```

### Results Don't Match macOS

**This is EXPECTED and CORRECT:**
- Different hardware (CUDA vs Metal)
- Different random number generators
- Different cuDNN algorithms
- → Linux results are authoritative, use these for paper

---

## After Training Completes

### 1. Copy Results to Mac

```bash
# On Linux, compress results
tar -czf linux_results.tar.gz results_linux/ *.log linux_authoritative_results_*.txt

# Copy to Mac (adjust paths)
scp linux_results.tar.gz username@mac:/path/to/mynanet/

# On Mac, extract
tar -xzf linux_results.tar.gz
```

### 2. Update Documentation

- [ ] Update `MYNANET_V1_RESULTS_SUMMARY.md` with Linux results
- [ ] Update `EVOLUTION_NARRATIVE.tex` tables with authoritative numbers
- [ ] Update `README.md` with final model specs
- [ ] Recompile LaTeX: `pdflatex EVOLUTION_NARRATIVE.tex`

### 3. Make Production Decision

Based on Linux results:
- **If v1 ≥ 94.5% and 1e ≤ 95.5%:** Stick with v1 (production ready)
- **If 1e > 96%:** Consider size increase if deployment allows
- **If 1e has high variance:** Definitely v1 (stability matters)

---

## Expected Outcome

### Best Case (v1 Validates)
```
MynaNet v1 (Linux):
  Seed 42:  94.7% @ 434KB
  Seed 100: 94.6% @ 434KB
  Seed 786: 94.8% @ 434KB
  Mean: 94.7% ± 0.1%
  → Production ready! ✓

Model 1e (Linux):
  Seed 42:  95.5% @ 529KB
  Seed 100: 95.4% @ 529KB
  Seed 786: 95.6% @ 529KB
  Mean: 95.5% ± 0.1%
  → Better accuracy but OVER BUDGET ✗

Decision: MynaNet v1 for production
```

### Alternate Case (1e Within Budget)
```
Model 1e (Linux):
  Seed 42:  95.7% @ 498KB  ← Under 512KB!
  Seed 100: 95.6% @ 498KB
  Seed 786: 95.8% @ 498KB
  Mean: 95.7% ± 0.1%
  → Fits in budget with +1% accuracy!

Decision: Consider 1e if consistent, or create v1.5 compromise
```

---

## Timeline

**Evening (Today):**
- ✓ Script prepared and paths updated
- ✓ Environment verified
- ✓ Data files confirmed accessible

**Start Training (Tomorrow Morning):**
```bash
nohup ./run_linux_authoritative.sh > authoritative_master.log 2>&1 &
```

**Check Progress (Tomorrow Evening):**
- Should be ~50% complete (3/6 models done)

**Training Complete (Next Day):**
- All 6 models trained
- Summary generated
- Results analyzed
- Production decision made

---

*Preparation completed: February 8, 2026, 10:30 PM*
*Ready for Linux authoritative training tomorrow morning*
