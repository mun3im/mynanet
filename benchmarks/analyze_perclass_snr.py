#!/usr/bin/env python3
"""Cross-reference per-clip SNR with per-clip prediction outcomes.

Addresses Reviewer-2 concern #10: per-class SNR distribution and whether
confusion concentrates in low-SNR clips, neither analysed in the manuscript.

Inputs:
  - clips.csv with columns: file_id, source_id, snr_db, ...
  - splits CSV with columns: file_id, split
  - The trained 1j INT8 .tflite for the target seed
  - The same dataset directory used at training time

Outputs:
  - perclip_snr_outcomes.csv     one row per test clip with
                                 (file_id, species, snr_db, predicted, correct)
  - perclass_snr_summary.csv     per-species: median/mean/std SNR, error rate
                                 in (low, mid, high) SNR terciles
  - snr_outcomes.pdf             scatter / box plot of SNR vs error rate

Usage:
  python3 analyze_perclass_snr.py [--seed 42] [--model 1j]

NOTE: This script needs TensorFlow to run inference; it is invoked under the
tf215_gpu conda env.
"""
from __future__ import annotations
import argparse
import csv
import os
import re
from pathlib import Path
import statistics
import sys


def load_model_classes(report_path: Path) -> list[str]:
    """Recover the model's class-index order from its sklearn classification
    report (the order rows appear in == the label-index order the model was
    trained with). This is NOT alphabetical, so it must not be reconstructed
    from sorted directory names."""
    classes = []
    row_re = re.compile(
        r"^\s*(.+?)\s+[01]\.\d{3,4}\s+[01]\.\d{3,4}\s+[01]\.\d{3,4}\s+\d+\s*$")
    for line in report_path.read_text().splitlines():
        m = row_re.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name not in ("accuracy", "macro avg", "weighted avg"):
            classes.append(name)
    return classes

DEFAULT_CLIPS_CSV  = Path(os.environ.get("CLIPS_CSV",  "/Volumes/Evo/MYGARDENBIRD/metadata16khz/clips.csv"))
DEFAULT_SPLITS_CSV = Path(os.environ.get("SPLITS_CSV", "/Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv"))
DEFAULT_FLAT_DIR   = Path(os.environ.get("FLAT_DIR",   "/Volumes/Evo/MYGARDENBIRD/mygardenbird16khz"))
# Portable results root: next to this script's parent, env-overridable.
RESULTS_ROOT = Path(
    os.environ.get(
        "RESULTS_ROOT_1",
        Path(__file__).resolve().parents[1] / "results_mygardenbird_1_linux",
    )
)


def load_clip_snr(clips_csv: Path) -> dict[str, float]:
    snr = {}
    with clips_csv.open() as f:
        for row in csv.DictReader(f):
            raw = (row.get("snr_db") or "").strip()
            if raw == "":
                continue  # some clips have no SNR estimate; skip them
            try:
                snr[row["file_id"]] = float(raw)
            except ValueError:
                continue
    return snr


def load_split(splits_csv: Path, which: str = "test") -> list[str]:
    items = []
    with splits_csv.open() as f:
        rdr = csv.reader(f)
        header = None
        for row in rdr:
            if not row or row[0].startswith("#"):
                continue
            if header is None:
                header = row
                col_id  = header.index("file_id") if "file_id" in header else 0
                col_spl = header.index("split")   if "split"   in header else 1
                continue
            if row[col_spl] == which:
                items.append(row[col_id])
    return items


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--model", default="1j")
    p.add_argument("--clips_csv",  type=Path, default=DEFAULT_CLIPS_CSV)
    p.add_argument("--splits_csv", type=Path, default=DEFAULT_SPLITS_CSV)
    p.add_argument("--flat_dir",   type=Path, default=DEFAULT_FLAT_DIR)
    return p.parse_args()


def find_tflite(model: str, seed: int) -> Path | None:
    # Find the matching result dir's INT8 tflite
    for d in RESULTS_ROOT.iterdir():
        if not d.is_dir():
            continue
        n = d.name
        if not n.startswith(model + "_"):
            continue
        if f"rand{seed}_" not in n:
            continue
        if "split80" not in n:
            continue
        # Canonical config used throughout the paper: 64 mels + mixup 0.2
        # (the front-end here generates 64-mel specs, so the model must match).
        if "mels64" not in n or "mixup0.2" not in n or "specaugment" in n:
            continue
        for tf in d.glob("*int8*.tflite"):
            return tf
    return None


def main():
    args = parse_args()
    print(f"Loading SNR from {args.clips_csv}")
    snr = load_clip_snr(args.clips_csv)
    print(f"  {len(snr)} clips with SNR")

    print(f"Loading test split from {args.splits_csv}")
    test_ids = load_split(args.splits_csv, "test")
    print(f"  {len(test_ids)} test clips")

    tflite_path = find_tflite(args.model, args.seed)
    if tflite_path is None:
        print(f"✗ No tflite found for model={args.model} seed={args.seed}",
              file=sys.stderr)
        sys.exit(1)
    print(f"Using INT8 model: {tflite_path}")

    # Recover the model's class-index order from the sibling classification
    # report — the prediction argmax indexes into THIS list, not the
    # alphabetically-sorted directory names (which gives a label permutation
    # and ~0% accuracy).
    report_path = tflite_path.parent / "classification_report_int8.txt"
    model_classes = load_model_classes(report_path) if report_path.exists() else []
    if len(model_classes) != 12:
        print(f"✗ could not recover 12-class order from {report_path}",
              file=sys.stderr)
        sys.exit(1)
    print(f"  model class order: {model_classes}")

    # Import TF lazily; pipeline must run inside tf215_gpu env
    import numpy as np
    import librosa
    import tensorflow as tf

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    print(f"  input  scale={in_d['quantization']}  dtype={in_d['dtype']}")
    print(f"  output scale={out_d['quantization']}  dtype={out_d['dtype']}")

    # Discover species list from the flat dir
    species = sorted(p.name for p in args.flat_dir.iterdir() if p.is_dir())
    print(f"  {len(species)} species: {species}")

    # Build {file_id -> (species, path)}
    fid_to_meta = {}
    for sp in species:
        sp_dir = args.flat_dir / sp
        for wav in sp_dir.glob("*.wav"):
            fid = wav.stem  # e.g. XC1002657_2860
            fid_to_meta[fid] = (sp, wav)

    # Iterate test clips and record outcome + SNR
    out_rows = []
    n_done = 0
    for fid in test_ids:
        meta = fid_to_meta.get(fid) or fid_to_meta.get(fid.lower())
        if meta is None:
            continue
        sp, wav = meta
        try:
            audio, sr = librosa.load(wav, sr=16000, mono=True)
        except Exception as e:
            print(f"  ✗ load failed {wav.name}: {e}", file=sys.stderr)
            continue
        if len(audio) < 48000:
            audio = np.pad(audio, (0, 48000 - len(audio)))
        audio = audio[:48000]
        spec = librosa.feature.melspectrogram(
            y=audio, sr=16000, n_fft=512, hop_length=160,
            win_length=400, n_mels=64, fmax=8000, center=True)
        log_spec = librosa.power_to_db(spec, ref=np.max)
        # Normalise to [0, 255] as in training
        log_spec = (log_spec - log_spec.min()) / (log_spec.max() - log_spec.min() + 1e-8)
        log_spec = (log_spec * 255).astype(np.uint8)
        log_spec = log_spec[..., np.newaxis][np.newaxis, ...]  # (1, 64, T, 1)
        # Truncate/pad to T=300
        T = 300
        if log_spec.shape[2] > T:
            log_spec = log_spec[:, :, :T, :]
        elif log_spec.shape[2] < T:
            pad = T - log_spec.shape[2]
            log_spec = np.pad(log_spec, ((0,0),(0,0),(0,pad),(0,0)))
        # Quantise to int8 range matching input details
        if in_d['dtype'] == np.int8:
            scale, zero = in_d['quantization']
            inp = (log_spec.astype(np.float32) / 255.0 / scale + zero).astype(np.int8)
        else:
            inp = (log_spec.astype(np.float32) / 255.0)
        interp.set_tensor(in_d['index'], inp)
        interp.invoke()
        logits = interp.get_tensor(out_d['index'])[0]
        pred_idx = int(np.argmax(logits))
        pred_sp = model_classes[pred_idx]
        clip_snr = snr.get(fid, snr.get(fid.upper()))
        out_rows.append({
            "file_id":   fid,
            "species":   sp,
            "snr_db":    clip_snr,
            "predicted": pred_sp,
            "correct":   pred_sp == sp,
        })
        n_done += 1
        if n_done % 50 == 0:
            print(f"  processed {n_done}/{len(test_ids)}")

    out_csv = Path(f"perclip_snr_outcomes_{args.model}_seed{args.seed}.csv")
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)
    print(f"✓ wrote {out_csv}  ({len(out_rows)} rows)")

    # Per-species SNR summary
    by_sp: dict[str, list[dict]] = {}
    for r in out_rows:
        by_sp.setdefault(r["species"], []).append(r)
    summary_csv = Path(f"perclass_snr_summary_{args.model}_seed{args.seed}.csv")
    with summary_csv.open("w") as f:
        f.write("species,n_clips,snr_median,snr_mean,snr_std,err_low_tercile,err_mid_tercile,err_high_tercile\n")
        all_snrs = sorted(r["snr_db"] for r in out_rows if r["snr_db"] is not None)
        t1, t2 = all_snrs[len(all_snrs)//3], all_snrs[2*len(all_snrs)//3]
        for sp, rs in by_sp.items():
            snrs = [r["snr_db"] for r in rs if r["snr_db"] is not None]
            if not snrs:
                continue
            errs = {"low": (0,0), "mid": (0,0), "high": (0,0)}
            for r in rs:
                if r["snr_db"] is None:
                    continue
                bucket = "low" if r["snr_db"] <= t1 else "mid" if r["snr_db"] <= t2 else "high"
                n_tot, n_err = errs[bucket]
                errs[bucket] = (n_tot + 1, n_err + (0 if r["correct"] else 1))
            def er(b):
                n_tot, n_err = errs[b]
                return f"{n_err/n_tot:.3f}" if n_tot else "n/a"
            f.write(f"{sp},{len(rs)},{statistics.median(snrs):.2f},{statistics.mean(snrs):.2f},"
                    f"{statistics.stdev(snrs) if len(snrs)>1 else 0:.2f},"
                    f"{er('low')},{er('mid')},{er('high')}\n")
    print(f"✓ wrote {summary_csv}")
    print(f"  SNR terciles: low ≤ {t1:.2f} dB, mid ≤ {t2:.2f} dB, high > {t2:.2f} dB")


if __name__ == "__main__":
    main()
