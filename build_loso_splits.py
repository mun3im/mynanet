#!/usr/bin/env python3
"""Build K leave-one-source-out splits from clips.csv.

Source = Xeno-canto recording (column `source_id`).  Each species' source
recordings are stratified into K folds so that every fold's test set is
source-disjoint from training, and every species is represented in roughly
equal numbers across folds.

Within each fold, the remaining (K-1) folds become train+val.  We further
hold out 10% of training sources (rounded down) as validation.

Output: one CSV per fold at splits_loso_k{K}_fold{i}.csv with columns
(file_id, split) where split ∈ {train, val, test}.
"""
from __future__ import annotations
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips_csv",  type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac_of_train", type=float, default=0.10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Read clips.csv → per-species list of (source_id, file_id) tuples.
    # We need a species column; clips.csv on disk has file_id, source_id, ...
    # but species is in the wav-file's parent directory in the flat_dir tree,
    # not clips.csv directly.  Workaround: clips.csv source IDs are stable
    # across species, and species is derivable from qc_report.csv.
    # Simpler: use the alternate qc_report.csv schema (species, file).
    qc_csv = args.clips_csv.parent / "qc_report.csv"
    species_of_file: dict[str, str] = {}
    with qc_csv.open() as f:
        for row in csv.DictReader(f):
            fid = Path(row["file"]).stem.upper()  # e.g. XC1002657_2860
            species_of_file[fid] = row["species"]

    sp_to_source_to_files: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list))
    with args.clips_csv.open() as f:
        for row in csv.DictReader(f):
            fid = row["file_id"]
            sp = species_of_file.get(fid.upper())
            if sp is None:
                continue
            sp_to_source_to_files[sp][row["source_id"]].append(fid)

    # Stratified K-fold by source within each species.
    fold_assignments: list[dict[str, str]] = [dict() for _ in range(args.k)]
    for sp, src_to_files in sp_to_source_to_files.items():
        sources = sorted(src_to_files)
        rng.shuffle(sources)
        # Round-robin assign sources to folds
        for i, src in enumerate(sources):
            test_fold = i % args.k
            for fid in src_to_files[src]:
                # Mark this clip as 'test' in fold = test_fold; for all other
                # folds it will be train or val.
                for f_idx in range(args.k):
                    if f_idx == test_fold:
                        fold_assignments[f_idx][fid] = "test"
                    else:
                        fold_assignments[f_idx].setdefault(fid, "train")

    # Carve a per-fold validation set out of training (10% of train sources).
    for f_idx, assign in enumerate(fold_assignments):
        train_files = [fid for fid, s in assign.items() if s == "train"]
        # Group by source for source-disjoint val
        src_of_file: dict[str, str] = {}
        with args.clips_csv.open() as f:
            for row in csv.DictReader(f):
                src_of_file[row["file_id"]] = row["source_id"]
        src_to_train: dict[str, list[str]] = defaultdict(list)
        for fid in train_files:
            src = src_of_file.get(fid)
            if src:
                src_to_train[src].append(fid)
        train_sources = sorted(src_to_train)
        rng_fold = random.Random(args.seed + f_idx)
        rng_fold.shuffle(train_sources)
        n_val = int(len(train_sources) * args.val_frac_of_train)
        val_sources = set(train_sources[:n_val])
        for src in val_sources:
            for fid in src_to_train[src]:
                assign[fid] = "val"

    # Write per-fold CSVs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for f_idx, assign in enumerate(fold_assignments):
        out = args.output_dir / f"splits_loso_k{args.k}_fold{f_idx}.csv"
        n = {"train": 0, "val": 0, "test": 0}
        with out.open("w") as fh:
            fh.write(f"# loso k={args.k} fold={f_idx} seed={args.seed}\n")
            fh.write("file_id,split\n")
            for fid, split in sorted(assign.items()):
                fh.write(f"{fid},{split}\n")
                n[split] += 1
        print(f"  fold {f_idx}: train={n['train']} val={n['val']} test={n['test']}  → {out.name}")


if __name__ == "__main__":
    main()
