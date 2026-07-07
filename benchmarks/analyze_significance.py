#!/usr/bin/env python3
"""Paired permutation tests + bootstrap CIs for the sub-1pp deltas.

Addresses Reviewer-2 concern #8: small accuracy gaps reported as conclusions
without significance tests.

Tests these claims from the manuscript:
  - 1j vs 1e   (n=5 seeds)
  - 1j vs 1p   (n=5 seeds)
  - 1j vs 1i   (n=5 seeds)
  - 1j vs 1q   (n=5 seeds)
  - 1e vs 1c   (n=3 seeds; 1c not re-seeded in E2)
  - 1g vs 1e   (n=3 seeds; 1g not re-seeded in E2)
  - 1f vs 1e   (n=3 seeds; 1f not re-seeded in E2)
  - 1q vs 1i   (n=5 seeds)

For each comparison, n = min(len(A), len(B)); zip truncates automatically.
  (a) paired permutation test over the 2^n sign assignments
  (b) percentile bootstrap CI from the per-seed paired diffs (10k resamples)

Minimum attainable two-sided p: n=3 → 2/8=0.250; n=5 → 2/32=0.0625.

Writes significance_results.csv and a Markdown summary.

Usage:
  python3 analyze_significance.py
"""
from __future__ import annotations
import csv
import itertools
import random
import statistics
from pathlib import Path

# INT8 per-seed accuracies extracted from training_report.txt files on disk.
# Seeds 42/100/786: original 3-seed runs (all models).
# Seeds 7/2024: E2 extension (1e, 1i, 1j, 1p, 1q only — verified 2026-07-01).
# Models 1c, 1f, 1g were NOT re-seeded in E2 (stay at n=3; footnote in paper).
PER_SEED_INT8 = {
    "1a": [93.06, 93.75, 92.50],           # n=3
    "1b": [93.75, 93.06, 92.36],           # n=3
    "1c": [94.44, 93.33, 93.06],           # n=3
    "1d": [92.50, 92.78, 94.03],           # n=3
    "1e": [94.44, 93.75, 94.17, 93.47, 93.06],   # n=5 (seeds 42,100,786,7,2024)
    "1f": [94.03, 93.89, 92.92],           # n=3
    "1g": [93.89, 94.03, 94.72],           # n=3
    "1i": [94.44, 94.17, 93.47, 93.61, 93.47],   # n=5
    "1j": [94.31, 95.42, 95.00, 95.14, 95.42],   # n=5
    "1p": [95.42, 94.72, 93.89, 94.86, 94.72],   # n=5
    "1q": [94.17, 93.75, 95.14, 93.19, 94.86],   # n=5
}

COMPARISONS = [
    ("1j", "1e", "MynaNet vs DS+SE+Res (best DS-CNN)"),
    ("1j", "1p", "MynaNet vs MBV3+hard-swish"),
    ("1j", "1i", "MynaNet vs MBV2+SE"),
    ("1j", "1q", "MynaNet vs EfficientNet-SE"),
    ("1e", "1c", "DS+SE+Res vs DS+SE"),
    ("1g", "1e", "DS+Att vs DS+SE+Res"),
    ("1f", "1e", "DS+SE+Res+Wide vs DS+SE+Res"),
    ("1q", "1i", "EfficientNet-SE vs MBV2+SE"),
]


def paired_permutation(a: list[float], b: list[float]) -> float:
    """Exact two-sided paired permutation test, n small."""
    diffs = [ai - bi for ai, bi in zip(a, b)]
    observed = sum(diffs)
    n = len(diffs)
    n_at_least = 0
    n_total = 0
    for signs in itertools.product([-1, 1], repeat=n):
        permuted = sum(s * d for s, d in zip(signs, diffs))
        if abs(permuted) >= abs(observed):
            n_at_least += 1
        n_total += 1
    return n_at_least / n_total


def bootstrap_ci(a: list[float], b: list[float], B: int = 10000,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean paired difference."""
    rng = random.Random(0)
    n = len(a)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    boots = []
    for _ in range(B):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistics.mean(sample))
    boots.sort()
    lo = boots[int(alpha / 2 * B)]
    hi = boots[int((1 - alpha / 2) * B)]
    return lo, hi


def main() -> None:
    out_csv = Path("significance_results.csv")
    out_md  = Path("significance_results.md")
    rows = []
    print(f"{'A':4s} {'B':4s} {'n':>3s} {'Δ̄ (pp)':>8s} {'95% CI':>18s} {'p (perm)':>10s}  description")
    print("-" * 95)
    for a, b, desc in COMPARISONS:
        va, vb = PER_SEED_INT8[a], PER_SEED_INT8[b]
        n = min(len(va), len(vb))
        va, vb = va[:n], vb[:n]
        diffs = [x - y for x, y in zip(va, vb)]
        mean_diff = statistics.mean(diffs)
        lo, hi = bootstrap_ci(va, vb)
        p = paired_permutation(va, vb)
        verdict = "ns" if (lo <= 0 <= hi) or p >= 0.05 else "*"
        rows.append([a, b, n, mean_diff, lo, hi, p, verdict, desc])
        print(f"{a:4s} {b:4s} {n:>3d} {mean_diff:>+7.2f}  ({lo:>+5.2f}, {hi:>+5.2f})  {p:>8.3f}  {verdict}  {desc}")

    with out_csv.open("w") as f:
        w = csv.writer(f)
        w.writerow(["A", "B", "n_seeds", "delta_mean_pp", "ci_lo", "ci_hi",
                    "p_permutation", "verdict", "description"])
        w.writerows(rows)
    print(f"\n✓ wrote {out_csv}")

    with out_md.open("w") as f:
        f.write("# Paired significance tests (n=3 or n=5 seeds)\n\n")
        f.write("Bootstrap 95% CI (10k resamples) + exact paired permutation test.\n")
        f.write("n=5 for comparisons where both models were re-seeded in E2 (seeds 42/100/786/7/2024); "
                "n=3 for comparisons involving 1c/1f/1g (not re-seeded).\n")
        f.write("`ns` = not significant at α=0.05 (either CI crosses 0 or p ≥ 0.05).\n\n")
        f.write("| A | B | n | Δ̄ (pp) | 95% CI | p | verdict | description |\n")
        f.write("|---|---|--:|--------:|--------|---|--------:|-------------|\n")
        for r in rows:
            a, b, n, d, lo, hi, p, v, desc = r
            f.write(f"| {a} | {b} | {n} | {d:+.2f} | ({lo:+.2f}, {hi:+.2f}) | {p:.3f} | {v} | {desc} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("Minimum attainable two-sided p: n=3 → 2/8=0.250; n=5 → 2/32=0.0625.  "
                "With n=5, a comparison where one model wins all five seeds would reach "
                "p=0.0625, approaching but not reaching the conventional α=0.05 threshold.  "
                "Bootstrap CIs are the more informative summary.  A CI that crosses zero "
                "indicates the on-paper delta is indistinguishable from noise under this "
                "seed budget; the paper should refrain from drawing conclusions from such deltas.  "
                "Models 1c, 1f, 1g were not re-seeded (n=3); their comparisons are footnoted "
                "accordingly.\n")
    print(f"✓ wrote {out_md}")


if __name__ == "__main__":
    main()
