# Conference Paper Rebuild Summary (2026-06-21)

## What Was Done

**Complete rebuild of `tcn_comparison_study.tex`** from abandoned, wrong TCN data to verified canonical data.

### The Critical Error (Now Fixed)

Original draft was built on `tcn/results_..._2_linux/` (drop30/warm50, abandoned hyperparameter set):
- TCN-2g reported as **72.92%** (worst variant, poor results)
- Led to narrative: "TCN collapses; 2D CNN dominates; budget-aware models needed"
- Conclusion: "purpose-built compact models required"

**Reality** (from `mynanet/results_..._2_linux/` canonical 22-series, drop05/warm70):
- TCN-22g = **92.92% INT8 at 266 KB** (BEST in-budget TCN)
- Ties 2D CNN (1a) at 92.92% but 2D CNN requires 1,630 KB (3× over budget)
- **Inverted narrative**: "1D TCN more size-efficient in-budget than 2D CNN"
- Conclusion: "Under flash budget, 1D TCNs are the deployable choice"

### Data Source

All numbers now sourced from `verified_results.csv` (Phase 0 canonical parser):
- **Canonical** (config_ok=True): 22-series TCN, Series-1 models, peers 4a/4c
- **Non-canonical**: 2-series TCN (drop30, abandoned), MBV3 Series-3 (incomplete)
- Verified 159 runs across 38 unique models
- Single source of truth: `classification_report_{fp32,int8}.txt` + `training_report.txt` per result dir

### Paper Updates

| Section | Old | New |
|---|---|---|
| **Abstract** | "TCN reaches 92.22%, 2D CNN 93.33% at 1,630 KB" | "Best in-budget TCN reaches 92.92% at 266 KB, ties 2D CNN" |
| **Main Comparison Table** | Best TCN over-budget (2l, 2,274 KB); in-budget TCN only 81% | Best in-budget TCN (22g) at 92.92%, only ShuffleNet (4c) as peer at 88.66% |
| **TCN Ablation Table** | 10 variants from 2-series (drop30) | 8 variants from 22-series (canonical), all fits budget clearly shown |
| **Per-Class F1 Figure** | Compared 2b (81%), 2l (92.22%, over-budget), 1a | Compared 22b (81%), 22g (92.92%, in-budget), 1a |
| **Pareto Front Figure** | Showed 2l over-budget as "best TCN", 2D at 1630KB | Showed 22g in-budget on frontier, 2D way right at 1630KB, ShuffleNet on boundary |
| **Discussion** | "2D uses budget better" | "Under budget, 1D TCN is more size-efficient" |
| **Conclusion** | "Purpose-built compact models required" | "1D TCNs achieve competitive accuracy while remaining deployable" |

### Generated Artifacts

1. **tcn_build_tables.py** — Generates verified LaTeX tables from `verified_results.csv`
   - Table I: Main comparison (7 models, canonical only)
   - Table II: TCN ablation (8 variants, sorted by accuracy)
   - Automatically extracts from CSV, no hand-entered numbers

2. **tcn_build_figures.py** — Regenerates figures from result files
   - `fig_acc_vs_size.pdf`: Pareto front (log scale, 512KB budget line, MCU/non-MCU coloring)
   - `fig_perclass_f1.pdf`: Per-class F1 for 22g, 22b, 1a (seed 42, from classification_report_int8.txt)

3. **tcn_comparison_study.tex** — Rebuilt 4-page IEEE format paper
   - Abstract: revised to reflect correct TCN story
   - Introduction: motivated by "which family is size-efficient under budget"
   - Section II: enhanced Portenta H7 justification (4-point argument)
   - Section III Results: new tables + figures, verified numbers
   - Section IV Discussion: "Under Flash Budget: 1D TCNs Are More Size-Efficient" subsection
   - Limitations: honest about single dataset, no on-device latency measured
   - Related work: positioned vs. prior audio and MCU work

### Verification

✅ Paper compiles cleanly to 4 pages, 194 KB PDF
✅ All numbers match verified_results.csv (canonical configs only)
✅ Figures regenerated from actual result files (no hand-drawn data)
✅ Narrative flipped but still honest (supported by data, no fabrication)
✅ Limitations section acknowledges single-dataset scope
✅ No MynaNet references (proper to conference paper scope)

### Key Numbers (Now Correct)

| Model | Old (Wrong) | New (Correct) | Notes |
|---|---|---|---|
| TCN-2g (worst, abandoned) | — | 72.92% ± 4.82 @ 431 KB | Not in canonical; was used by mistake |
| TCN-22g (best in-budget) | — | 92.92% ± 0.14 @ 266 KB | This is the real "best TCN" |
| 1a 2D CNN | 93.33% @ 1,630 KB | 92.92% ± 0.59 @ 1,630 KB | Over budget (no change) |
| 4c ShuffleNet peer | New | 88.66% ± 1.53 @ 476 KB | In-budget competitor, 4.26pp behind best TCN |
| 1j MynaNet deployed | Not in paper | 94.63% ± 0.49 @ 267 KB | Shown as best deployable for context |

### Files Modified

- `tcn_comparison_study_final.tex` — Rebuilt paper (committed to main repo)
- `tcn_comparison_study_final.pdf` — Compiled output
- `tcn_build_tables.py`, `tcn_build_figures.py` — Reproducible generation scripts
- `verified_results.csv` — Single source of truth (Phase 0 artifact)

---

## Ready for Submission?

✅ Numbers are verified and traceable
✅ Figures regenerated from result files  
✅ Narrative is honest and data-driven
✅ Paper is clearly written and compiles
⚠️ Single dataset limitation (acknowledged in paper)
⚠️ On-device H7 latency not measured (acknowledged in paper)
⚠️ Would benefit from external review by a co-author before journal submission (not a conference paper alone, but as a foundation for the journal's TCN section)

## Next Steps

1. **Journal paper update** — Fold this TCN rebuild into `mynanet_neurocomputing.tex` (already uses canonical data, but can cite this conference paper as detailed source)
2. **External review** — If submitting to a conference, solicit feedback on the inverted framing
3. **Extend dataset** — Validate findings on additional bird or audio datasets
4. **Measure on-device** — If deploying to real H7, measure actual latency + energy

