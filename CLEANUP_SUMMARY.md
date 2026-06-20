# Directory Cleanup Summary (2026-06-20)

## Objective
Clean up sibling directories (`../tcn/` and `../mobilenet-inspired/`) by removing archival/experimental scripts while preserving all MynaNet-relevant models used in paper and benchmarking.

## Changes

### TCN Directory (`../tcn/`)

**Kept: 8 scripts** (Series 2 core ablations)
- `2a_tcn_baseline.py` — baseline TCN, 7 dilated blocks
- `2b_tcn_shallow.py` — reduced depth variant
- `2c_tcn_no_residual.py` — ablation: remove residual connections
- `2d_tcn_lightweight.py` — reduced channel width
- `2e_tcn_wide.py` — increased channel width
- `2f_tcn_deep.py` — increased depth
- `2g_tcn_kernel2.py` — kernel size=2 ablation
- `2h_tcn_kernel5.py` — kernel size=5 ablation

**Removed: 15 scripts** (archival/experimental)
- `2i_tcn_specaugment.py` — archival augmentation test
- `2j_tcn_combined.py` — combined variants
- `2k_tcn_se.py` — SE block experiment
- `2l_pure1d_tcn.py`, `2l_tcn_multiscale.py`, `2l_tcn_residual.py` — multiple 2l versions
- `2m_tcn1d_raw.py`, `2m_tcn_optimized.py` — multiple 2m versions
- `2n_tcn_mild_augmentation.py` — augmentation variant
- `2o_tcn_distillation.py` — distillation experiment
- `2p_tcn_advanced_distillation.py` — advanced distillation
- `2q_tcn_optimal.py`, `2r_tcn_ultra.py`, `2s_tcn_optimized_v2.py` — optimized variants
- `2t_tcn1d_classifier_dual_stage.py` — dual-stage classifier

**Rationale:** Keep only the systematic 2a-2h ablation series documented in CLAUDE.md and used for paper results. Remove redundant numbered versions and experimental variants.

### Mobilenet-Inspired Directory (`../mobilenet-inspired/`)

**Kept in development directory: 7 scripts** (Series 3 — actual development lineage)
- `3a_mobilenetv3_pretrained_224x224.py` — ImageNet pretrained (transfer baseline)
- `3b_mobilenetv3_64x300.py` — native 64×300 input
- `3c_mobilenetv3_48x300.py` — 48-mel variant
- `3d_mobilenetv3_80x300.py` — 80-mel variant
- `3e_mobilenetv3_width075_64x300.py` — width×0.75 (Series 3 core)
- `3f_mobilenetv3_width075_48x300.py` — width×0.75 at 48-mel
- `3g_mobilenetv3_optimized.py` — optimized variant

**Moved to `./benchmarks/`: 8 scripts** (peer architectures for comparison)
- `3a_transformer_encoder.py` — pure Transformer baseline
- `4a_squeezenet_v11.py` — SqueezeNet v1.1
- `4b_shufflenetv2_v11.py` — ShuffleNetV2 v1.1
- `4c_shufflenetv2_compact.py` — ShuffleNetV2 Compact
- `4f_matchboxnet_64x300.py` — MatchBoxNet variant A
- `4g_matchboxnet_regularized_64x300.py` — MatchBoxNet variant B
- `4h_matchboxnet_wider_64x300.py` — MatchBoxNet variant C
- `4i_transformer_encoder.py` — Transformer (alternative implementation)

**Removed: 5 scripts** (older variants, not in paper)
- `3h_mobilenetv1_narrow_64x300.py` — deprecated MobileNetV1
- `3i_mobilenetv1_regularized_64x300.py` — deprecated MobileNetV1
- `3j_mobilenetv1_narrow_regularized_v2.py` — deprecated MobileNetV1
- `3k_mobilenetv1_width035.py` — deprecated MobileNetV1
- `3l_mobilenetv2_narrow_64x300.py` — deprecated MobileNetV2

**Rationale:** Series 3 (3a-3g) represents the actual development path explored during MynaNet research. Peer architectures (Series 4 + Transformers) are purely for benchmarking/comparison and belong in `./benchmarks/` for a cohesive validation suite.

## Impact

- **Removed:** 20 archival/experimental scripts (~1.3 MB)
- **Reorganized:** 8 peer architecture scripts moved to `./benchmarks/` for cohesion
- **Development lineage preserved:** Series 3 (3a-3g) remains in `../mobilenet-inspired/`
- **Benchmarking suite complete:** All peer models (4a-4i) + Transformers now in `./benchmarks/`
- **Directory structure clarified:**
  - `../mobilenet-inspired/` = development path (Series 3 MobileNetV3 evolution)
  - `../tcn/` = development path (Series 2 TCN ablations)
  - `./benchmarks/` = validation/comparison (YAMNet, MatchBoxNet, peers, analysis)
- **No impact on paper results:** All kept scripts correspond to experiments in CLAUDE.md
- **Total active scripts:** 30 MynaNet-relevant training + benchmarking scripts

## Cross-Reference
- TCN results: CLAUDE.md § "SERIES 2" (2a-2h ablations)
- Mobilenet results: CLAUDE.md § "SERIES 3" (3a-3g MobileNetV3), benchmarks/README.md (4a-4i peers)
- Benchmarking: benchmarks/README.md results table
