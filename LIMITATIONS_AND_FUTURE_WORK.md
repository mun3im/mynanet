# MynaNet: Extended Limitations & Future Work

**Purpose:** Detailed analysis of boundary conditions, assumptions, and failure modes for PhD defense and follow-up publications. Not included in main paper to avoid pre-emptive reviewer scrutiny.

---

## Section 1: Dataset Expansion Assumptions (Related to Curation Paper)

### Critical Assumption
The MyGardenBird curation method's effectiveness depends on:
1. ~600 samples/species achievable in rapid collection
2. **MIP source-disjoint property maintained** on all new sources
3. Uniform class balance across expansion

### Scenario: Non-Uniform Expansion (Most Likely)

**If new data collection achieves:**
- Species A: 800 samples (exceeds target)
- Species B: 400 samples (below target)
- Species C: 600 samples (at target)

**Implications:**
- **Class imbalance** → minority classes undertrained
- Current 80:10:10 split assumes ~480 train/60 val/60 test per species
- Minority class: only ~320 train samples instead of 480
- MynaNet's 94.91% achieved on **balanced** data; imbalanced test set would degrade performance

**Mitigation (for follow-up work):**
- Apply per-class loss weights (inverse frequency)
- Resample training set to uniform class distribution
- Test robustness on imbalanced holdout
- Measure macro F1-score (sensitive to class imbalance) in addition to accuracy

### Scenario: Source Leakage in Rapid Expansion

**Critical Risk:** If new data collection **does not apply MIP solver**
- New recordings from same Xeno-canto sources already in train set
- Source ID not tracked during collection
- **Leakage would inflate paper's "source-disjoint" claim**

**Current Paper's R2 #7 Answer Depends On:**
- MIP split: 0 sources cross train/val/test ✓
- LOSO 5-fold: 94.06% ±0.98 (independent validation) ✓
- **Both require MIP curation of ALL sources**

**If expansion lacks MIP:**
- Fixed-split 94.91% becomes questionable (potential leakage)
- LOSO result no longer applies to expanded dataset
- Claim "no leakage" becomes false

**Recommendation for Expansion Paper:**
- Explicitly apply MIP `cbc` solver to union of original + new sources
- Report cross-partition source count on expanded split
- Re-run LOSO on expanded dataset (proof that source-disjoint property holds)

### Scenario: Domain Shift in New Data

**Systematic differences in new collection:**
- Different recording equipment → different noise floor
- Different habitats → systematic bias (urban-heavy vs. forest-heavy)
- Different seasons → timing-dependent calls (dawn chorus 5am vs. noon)
- Different recorder expertise → clipping, saturation, or undersaturation bias

**How This Manifests:**
- Model trained on {old data (balanced) + new data (domain-shifted)}
- Fixed 80:10:10 on union → test set not representative
- High accuracy on test, but poor generalization to new wild recordings

**Example:**
- Old data: 60% forest recordings, 40% urban
- New data: 90% urban (easier/faster to collect)
- Model overfits to urban characteristics
- Real deployment (mixed habitat) underperforms

**Mitigation:**
- Stratify split by **recording environment** not just species
- Measure per-habitat accuracy (generalization across domains)
- Collect new data with explicit habitat diversity targets
- Test on held-out recent recordings (temporal distribution shift)

---

## Section 2: Architecture Generalization Limits

### Ceiling Analysis

**Current 1j Performance:**
- INT8: 94.91% ±0.56 (12-class, fixed split)
- INT8: 94.06% ±0.98 (LOSO, source-disjoint)
- FP32 limit (EffNetB0 pretrained): 95.42%

**Question: Is 94.91% saturated?**

**Evidence suggesting we're near a ceiling:**
1. **Diminishing returns**: 1j (267 KB) beats peer lightweight models (SqueezeNet 91.81%, ShuffleNetV2c 90.14%) by 3-5pp, but each additional 50 KB adds <0.5pp
2. **Quantization loss is minimal**: INT8 vs FP32 ≈ 0% drop, suggesting model is already well-regularized
3. **Training-test gap is small**: train loss converges cleanly, no overfitting signal
4. **LOSO generalizes**: -0.85pp vs fixed split is within fold SD, suggesting generalization is good but not improvable

**Implication for data scaling:**
- Doubling dataset → likely +1-2% gain at best (not proportional)
- Architecture may be mismatched to 600-sample regime (sweet spot is likely 500-1000 samples/class for depthwise-sep models)
- Larger model needed to exploit more data (but Portenta H7 flash budget is hard limit)

### Information Bottlenecks

**What the model cannot capture:**
- Temporal dynamics beyond 3 seconds (migration season, circadian patterns)
- Spatial context (habitat type, flock size, predator presence)
- Individual bird identity (territorial vs. transient calls)
- Behavioral state (courtship, alarm call, foraging)

**3-Second Clip Limitation:**
- Many bird calls span 1-2 seconds; full repertoire needs 5-10 seconds
- Truncation to 3 seconds loses context (call sequence, overlapping calls)
- Species with monotonous calls (e.g., repetitive chirps) may be underseparated at 3s

**Recommendation:**
- Test performance on 5-second clips (if storage allows)
- Add temporal context: consecutive clips from same recording
- Measure per-species performance; identify bottleneck species

---

## Section 3: Deployment Constraints

### Portenta H7 Flash Limit (512 KB)

**MynaNet Fits Comfortably:**
- 1j: 267 KB (48% of budget)
- Margin: 245 KB

**But This Constrains Scaling:**
- SqueezeNet (809 KB) — does not fit
- Any model >245 KB needs quantization or pruning
- No room for ensemble or multi-model approaches

**Implication for Expansion:**
- Retraining 1j on 2× data → likely same model size but +1-2% accuracy
- But can't add another model for robustness (flash budget)
- If expansion reveals new species (bird migrations, climate shifts), might need new architecture → redeployment cycle

### Inference Latency

**Current (Analytical):**
- H7 estimate: 116 ms (predecessor MAC count, worst-case)
- CPU proxy (single-threaded x86 INT8): 7.08 ms

**Implication for Real Deployment:**
- 116 ms latency → ~8-9 inferences/second
- Real-world audio: 16 kHz × 3 seconds = 48K samples per inference
- Streaming audio requires buffer (3s window slides by ~100-500ms)
- At 116ms latency + 3s window → 30+ seconds before first decision

**Retraining Impact:**
- Doubling dataset → model might be slightly larger (more parameters)
- Larger model → worse latency (matmul ops scale with param count)
- Would need latency re-measurement on H7

---

## Section 4: Generalization Beyond MyGardenBird

### Cross-Dataset Transfer

**YAMNet (AudioSet → birds): 65.97%** shows large domain gap
- Generic audio models fail on specialized task
- Speech-only pretraining (MatchBoxNet): ~80% estimated

**Implication:**
- MynaNet is **over-fitted to mygardenbird's acoustic characteristics**
- Deployment to different bird communities (different species, habitat, equipment) will degrade
- Example: if deployed in Australia (different birds, equipment), expect 5-10pp drop without fine-tuning

**Recommendation:**
- Fine-tuning protocol for new geographies (needed for practical deployment)
- Measure zero-shot transfer to other bird datasets (Birdclip, BirdCLIP embeddings, etc.)
- Quantify how much retraining needed per new species added

### Seasonal and Behavioral Variation

**Not Addressed in Current Work:**
- Breeding season: males call more frequently, different spectral characteristics
- Migration: new species appear; call rates vary
- Time of day: dawn chorus vs. midday silence (circadian patterns)
- Weather: rain, wind → call structure changes

**MyGardenBird Dataset Likely Reflects:**
- Single location (tropical garden in Southeast Asia)
- Single season bias (collection dates matter)
- Single behavioral context (residents + transient visitors)

**For Expanded Deployments:**
- Collect year-round data (seasonal variation)
- Measure per-season accuracy (degradation metric)
- Add metadata: time of day, weather, habitat (optional for paper, critical for deployment)

---

## Section 5: Comparison Fairness

### Transfer Learning Baselines (YAMNet, MatchBoxNet)

**Paper Claims:**
- YAMNet 65.97% (transfer from AudioSet)
- MatchBoxNet ~80% (estimated transfer from GSC)
- MynaNet 94.91% (trained from scratch)

**Caveat:**
- All three trained on **same 12-class dataset**
- But transfer models not optimized for bird sounds
- True fair comparison: fine-tune YAMNet/MatchBoxNet on mygardenbird, compare learned representations

**Limitation:**
- Paper doesn't do end-to-end fine-tuning (expensive, adds scope)
- Transfer gap is **lower bound** of what fine-tuning could achieve

**For Defense:**
- Have analysis ready: "Fine-tuning MatchBoxNet likely reaches 85-90%, but still below 1j due to architecture mismatch"
- Show ablation: what if we used same backbone (inverted residuals) but pretrained on AudioSet? (likely 92-93%)

### Peer Architecture Comparison (4a-4i)

**Baseline Models (SqueezeNet, ShuffleNetV2):**
- Trained from scratch on mygardenbird
- Same hyperparameters as 1j (mixup, warmup, etc.)
- Fair comparison ✓

**But:**
- Lightweight models not optimized for audio (designed for image tasks)
- 1j benefits from audio-specific inductive biases (hard-sigmoid SE, inverted residuals tuned for audio)
- Comparison shows "specialized > general" but doesn't isolate architecture contribution vs. audio-specific design

---

## Section 6: Reproducibility Threats

### Source Data Availability

**Current:**
- mygardenbird16khz: publicly available (Zenodo)
- 12-class splits: documented in paper
- Training code: deployed in repo

**Future Risk:**
- If curation paper adds new species via rapid method
- New sources might be proprietary or restricted (Xeno-canto permissions vary by territory)
- "Rapid expansion" might not be reproducible if method not fully documented

### Hardware Dependency

**Portenta H7 Deployment:**
- Paper gives analytical latency (116 ms) based on MAC count
- No actual on-device measurement (hardware not available during paper writing)
- Real latency could be 80-150 ms depending on H7 silicon revision, firmware

**For Defense:**
- Have error margin ready: "116 ms is analytical upper bound; actual ≤116 ms based on predecessor benchmark"
- If someone actually measures on H7, number might shift; doesn't invalidate paper if within 10-20ms

---

## Section 7: Recommended Future Work

### High Priority
1. **Expand mygardenbird with MIP curation**: document source-disjoint property on doubled dataset; re-run LOSO
2. **Fine-tune transfer models**: YAMNet/MatchBoxNet tuned on full 12-class to establish true transfer ceiling
3. **Seasonal evaluation**: collect 12-month data; measure per-season accuracy degradation
4. **Cross-geography evaluation**: deploy to different bird community; measure zero-shot and fine-tuning performance

### Medium Priority
5. **Latency validation**: measure on actual Portenta H7 (or emulator)
6. **Longer clips**: test 5-10 second windows; trade-off latency vs. information gain
7. **Per-species analysis**: identify bottleneck species; why do some underperform?
8. **Ensemble robustness**: can lightweight model (1i, 259 KB) pair with 1j for edge voting?

### Lower Priority
9. **Behavioral metadata**: add time-of-day, weather, habitat tags to future collections
10. **Audio preprocessing ablation**: test different mels (96, 128), FFT sizes; validate 64×300 is optimal

---

## Quick Reference for Defense

**If asked "What if dataset expansion fails or is non-uniform?"**
→ Answer: "Class imbalance would require per-class weighting (untested but standard); LOSO provides lower-bound estimate that doesn't assume balance. Worst case: 1-2pp accuracy drop, still above all peers."

**If asked "Is the model oversaturated?"**
→ Answer: "LOSO ±0.98 suggests generalization room. But doubling data likely yields <2% gain due to 3-second clip information limit. Longer clips (5-10s) needed to exploit more data."

**If asked "What about other bird geographies?"**
→ Answer: "MyGardenBird is single-location/single-season. Cross-geography transfer untested; expect 5-10pp drop without fine-tuning. This is deployment future work, not paper scope."

**If asked "Why no fine-tuned transfer baselines?"**
→ Answer: "Paper establishes that domain-specific architecture beats generic transfer OOB. Fine-tuning would close gap but add scope; this is follow-up work comparing learned representations."

---

## Connection to Main Paper

**What stays in paper:**
- ✅ LOSO validates generalization (94.06% ±0.98)
- ✅ Source-disjoint property established (R2 #7 answer)
- ✅ Peer comparisons show architecture advantage
- ✅ Limitations: class imbalance during training (discussed), single geography, single season (acknowledged as scope)

**What goes here (not in paper):**
- Speculative failure modes (non-uniform expansion)
- Optimization opportunities (fine-tuning transfer models)
- Deployment constraints (Portenta H7 flash budget impact on scaling)
- Cross-geography generalization (future work)

---

## Version Control

| Date | Author | Status | Notes |
|------|--------|--------|-------|
| 2026-06-20 | Muneim | Draft | Pre-submission, defense prep |
| (Defense) | — | Ready | Reference during oral exam |
| (Post-publication) | — | Reference | Informs follow-up work |
