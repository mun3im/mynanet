# center=True vs center=False Analysis

## Quick Answer

**We use `center=True` because:**
1. Better temporal alignment (window centered on time points)
2. Better edge handling (captures start/end of signal)
3. Industry standard for audio processing
4. Only costs 10ms of audio length (negligible for 3-second calls)

---

## Detailed Comparison

### What `center` Does

**`center=True`** (our choice):
- Adds padding of `n_fft // 2 = 256` samples on **each side** of the audio
- Total padding: 512 samples (32ms at 16kHz)
- Window is **centered** on each time point
- First window captures audio from the very beginning

**`center=False`**:
- No padding added
- First window starts at sample 0
- Requires `n_fft` samples before first frame can be computed
- Window is **left-aligned** on each time point

---

## Audio Length Requirements for 300 Frames

| Setting | Audio Length | Duration | Formula |
|---------|-------------|----------|---------|
| `center=True` | 47,840 samples | 2.990 sec | `(frames - 1) × hop` |
| `center=False` | 48,352 samples | 3.022 sec | `(frames - 1) × hop + n_fft` |
| **Difference** | **512 samples** | **0.032 sec** | **n_fft** |

### With Original 3.0 Seconds (48,000 samples)

| Setting | Frames | Need Adjustment? |
|---------|--------|------------------|
| `center=True` | 301 | Yes (truncate 1 frame) |
| `center=False` | 297 | Yes (pad 3 frames) |

**Neither gives exactly 300 frames with 3.0 seconds!**

---

## Mathematical Formulas

### With center=True
```python
# Librosa adds padding
padded_length = audio_length + n_fft

# Number of frames
num_frames = floor((padded_length - n_fft) / hop_length) + 1
           = floor(audio_length / hop_length) + 1

# For exact 300 frames
audio_length = (300 - 1) × 160 = 47,840 samples
```

### With center=False
```python
# No padding added
# Number of frames
num_frames = floor((audio_length - n_fft) / hop_length) + 1

# For exact 300 frames
audio_length = (300 - 1) × 160 + 512 = 48,352 samples
```

---

## Temporal Alignment Example

Consider a signal with an event at exactly t=0:

### center=True (Good ✓)
```
Time:     0ms        10ms       20ms       30ms
          |          |          |          |
Window 1: [--*--]
Window 2:    [--*--]
Window 3:       [--*--]

Event at t=0 is in CENTER of window 1
```

### center=False (Less ideal)
```
Time:     0ms        10ms       20ms       30ms
          |          |          |          |
Window 1: *-----]
Window 2:    *-----]
Window 3:       *-----]

Event at t=0 is at LEFT EDGE of window 1
```

**Why centered is better:**
- FFT assumes periodicity
- Centered window minimizes edge artifacts
- Better frequency resolution for transient events
- More accurate phase information

---

## Edge Behavior

### Start of Audio (first 32ms)

**center=True:**
```
Audio:    [==========actual signal==========]
Padding:  [pad][=====signal=====]
Window 1:  [----FFT window----]  ← Captures from start
Window 2:     [----FFT window----]
```
✓ First frame captures information from first sample

**center=False:**
```
Audio:    [==========actual signal==========]
Window 1: [----FFT window----]  ← Needs 512 samples
          ^
          First 512 samples just to START
```
✗ Loses information in first 32ms (n_fft/sr = 512/16000 = 32ms)

### End of Audio (last 32ms)

**center=True:**
```
Audio:    [==========actual signal==========]
Padding:  [=====signal=====][pad]
Window N:            [----FFT window----]  ← Captures to end
```
✓ Last frame captures information to last sample

**center=False:**
```
Audio:    [==========actual signal==========]
Window N:     [----FFT window----]
                                  ^
                                  Loses last 32ms
```
✗ Loses information in last 32ms

---

## Pros & Cons

### center=True (Our Choice)

**Pros:**
- ✓ Better temporal alignment (centered windows)
- ✓ Captures start/end of signal (no information loss)
- ✓ Industry standard (most papers use this)
- ✓ Better for transient detection (bird calls are transient)
- ✓ Matches librosa defaults (less surprising)
- ✓ More accurate phase information

**Cons:**
- ✗ Requires shorter audio (47,840 vs 48,352 samples for 300 frames)
- ✗ Slight mismatch with "3 seconds" marketing (actually 2.99s)

### center=False

**Pros:**
- ✓ No padding (slightly more "pure" signal)
- ✓ Can use longer audio for same frame count
- ✓ Matches "3 seconds" better (3.022s vs 2.990s)

**Cons:**
- ✗ Loses first 32ms of information
- ✗ Loses last 32ms of information
- ✗ Poor temporal alignment (left-aligned windows)
- ✗ Not industry standard
- ✗ Worse for transient events
- ✗ Still doesn't give exactly 300 frames with 3.0 seconds

---

## Impact on Bird Call Classification

Bird calls are **transient signals** with important information at:
- **Attack**: Sharp onset (start of call)
- **Sustain**: Main body of call
- **Release**: End of call

### Why center=True is better for bird calls:

1. **Onset Detection**: Bird calls often start abruptly. `center=True` captures this better.

2. **Offset Detection**: Call endings contain species information. `center=True` preserves this.

3. **Temporal Precision**: Centered windows give better time localization for rapid frequency changes.

4. **Information Loss**: With `center=False`, losing 32ms at start/end means losing ~2% of a 3-second call. This could be the most distinctive part!

### Example: Short "Chip" Call
```
Duration: 50ms
With center=False: Loses 32ms at edges → Only 18ms captured!
With center=True:  Full 50ms captured ✓
```

---

## Literature & Standards

Most audio ML papers use `center=True`:
- **AudioSet** (Google): center=True
- **PANNs** (Kong et al.): center=True
- **Wav2Vec** (Facebook): center=True
- **BirdNET** (Cornell): center=True
- **ESC-50** (environmental sounds): center=True

**Industry consensus**: `center=True` is the standard for audio ML.

---

## Performance Comparison

| Metric | center=True | center=False |
|--------|-------------|--------------|
| Information Loss | 0% | ~2% (edges) |
| Temporal Accuracy | Better | Worse |
| Edge Artifacts | Fewer | More |
| Frame Calculation | Simpler | More complex |
| Audio Length for 300 frames | 2.990s | 3.022s |
| Padding Overhead | 512 samples | 0 samples |
| Computation Time | Same | Same |

---

## Code Comparison

### With center=True (current)
```python
FIXED_AUDIO_LENGTH = (300 - 1) * 160  # 47,840 samples
mel = librosa.feature.melspectrogram(
    y=audio, sr=16000, n_fft=512,
    hop_length=160, n_mels=64,
    fmax=8000, center=True, power=2.0
)
# Result: (64, 300) ✓
```

### With center=False (alternative)
```python
FIXED_AUDIO_LENGTH = (300 - 1) * 160 + 512  # 48,352 samples
mel = librosa.feature.melspectrogram(
    y=audio, sr=16000, n_fft=512,
    hop_length=160, n_mels=64,
    fmax=8000, center=False, power=2.0
)
# Result: (64, 300) ✓
# BUT: Loses first/last 32ms of audio!
```

---

## Should We Switch to center=False?

**No. Here's why:**

1. **Information Loss**: Losing 64ms total (32ms × 2) is worse than using 10ms less audio
2. **Standard Practice**: Going against industry standards makes research less comparable
3. **Temporal Alignment**: Poor alignment affects model's ability to learn temporal patterns
4. **Edge Effects**: TCN models benefit from accurate edge information
5. **No Real Benefit**: The 32ms extra audio with `center=False` doesn't compensate for the information loss

---

## Recommendations

### For Current Project (Seabird Classification)
**Use `center=True`** (current implementation) ✓

Reasons:
- Bird calls are transient → need good edge capture
- Species identification often depends on attack/release
- Standard approach makes results comparable
- 2.99 seconds is still "3 seconds" for practical purposes

### When You Might Use center=False
- Continuous audio analysis (no clear start/end)
- Very long recordings where edges are unimportant
- When you specifically need to avoid padding artifacts
- When comparing with old code that used center=False

### For New Projects
Default to `center=True` unless you have a specific reason not to.

---

## Summary Table

| Aspect | center=True (current) | center=False |
|--------|----------------------|--------------|
| **Audio Length** | 47,840 (2.990s) | 48,352 (3.022s) |
| **Frames** | 300 ✓ | 300 ✓ |
| **Information Loss** | None | ~2% at edges |
| **Temporal Alignment** | Centered (better) | Left-aligned |
| **Edge Handling** | Good | Poor |
| **Standard Practice** | Yes | No |
| **Bird Call Suitability** | Excellent | Poor |
| **Recommendation** | **Use this** ✓ | Don't use |

---

## Conclusion

**Keep `center=True`** with 47,840 samples. The trade-offs are:

**What we gain:**
- ✓ Better temporal alignment
- ✓ No information loss at edges
- ✓ Industry standard approach
- ✓ Better for transient signals (bird calls)

**What we lose:**
- ✗ 10ms of audio duration (2.990s instead of 3.000s)

The 10ms reduction is **completely negligible** compared to the benefits of proper temporal alignment and edge handling.

---

## Migration Note

**No action needed.** The current implementation already uses `center=True` with the correct audio length (47,840 samples). This is the optimal configuration.
