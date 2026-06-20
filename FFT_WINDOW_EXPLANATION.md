# FFT Windowing: Why We Use Hann Window (YAMNet Standard)

## Quick Summary

**We explicitly use `window='hann'` in our mel spectrogram computation**, following YAMNet's design. While this is librosa's default, we make it explicit for:
1. Code clarity and documentation
2. Compatibility with YAMNet preprocessing
3. Industry best practice for audio ML

---

## What is Windowing?

### The Problem: Spectral Leakage

When computing FFT on a segment of audio:

**Without windowing (rectangular window):**
```
Audio segment: [====signal====]
                ^              ^
                Abrupt edges cause discontinuities
```

The abrupt start/end creates **spectral leakage**: energy "leaks" into frequencies that aren't actually present in the signal.

**Example:**
- Real signal: Pure 1000 Hz sine wave
- Without window: FFT shows energy at 1000 Hz + many spurious frequencies
- With Hann window: FFT shows clean peak at 1000 Hz

### The Solution: Tapering Windows

Windowing multiplies the audio segment by a tapering function that smoothly goes to zero at the edges:

```
Hann window shape:
     /‾‾‾‾‾‾‾‾‾‾‾\
    /             \
   /               \
  /                 \
 0                   0

Applied to audio:
Original:  [====signal====]
Windowed:  [  =signal=  ]
              ^smooth^
```

This eliminates edge discontinuities and reduces spectral leakage.

---

## Why Hann Window Specifically?

### Common Window Types

| Window | Shape | Spectral Leakage | Frequency Resolution | Use Case |
|--------|-------|------------------|---------------------|----------|
| **Rectangular** | Flat (no taper) | High | Best | Only for continuous signals |
| **Hann** | Smooth taper | Low | Good | General audio (YAMNet, most ML) |
| **Hamming** | Similar to Hann | Very low | Good | Legacy audio processing |
| **Blackman** | Very smooth | Very low | Worse | High-precision spectral analysis |
| **Kaiser** | Adjustable | Adjustable | Adjustable | Research/specialized |

### Why YAMNet Chose Hann

From YAMNet's design decisions:

1. **Good balance**: Low spectral leakage + reasonable frequency resolution
2. **Standard in audio ML**: AudioSet, PANNs, VGGish all use Hann
3. **Smooth taper**: Goes to exactly zero at edges (unlike Hamming)
4. **Well-studied**: Extensive literature on its properties
5. **Efficient**: Simple to compute

### Mathematical Definition

**Hann window:**
```python
w[n] = 0.5 * (1 - cos(2π * n / (N-1)))    for n = 0, 1, ..., N-1
```

Where N is window length (n_fft = 512 in our case).

**Properties:**
- Starts at 0, peaks at 0.5 in middle, ends at 0
- Symmetric around center
- Smooth first and second derivatives
- Main lobe width: 8π/N (in frequency domain)

---

## YAMNet's Exact Parameters

YAMNet uses these audio preprocessing parameters:

```python
# YAMNet specification
sample_rate = 16000         # 16 kHz
window_length = 400         # 25 ms (different from ours)
hop_length = 160            # 10 ms (same as ours)
n_mels = 64                 # 64 mel bins (same as ours)
fmin = 125                  # 125 Hz (different from ours)
fmax = 7500                 # 7500 Hz (different from ours)
window = 'hann'             # Hann window (same as ours)
```

**Our parameters (slightly different):**
```python
sample_rate = 16000         # Same
window_length = 512         # 32 ms (longer than YAMNet)
hop_length = 160            # Same
n_mels = 64                 # Same
fmin = 0                    # Different (we include lowest freqs)
fmax = 8000                 # Similar (8kHz vs 7.5kHz)
window = 'hann'             # Same ✓
```

**Key similarity:** Both use **Hann window** for FFT.

---

## Impact of Window Choice

### Test: Pure 1000 Hz Sine Wave

**With Hann window:**
```
Frequency (Hz): ... 900  950  1000  1050  1100 ...
Magnitude:      ... 0.1  0.3  1.0   0.3   0.1  ...
                          ^
                      Clean peak at 1000 Hz
```

**With Rectangular window:**
```
Frequency (Hz): ... 900  950  1000  1050  1100 ...
Magnitude:      ... 0.4  0.6  1.0   0.6   0.4  ...
                    ^^^ Spectral leakage ^^^
```

**Impact on bird calls:**
- Bird calls have rapid frequency changes
- Spectral leakage would blur frequency information
- Hann window preserves clean frequency representation

---

## Code Implementation

### In Our Code (4d_tcn_mel_64x300.py)

**Before (implicit default):**
```python
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS,
    fmax=FMAX, center=True, power=2.0
    # window defaults to 'hann' but not explicit
)
```

**After (explicit, YAMNet-compatible):**
```python
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS,
    fmax=FMAX, center=True, power=2.0,
    window='hann'  # Explicit window function (YAMNet standard)
)
```

### Why Make It Explicit?

1. **Documentation**: Clear what preprocessing we use
2. **Reproducibility**: No ambiguity about defaults
3. **Compatibility**: Easy to verify YAMNet compatibility
4. **Best practice**: Explicit is better than implicit (PEP 20)

---

## Spectral Leakage Explained

### Visual Example

Imagine a bird call with a note at exactly 2000 Hz:

**Ideal (no leakage):**
```
Mel bin frequency ranges:
[1900-2000]: 0.0
[2000-2100]: 1.0  ← All energy here
[2100-2200]: 0.0
```

**With spectral leakage (rectangular window):**
```
Mel bin frequency ranges:
[1900-2000]: 0.3  ← Leaked energy
[2000-2100]: 1.0
[2100-2200]: 0.3  ← Leaked energy
```

**Impact:**
- Model sees energy in wrong frequency bins
- Harder to distinguish similar species
- Reduced classification accuracy

**With Hann window:**
- Minimal leakage (similar to ideal case)
- Clean frequency representation
- Better classification

---

## Comparison with Other Windows

### Hann vs Hamming

**Hann (our choice):**
```python
w[n] = 0.5 * (1 - cos(2π*n/(N-1)))
# Goes to exactly 0 at edges
```

**Hamming:**
```python
w[n] = 0.54 - 0.46 * cos(2π*n/(N-1))
# Goes to ~0.08 at edges (not zero)
```

**Trade-offs:**
- Hann: Better time localization (goes to zero)
- Hamming: Slightly better spectral leakage suppression
- **Industry prefers Hann** for audio ML (cleaner edges)

### Hann vs Blackman

**Blackman:**
```python
w[n] = 0.42 - 0.5*cos(2π*n/(N-1)) + 0.08*cos(4π*n/(N-1))
# Even smoother than Hann
```

**Trade-offs:**
- Blackman: Best spectral leakage suppression
- Blackman: Worst frequency resolution (wider main lobe)
- **Hann is better balanced** for audio signals

---

## Impact on Our Model

### Why It Matters for Bird Call Classification

1. **Frequency precision**: Bird species often distinguished by subtle frequency differences
   - Example: Two warblers might differ by only 200 Hz in peak frequency
   - Spectral leakage could blur this distinction

2. **Harmonic structure**: Many bird calls have harmonics
   - Clean harmonics → easier species identification
   - Leaked energy → harmonics less distinct

3. **Transient events**: Bird calls start/stop abruptly
   - Good windowing captures clean onset/offset
   - Poor windowing creates artifacts

4. **Model training**: Clean spectrograms → better learned features
   - Less noise in training data
   - More accurate learned representations

---

## Validation: Does It Match librosa Default?

```python
import librosa
import numpy as np

audio = np.random.randn(16000)

# Implicit default
mel1 = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512)

# Explicit Hann
mel2 = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512, window='hann')

# Check equality
print(np.allclose(mel1, mel2))  # True ✓
```

**Result:** Yes, librosa defaults to Hann, but we make it explicit for clarity.

---

## Alternative Windows (Not Used)

### Why Not Hamming?
- Doesn't go to exactly zero at edges
- Very similar to Hann in practice
- Less standard in modern audio ML

### Why Not Blackman?
- Too much frequency resolution loss
- Overkill for our application
- Not used by YAMNet or AudioSet

### Why Not Kaiser?
- Requires tuning β parameter
- More complex than needed
- Not standard in audio ML

### Why Not Rectangular?
- Terrible spectral leakage
- Only for continuous periodic signals
- Never use for real audio

---

## Frequency Domain Characteristics

### Hann Window in Frequency Domain

**Main lobe width:** 8π/N radians (at -3dB points)
**Side lobe level:** -31.5 dB (first side lobe)
**Side lobe fall-off:** -18 dB/octave

**What this means:**
- Main lobe: How much frequency blurring occurs
- Side lobes: How much spectral leakage occurs
- Fall-off: How quickly leakage decreases

**Hann performance:**
- Good main lobe (not too wide)
- Good side lobe suppression (-31.5 dB is excellent)
- Good fall-off rate (spectral leakage decreases quickly)

---

## Practical Example: Spectrogram Quality

### Bird Call Spectrogram Comparison

**With Rectangular Window (BAD):**
```
Frequency
  ^
8kHz |           ##
     |          #  #
4kHz |    ###  #    #  ###
     |   #   ##      ##   #
2kHz | ##                 ##
     +-----|-----|-----|-----> Time
          0.5s  1.0s  1.5s

Notice: Fuzzy boundaries, leaked energy
```

**With Hann Window (GOOD):**
```
Frequency
  ^
8kHz |           ##
     |          #  #
4kHz |      ##  #    #  ##
     |      # #      # #
2kHz |      #          #
     +-----|-----|-----|-----> Time
          0.5s  1.0s  1.5s

Notice: Clean boundaries, concentrated energy
```

---

## YAMNet Compatibility

### What We Match
✓ Window function: Hann
✓ Sample rate: 16 kHz
✓ Hop length: 160 samples (10 ms)
✓ Mel bins: 64

### What We Don't Match (Minor Differences)
- Window length: 512 vs 400 samples
  - Ours: 32 ms (better frequency resolution)
  - YAMNet: 25 ms (better time resolution)
  - Both valid choices

- Frequency range:
  - Ours: 0-8000 Hz
  - YAMNet: 125-7500 Hz
  - Trade-off: We capture lower frequencies (some bird calls use them)

**Bottom line:** Core windowing approach matches YAMNet standard.

---

## Implementation Details

### Where Window is Applied

```python
# In librosa.feature.melspectrogram internals:
1. Apply Hann window to each frame
2. Compute FFT of windowed frame
3. Convert to mel scale
4. Average energy in each mel bin
```

### Computational Cost

**Hann window overhead:** Negligible
- Pre-computed once: `w = hann(512)`
- Per-frame multiplication: O(N) = O(512)
- Total overhead: < 1% of FFT cost

**Verdict:** Free lunch - huge quality improvement, minimal cost.

---

## References & Standards

### Papers Using Hann Window
- **YAMNet** (Google, 2020): Hann window, 25ms
- **AudioSet** (Google, 2017): Hann window
- **PANNs** (Kong et al., 2020): Hann window
- **VGGish** (Google, 2017): Similar to Hann
- **BirdNET** (Cornell, 2021): Hann window

### Why It's Standard
1. Good balance of properties
2. Well-documented behavior
3. Reproducible results across platforms
4. Easy to implement (librosa default)

---

## Cache Impact

### Changed Parameters
Adding explicit `window='hann'` changes cache hash:

**Before:**
```python
cache_key = {
    'n_fft': 512,
    # ... other params
    'center': True
}
```

**After:**
```python
cache_key = {
    'n_fft': 512,
    # ... other params
    'center': True,
    'window': 'hann'  # New
}
```

**Impact:** Cache invalidated (will regenerate)
**Why:** Ensures cache reflects exact preprocessing

**Note:** Actual preprocessing doesn't change (Hann was already default), only documentation/cache validation improves.

---

## Testing Window Effect

### Quick Test Script
```python
import librosa
import numpy as np
import matplotlib.pyplot as plt

# Generate test signal: 2kHz sine wave
sr = 16000
t = np.linspace(0, 0.1, int(sr * 0.1))
signal = np.sin(2 * np.pi * 2000 * t)

# Compare windows
windows = ['hann', 'hamming', 'blackman', None]  # None = rectangular

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, win in enumerate(windows):
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=512,
        hop_length=160, window=win
    )
    mel_db = librosa.power_to_db(mel)

    ax = axes[i // 2, i % 2]
    librosa.display.specshow(mel_db, ax=ax)
    ax.set_title(f'Window: {win or "Rectangular"}')

plt.tight_layout()
plt.savefig('window_comparison.png')
```

---

## Recommendations

### For Current Project (Seabird)
✓ **Use Hann window** (as implemented)

Reasons:
- Matches YAMNet standard
- Good for transient bird calls
- Industry best practice
- Clean frequency representation

### When to Use Different Windows

**Hamming:**
- Legacy compatibility
- Marginally better side lobe suppression
- (We don't need this)

**Blackman:**
- Very high-precision spectral analysis
- When frequency resolution less critical
- (Overkill for our use)

**Kaiser:**
- Research on window trade-offs
- Custom applications
- (Too complex for production)

**Rectangular:**
- Never for real audio
- Only for perfectly periodic synthetic signals
- (Don't use)

---

## Summary

### What We Did
✓ Made `window='hann'` explicit in code
✓ Added to cache hash validation
✓ Documented in training report
✓ Explained why in code comments

### Why It Matters
✓ Reduces spectral leakage (cleaner frequency info)
✓ Matches YAMNet preprocessing
✓ Industry best practice
✓ Better classification accuracy

### Impact
✓ Code clarity: Explicit > implicit
✓ Cache: Invalidated (will regenerate with documentation)
✓ Results: Unchanged (was already default)
✓ Compatibility: Better YAMNet alignment

---

## Conclusion

**The Hann window is the right choice** for our seabird classification task because:

1. **Reduces spectral leakage**: Clean frequency representation
2. **YAMNet standard**: Matches proven audio ML architecture
3. **Industry consensus**: Most audio ML uses Hann
4. **Good for transients**: Bird calls are transient signals
5. **Balanced trade-offs**: Good frequency + time resolution

By making it explicit in our code, we improve documentation and ensure future maintainers understand our preprocessing choices.

**No changes to actual processing** - librosa already defaulted to Hann. We just made it clear and documented.
