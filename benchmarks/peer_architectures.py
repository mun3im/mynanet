"""
peer_architectures.py — drop-in architecture definitions for MyGardenBird
peer benchmarks (Tier-A, paper Table 8).

USAGE
=====
Each function builds a Keras model with the same I/O signature as the
existing 4a_squeezenet_v11.py / 4b_shufflenetv2_v11.py models:

    model = build_<peer>(num_classes=12, input_shape=(64, 300, 1), dropout=0.05)

The intended workflow is:

  1.  cp 4a_squeezenet_v11.py 5b_dscnn_l_helloedge.py   (one copy per peer)
  2.  In the copy, replace the `create_squeezenet_v11_64x300(...)` call in
      main() with the relevant build_* function from this module.
  3.  Adjust:
          model_id           (e.g. "5b_dscnn_l_helloedge")
          output_dir_name    (prefix used by run_peer_benchmarks.sh)
          *_int8.tflite      output filename
  4.  Run the canonical CLI:
          python3 5b_dscnn_l_helloedge.py \
              --splits_csv /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv \
              --flat_dir   /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz \
              --dropout 0.05 --warmup_epochs 70 --mixup 0.2 --random_seed 42

Rationale: every 4*.py script duplicates ~1000 lines of pipeline code.
Rather than introduce a 5th copy per peer, we keep the pipeline copies
local but factor only the architecture out, so the four Tier-A peers
can share one source of truth for their models.

Numbers below are author-reported on each peer's native task and are
included only as a sanity check on the implementation, NOT as
ground-truth for MyGardenBird.

REFERENCES
==========
DS-CNN-L     : Zhang et al., "Hello Edge: Keyword Spotting on MCUs",
               arXiv:1711.07128, 2017.
BC-ResNet    : Kim et al., "Broadcasted Residual Learning for Efficient
               Keyword Spotting", Interspeech 2021. arXiv:2106.04140.
TC-ResNet    : Choi et al., "Temporal Convolution for Real-Time Keyword
               Spotting on Mobile Devices", Interspeech 2019.
MatchboxNet  : Majumdar & Ginsburg, Interspeech 2020. arXiv:2004.08531.
"""

from __future__ import annotations

import tensorflow as tf

# IMPORTANT: the training pipeline (5*.py) imports `tf_keras` rather than the
# `tensorflow.keras` module that ships embedded in TF 2.15.  These are two
# distinct Keras runtimes that cannot interoperate — an optimiser built by
# `tf_keras` will not compile against a model built by `tensorflow.keras`
# (ValueError: Could not interpret optimizer identifier).  We therefore build
# every peer architecture against `tf_keras` so it matches the rest of the
# pipeline exactly.
import tf_keras
from tf_keras import layers, regularizers
from tf_keras.models import Model

# ---------------------------------------------------------------------------
# 5a — MatchboxNet 3×2×64
# ---------------------------------------------------------------------------
def build_matchboxnet_3x2x64(num_classes: int = 12,
                             input_shape=(64, 300, 1),
                             dropout: float = 0.05) -> Model:
    """MatchboxNet B=3, R=2, C=64 (≈93k params, 1D time-channel-separable).

    Input is the (mel, time, 1) spectrogram; we treat the mel axis as the
    channel dimension by squeezing the trailing channel and rearranging.
    """
    inp = layers.Input(shape=input_shape, name="input")
    # (mel=64, time=300, 1) -> (time=300, channels=64)
    x = layers.Permute((2, 1, 3))(inp)
    x = layers.Reshape((input_shape[1], input_shape[0]))(x)

    # Prologue
    x = layers.SeparableConv1D(128, 11, strides=2, padding="same",
                               depthwise_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    # 3 blocks × 2 sub-blocks, growing kernel size
    for block_kernel in (13, 15, 17):
        residual = layers.Conv1D(64, 1, padding="same")(x)
        residual = layers.BatchNormalization()(residual)
        for sub in range(2):
            x = layers.SeparableConv1D(64, block_kernel, padding="same",
                                       depthwise_regularizer=regularizers.l2(1e-3))(x)
            x = layers.BatchNormalization()(x)
            if sub == 1:
                x = layers.Add()([x, residual])
            x = layers.ReLU()(x); x = layers.Dropout(dropout)(x)

    # Epilogue
    x = layers.SeparableConv1D(128, 29, padding="same",
                               depthwise_regularizer=regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.Conv1D(128, 1, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="MatchboxNet_3x2x64")


# ---------------------------------------------------------------------------
# 5b — DS-CNN-L (Hello Edge, Zhang 2017)
# ---------------------------------------------------------------------------
def build_dscnn_l_helloedge(num_classes: int = 12,
                            input_shape=(64, 300, 1),
                            dropout: float = 0.05) -> Model:
    """DS-CNN-L from Hello Edge (Zhang et al. 2017).

    Five depthwise-separable blocks with 276 channels. Original was tuned
    for 49×10 MFCC; here we use 64×300 log-mel. Use strides to bring the
    spatial dims down to the original receptive-field-per-param ratio.
    """
    def ds_block(x, filters, stride):
        x = layers.DepthwiseConv2D(3, strides=stride, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        return x

    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Conv2D(276, (10, 4), strides=(2, 2), padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    for i, stride in enumerate([(1, 1), (1, 1), (1, 1), (1, 1)]):
        x = ds_block(x, 276, stride=stride)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="DSCNN_L_HelloEdge")


# ---------------------------------------------------------------------------
# 5c — BC-ResNet-8 (Kim 2021)
# ---------------------------------------------------------------------------
def _bc_block(x, channels, dilation=1, stride=1, dropout=0.05):
    """Broadcasted Residual block (Kim 2021).

    Applies a frequency-DWS then broadcasts a time-DWS over frequency.
    Implementation follows Eq.(3) in the paper.

    NOTE: TF/Metal requires equal strides in (H,W) and dilation_rate=1 along
    one axis to be expressed as a 1-element kernel.  Downsampling is therefore
    delegated to a MaxPool2D after the depthwise convs.
    """
    # frequency branch: depthwise on the mel axis (stride=1; downsample later)
    y = layers.DepthwiseConv2D((3, 1), padding="same", strides=1)(x)
    y = layers.BatchNormalization()(y)
    # bring y to the target channel count so broadcast addition is well-defined
    y = layers.Conv2D(channels, 1, padding="same")(y)

    # collapse mel to broadcast over frequency (avg over the frequency axis)
    z = layers.Lambda(
        lambda t_: tf.reduce_mean(t_, axis=1, keepdims=True))(y)

    # time branch (1D): dilated depthwise on the time axis.
    # dilation_rate must be equal across spatial dims for Metal; emulate the
    # "dilate along time only" pattern by using a 1×K kernel with uniform
    # dilation_rate=dilation (the H-side has size 1 so dilation along H is a
    # no-op).
    z = layers.DepthwiseConv2D((1, 3), padding="same",
                               dilation_rate=dilation)(z)
    z = layers.BatchNormalization()(z); z = layers.Activation("swish")(z)
    z = layers.Conv2D(channels, 1, padding="same")(z)
    z = layers.BatchNormalization()(z)

    # broadcast (1, T, C) back over the frequency axis to match y's (F, T, C)
    freq_size = y.shape[1]
    z = layers.Lambda(
        lambda t_: tf.tile(t_, [1, freq_size, 1, 1]))(z)

    out = layers.Add()([y, z])
    out = layers.Activation("swish")(out)
    out = layers.Dropout(dropout)(out)

    # Downsample by stride if requested (frequency axis only, like Kim 2021).
    if stride > 1:
        out = layers.MaxPool2D(pool_size=(stride, 1))(out)
    return out


def build_bcresnet8(num_classes: int = 12,
                   input_shape=(64, 300, 1),
                   dropout: float = 0.05,
                   width_mult: float = 8.0) -> Model:
    """BC-ResNet-N: 8 broadcasted residual blocks (Kim et al. 2021).

    width_mult selects the variant:
        1.0 → BC-ResNet-1 (≈9.2 k params)
        2.0 → BC-ResNet-2
        3.0 → BC-ResNet-3
        8.0 → BC-ResNet-8 (≈317 k params)
    Channel widths from Kim 2021 Table 1: base × {1, 1.5, 2.5, 3.5}.
    """
    base = max(1, int(round(8 * width_mult)))
    widths = (base,
              int(round(base * 1.5)),
              int(round(base * 2.5)),
              int(round(base * 3.5)))

    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Conv2D(16 * max(1, int(round(width_mult))), 5,
                      strides=(2, 1), padding="same")(inp)
    x = layers.BatchNormalization()(x); x = layers.Activation("swish")(x)

    config = [
        (widths[0], 1, 1), (widths[0], 1, 1),
        (widths[1], 2, 2), (widths[1], 2, 1),
        (widths[2], 4, 2), (widths[2], 4, 1),
        (widths[3], 8, 1), (widths[3], 8, 1),
    ]
    for ch, dil, stride in config:
        x = _bc_block(x, channels=ch, dilation=dil, stride=stride,
                      dropout=dropout)

    x = layers.Conv2D(widths[3] * 2, 1, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("swish")(x)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="BC_ResNet_8")


# ---------------------------------------------------------------------------
# 5d — TC-ResNet-14-1.5 (Choi 2019)
# ---------------------------------------------------------------------------
def build_tcresnet14_15(num_classes: int = 12,
                       input_shape=(64, 300, 1),
                       dropout: float = 0.05,
                       width_mult: float = 1.5) -> Model:
    """TC-ResNet-14: 1D temporal CNN treating mel bins as channels."""
    inp = layers.Input(shape=input_shape, name="input")
    x = layers.Permute((2, 1, 3))(inp)
    x = layers.Reshape((input_shape[1], input_shape[0]))(x)   # (time, mel) as (T, C)

    base_filters = [16, 24, 32, 48]
    base_filters = [int(f * width_mult) for f in base_filters]

    x = layers.Conv1D(base_filters[0], 3, padding="same")(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    def res_block(x, filters, stride):
        shortcut = x
        x = layers.Conv1D(filters, 9, strides=stride, padding="same")(x)
        x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
        x = layers.Conv1D(filters, 9, padding="same")(x)
        x = layers.BatchNormalization()(x)
        if stride != 1 or shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, strides=stride, padding="same")(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)
        x = layers.Add()([x, shortcut]); x = layers.ReLU()(x)
        return x

    # 3 stages × 2 blocks (= 6 blocks; with initial conv and head ≈ 14 layers)
    for i, filters in enumerate(base_filters[1:], start=1):
        for j in range(2):
            stride = 2 if j == 0 else 1
            x = res_block(x, filters, stride=stride)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="TC_ResNet_14_1.5")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name, fn in [
        ("MatchboxNet 3x2x64", build_matchboxnet_3x2x64),
        ("DS-CNN-L HelloEdge", build_dscnn_l_helloedge),
        ("BC-ResNet-8",         build_bcresnet8),
        ("TC-ResNet-14 1.5x",  build_tcresnet14_15),
    ]:
        m = fn()
        params = m.count_params()
        size_kb = params * 1 / 1024  # INT8 byte/param
        print(f"{name:24s}  params={params:>8,d}  est. INT8 ≈ {size_kb:5.1f} KB")
