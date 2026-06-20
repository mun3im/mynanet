# mynanet-benchmarking

Peer-benchmark training for the MynaNet paper's `tab:comparison` (Table 8).
Trains MCU-class / lightweight-KWS architectures on the **same** MyGardenBird
12-class protocol used by 1j, so the published comparison is apples-to-apples
instead of literature-reported peer numbers on disjoint tasks.

## Layout

```
mynanet-benchmarking/
├── peer_architectures.py        # Tier-A model defs (Matchbox, DS-CNN-L, BC-ResNet, TC-ResNet)
├── run_peer_benchmarks.sh       # driver — same canonical CLI as 1j/4a-4e
├── 5a_matchboxnet_3x2x64.py     # *** to be created (fork 4a, swap arch) ***
├── 5b_dscnn_l_helloedge.py      # *** to be created ***
├── 5c_bcresnet8.py              # *** to be created ***
├── 5d_tcresnet14.py             # *** to be created ***
└── results_mygardenbird_5_<platform>/   # populated by the driver
```

## How to add a peer (Tier A)

Each `5*.py` follows the same self-contained pipeline as
`mobilenet-inspired/4a_squeezenet_v11.py`. To minimise drift, fork that file
and replace **only** the architecture function:

```bash
cp ../mobilenet-inspired/4a_squeezenet_v11.py 5b_dscnn_l_helloedge.py
```

In the copy, edit:

1.  Import: `from peer_architectures import build_dscnn_l_helloedge`
2.  Inside `main()`, replace
       `model = create_squeezenet_v11_64x300(...)`
    with
       `model = build_dscnn_l_helloedge(num_classes=n_classes, input_shape=(N_MELS, TIME_FRAMES, 1), dropout=args.dropout)`
3.  `output_dir_name` prefix → `5b_dscnn_l_helloedge` (must match `PREFIX_OF[5b]` in the driver).
4.  Output TFLite filename → e.g. `dscnn_l_int8.tflite`.

Smoke-test the architectures from this directory:

```bash
python3 peer_architectures.py
```

## How to run

```bash
# all Tier-A peers × 3 seeds
bash run_peer_benchmarks.sh

# only DS-CNN-L
bash run_peer_benchmarks.sh 5b

# include Tier-B (WrenNet, TinyChirp) if you've written the .py
bash run_peer_benchmarks.sh --tier-b
```

The driver is idempotent: any run whose result directory already contains
`training_report.txt` is skipped.

## Canonical CLI (identical to 1j and 4a-4e)

```
--splits_csv  /Volumes/Evo/MYGARDENBIRD/metadata16khz/splits_mip_80_10_10.csv
--flat_dir    /Volumes/Evo/MYGARDENBIRD/mygardenbird16khz
--dropout     0.05
--warmup_epochs 70
--mixup       0.2
--random_seed {42, 100, 786}
```

## Tier-A peers (drop-in fair comparisons)

| ID | Architecture | Native task | Source paper |
|----|--------------|-------------|--------------|
| 5a | MatchboxNet 3×2×64 | KWS | Majumdar & Ginsburg 2020 |
| 5b | DS-CNN-L (Hello Edge) | KWS | Zhang et al. 2017 |
| 5c | BC-ResNet-8 | KWS | Kim et al. 2021 |
| 5d | TC-ResNet-14-1.5 | KWS | Choi et al. 2019 |

## Tier-B peers (bioacoustic specialists; require porting effort)

| ID | Architecture | Friction |
|----|--------------|----------|
| 5e | WrenNet | Custom semi-learnable filterbank — verify TFLM op support before retraining |
| 5f | TinyChirp CNN-Time | Designed for binary; expand head to 12 classes |

## See also

- `Paper4_MynaNet_EcolInf/BENCHMARK_PEERS.md` — survey & selection rationale
- `Paper4_MynaNet_EcolInf/run_paper_revalidation.sh` — top-level driver
  (stage name: `peers`)
