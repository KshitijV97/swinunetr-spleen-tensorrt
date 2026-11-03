# SwinUNETR Spleen Segmentation → ONNX → TensorRT

[![tests](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml/badge.svg)](https://github.com/kshitijv97/swinunetr-spleen-tensorrt/actions/workflows/tests.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kshitijv97/swinunetr-spleen-tensorrt/blob/main/notebooks/colab_train_and_benchmark.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Train a **SwinUNETR** (Swin Transformer encoder + CNN decoder) to segment the
spleen in 3D abdominal CT scans with **MONAI**. Then export it to **ONNX**,
compile it with **TensorRT FP16**, and measure what that does to inference
**latency** and segmentation accuracy (**Dice**) on an NVIDIA T4 GPU.

The project is small on purpose: one dataset, one model, one deployment path,
each step explained and checked. The [`docs/`](docs/) folder explains every
design decision in plain language.

> **Status:** code, tests and the local smoke test are done. The full
> training and T4 benchmark numbers are **not filled in yet** (marked
> *TBD* below). They will come from running the Colab notebook, and nothing in
> this README is estimated or invented.

---

## Contents

1. [Motivation](#motivation-segmentation-for-surgical-robotics)
2. [What this project does](#what-this-project-does)
3. [Architecture overview](#architecture-overview)
4. [Results](#results)
5. [Quick start: local smoke test (Mac / CPU)](#quick-start-local-smoke-test-mac--cpu)
6. [Full run on Google Colab (T4)](#full-run-on-google-colab-t4)
7. [Repository layout](#repository-layout)
8. [Configuration](#configuration)
9. [Limitations](#limitations)
10. [Ideas for extension](#ideas-for-extension)
11. [Documentation index](#documentation-index)
12. [References and acknowledgements](#references-and-acknowledgements)

---

## Motivation: segmentation for surgical robotics

Robot-assisted and image-guided surgery depend on knowing **where the organs
are**. 3D segmentation of pre-operative CT feeds several parts of that loop:

- **Planning:** a patient-specific 3D model of target and at-risk organs,
  used to plan port placement, trajectories and safety margins.
- **Registration and navigation:** aligning pre-operative anatomy with the
  intra-operative view, so instruments can be shown relative to structures the
  camera cannot see.
- **Safety constraints:** virtual "no-go" boundaries (virtual fixtures)
  around delicate organs. The spleen is a good example: it is highly vascular,
  and accidental splenic injury is a recognised complication of some left
  upper-abdominal and colorectal procedures.

Two properties matter for this use:

1. **Accuracy of the 3D shape**, measured here with the Dice overlap score.
2. **Inference speed on the GPU that will actually run it.** A model that
   takes many seconds per scan is fine for offline planning but not for
   anything interactive. That is why this project goes past training and
   optimises the model with TensorRT, NVIDIA's deployment compiler (the same
   toolchain used on embedded Jetson GPUs in robots).

The spleen task is a deliberately well-understood, public benchmark, so the
effort goes into a correct, reproducible pipeline and an honest measurement
rather than into dataset wrangling.

## What this project does

```
MSD Spleen CT ──▶ MONAI preprocessing ──▶ SwinUNETR training ──▶ best.pt
                                                                  │
                     ┌────────────────────────────────────────────┤
                     ▼                                            ▼
          ONNX export (fixed 4×1×96³ input)              PyTorch FP32 / AMP
          + verification vs PyTorch                               │
                     │                                            │
                     ▼                                            │
          TensorRT FP16 engine ───────────▶ benchmark: latency per volume + Dice
                                                       results/benchmark.{json,md}
```

| Step | Script | Details |
|---|---|---|
| Download + split the data | `src/data.py` | 41 labelled CT volumes → 33 train / 8 validation (fixed seed) |
| Train | `src/train.py` | DiceCE loss, AdamW, cosine LR, AMP, sliding-window validation, best-checkpoint saving |
| Export | `src/export_onnx.py` | ONNX opset 18, fixed shape, checked against PyTorch with ONNX Runtime |
| Optimise + benchmark | `src/benchmark.py`, `src/trt_utils.py` | TensorRT FP16 engine. Compares PyTorch FP32, PyTorch AMP and TensorRT FP16 on mean/p95 latency per volume and Dice |

## Architecture overview

SwinUNETR ([Hatamizadeh et al., 2022](https://arxiv.org/abs/2201.01266))
replaces the CNN encoder of a U-Net with a **Swin Transformer**:

- The 96³ CT crop is cut into 2×2×2 patches → 48³ tokens.
- Four transformer stages compute **self-attention inside local 7×7×7
  windows**, alternating with **shifted windows** so information crosses
  window borders. Between stages, **patch merging** halves the resolution and
  doubles the channels (48³ → 24³ → 12³ → 6³ → 3³).
- A **convolutional decoder** upsamples back to 96³, with **skip connections**
  from every encoder stage, and outputs 2 logits per voxel (background / spleen).

With `feature_size = 48` the model has **62.2 M parameters**. Whole scans
are segmented with **sliding-window inference**: overlapping 96³ windows,
blended with Gaussian weights.

Full explanation with a shape diagram: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Results

> ⚠️ **TBD: run notebook.** Every value in this section is a placeholder until
> `notebooks/colab_train_and_benchmark.ipynb` has been run on a Colab T4. The
> notebook writes the real values to `results/`, and
> [docs/REPORTING_RESULTS.md](docs/REPORTING_RESULTS.md) says which number goes where.

### Training

| | Value |
|---|---|
| Train / validation volumes | 33 / 8 |
| Model | SwinUNETR, feature size 48, 62.2 M parameters |
| Epochs trained | TBD: run notebook |
| Training time (T4) | TBD: run notebook |
| Best validation Dice (PyTorch, during training) | TBD: run notebook |

<!-- After the Colab run, uncomment: ![Training curve](results/training_curve.png) -->

### Inference benchmark (NVIDIA T4, 8 validation volumes × 3 repeats)

Latency = sliding-window inference over one whole preprocessed volume (96³
windows, batch 4, 50 % overlap), after warm-up, with CUDA synchronisation.
Preprocessing and I/O are excluded. See [docs/BENCHMARKING.md](docs/BENCHMARKING.md).

| Backend | Mean latency / volume (ms) | p95 latency (ms) | Speed-up vs FP32 | Mean Dice |
|---|---:|---:|---:|---:|
| PyTorch FP32 | TBD: run notebook | TBD: run notebook | 1.00× | TBD: run notebook |
| PyTorch AMP (FP16 autocast) | TBD: run notebook | TBD: run notebook | TBD: run notebook | TBD: run notebook |
| TensorRT FP16 | TBD: run notebook | TBD: run notebook | TBD: run notebook | TBD: run notebook |

### Export check

| | Value |
|---|---|
| ONNX vs PyTorch max abs. difference (logits) | TBD: run notebook |
| ONNX vs PyTorch argmax agreement | TBD: run notebook |
| TensorRT engine build time / size | TBD: run notebook |
| Software (CUDA / PyTorch / TensorRT) | TBD: run notebook |

## Quick start: local smoke test (Mac / CPU)

The local run proves the **pipeline** works end to end on a laptop with no
NVIDIA GPU: a tiny model (`feature_size = 12`), **2 volumes** (1 train,
1 validation), 64³ crops, 2 epochs. It is not meant to produce a useful
model, and its numbers go to the git-ignored `results/smoke/`.

```bash
git clone https://github.com/kshitijv97/swinunetr-spleen-tensorrt.git
cd swinunetr-spleen-tensorrt

# Python 3.12 recommended (PyTorch wheels lag behind the newest Python releases)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Unit tests (synthetic data, ~30 s)
pytest -q

# Smoke test on 2 real volumes (downloads the 1.5 GB dataset on first run)
python -m src.train --smoke-test
python -m src.export_onnx --smoke-test
python -m src.benchmark --smoke-test

# ...or the same pipeline without any download, on synthetic volumes
python -m src.train --smoke-test --synthetic
python -m src.export_onnx --smoke-test
python -m src.benchmark --smoke-test --synthetic
```

On a Mac, `--device auto` uses the Apple GPU (MPS) when available. Add
`--device cpu` to force the CPU. The benchmark runs PyTorch FP32 only and
records AMP and TensorRT as *skipped: needs CUDA*; TensorRT is never
imported without CUDA.

## Full run on Google Colab (T4)

1. Click the **Open in Colab** badge at the top, or open
   `https://colab.research.google.com/github/kshitijv97/swinunetr-spleen-tensorrt/blob/main/notebooks/colab_train_and_benchmark.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. **Runtime → Run all.** The notebook clones this repo, installs pinned
   dependencies + TensorRT, runs the tests, downloads the data, trains,
   exports, benchmarks, plots the training curve, and offers `results.zip`
   for download. It can optionally save checkpoints to Google Drive so a
   disconnect does not lose the trained model.
4. Unzip into `results/`, paste the numbers into the [Results](#results)
   section, commit and push.

Step-by-step guide and troubleshooting: **[docs/COLAB.md](docs/COLAB.md)**.

The same commands work on any Linux machine with an NVIDIA GPU:

```bash
pip install -r requirements-colab.txt   # plus a CUDA build of torch; includes TensorRT 10
python -m src.data --download
python -m src.train
python -m src.export_onnx
python -m src.benchmark
```

## Repository layout

```
├── configs/default.yaml                 hyperparameters (+ smoke-test overrides)
├── src/
│   ├── data.py                          download, split, MONAI transforms, synthetic data
│   ├── model.py                         SwinUNETR construction, checkpoint loading
│   ├── evaluation.py                    sliding-window inference + Dice
│   ├── train.py                         training loop (CLI, --smoke-test)
│   ├── export_onnx.py                   ONNX export + verification
│   ├── trt_utils.py                     TensorRT engine build/run (CUDA only)
│   ├── benchmark.py                     PyTorch FP32 vs AMP vs TensorRT FP16
│   └── utils.py                         config, device, seeding, JSON helpers
├── tests/                               pytest: transforms, split, model shapes, TRT guard
├── notebooks/colab_train_and_benchmark.ipynb
├── docs/                                in-depth documentation (see index below)
├── results/                             real results, committed after the Colab run
├── requirements.txt                     pinned local (CPU/MPS) environment
└── requirements-colab.txt               pinned Colab extras (keeps Colab's CUDA torch)
```

Never committed (see `.gitignore`): the dataset (`data/`), checkpoints
(`*.pt`), ONNX files and TensorRT engines.

## Configuration

Everything lives in [`configs/default.yaml`](configs/default.yaml) and is
documented inline. Key defaults:

| Setting | Default | |
|---|---|---|
| Voxel spacing | 1.5 × 1.5 × 2.0 mm | resampled with bilinear (image) / nearest (label) |
| CT window | −57 … 164 HU → [0, 1] | soft-tissue window |
| Crop / window size | 96³ | 4 crops per volume per step, 1 : 1 spleen : background |
| Model width | `feature_size = 48` | 62.2 M parameters |
| Optimiser | AdamW, lr 1e-4, wd 1e-5, cosine schedule | |
| Loss | Dice + cross-entropy | |
| Epochs | 150, validation every 5 | |
| Inference | sliding window, batch 4, overlap 0.5, Gaussian blending | |

Common overrides go on the command line: `python -m src.train --epochs 80 --feature-size 24`.
Every setting is explained in [docs/TRAINING.md](docs/TRAINING.md).

## Limitations

Read these before quoting any number from this repository.

- **One small dataset.** 41 labelled CT volumes from a single institution
  and a single patient population (patients with liver metastases). Nothing
  here shows generalisation to other scanners, protocols, pathologies or
  modalities.
- **Validation, not test.** There is one fixed 33 / 8 split and no held-out
  test set. The best checkpoint is *selected* on the same 8 volumes that the
  Dice is reported on, which is slightly optimistic. Cross-validation would be
  more robust.
- **Dice in resampled space.** Dice is computed on the 1.5 × 1.5 × 2 mm
  resampled, foreground-cropped grid, not on each scan's native grid
  ([details](docs/DATA.md#5-where-dice-is-measured-an-honest-caveat)).
- **T4 benchmarks only.** All latencies come from one GPU type (a shared,
  virtualised Colab T4) with Colab's software versions on the day of the run.
  Absolute numbers and speed-ups will differ on other GPUs (A100, Jetson Orin,
  ...) and other TensorRT versions.
- **Model latency, not system latency.** Preprocessing, file I/O and
  post-processing are excluded from the timings.
- **Fixed input shape.** The ONNX/TensorRT model accepts exactly
  `4 × 1 × 96³`. That is all sliding-window inference needs, but it is not
  a general-purpose dynamic-shape engine.
- **No clinical validation.** This is a research and learning project, not a
  medical device, and it must not be used for clinical decisions.

## Ideas for extension

- 5-fold cross-validation and Dice in native space (`Invertd`), plus a surface
  metric (HD95).
- Self-supervised pre-trained SwinUNETR weights (Tang et al., 2022) and a
  comparison with a 3D U-Net baseline.
- INT8 TensorRT with calibration. A dynamic-batch optimisation profile.
- GPU-side preprocessing and end-to-end latency.
- Deployment on a Jetson Orin and/or as a ROS 2 node in a surgical-robotics simulator.
- Multi-organ segmentation (BTCV / AMOS / TotalSegmentator).

## Documentation index

| Document | What it covers |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | SwinUNETR from input to logits: patch embedding, (shifted) window attention, patch merging, decoder, shapes, parameter counts |
| [docs/DATA.md](docs/DATA.md) | The MSD Spleen dataset, a CT primer, every preprocessing transform and why, caching, synthetic data |
| [docs/TRAINING.md](docs/TRAINING.md) | Commands, every hyperparameter, loss, optimiser, AMP, sliding-window inference, Dice, troubleshooting |
| [docs/ONNX_TENSORRT.md](docs/ONNX_TENSORRT.md) | ONNX export decisions, verification, how TensorRT builds an engine, FP16 accuracy, the CUDA guard |
| [docs/BENCHMARKING.md](docs/BENCHMARKING.md) | Exactly what the latency numbers include and exclude, and how to read them |
| [docs/COLAB.md](docs/COLAB.md) | Step-by-step Colab guide and troubleshooting |
| [docs/REPORTING_RESULTS.md](docs/REPORTING_RESULTS.md) | Which numbers go into the README and a résumé, and how to phrase them honestly |
| [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md) | File-by-file tour of the code and tests |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Definitions of every technical term used |

## References and acknowledgements

- Hatamizadeh, A. et al. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.* BrainLes 2021 / arXiv:2201.01266.
- Tang, Y. et al. *Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis.* CVPR 2022.
- Liu, Z. et al. *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021.
- Antonelli, M. et al. *The Medical Segmentation Decathlon.* Nature Communications 13, 4128 (2022). Data: <http://medicaldecathlon.com> (CC-BY-SA 4.0).
- [MONAI](https://monai.io): model implementation, transforms and the spleen tutorial whose preprocessing defaults this project follows.
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) and [ONNX](https://onnx.ai).

Code released under the [MIT License](LICENSE). The dataset is not included
and keeps its own license.
